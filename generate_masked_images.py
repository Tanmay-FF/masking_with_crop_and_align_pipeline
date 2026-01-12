
#!/usr/bin/env python3
"""
Single-process optimized version of the mask/augment pipeline.
Main optimizations performed:
- Removed multiprocessing / ProcessPoolExecutor: single process loop with tqdm progress.
- Initialize heavy resources (dlib detector & predictor) ONCE and reuse for all images.
- Reduced unnecessary deep copies and I/O where possible.
- Use OpenCV threading controls to avoid oversubscription.
- Simpler, clearer flow and robust logging to CSV.

Drop-in replacement for the original multiprocess script. Keep your params.yaml file path same.
"""

import os
import sys
import time
import yaml
import cv2
import csv
import copy
import zipfile
import numpy as np
from tqdm import tqdm

# if your project layout requires utils in parent folder, keep this
sys.path.append(os.path.abspath("../"))
from utils.aux_functions import mask_image  # only import what's needed

import dlib

# Color palette used previously
COLOR = [
    "#fc1c1a",
    "#177ABC",
    "#94B6D2",
    "#A5AB81",
    "#DD8047",
    "#6b425e",
    "#e26d5a",
    "#c92c48",
    "#6a506d",
    "#ffc900",
    "#ffffff",
    "#000000",
    "#49ff00",
]


def getConfig(path=None):
    path = path or r'params.yaml'
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded config from {path}")
        return cfg
    except Exception as e:
        print(f"Failed to load config file: {e}")
        raise


def zip_output_folder(folder_path: str) -> str:
    zip_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=folder_path)
                zipf.write(abs_path, arcname=rel_path)
    return zip_path


def listFiles(directory):
    filePaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filePaths.append(os.path.join(root, file))
    return filePaths


# Modified applyMask: expects detector & predictor preloaded in args
def applyMask(args, img_in, out_dir=None):
    """
    Apply mask to a single image (img_in is an ndarray already read by cv2).
    This version *reuses* args['detector'] and args['predictor'] which must be created once.
    Returns the masked image (single image) or None on failure.
    """
    # keep the API similar to your original mask_image wrapper
    if img_in is None:
        return None

    # mask_image in utils.aux_functions previously returned (masked_image, mask, mask_binary_array, original_image)
    # for a single image input it seems masked_image is either a list or ndarray; we adapt to return a single ndarray
    masked_image, mask, mask_binary_array, original_image = mask_image(img_in, args)

    if mask is None or masked_image is None:
        return None

    # If masked_image is a list (multiple faces), pick first one. If it's an ndarray, return directly.
    if isinstance(masked_image, (list, tuple)):
        if len(masked_image) == 0:
            return None
        return masked_image[0]
    return masked_image


def process_image_single(img_path, base_config, detector, predictor):
    """
    Process a single image path using preloaded detector/predictor.
    Returns a log entry dict and missed flag (True if face analysis/alignment failed).
    """
    config = base_config.copy()  # shallow copy is enough
    output_dir = os.path.join(config['output_dir'], os.path.split(os.path.relpath(img_path, os.path.split(config['input_dir'])[0]))[0])
    os.makedirs(output_dir, exist_ok=True)

    log_entry = {'image_path': img_path, 'mask_type': '', 'mask_success': '', 'error': ''}
    missed = False

    pipeline_image = cv2.imread(img_path)
    if pipeline_image is None:
        log_entry['error'] = 'cv2.imread returned None (file may be corrupted)'
        log_entry['mask_success'] = False
        return log_entry, True

    try:
        config['mask_type'] = np.random.choice(["surgical", "N95", "cloth"])
        config['color'] = np.random.choice(COLOR)
        config['mask_y_offset'] = int(np.random.randint(-10, 20))
        # config['mask_y_offset'] = -50
        config['mask_rotation_angle'] = int(np.random.randint(-5, 5))

        log_entry.update({'mask_type': config['mask_type'], 'mask_y_offset': config['mask_y_offset'],
                          'mask_rotation_angle': config['mask_rotation_angle']})

        # attach preloaded detector/predictor and other fields expected by mask_image
        config['detector'] = detector
        config['predictor'] = predictor
        config['model'] = config.get('model', '')  # keep for compatibility

        masked_image = applyMask(config, pipeline_image, output_dir)

        if masked_image is not None:
            config['mask_success'] = True
            output_filename = f"{os.path.basename(img_path).split('.')[0]}_augmented.png"
            output_path = os.path.join(output_dir, output_filename)
            # Use efficient PNG write params (if desired); default is fine
            cv2.imwrite(output_path, masked_image)
            config['output_path'] = output_path
        else:
            config['mask_success'] = False
            log_entry['error'] += "dlib unable to detect face in image to put mask on.\n"

    except Exception as e:
        config['mask_success'] = False
        log_entry['error'] += f"Mask Error: {e}\n"

    log_entry.update({'mask_success': config.get('mask_success', 'NA'), 'output_path': config.get('output_path', 'None')})
    return log_entry, missed


def process_images_single(config_path=None, verbose=True):
    start = time.time()
    base_config = getConfig(config_path)

    base_output_dir = base_config['output_dir']
    folder_path = base_config['input_dir']

    os.makedirs(base_output_dir, exist_ok=True)

    image_paths = listFiles(folder_path)
    total = len(image_paths)
    print(f"TOTAL IMAGES TO BE PROCESSED: {total}")

    # Open heavy resources once
    predictor_path = base_config.get('model')
    if not predictor_path or not os.path.exists(predictor_path):
        raise FileNotFoundError(f"dlib predictor model not found at: {predictor_path}")

    # Limit OpenCV threads to 1 to avoid oversubscription (single-process)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    log_entries = []
    missed_img = []

    # Write CSV header now
    csv_path = os.path.join(base_output_dir, "logs.csv")
    csv_fieldnames = ['image_path', 'mask_type', 'mask_y_offset', 'mask_rotation_angle', 'error', 'mask_success', 'output_path']
    csv_file_handle = open(csv_path, mode='a', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file_handle, fieldnames=csv_fieldnames)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writeheader()

    # Process images sequentially with progress bar
    for img_path in tqdm(image_paths, desc="Processing", total=total):
        log_entry, missed = process_image_single(img_path, base_config, detector, predictor)
        log_entries.append(log_entry)
        csv_writer.writerow(log_entry)
        csv_file_handle.flush()
        if missed:
            missed_img.append(log_entry['image_path'])

    csv_file_handle.close()

    if missed_img:
        missed_path = os.path.join(base_output_dir, 'missed_images.txt')
        with open(missed_path, 'a', encoding='utf-8') as f:
            for m in missed_img:
                f.write(m + "\n")

    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f}s")

    # Optionally zip outputs
    try:
        zip_path = zip_output_folder(base_output_dir)
        if verbose:
            print(f"Saved ZIP: {zip_path}")
        return zip_path
    except Exception as e:
        print(f"Failed to create zip: {e}")
        return base_output_dir


if __name__ == '__main__':
    # default: read params.yaml from the getConfig default location
    process_images_single()
