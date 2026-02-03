
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
import os
import sys
import time
import cv2
import csv
import copy
import numpy as np
import argparse
from tqdm import tqdm
from cropping_utils import FaceCropper
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

    if img_in is None:
        return None

    masked_image, mask, mask_binary_array, original_image = mask_image(img_in, args)

    if mask is None or masked_image is None:
        return None

    if isinstance(masked_image, (list, tuple)):
        if len(masked_image) == 0:
            return None
        return masked_image[0]
    return masked_image


def process_image_single(img_path, base_config, detector, predictor, face_cropper, augment_idx=0, do_mask=True, do_crop=True):
    """
    Process a single image path using preloaded detector/predictor/face_cropper.
    Supports toggling masking and cropping, and handling augmentation index.
    """
    config = base_config.copy()  # shallow copy is enough
    
    # Determine output directory structure
    rel_path = os.path.relpath(img_path, start=base_config['input_root'])

    if rel_path.startswith('..'): 
        # Fallback if relative path calculation fails (e.g. cross-drive or outside root)
        rel_path = os.path.basename(img_path)
        
    sub_dir = os.path.split(rel_path)[0]
    output_dir = os.path.join(config['output_dir'], sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_entry = {'image_path': img_path, 'mask_type': 'NA', 'mask_success': 'NA', 'error': '', 'augmentation': augment_idx}
    missed = False

    pipeline_image = cv2.imread(img_path)
    if pipeline_image is None:
        log_entry['error'] = 'cv2.imread returned None (file may be corrupted)'
        log_entry['mask_success'] = False
        return log_entry, True

    try:
        # ----------------------------------------------------------------------
        # 1. VISUAL AUGMENTATION / MASKING
        # ----------------------------------------------------------------------
        current_image = pipeline_image
        
        if do_mask:
            config['mask_type'] = np.random.choice(["surgical", "N95", "cloth"])
            config['color'] = np.random.choice(COLOR)
            config['mask_y_offset'] = int(np.random.randint(-10, 20))
            config['mask_rotation_angle'] = int(np.random.randint(-5, 5))
            
            log_entry.update({'mask_type': config['mask_type'], 'mask_y_offset': config['mask_y_offset'],
                              'mask_rotation_angle': config['mask_rotation_angle']})

            # attach preloaded detector/predictor
            config['detector'] = detector
            config['predictor'] = predictor
            config['model'] = config.get('model', '')

            masked_result = applyMask(config, pipeline_image, output_dir)
            
            if masked_result is not None:
                current_image = masked_result
                config['mask_success'] = True
            else:
                config['mask_success'] = False
                log_entry['error'] += "Mask application failed. "
                return log_entry, True
        else:
            log_entry['mask_type'] = "None (Disabled)"
            config['mask_success'] = "Skipped"

        # Save the intermediate (or final) full image
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        aug_suffix = f"_aug{augment_idx}" if augment_idx >= 0 else ""
        
        if do_mask:
             base_filename = f"{base_name}{aug_suffix}_masked"
             output_path = os.path.join(output_dir, f"{base_filename}.png")
             cv2.imwrite(output_path, current_image)
             config['output_path'] = output_path
        elif not do_crop:
             # If no mask AND no crop, save original with suffix
             base_filename = f"{base_name}{aug_suffix}_original"
             output_path = os.path.join(output_dir, f"{base_filename}.png")
             cv2.imwrite(output_path, current_image)
             config['output_path'] = output_path
        
        # ----------------------------------------------------------------------
        # 2. CROPPING & ALIGNMENT
        # ----------------------------------------------------------------------
        if do_crop:
            crop_style_req = base_config.get('crop_style', 'all')
            
            # Detect faces in the current image (masked or original)
            boxes = face_cropper.detect(current_image)
            
            if boxes:
                for i, bbox in enumerate(boxes):
                    aligned_results = face_cropper.align(current_image, bbox, style=crop_style_req)
                    
                    for style_name, cropped_img in aligned_results.items():
                        suffix = f"_{style_name}"
                        if len(boxes) > 1: suffix += f"_face{i}"
                        
                        # logic to keep filename clean
                        save_name = f"{base_name}{aug_suffix}{suffix}.jpg"
                        crop_save_path = os.path.join(output_dir, save_name)
                        cv2.imwrite(crop_save_path, cropped_img)
                        
                if not log_entry.get('output_path'):
                     log_entry['output_path'] = "Cropped output(s)"
            else:
                log_entry['error'] += "Face detection failed for cropping. "
                missed = True

    except Exception as e:
        config['mask_success'] = False
        log_entry['error'] += f"Processing Error: {e}\n"
        missed = True

    log_entry.update({'mask_success': config.get('mask_success', 'NA'), 'output_path': config.get('output_path', 'None')})
    return log_entry, missed


def run_pipeline(args):
    if args.n_augmentations < 1:
        print("*"*100)
        print("Number of augmentations cannot be less than 1. Exiting...")
        print("*"*100)
        exit()
    start = time.time()

    # --------------------------------------------------------------------------
    # HARDCODED DEFAULTS (DO NOT CHANGE)
    # --------------------------------------------------------------------------
    base_config = {
        'verbose': False,
        'code': '',            # required by aux_functions
        'pattern': 'random',   # required by aux_functions logic
        'pattern_weight': 0.5,
        'color_weight': 0.5,
        'model': "assets/models/shape_predictor_68_face_landmarks.dat"
    }
    # --------------------------------------------------------------------------
    
    # Override/Extend with CLI args
    base_config['output_dir'] = args.output
    base_config['crop_style'] = args.crop_style
    
    # Determine Input Mode (File vs Folder)
    if os.path.isfile(args.input):
        image_paths = [args.input]
        base_config['input_root'] = os.path.dirname(args.input) # treat parent dir as root for single file
    elif os.path.isdir(args.input):
        image_paths = listFiles(args.input)
        base_config['input_root'] = args.input
    else:
        print(f"Error: Input {args.input} not found.")
        return

    total_images = len(image_paths)
    
    actual_augmentations = 1 if args.no_mask else args.n_augmentations
    
    total_ops = total_images * actual_augmentations
    print(f"PIPELINE STARTED")
    print("*"*60)
    print(f"Input: {args.input}")
    print(f"Images Found: {total_images}")
    print(f"Augmentations per Image: {args.n_augmentations}")
    print(f"Masking Enabled: {not args.no_mask}")
    print(f"Cropping Enabled: {not args.no_crop}")
    print(f"Crop Style: {args.crop_style}")
    print(f"Total Operations: {total_ops}")
    print("*"*60)
    os.makedirs(base_config['output_dir'], exist_ok=True)

    # Open heavy resources once
    predictor_path = base_config['model']
    if not os.path.exists(predictor_path):

        if not os.path.exists(predictor_path):
             raise FileNotFoundError(f"dlib predictor model not found at: {predictor_path}")

    # Limit OpenCV threads
    try: cv2.setNumThreads(1)
    except: pass

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    print("Initializing ONNX FaceCropper...")
    face_cropper = FaceCropper() 

    # CSV Logging
    csv_path = os.path.join(base_config['output_dir'], "logs.csv")
    csv_fieldnames = ['image_path', 'mask_type', 'mask_y_offset', 'mask_rotation_angle', 'error', 'mask_success', 'output_path', 'augmentation']
    
    # Initialize CSV if new
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    csv_file_handle = open(csv_path, mode='a', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file_handle, fieldnames=csv_fieldnames)
    if not file_exists:
        csv_writer.writeheader()

    missed_img = []

    # Main Loop
    with tqdm(total=total_ops, desc="Processing") as pbar:
        for img_path in image_paths:
            
            for i in range(actual_augmentations):
                log_entry, missed = process_image_single(
                    img_path=img_path, 
                    base_config=base_config, 
                    detector=detector, 
                    predictor=predictor, 
                    face_cropper=face_cropper,
                    augment_idx=i,
                    do_mask=(not args.no_mask),
                    do_crop=(not args.no_crop)
                )
                
                csv_writer.writerow(log_entry)
                if missed:
                    missed_img.append(f"{img_path} (Aug {i})")
                
                pbar.update(1)
            

            
            csv_file_handle.flush()

    csv_file_handle.close()

    if missed_img:
        missed_path = os.path.join(base_config['output_dir'], 'missed_images.txt')
        with open(missed_path, 'a', encoding='utf-8') as f:
            for m in missed_img:
                f.write(m + "\n")

    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f}s")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mask Augmentation & Face Cropping Pipeline")
    
    parser.add_argument('--input', type=str, required=True, help="Input image file or directory")
    parser.add_argument('--output', type=str, default='output', help="Output directory")
    parser.add_argument('--n_augmentations', type=int, default=1, help="Number of variations per image")
    parser.add_argument('--no_mask', action='store_true', help="Disable masking")
    parser.add_argument('--no_crop', action='store_true', help="Disable cropping and alignment")
    parser.add_argument('--crop_style', type=str, nargs='+', default=['all'], choices=['buffalo', 'buffalo_chin', 'original_cropping', 'scale_shift', 'all'], help="Style of cropping to apply")

    args = parser.parse_args()
    
    run_pipeline(args)
