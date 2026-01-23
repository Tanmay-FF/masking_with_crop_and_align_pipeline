import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
from skimage import transform as trans


class FaceCropper:
    # --------------------------------------------------------------------------
    # Base Configuration (Class Attributes)
    # --------------------------------------------------------------------------
    _TEMPLATE = np.float32([
        (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
        (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
        (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
        (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
        (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
        (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
        (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
        (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
        (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
        (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
        (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
        (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
        (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
        (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
        (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
        (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
        (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
        (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
        (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
        (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
        (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
        (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
        (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
        (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
        (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
        (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
        (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
        (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
        (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
        (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
        (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
        (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
        (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
        (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)
    ])
    _TPL_MIN = np.min(_TEMPLATE, axis=0)
    _TPL_MAX = np.max(_TEMPLATE, axis=0)
    _MINMAX_TEMPLATE = (_TEMPLATE - _TPL_MIN) / (_TPL_MAX - _TPL_MIN)
    _INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

    # --------------------------------------------------------------------------
    # Buffalo Style Configuration (Class Attributes)
    # --------------------------------------------------------------------------
    _BUFFALO_SCALE_RATIO = 0.65
    _BUFFALO_SHIFT_Y     = 0.16
    _BUFFALO_IMG_SIZE    = 112

    # Calculate Destination Points Once
    _base_pts = _MINMAX_TEMPLATE[_INNER_EYES_AND_BOTTOM_LIP].copy()
    _base_pts = (_base_pts - 0.5) * _BUFFALO_SCALE_RATIO + 0.5
    _base_pts[:, 1] += _BUFFALO_SHIFT_Y
    _BUFFALO_DST_PTS = _base_pts * _BUFFALO_IMG_SIZE
    # --------------------------------------------------------------------------

    def __init__(self, detector_path='assets/models/onnx_models/RFB_finetuned_with_postprocessing.onnx', 
                 landmark_path='assets/models/onnx_models/landmark_model.onnx'):
        
        self.so = ort.SessionOptions()
        # Try CUDA, fall back to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(detector_path, sess_options=self.so, providers=providers)
        self.session_lm = ort.InferenceSession(landmark_path, sess_options=self.so, providers=providers)
        
        self.transform = transforms.Compose([
            transforms.Resize((360, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def detect(self, img_bgr, threshold=0.5, iou_threshold=0.4):
        """
        Detects faces in the image.
        Returns boxes, probs, and inference time.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        transformed_image = self.transform(img_pil)
        image_array = np.expand_dims(np.array(transformed_image), axis=0).astype(np.float32)

        input_image_name = self.session.get_inputs()[0].name
        inputs = {
            input_image_name: image_array,
            'iou_threshold': np.array([iou_threshold], dtype=np.float32),
            'score_threshold': np.array([threshold], dtype=np.float32)
        }
        
        boxes, labels, probs = self.session.run(None, inputs)
        
        # Scale boxes back to original image size
        image_h, image_w, _ = img_bgr.shape
        scaled_boxes = []
        if boxes is not None:
            for bbox in boxes:
                x_min, y_min, x_max, y_max = map(float, bbox)
                x_min = x_min * image_w if x_min <= 1 else x_min
                y_min = y_min * image_h if y_min <= 1 else y_min
                x_max = x_max * image_w if x_max <= 1 else x_max
                y_max = y_max * image_h if y_max <= 1 else y_max
                scaled_boxes.append(np.array([x_min, y_min, x_max, y_max], dtype=np.float32))
                
        return scaled_boxes

    def _get_landmarks(self, img, bbox, margin):
        """
        Extracts landmarks for a given bbox and margin.
        Returns (landmarks_in_original_coords, face_crop) or None.
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox[:4].astype(int)
        
        # Apply specific margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        ch, cw = y2 - y1, x2 - x1
        scaler = np.array([ch, cw])
        
        if x1 >= x2 or y1 >= y2: return None
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0: return None

        crop_resized = cv2.resize(face_crop, (64, 64))
        crop_resized = np.expand_dims(crop_resized, axis=0).astype(np.uint8)
        
        inputs = {self.session_lm.get_inputs()[0].name: crop_resized}
        ort_outputs = self.session_lm.run(None, inputs)
        keypoints = np.array(ort_outputs).reshape(98, 2)
        
        landmarks = (keypoints * scaler) + (y1, x1)
        
        # Extract specific landmarks (INNER_EYES_AND_BOTTOM_LIP logic)
        landmarks_xy = []
        lm_cnt=0
        for y , x in landmarks:
                lm_cnt += 1
                if(lm_cnt==65 or lm_cnt==69 or lm_cnt==86):
                    landmarks_xy.append([x , y])
        
        return np.float32(landmarks_xy), face_crop

    def _align_buffalo(self, img, npLandmarks, face_crop):
        """
        Buffalo Style: Scale 0.65, Shift 0.16, Similarity Transform (skimage)
        """
        # InsightFace style aligner adapted for 3 points.
        # Uses SimilarityTransform (Rotation, Scale, Translation) but NO SHEAR.
        
        try:
            tform = trans.SimilarityTransform()
            tform.estimate(npLandmarks, self._BUFFALO_DST_PTS)
            M = tform.params[0:2, :]
            
            # Warp with REFLECTION to fix black pixels
            return cv2.warpAffine(img, M, (self._BUFFALO_IMG_SIZE, self._BUFFALO_IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
        except Exception as e:
            print(f"Alignment failed: {e}")
            return cv2.resize(face_crop, (112, 112))

    def _align_production_original(self, img, npLandmarks, face_crop):
        """
        Production Original: No extra scale/shift, Affine Transform (getAffineTransform)
        """
        landmarkIndices = self._INNER_EYES_AND_BOTTOM_LIP
        npLandmarkIndices = np.array(landmarkIndices)
        imgDim1, imgDim2 = 112, 112
        T = self._MINMAX_TEMPLATE[npLandmarkIndices].copy()
        T[:, 0] = imgDim1 * T[:, 0]
        T[:, 1] = imgDim2 * T[:, 1]
        
        try:
            H = cv2.getAffineTransform(npLandmarks, T)
            return cv2.warpAffine(img, H, (imgDim1, imgDim2))
        except Exception:
            return cv2.resize(face_crop, (112, 112))

    def _align_production_scale_shift(self, img, npLandmarks, face_crop):
        """
        Production w/ Scale & Shift: Scale 0.85, Shift -0.085, Similarity Transform
        """
        landmarkIndices = self._INNER_EYES_AND_BOTTOM_LIP
        npLandmarkIndices = np.array(landmarkIndices)
        
        T = self._MINMAX_TEMPLATE[npLandmarkIndices].copy()
        scale = 0.85
        shift_y = -0.085
        T = (T - 0.5) * scale + 0.5
        T[:, 1] += shift_y
        T[:, 0] *= 112
        T[:, 1] *= 112
        
        H = cv2.getAffineTransform(npLandmarks, T)
        return cv2.warpAffine(img, H, (112, 112))


    def align(self, img, bbox, style='all'):
        """
        Perform alignment based on style.
        style: 'buffalo', 'original_cropping', 'scale_shift', or 'all'
        Returns a dict of {style_name: cropped_img}
        """
        results = {}
        
        # Buffalo Style - Margin 0
        if style == 'buffalo' or style == 'all':
            res = self._get_landmarks(img, bbox, margin=0)
            if res is not None:
                npLandmarks, face_crop = res
                results['buffalo'] = self._align_buffalo(img, npLandmarks, face_crop)
        
        # Production Styles - Margin 0 (for both Original and Scale/Shift)
        if style in ['original_cropping', 'scale_shift', 'all']:
            # Only compute if we need original or scale_shift or both
            res = self._get_landmarks(img, bbox, margin=0)
            if res is not None:
                npLandmarks, face_crop = res
                
                if style == 'original_cropping' or style == 'all':
                    results['original_cropping'] = self._align_production_original(img, npLandmarks, face_crop)
                    
                if style == 'scale_shift' or style == 'all':
                    results['scale_shift'] = self._align_production_scale_shift(img, npLandmarks, face_crop)
            
        return results
