# slam/frontend/vision/feature_extractor.py
import cv2
import numpy as np
from typing import Optional, List, Tuple


class FeatureExtractor:
    """
    Wrapper around AKAZE or SuperPoint feature extractor.
    """
    def __init__(self, threshold: float = 0.001,
                 detector_type: str = "akaze",
                 superpoint_cfg: Optional[dict] = None,
                 **kwargs):
        self.detector_type = detector_type.lower()
        if self.detector_type == "akaze":
            # classical AKAZE
            self.threshold = threshold
            self.detector = cv2.AKAZE_create(threshold=threshold, **kwargs)
        elif self.detector_type == "superpoint":
            # SuperPoint configuration
            from superpoint_superglue_deployment import SuperPointHandler
            if superpoint_cfg is None:
                superpoint_cfg = {}
            # allow passing nms_radius, keypoint_threshold, max_keypoints, etc.
            self.superpoint_handler = SuperPointHandler(superpoint_cfg)
        else:
            raise ValueError(f"Unsupported detector_type: {detector_type}")


    def detect_and_compute(
        self, image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect keypoints and compute descriptors.
        For AKAZE a mask may be provided. SuperPoint does not support masks.
        """
        if self.detector_type == "akaze":
            kps, descs = self.detector.detectAndCompute(image, mask)
            if descs is None:
                desc_length = self.detector.descriptorSize()
                descs = np.empty((0, desc_length), dtype=np.uint8)
            return kps, descs
        else:
            # superpoint expects single‑channel uint8 image
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kps, descs = self.superpoint_handler.detect_and_compute(image)
            # descs is an array of shape (num_kpts, 256)
            return kps, descs


    def set_threshold(self, threshold: float) -> None:
        """Adjust AKAZE threshold in-place."""
        if self.detector_type != "akaze":
            raise AttributeError("Threshold adjustment only applies to AKAZE.")
        try:
            self.detector.setThreshold(threshold)
        except AttributeError:
            raise AttributeError("Detector does not support threshold adjustment")
