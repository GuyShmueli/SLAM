"""
slam/frontend/vision/feature_extractor.py

Objective:
    Given an image, perform feature-detection and descriptor computation for stereo SLAM.
"""
import cv2
import numpy as np
from typing import Optional

class FeatureExtractor:
    """
    Wrapper around OpenCV Feature2D Accelerated-KAZE (AKAZE) algorithm

    ---
    Attributes:
    threshold
    detector
    """
    def __init__(self, threshold,
                 **kwargs           # any other parameters for AKAZE-detector
                 ):

        self.threshold = threshold
        self.detector = cv2.AKAZE_create(threshold=threshold, **kwargs)


    def detect_and_compute(self, image,
                           mask: Optional[np.ndarray] = None  # optional binary mask to specify where to detect
                           ):
        """ Detect keypoints and compute descriptors on a single image. """
        kps, descs = self.detector.detectAndCompute(image, mask)
        if descs is None:
            # No keypoints found: return empty descriptor array
            desc_length = self.detector.descriptorSize()
            dtype = np.uint8
            descs = np.empty((0, desc_length), dtype=dtype)   # shape (0, d)
        return kps, descs


    def set_threshold(self, threshold):
        """ Adjust AKAZE threshold in-place. """
        try:
            self.detector.setThreshold(threshold)
        except AttributeError:                # method does not exist
            raise AttributeError("Detector does not support threshold adjustment")
        except cv2.error as e:                # invalid value etc.
            raise ValueError(f"Failed to set threshold={threshold}") from e
