# slam/frontend/vision/descriptor_matcher.py
import numpy as np
import cv2
class DescriptorMatcher:
    """
    Given 2 descriptors, perform matching.
    Centralizes all of feature‐descriptor matching logic.
    Supports both classical matchers and 'SuperGlue' neural-network matcher.
    """
    def __init__(self, cross_check, matcher_type="bf", superglue_cfg=None):
        self.matcher_type = matcher_type.lower()
        self.cross_check = cross_check
        if self.matcher_type == "bf":
            self._bf = None
            self._bf_norm = None
        elif self.matcher_type == "superglue":
            from superpoint_superglue_deployment import SuperGlueHandler, SuperPointHandler
            self.superglue_handler = SuperGlueHandler(superglue_cfg or {})
            self.sp_handler = SuperPointHandler({'use_gpu': (superglue_cfg or {}).get('use_gpu', False)})
        else:
            raise ValueError(f"Unsupported matcher_type: {matcher_type}")

    def _get_bf(self, desL):
        norm = cv2.NORM_HAMMING if desL.dtype == np.uint8 else cv2.NORM_L2
        if (self._bf is None) or (self._bf_norm != norm):
            self._bf = cv2.BFMatcher(norm, crossCheck=self.cross_check)
            self._bf_norm = norm
        return self._bf

    def match(self, desL, desR, kpsL=None, kpsR=None, shapeL=None, shapeR=None):
        if self.matcher_type == "bf":
            bf = self._get_bf(desL)
            matches = bf.match(desL, desR)
            return sorted(matches, key=lambda x: x.distance)
        else:
            if kpsL is None or kpsR is None or shapeL is None or shapeR is None:
                raise ValueError("SuperGlue requires keypoints and image shapes.")
            pred0 = self.sp_handler.to_prediction(kpsL, desL)
            pred1 = self.sp_handler.to_prediction(kpsR, desR)
            matches = self.superglue_handler.match(pred0, pred1, shapeL, shapeR)
            return sorted(matches, key=lambda x: x.distance)  # keep behavior consistent


    @staticmethod
    def extract_matched_pixels(kpsL, kpsR, matches):
        """Turn a list of DMatch into two (N,2) arrays of pixel coordinates."""
        pix1, pix2 = [], []
        for m in matches:
            pix1.append(kpsL[m.queryIdx].pt)
            pix2.append(kpsR[m.trainIdx].pt)
        return (np.array(pix1, dtype=np.float32),
                np.array(pix2, dtype=np.float32))


    @staticmethod
    def extract_matched_keypoints(matches, kpsL, kpsR):
        """Return two parallel lists of cv2.KeyPoint corresponding to each match."""
        return ([kpsL[m.queryIdx] for m in matches],
                [kpsR[m.trainIdx] for m in matches])
