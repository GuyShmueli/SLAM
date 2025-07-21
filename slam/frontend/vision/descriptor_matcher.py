"""
slam/frontend/vision/descriptor_matcher.py

Objective:
    Given 2 descriptors, perform matching.
    Centralizes all of feature‐descriptor matching logic.
"""
import cv2
import numpy as np

class DescriptorMatcher:
    """
    Wraps OpenCV descriptor matching and provides helpers for
    ratio tests and stereo-pixel extraction.

    Attributes:
      matcher: cv2.BFMatcher instance (Hamming + optional crossCheck)
    """
    def __init__(self, cross_check=True):
        """
        Args:
          cross_check: if True, only mutual best matches are returned.
        """
        # Because of the binary character of each des1[i], des2[j],
        # it's natural to choose Hamming Distance (which Xors them and then sums up the Xored elemented)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)


    def match(self, desL, desR):
        """ Return a list of DMatch sorted by increasing distance. """
        matches = self.matcher.match(desL, desR)
        return sorted(matches, key=lambda x: x.distance)


    def knn_match(self, desL, desR, k):
        """ Return a list (len=des1) of up to k matches for each descriptor in des1. """
        knn_matches = self.matcher.knnMatch(desL, desR, k=k)
        return knn_matches


    def ratio_test(self, knn_matches, ratio=0.75):
        """
        Lowe’s ratio test for filtering ambiguous matches.

        Args:
          knn_matches: output of knn_match(des1, des2, k=2)
          ratio: keep matches where best_distance < ratio * second_best_distance

        Returns:
          (good_matches, bad_matches)
        """
        good, bad = [], []
        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n[0], m_n[1]
            if m.distance < ratio * n.distance:
                good.append(m)
            else:
                bad.append(m)
        return good, bad


    @staticmethod
    def extract_matched_pixels(kpsL, kpsR, matches):
        """ Turn a list of DMatch into two (Nx2) arrays of pixel coords. """
        pix1, pix2 = [], []
        for m in matches:
            pix1.append(kpsL[m.queryIdx].pt)
            pix2.append(kpsR[m.trainIdx].pt)
        return (np.array(pix1, dtype=np.float32),
                np.array(pix2, dtype=np.float32))
    
    
    @staticmethod
    def extract_matched_keypoints(matches, kpsL, kpsR):
        """ Return two parallel lists of KeyPoints corresponding to each match. """
        return ([kpsL[m.queryIdx] for m in matches],
                [kpsR[m.trainIdx] for m in matches])
