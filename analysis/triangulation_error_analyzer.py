import numpy as np
from slam.frontend.vision.feature_extractor import FeatureExtractor
from slam.frontend.vision.descriptor_matcher import DescriptorMatcher
from slam.frontend.io.image_sequence import ImageSequence
from slam.frontend.geometry.triangulation import Triangulation


class TriangErrorAnalyzer:
    """
    A wrapper for analyzing triangulation over a given range of images.
    """
    def __init__(self,
                 sequence: ImageSequence,
                 triangulator: Triangulation,
                 pixel_thresh=2.0,
                 disparity_min=1.0):
        self.seq = sequence
        self.triang = triangulator
        self.fe = FeatureExtractor()
        self.matcher = DescriptorMatcher()
        self.pixel_thresh  = pixel_thresh
        self.disparity_min = disparity_min


    def count_erroneous(self, pts3d: np.ndarray, min_z=0.0):
        """ Count points whose Z < min_z. """
        return int((pts3d[:,2] < min_z).sum())


    def stereo_inliers(self, matches, kpsL, kpsR):
        return [
            m for m in matches
            if abs(kpsL[m.queryIdx].pt[1] - kpsR[m.trainIdx].pt[1])
               <= self.pixel_thresh
            and (kpsL[m.queryIdx].pt[0] - kpsR[m.trainIdx].pt[0])
               >= self.disparity_min
        ]

    def erroneous_in_image_pair(self, frame_idx):
        """
        Given an image pair, triangulate a 3D-point and count erroneous points:
        1) load L/R
        2) detect, describe, match
        3) stereo-filter
        4) triangulate
        5) count negative-Z
        """
        imgL, imgR = self.seq[frame_idx]                # returns a (left, right) pair
        kpL, desL = self.fe.detect_and_compute(imgL)
        kpR, desR = self.fe.detect_and_compute(imgR)
        matchesLR = self.matcher.match(desL, desR)
        stereo_inl = self.stereo_inliers(matchesLR, kpL, kpR)
        pts3d = self.triang.triangulate_matches_batched(kpL, kpR, stereo_inl)
        return self.count_erroneous(pts3d)


    def scan_erroneous_range(self, low: int, high: int):
        """
        Run `erroneous_in_image_pair` on frames [low..high)
        and return the index with the maximum bad-point count + the full list.
        """
        results = [(i, self.erroneous_in_image_pair(i, self.pixel_thresh, self.disparity_min))
                   for i in range(low, high)]
        worst = max(results, key=lambda x: x[1])
        print(f"In frames {low}–{high-1}, frame {worst[0]} has the most bad points ({worst[1]}).")
        return worst, results
