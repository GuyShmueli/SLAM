"""
slam/frontend/geometry/triangulation.py

Objective:
    Given
"""
import cv2
import numpy as np
from slam.frontend.io.camera_model import CameraModel
from slam.frontend.vision.descriptor_matcher import DescriptorMatcher


class Triangulation:
    """

    ---
    Attributes:
        cam      CameraModel
    """
    def __init__(self, cam: CameraModel,
                 disparity_min,
                 pixel_thresh):
        self.cam = cam
        # Precompute projection matrices
        self.P_left = cam.P_left
        self.P_right = cam.P_right
        self.disparity_min = disparity_min
        self.pixel_thresh = pixel_thresh


    ## Helpers for triangulate_matches_batched ##
    def _build_single_camera_A_matrix(self, x_pix, y_pix, camera_matrix):
        """
        Build the (2x4) linear block for ONE camera / ONE point.
        camera_matrix : (3,4)
        x_pix, y_pix  : scalars (pixel coordinates)
        """
        P1, P2, P3 = camera_matrix[0], camera_matrix[1], camera_matrix[2]
        first_row = (y_pix * P3) - P2
        second_row = P1 - (x_pix * P3)
        return np.vstack([first_row, second_row])  # (2,4)


    def _build_A_matrix(self, A1, A2):
        """Stack the two cameras’ (2x4) blocks  ->  (4x4) matrix. Still ONE point"""
        return np.vstack([A1, A2])  # (4,4)


    def triangulate_matches_batched(self, kps_left, kps_right, matches):
        """
        Compute the world 3D points corresponding to pairs of pixels, manually (least squares).
        """
        # Create a 3D-tensor, the first dim is the number of points
        N = len(matches)
        A_3d = np.empty((N, 4, 4), dtype=float)

        for k, m in enumerate(matches):
            uL, vL = kps_left[m.queryIdx].pt
            uR, vR = kps_right[m.trainIdx].pt

            A_left = self._build_single_camera_A_matrix(uL, vL, self.P_left)
            A_right = self._build_single_camera_A_matrix(uR, vR, self.P_right)

            A_for_point_k = self._build_A_matrix(A_left, A_right)
            A_3d[k] = A_for_point_k

        _, _, Vt = np.linalg.svd(A_3d)
        # For each point, we take the last row of Vt (which is equivalent to last col of V)
        Xh_opt = Vt[:, -1, :]  # (N,4)
        # Transforming from P^3 back to R^3, equivalent to:
        # X_opt = Xh_opt[:, :3] / Xh_opt[:, 3:4]  # (N,3)
        X_opt = np.squeeze(cv2.convertPointsFromHomogeneous(Xh_opt))
        return X_opt


    def triangulate_opencv(self, kps_left, kps_right, matches):
        """
        Compute the world 3D points corresponding to pairs of pixels,
        using OpenCV triangulation function.
        """
        # (2xN) matrices:  [[u0 u1 ... uN-1],
        #                   [v0 v1 ... vN-1]]
        pts_L, pts_R = DescriptorMatcher.extract_matched_pixels(kps_left, kps_right, matches)
        # OpenCV returns homogeneous coords (4xN)
        Xh_cv = cv2.triangulatePoints(self.P_left, self.P_right, pts_L.T, pts_R.T)  # (4xN)
        X_cv = (Xh_cv[:3] / Xh_cv[3]).T     # (Nx3)  in left-cam frame
        return X_cv


    def compare_triangulations(self, X_custom, X_cv):
        """Compare OpenCV with my SVD solution."""
        dists = np.linalg.norm(X_custom - X_cv, axis=1)
        print(f"Median L2 distance (custom vs OpenCV): {np.median(dists):.8f}")


    def create_inlier_outlier_matches(self, matches, kpR, kpL):
        """ Classify matches by whether the gap |vL−vR| <= pixel_thresh. """
        inlier_matches = []
        outlier_matches = []
        for m in matches:
            uL, vL = kpR[m.queryIdx].pt
            uR, vR = kpL[m.trainIdx].pt

            if abs(vR - vL) <= self.pixel_thresh and (uL - uR) >= self.disparity_min:
                inlier_matches.append(m)
            else:
                outlier_matches.append(m)

        return inlier_matches, outlier_matches
