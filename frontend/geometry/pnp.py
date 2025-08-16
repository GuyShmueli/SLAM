"""
slam/frontend/geometry/pnp.py

Objective:
    Given a set of landmarks (3D-points) and their corresponding measured pixels,
    estimate the pose of a camera.
"""
from slam.frontend.io.camera_model import CameraModel
from slam.utils.graph_utils import compute_err, homogenous, non_homogenous
import numpy as np
import cv2
import math

class PnP:
    """

    """
    def __init__(self,
                 cam: CameraModel,
                 pixel_thresh):
        self.K = cam.K
        self.M_left = cam.M_left
        self.M_right = cam.M_right
        self.P_left = cam.P_left
        self.P_right = cam.P_right

        self.pixel_thresh = pixel_thresh

    def _rotation_vec_to_mat(self, R_vec, t_vec):
        """ A helper that takes a vectoric rotation and translation
        and turns into an extrinsic matrix [R|t]. """
        R_mat, _ = cv2.Rodrigues(R_vec)     # 3x3
        return np.hstack((R_mat, t_vec))    # 3x4


    def solve_pnp(self, obj_pts, img_pts, flag):
        """ Perform a PnP algorithm. """
        is_ok, R_vec, t_vec = cv2.solvePnP(obj_pts, img_pts, self.K,
                                           None, flags=flag)
        extrinsic_mat = None
        if is_ok:
            extrinsic_mat = self._rotation_vec_to_mat(R_vec, t_vec)
        return extrinsic_mat


    def _convert_left_to_right(self, intra_matches):
        """ A helper that creates a dictionary mapping the left
        image's descriptor indices to the right ones."""
        L_to_R = {m.queryIdx: m.trainIdx for m in intra_matches}
        return L_to_R


    def create_L_R_pixels(self, matches01, intra_matches, intra_inlier_matches_indices, kps_L, kps_R, pair_0_or_1):
        """ Within the stereo pair, take just those matches that correspond
        to the obtained inter-pair matches. """
        # 1) build a map: left1_kp_idx  ->  right1_kp_idx
        L_to_R = self._convert_left_to_right(intra_matches)

        # 2) recover original left-1 kp index for each frame-to-frame match
        pix_L = []  # left image pixels list
        pix_R = []  # right image pixels list

        for m in matches01:
            # L0 indices that were both used to create 01
            if pair_0_or_1 == 0:
                L_index = intra_inlier_matches_indices[m.queryIdx]
            # L1
            elif pair_0_or_1 == 1:
                L_index = intra_inlier_matches_indices[m.trainIdx]
            else:
                raise ValueError("pair_0_or_1 should be either 0 or 1")

            # left pixel
            pix_L.append(kps_L[L_index].pt)

            # right pixel
            R_idx = L_to_R[L_index]
            pix_R.append(kps_R[R_idx].pt)

        pix_L = np.asarray(pix_L, dtype=np.float32)
        pix_R = np.asarray(pix_R, dtype=np.float32)

        return pix_L, pix_R


    def supporters_pair(self, meas_L, proj_L, meas_R, proj_R):
        """ Create a mask, containing 'True' for error lower than thresh for
          both left-right images. """
        err_L = compute_err(meas_L, proj_L)
        err_R = compute_err(meas_R, proj_R)
        return (err_L <= self.pixel_thresh) & (err_R <= self.pixel_thresh)


    @staticmethod
    def project(Xw, P):
        """ Project a world-point onto an image with camera matrix P. """
        Xw_h = np.column_stack([Xw, np.ones(len(Xw))])  # turn points homogenous (Nx4)
        pix = (P @ Xw_h.T).T
        return pix[:, :2] / pix[:, 2:3]  # Nx2


    def ransac_pnp_pair1(self, obj_pts,
                         meas_L1, meas_R1,
                         p_success=0.99999,    # desired global confidence
                         n_iter_max=2500,     # hard cap
                         early_inliers=None,  # stop when reached
                         ):
        """ Adaptive RANSAC-PnP: update the required iteration count
        as the inlier ratio estimate improves. """
        N = len(obj_pts)
        m = 4              # minimal sample size (AP3P)
        best_mask = None
        best_count = 0
        iter_done = 0
        M_right_h = homogenous(self.M_right)  # cache once

        # until we know the inlier ratio, assume the worst -> run n_iter_max
        n_iter_required = n_iter_max

        # rng = np.random.default_rng(12345)

        rng = np.random.default_rng()
        while iter_done < n_iter_required and iter_done < n_iter_max:
            iter_done += 1

            idx4 = rng.choice(N, m, replace=False)
            Rt_L1 = self.solve_pnp(obj_pts[idx4], meas_L1[idx4],
                                   cv2.SOLVEPNP_AP3P)
            if Rt_L1 is None:
                continue

            # --- reprojection ---
            P_L1 = self.K @ Rt_L1
            proj_L1 = PnP.project(obj_pts, P_L1)

            Rt_L1_h = homogenous(Rt_L1)                    # 4x4
            Rt_R1 = non_homogenous(M_right_h @ Rt_L1_h)    # 3x4
            P_R1 = self.K @ Rt_R1
            proj_R1 = PnP.project(obj_pts, P_R1)

            mask = self.supporters_pair(meas_L1, proj_L1,
                                   meas_R1, proj_R1)
            inliers = mask.sum()
            if inliers > best_count:
                best_count, best_mask = inliers, mask

                # --- adaptive update of required iterations ---
                inlier_ratio = best_count / N
                if inlier_ratio > 0:  # avoid log(0)
                    prob_hit_all_inliers = inlier_ratio ** m
                    # formula:  k ≥ log(1-p) / log(1 - w^m)
                    n_iter_required = math.ceil(
                        math.log(1 - p_success) /
                        math.log(1 - prob_hit_all_inliers)
                    )

            if early_inliers is not None and best_count >= early_inliers:
                break

        return best_mask



