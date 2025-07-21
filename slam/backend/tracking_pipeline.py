"""
slam/backend/tracking_pipeline.py

"""
from slam.frontend.io.camera_model import CameraModel
from slam.frontend.io.image_sequence import ImageSequence
from slam.frontend.vision.descriptor_matcher import DescriptorMatcher
from slam.frontend.vision.feature_extractor import FeatureExtractor
from slam.frontend.geometry.triangulation import Triangulation
from slam.frontend.geometry.pnp import PnP
from slam.backend.tracking_database import TrackingDB
import cv2
import numpy as np


class TrackingPipeline:
    """

    """
    def __init__(self,
                 cam: CameraModel,
                 img_seq: ImageSequence,
                 akaze_thresh,
                 disparity_min,
                 pixel_thresh,
                 stereo_cross_check,
                 frame_cross_check    # rely on RANSAC-PnP test
                 ):
        self.cam = cam           # one source of truth
        self.img_seq = img_seq

        # Helpers
        self.feature_extractor = FeatureExtractor(threshold=akaze_thresh)
        self.stereo_matcher = DescriptorMatcher(cross_check=stereo_cross_check)
        self.frame_matcher = DescriptorMatcher(cross_check=frame_cross_check)
        self.triang = Triangulation(cam, disparity_min, pixel_thresh)   # pixels   ->  3-D
        self.pnp = PnP(cam, pixel_thresh)     # cam is for K, M_right   # (3-D,2-D) -> pose

        # Cache later in 'build'
        self.relative_Rts = None

    def build(self,
            max_frame=2_600,
            out_path="kitti05_tracks"):
        """ Build a TrackingDB over frames [0 .. max_frame-1] using AKAZE + stereo + RANSAC‐PnP. """
        # 0) Initialize empty DB and empty list to store relative poses
        db = TrackingDB()
        relative_Rts = []

        # 1) Frame 0: do stereo detection + inlier‐filter + triangulation
        img_L_prev, img_R_prev = self.img_seq[0]

        # Detect AKAZE keypoints + descriptors on left & right images of frame 0
        kp_L_prev, des_L_prev = self.feature_extractor.detect_and_compute(img_L_prev)
        kp_R_prev, des_R_prev = self.feature_extractor.detect_and_compute(img_R_prev)
        matches_LR_prev = self.stereo_matcher.match(des_L_prev, des_R_prev)

        inliers_LR_prev = self.triang.create_inlier_outlier_matches(
                                                matches_LR_prev, kp_L_prev, kp_R_prev)[0]

        # Triangulate all those stereo‐inliers to get a 3D cloud, in exactly the same order:
        # X_prev_unsorted[i]  <-  3D point for inliers_LR_prev[i]
        X_prev_unsorted = self.triang.triangulate_opencv(
            kp_L_prev, kp_R_prev,
            inliers_LR_prev
        )  # shape = (N_prev x 3), in the same order as inliers_LR_prev

        # Use create_links() to build (inlier_des_L_prev, links_prev) in exactly the same order as inliers_LR_prev:
        inlier_des_L_prev, links_prev = TrackingDB.create_links(
            des_L_prev,       # full descriptor array for left image 0
            kp_L_prev,        # keypoints on left image 0
            kp_R_prev,        # keypoints on right image 0
            inliers_LR_prev,  # list[cv2.DMatch] of just the stereo‐inliers
            inliers=None,     # since we only passed the inliers themselves, no extra mask is needed
            keep_match_order=True
        )
        # Now: inlier_des_L_prev[i]  <-->  links_prev[i]  <-->  X_prev_unsorted[i]

        # We can treat X_prev_unsorted as “X_prev” because it’s already aligned with inlier_des_L_prev
        X_prev = X_prev_unsorted.copy()  # shape = (N_prev x 3)

        # Add frame-0 into the DB (no “matches_to_previous” because it’s the first frame)
        db.add_frame(links_prev, inlier_des_L_prev, None)

        # 2) Loop over frames 1 … max_frame-1
        for idx in range(1, max_frame):
            # 2a. Load frame‐idx’s stereo pair, detect and compute descriptors
            img_L, img_R = self.img_seq[idx]
            kp_L, des_L = self.feature_extractor.detect_and_compute(img_L)
            kp_R, des_R = self.feature_extractor.detect_and_compute(img_R)
            matches_LR = self.stereo_matcher.match(des_L, des_R)
            # Filter stereo‐inliers (left vs. right)
            inliers_LR = self.triang.create_inlier_outlier_matches(
                matches_LR, kp_L, kp_R)[0]

            # 2b. Triangulate all those stereo‐inliers to get 3D points for this frame:
            X_curr_unsorted = self.triang.triangulate_opencv(
                kp_L, kp_R,
                inliers_LR
            )  # shape = (N_curr x 3), in the same order as inliers_LR

            # 2c. Build (inlier_des_L, links_curr) in exactly the same “inlier” order:
            inlier_des_L, links_curr = TrackingDB.create_links(
                des_L,       # full descriptor array for left image of frame idx
                kp_L,        # keypoints on left image idx
                kp_R,        # keypoints on right image idx
                inliers_LR,  # list[cv2.DMatch] for stereo‐inliers only
                inliers=None,
                keep_match_order=True
            )
            # Now X_curr_unsorted[i] corresponds exactly to inlier_des_L[i] <--> links_curr[i].

            X_curr = X_curr_unsorted.copy()

            # 2d. Match “previous‐frame inliers” <--> “current‐frame inliers” on the left descriptors:
            matches01 = self.frame_matcher.match(
                inlier_des_L_prev,  # descriptors from frame‐(idx-1) after stereo‐filter
                inlier_des_L,       # descriptors from frame‐idx after stereo‐filter
            )

            # Build an array of 3D points corresponding to each match’s “previous‐frame” end:
            matches01_idx_prev = [m.queryIdx for m in matches01]
            X_corr = X_prev[matches01_idx_prev]
            # shape = (len(matches01) × 3), aligned with `matches01`

            # 2e. Build “current‐frame” 2D pixel arrays for those same matches (needed by RANSAC‐PnP):
            # We need the left and right‐pixel coordinates in frame idx for each match01.
            inlier_idx_L = [m.queryIdx for m in inliers_LR]  # indices of left‐keypoints that survived stereo
            pix_L_curr, pix_R_curr = self.pnp.create_L_R_pixels(
                matches01,
                inliers_LR,
                inlier_idx_L,
                kp_L, kp_R,
                pair_0_or_1=1
            )
            # pix_L_curr.shape == (len(matches01) × 2), pix_R_curr likewise.

            # 2f. Run RANSAC‐PnP to filter out 3D -> 2D outliers:
            inlier_mask = self.pnp.ransac_pnp_pair1(
                X_corr,
                pix_L_curr,
                pix_R_curr,
            )

            # 2g. Refine estimated pose by using an iterative PnP solution
            Rt_prev_to_curr = self.pnp.solve_pnp(X_corr[inlier_mask],
                                            pix_L_curr[inlier_mask],
                                            cv2.SOLVEPNP_ITERATIVE)
            relative_Rts.append(Rt_prev_to_curr)    # add on-the-fly the poses

            # 2h. Add frame‐idx into the DB.
            db.add_frame(links_curr, inlier_des_L, matches01, inlier_mask)

            # 2i. Shift “current” -> “previous” for next iteration:
            X_prev = X_curr
            inlier_des_L_prev = inlier_des_L

        # 3) After the loop, save the DB to “kitti05_tracks.pkl”
        db.serialize(out_path)

        # 4) Cache the relative poses
        self.relative_Rts = np.stack(relative_Rts, axis=0)
        # return relative_Rts

    def get_relative_Rts(self):
        return self.relative_Rts
