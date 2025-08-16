"""
slam/backend/tracking_pipeline.py

"""
from slam.frontend.io.camera_model import CameraModel
from slam.frontend.io.image_sequence import ImageSequence
from slam.frontend.vision.descriptor_matcher import DescriptorMatcher
from slam.frontend.vision.feature_extractor import FeatureExtractor
from slam.frontend.geometry.triangulation import Triangulation
from slam.frontend.geometry.pnp import PnP
from slam.backend.tracking.database import TrackingDB

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt


class TrackingPipeline:
    """
    Builds a per-frame tracking database using stereo matching (L<->R),
    temporal matching (prev-L <-> curr-L), triangulation, and PnP.
    Supports AKAZE+BF or SuperPoint+SuperGlue. If args.use_gpu is True
    and CUDA is available, SuperPoint/SuperGlue will run on GPU.
    """

    def __init__(self,
                 cam: CameraModel,
                 img_seq: ImageSequence,
                 akaze_thresh,
                 disparity_min,
                 pixel_thresh,
                 stereo_cross_check,
                 frame_cross_check,
                 feature_type: str,
                 matcher_type: str,
                 plot: bool,
                 use_gpu=True):
        self.cam = cam
        self.img_seq = img_seq
        self.plot = plot
        self.relative_Rts = None  # cached in 'build'
        self.matcher_type = matcher_type

        # Decide GPU usage safely
        if feature_type=="superpoint" and use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[TrackingPipeline] CUDA not available, running on CPU...")
                    use_gpu = False
                print("CUDA is available, running on GPU...")
            except Exception:
                use_gpu = False

        # Feature extractor (AKAZE or SuperPoint)
        self.feature_extractor = FeatureExtractor(
            threshold=akaze_thresh,
            detector_type=feature_type,
            superpoint_cfg={
                'keypoint_threshold': 0.005,
                'max_keypoints': 2048,
                'use_gpu': use_gpu
            },
        )

        # Use separate matchers for stereo and temporal so cross_check settings don't leak
        sg_cfg = {'weights': 'outdoor', 'match_threshold': 0.2, 'use_gpu': use_gpu}
        self.stereo_matcher = DescriptorMatcher(
            cross_check=stereo_cross_check,
            matcher_type=matcher_type,
            superglue_cfg=sg_cfg,
        )
        self.frame_matcher = DescriptorMatcher(
            cross_check=frame_cross_check,
            matcher_type=matcher_type,
            superglue_cfg=sg_cfg,
        )

        # Geometric modules
        self.triang = Triangulation(cam, disparity_min, pixel_thresh)  # pixels -> 3-D
        self.pnp = PnP(cam, pixel_thresh)                              # (3-D,2-D) -> pose


    def build(self, max_frame=2_600, out_path="kitti05_tracks"):
        """
        Build a TrackingDB over frames [0 .. max_frame-1].
        - Stereo (L<->R) matching per frame -> triangulate to 3D
        - Temporal (prev-L <-> curr-L) matching on stereo-inlier subsets
        - RANSAC-PnP + refinement to estimate relative poses
        """
        db = TrackingDB()
        relative_Rts = []
        matches_per_frame = []   # for frames 1..N-1
        inliers_per_frame = []

        total_frames = min(max_frame, len(self.img_seq))
        if total_frames < 2:
            raise ValueError("Need at least 2 frames to build tracking database.")

        # ---- Frame 0: stereo detection + matching + triangulation ----
        img_L_prev, img_R_prev = self.img_seq[0]

        kp_L_prev, des_L_prev = self.feature_extractor.detect_and_compute(img_L_prev)
        kp_R_prev, des_R_prev = self.feature_extractor.detect_and_compute(img_R_prev)

        if self.matcher_type == 'superglue':
            matches_LR_prev = self.stereo_matcher.match(
                des_L_prev, des_R_prev,
                kpsL=kp_L_prev, kpsR=kp_R_prev,
                shapeL=img_L_prev.shape[:2], shapeR=img_R_prev.shape[:2],
            )
        else:
            matches_LR_prev = self.stereo_matcher.match(des_L_prev, des_R_prev)

        inliers_LR_prev = self.triang.create_inlier_outlier_matches(
            matches_LR_prev, kp_L_prev, kp_R_prev
        )[0]

        # Triangulate stereo inliers (order aligned with inliers_LR_prev)
        X_prev = self.triang.triangulate_opencv(kp_L_prev, kp_R_prev, inliers_LR_prev)

        # Create DB links and aligned descriptor subset for left-inliers
        inlier_des_L_prev, links_prev = TrackingDB.create_links(
            des_L_prev,        # full left descriptors (frame 0)
            kp_L_prev,         # left keypoints (frame 0)
            kp_R_prev,         # right keypoints (frame 0)
            inliers_LR_prev,   # stereo inliers (list of DMatch)
            inliers=None,
            keep_match_order=True
        )

        db.add_frame(links_prev, inlier_des_L_prev, None)

        # Cache prev-left inlier keypoints & shape for temporal matching
        kpsL_prev_inliers = [kp_L_prev[m.queryIdx] for m in inliers_LR_prev]
        shapeL_prev = img_L_prev.shape[:2]

        # ---- Loop over frames 1..total_frames-1 ----
        for idx in range(1, total_frames):
            img_L, img_R = self.img_seq[idx]

            # Feature extraction
            kp_L, des_L = self.feature_extractor.detect_and_compute(img_L)
            kp_R, des_R = self.feature_extractor.detect_and_compute(img_R)

            # Stereo matching (L <-> R) for current frame
            if self.matcher_type == 'superglue':
                matches_LR = self.stereo_matcher.match(
                    des_L, des_R,
                    kpsL=kp_L, kpsR=kp_R,
                    shapeL=img_L.shape[:2], shapeR=img_R.shape[:2],
                )
            else:
                matches_LR = self.stereo_matcher.match(des_L, des_R)

            # Stereo inliers for current frame
            inliers_LR = self.triang.create_inlier_outlier_matches(
                matches_LR, kp_L, kp_R
            )[0]

            # Triangulate current frame's stereo inliers (aligned to inliers_LR)
            X_curr = self.triang.triangulate_opencv(kp_L, kp_R, inliers_LR)

            # Build aligned descriptor subset and links for current frame (left inliers)
            inlier_des_L, links_curr = TrackingDB.create_links(
                des_L, kp_L, kp_R, inliers_LR,
                inliers=None, keep_match_order=True
            )

            # Prepare left inlier keypoints for current frame (same order as inlier_des_L)
            kpsL_curr_inliers = [kp_L[m.queryIdx] for m in inliers_LR]

            # ---- 2d. Temporal matching: prev-left-inliers <-> curr-left-inliers ----
            if self.matcher_type == 'superglue':
                matches01_compact = self.frame_matcher.match(
                    inlier_des_L_prev, inlier_des_L,
                    kpsL=kpsL_prev_inliers,        # prev-L inlier keypoints
                    kpsR=kpsL_curr_inliers,        # curr-L inlier keypoints
                    shapeL=shapeL_prev,            # prev-L image shape (H, W)
                    shapeR=img_L.shape[:2],        # curr-L image shape (H, W)
                )
            else:
                # BF matching on the stereo-inlier subsets (prev-L vs curr-L)
                matches01_compact = self.frame_matcher.match(inlier_des_L_prev, inlier_des_L)

            # --- Build 3D-2D correspondences for PnP from COMPACT matches ---
            # Indices into prev-left-inlier set (aligned with X_prev)
            matches01_idx_prev = [m.queryIdx for m in matches01_compact]
            X_corr = X_prev[matches01_idx_prev]

            # Current-frame 2D pixels (L and R) for those same temporal matches
            inlier_idx_L = [m.queryIdx for m in inliers_LR]  # indices of left keypoints that survived stereo (curr)

            pix_L_curr, pix_R_curr = self.pnp.create_L_R_pixels(
                matches01_compact, inliers_LR, inlier_idx_L, kp_L, kp_R, pair_0_or_1=1
            )

            # RANSAC PnP to filter out outliers (mask is COMPACT)
            inlier_mask_compact = self.pnp.ransac_pnp_pair1(
                X_corr, pix_L_curr, pix_R_curr
            )
            # Refine pose using iterative PnP on inliers
            Rt_prev_to_curr = self.pnp.solve_pnp(
                X_corr[inlier_mask_compact], pix_L_curr[inlier_mask_compact],
                cv2.SOLVEPNP_ITERATIVE
            )

            if self.matcher_type == 'superglue' or self.frame_matcher.cross_check:
                # --- Densify matches and mask to previous-frame length ---
                prev_n = len(inlier_des_L_prev)
                dense_matches = [cv2.DMatch(i, -1, 0, float("inf"))      # (queryIdx, trainIdx, imgIdx, distance)
                                 for i in range(prev_n)]
                dense_inliers = np.zeros(prev_n, dtype=bool)

                for k, m in enumerate(matches01_compact):
                    p = m.queryIdx  # index into prev inlier set
                    dense_matches[p] = m
                    if inlier_mask_compact[k]:
                        dense_inliers[p] = True
            else:
                dense_matches = matches01_compact
                dense_inliers = inlier_mask_compact

            # Log stats
            if self.plot:
                matches_per_frame.append(len(matches01_compact))
                inliers_per_frame.append(int(np.count_nonzero(inlier_mask_compact)))

            # Store current frame into DB (expects dense alignment to PREV)
            db.add_frame(links_curr, inlier_des_L, dense_matches, dense_inliers)

            # ---- Roll "current" -> "previous" for next iteration ----
            X_prev = X_curr
            inlier_des_L_prev = inlier_des_L
            kpsL_prev_inliers = kpsL_curr_inliers
            shapeL_prev = img_L.shape[:2]

            # Cache relative pose
            relative_Rts.append(Rt_prev_to_curr)

        # Optional plots
        if self.plot:
            db.plot_tracks_len_histogram()
            db.plot_connectivity()
            self.present_stats(db, matches_per_frame, inliers_per_frame)

        # Persist DB
        db.serialize(out_path)

        # Cache relative poses, save the DB to “kitti05_tracks.pkl”
        self.relative_Rts = np.stack(relative_Rts, axis=0)
        rel_path = "relative_Rts"
        with open(rel_path, 'wb') as f:
            pickle.dump(self.relative_Rts, f)


    def get_relative_Rts(self):
        return self.relative_Rts


    def present_stats(self, db: TrackingDB, matches_per_frame, inliers_per_frame):
        num_frames = db.frame_num()
        total_tracks = db.track_num()

        track_lengths = [db.track_length(tid) for tid in db.all_tracks()]
        mean_track_length = float(np.mean(track_lengths)) if track_lengths else 0.0

        links_by_frame = db.links_by_frame()
        links_per_frame = [len(links_by_frame.get(fid, {})) for fid in db.all_frames()]
        mean_frame_links = float(np.mean(links_per_frame)) if links_per_frame else 0.0

        # Print summary
        print("\n--- Tracking Statistics ---")
        print(f"Total number of tracks: {total_tracks}")
        print(f"Number of frames: {num_frames}")
        print(f"Mean track length (frames): {mean_track_length:.2f}")
        print(f"Mean number of frame links: {mean_frame_links:.2f}")

        # Prepare per-frame arrays aligned to frame IDs [0..N-1]
        matches_full = np.zeros(num_frames, dtype=int)
        matches_full[1:1 + len(matches_per_frame)] = np.array(matches_per_frame, dtype=int)

        mp = np.array(matches_per_frame, dtype=float)
        ip = np.array(inliers_per_frame, dtype=float)
        inlier_pct_full = np.zeros(num_frames, dtype=float)
        inlier_pct_full[1:1 + len(mp)] = 100.0 * (ip / np.maximum(1.0, mp))

        frames = np.arange(num_frames)
        mean_matches_per_frame = np.full(len(matches_full), matches_full.mean(), dtype=float)
        mean_inlier_pct_per_frame = np.full(len(matches_full), inlier_pct_full.mean(), dtype=float)

        # Plot number of matches per frame
        plt.figure(figsize=(10, 4))
        plt.plot(frames, matches_full, linewidth=1)
        plt.plot(frames, mean_matches_per_frame, linewidth=1, color='r')
        plt.xlabel("Frame ID"); plt.ylabel("# matches per frame")
        plt.title("Number of Matches per Frame")
        plt.grid(True); plt.tight_layout(); plt.show()

        # Plot percentage of inliers per frame
        plt.figure(figsize=(10, 4))
        plt.plot(frames, inlier_pct_full, linewidth=1)
        plt.plot(frames, mean_inlier_pct_per_frame, linewidth=1, color='r')
        plt.xlabel("Frame ID"); plt.ylabel("PnP inlier %")
        plt.title("Percentage of PnP Inliers per Frame")
        plt.ylim(0, 100)
        plt.grid(True); plt.tight_layout(); plt.show()
