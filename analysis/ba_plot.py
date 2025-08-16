# slam/analysis/ba_plot.py
from __future__ import annotations
from .optimizers_analysis import OptimizersPlot
from slam.utils.graph_utils import pose_nextKF_to_currKF, rel_to_vals
from dataclasses import dataclass
from typing import List
import numpy as np
import gtsam
import matplotlib.pyplot as plt


@dataclass
class BAResults:
    """Immutable results and logs from bundle adjustment."""
    KF_indices: List[int]                        # frame ids where each window is rooted
    bundle_frames: List[List[int]]               # frames per window (including root)
    mean_init_err: List[float]                   # mean factor error before BA per window
    mean_opt_err: List[float]                    # mean factor error after BA per window
    median_init_reproj: List[float]              # median stereo reprojection error before BA
    median_opt_reproj: List[float]               # median stereo reprojection error after BA
    errs_L_by_frame: List[List[float]]           # per-frame L image reprojection error post BA
    errs_R_by_frame: List[List[float]]           # per-frame R image reprojection error post BA
    graphs: List[gtsam.NonlinearFactorGraph]     # graphs per window
    init_values: List[gtsam.Values]              # initial values per window
    ba_values: List[gtsam.Values]                # optimized values per window


class BAAnalyzer:
    @staticmethod
    def reproj_error_px(graph, values, K_gtsam):
        """
        Median left-image reprojection error (px)  ||z_meas - pi(K[R|t]x_point)||
        over a 'GenericStereoFactor3D' in a single graph.
        """
        errs_L = []  # cache reprojection errors
        errs_R = []
        errs = []
        for j in range(graph.size()):
            f = graph.at(j)
            if isinstance(f, gtsam.GenericStereoFactor3D):   # skip Prior factor
                k_pose, k_point = f.keys()[0], f.keys()[1]
                cam = gtsam.StereoCamera(values.atPose3(k_pose), K_gtsam)  # Cal3_S2Stereo
                z = f.measured()  # graph holds the measurements (StereoPoint2 (uL, uR, v))
                zh = cam.project(values.atPoint3(k_point))  # StereoPoint2
                duL = z.uL() - zh.uL()
                duR = z.uR() - zh.uR()
                dv = z.v() - zh.v()
                errs_L.append(np.linalg.norm([duL, dv]))     # L2-norm
                errs_R.append(np.linalg.norm([duR, dv]))
                errs.append(np.linalg.norm([duL, duR, dv]))
        return errs_L, errs_R, np.median(errs)    # median of the reprojection errors in the window

    @staticmethod
    def framewise_reproj_errors(graph, values, frames, K_gtsam):
        # Collect L/R reprojection errors per frame id
        L_lists = {fid: [] for fid in frames}
        R_lists = {fid: [] for fid in frames}

        for j in range(graph.size()):
            f = graph.at(j)
            if isinstance(f, gtsam.GenericStereoFactor3D):
                k_pose, k_point = f.keys()[0], f.keys()[1]
                fid = gtsam.Symbol(k_pose).index()  # recover frame id

                cam = gtsam.StereoCamera(values.atPose3(k_pose), K_gtsam)
                z = f.measured()
                zh = cam.project(values.atPoint3(k_point))

                L_lists[fid].append(np.hypot(z.uL() - zh.uL(), z.v() - zh.v()))
                R_lists[fid].append(np.hypot(z.uR() - zh.uR(), z.v() - zh.v()))

        L_per_frame = [np.median(L_lists[fid]) if L_lists[fid] else np.nan for fid in frames]
        R_per_frame = [np.median(R_lists[fid]) if R_lists[fid] else np.nan for fid in frames]
        return L_per_frame, R_per_frame

    @staticmethod
    def poses_next_to_curr(vals, KF_indices):
        """Relative poses for every KF pair."""
        KF_idx_pairs = [
                        (KF_indices[i], KF_indices[i + 1])
                        for i in range(len(KF_indices) - 1)
                       ]

        rel_poses =  [
                      pose_nextKF_to_currKF(k0, k1, res)
                      for (k0, k1), res in zip(KF_idx_pairs, vals)
                     ]
        return rel_poses


class BA_Plotter(OptimizersPlot):
    """BA-specific plots layered on top of the shared OptimizersPlot base."""
    def __init__(self, res, poses_gt, Rts_abs_all):
        self.val_poses = rel_to_vals(BAAnalyzer.poses_next_to_curr(res.ba_values,
                                                                   res.KF_indices))
        super().__init__(self.val_poses, res.KF_indices, poses_gt,
                         Rts_abs_all,"BA")
        self.res = res

    # --- API ---
    def mean_error_per_bundle(self):
        x = list(self.res.KF_indices)
        if not self.res.mean_init_err or not self.res.mean_opt_err:
            raise RuntimeError("No error logs - run optimize_bundles with plotting enabled.")
        plt.figure(figsize=(8, 5))
        plt.plot(x, self.res.mean_init_err, label="Initial Error")
        plt.plot(x, self.res.mean_opt_err, label="Optimized Error")
        plt.xlabel("Bundle Starting at Frame Idx")
        plt.ylabel("Mean Factor Error")
        plt.title("Mean Factor Error Before and After BA")
        plt.legend(); plt.tight_layout(); plt.show()

    def median_error_per_bundle(self):
        x = list(self.res.KF_indices)
        if not self.res.median_init_reproj or not self.res.median_opt_reproj:
            raise RuntimeError("No reprojection logs - run optimize_bundles with plotting enabled.")
        plt.figure(figsize=(8, 5))
        plt.plot(x, self.res.median_init_reproj, label="Initial Error")
        plt.plot(x, self.res.median_opt_reproj, label="Optimized Error")
        plt.ylim(bottom=0)
        plt.xlabel("Bundle Starting at Frame Idx")
        plt.ylabel("Median Factor Error")
        plt.title("Median Factor Error Before and After BA")
        plt.legend(); plt.tight_layout(); plt.show()

    def reproj_vs_distance(self, window_idx=None):
        if window_idx is None:
            window_idx = max(range(len(self.res.bundle_frames)),
                             key=lambda i: len(self.res.bundle_frames[i]))
        frames = self.res.bundle_frames[window_idx]
        x = list(range(len(frames)))
        yL = np.asarray(self.res.errs_L_by_frame[window_idx], dtype=float)
        yR = np.asarray(self.res.errs_R_by_frame[window_idx], dtype=float)
        plt.figure(figsize=(12, 6))
        plt.plot(x, yL, label='Left'); plt.plot(x, yR, label='Right')
        plt.xlabel("Distance from keyframe")
        plt.ylabel("BA reprojection error (L$_2$ px)")
        plt.title("BA reprojection error vs. distance from reference")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()