# slam/backend/optimizers/bundle_adjustment.py
from collections import defaultdict
import gtsam
import numpy as np
from slam.frontend.io.camera_model import CameraModel
from slam.analysis.ba_plot import BAResults, BAAnalyzer
from slam.utils.graph_utils import (bundles_split_Rt, Rt_c2w_gtsam, invert_pose,
                                                 extract_intrinsic_param, single_backproject_gtsam,
                                                 Rts_to_homogenous)

class BundleAdjustment:
    """

    """
    def __init__(self,
                 cam: CameraModel,
                 Rt_curr_to_next,
                 links_by_frame,
                 plot: bool):
        self.cam = cam
        self.K_gtsam = gtsam.Cal3_S2Stereo(extract_intrinsic_param(cam.K, cam.M_right))
        self.pixel_noise = 4.0
        self.Rt_curr_to_next = Rt_curr_to_next
        self.links_by_frame = links_by_frame
        self.KF_indices = None      # to be cached in 'build()'
        self.bundle_frames = []
        self.plot = plot


    def _set_keyframe_criterion(self, t_rel):
        """ Label keyframes by accumulating translation until a threshold is exceeded. """
        # All the translation norms between consecutive frames (in meters)
        t_norms = np.linalg.norm(t_rel, axis=1)

        # Mean translation between consecutive frames
        # t_mean = np.mean(t_norms)  # obtained ~0.8 meters

        # We want 5-20 frames in each 'bundle window'
        # ~16 m between keyframe_i to keyframe_i+1 performed well
        KF_idx = 0
        fid_to_KF = {}
        fid_to_KF[0] = KF_idx
        accum_t_norm = 0
        for fid, t_norm in enumerate(t_norms):
            accum_t_norm += t_norm
            if accum_t_norm > 14:
                KF_idx += 1
                fid_to_KF[fid] = KF_idx
                accum_t_norm = 0
        return fid_to_KF    # frame_idx  ->  keyframe_idx


    def make_bundle_graph(self, bundle_fids,
                          links_cache,  # dict fid -> {tid:Link}
                          init_poses,  # dict fid -> Pose3
                          ):
        """Build a gtsam factor-graph + initial Values for one bundle-adjustment window."""
        obs_by_tid = defaultdict(dict)
        for fid in bundle_fids:  #  O(#frames in window)
            for tid, link in links_cache[fid].items():
                obs_by_tid[tid][fid] = link

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        fid0 = bundle_fids[0]
        c0 = gtsam.symbol('c', fid0)
        pose0 = init_poses[fid0]
        sigmas = np.array([np.deg2rad(0.5),  # roll: car almost flat
                           np.deg2rad(0.5),  # pitch: car almost flat
                           np.deg2rad(2.0),  # yaw: heading is less certain
                           0.25,  # x (lateral): can drift a bit
                           0.10,  # y (vertical): ground height well known
                           0.35])  # z (forward): can drift a bit


        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)

        graph.add(gtsam.PriorFactorPose3(
            c0, pose0, prior_noise))
        values.insert(c0, pose0)

        pose_used = {fid0}
        meas_noise = gtsam.noiseModel.Isotropic.Sigma(3, self.pixel_noise)

        for tid, obs in obs_by_tid.items():
            if len(obs) < 2:  # need >=2 frames for BA
                continue
            qk = gtsam.symbol('q', tid)
            fid_ref = max(obs)  # last obs - purely a heuristic
            link = obs[fid_ref]

            cam_ref = gtsam.StereoCamera(init_poses[fid_ref], self.K_gtsam)
            z_ref = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
            p_xyz = single_backproject_gtsam(cam_ref, z_ref)
            values.insert(qk, gtsam.Point3(p_xyz))

            for fid, lnk in obs.items():
                ck = gtsam.symbol('c', fid)
                pose_used.add(fid)
                z = gtsam.StereoPoint2(lnk.x_left, lnk.x_right, lnk.y)
                graph.add(gtsam.GenericStereoFactor3D(
                    z, meas_noise, ck, qk, self.K_gtsam))

        for fid in pose_used:
            ck = gtsam.symbol('c', fid)
            if not values.exists(ck):
                values.insert(ck, init_poses[fid])

        return graph, values



    def _bundle_window(self, T_curr_to_next, curr_KF):
        """For frames k0..k1 (k0 itself is I), map current camera coords to coords of frame k0."""
        k0 = self.KF_indices[curr_KF]
        k1 = self.KF_indices[curr_KF + 1] if curr_KF < len(self.KF_indices) - 1 else len(T_curr_to_next)
        T_window = [np.eye(4, dtype=T_curr_to_next.dtype)]
        running_T = np.eye(4, dtype=T_curr_to_next.dtype)
        for fid in range(k0 + 1, k1 + 1):
            running_T = T_curr_to_next[fid - 1] @ running_T  # accumulate k0->fid
            T_window.append(invert_pose(running_T))  # store fid->k0
        return T_window


    def build_windows(self):
        """
        Partition the trajectory into consecutive bundle‑adjustment windows
        (each rooted at a keyframe), convert every frame’s pose to its local
        window‑anchor coordinates, wrap the poses into 'gtsam.Pose3' objects,
        and construct a 'gtsam.NonlinearFactorGraph' plus matching initial
        'Values' container for each window.
        """
        t_curr_to_next = self.Rt_curr_to_next[:, :, 3]              # translation part
        T_curr_to_next = Rts_to_homogenous(self.Rt_curr_to_next)    # 4x4 pose

        fid_to_KF = self._set_keyframe_criterion(t_curr_to_next)
        KF_indices = list(fid_to_KF.keys())
        self.KF_indices = KF_indices        # cache 'KF_indices' for other functions
        T_bundles = [self._bundle_window(T_curr_to_next, kf_i) for kf_i, _ in enumerate(KF_indices)]

        self.bundle_frames = [list(range(KF_indices[kf_i], KF_indices[kf_i + 1] + 1))
                         for kf_i, _ in enumerate(KF_indices[:-1])]
        self.bundle_frames.append(list(range(KF_indices[-1], len(T_curr_to_next) + 1)))

        R_cam_to_window, t_cam_to_window = bundles_split_Rt(T_bundles)
        camera_poses_bundles = [Rt_c2w_gtsam(R, t, self.K_gtsam) for R, t in zip(R_cam_to_window, t_cam_to_window)]
        poses_bundles = [pose for cam, pose in camera_poses_bundles]
        fid_to_pose_bundles = []
        for frames, poses in zip(self.bundle_frames, poses_bundles):
            fid_to_pose_bundles.append({fid: pose for (fid, pose) in zip(frames, poses)})

        # Create graph & value objects for each bundle window
        graphs, values = [], []
        for frames, init_map in zip(self.bundle_frames, fid_to_pose_bundles):
            g, v = self.make_bundle_graph(frames, self.links_by_frame, init_map)
            graphs.append(g)
            values.append(v)
        return graphs, values


    def optimize_bundles(self, graphs, values):
        """Perform a Levenberg-Marquardt optimization to each bundle-adjustment window,
        returning a BAResults object with logs and the per-window optimized Values."""
        self.mean_init_err, self.mean_opt_err = [], []
        self.median_init_reproj, self.median_opt_reproj = [], []
        self.errs_L_by_frame, self.errs_R_by_frame = [], []

        ba_values = []
        for i, (graph, val) in enumerate(zip(graphs, values)):
            # Optimize
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, val)
            ba_val = optimizer.optimize()
            ba_values.append(ba_val)

            if self.plot:
                # 0) Call error-calculation static methods
                reproj_error_px = BAAnalyzer.reproj_error_px
                framewise_reproj_errors = BAAnalyzer.framewise_reproj_errors

                # 1) Error before optimization
                total_init_err = graph.error(val)
                n_keyframes = len(self.bundle_frames[i])
                mean_init_err = total_init_err / n_keyframes
                self.mean_init_err.append(mean_init_err)
                _, _, median_init_errs = reproj_error_px(graph, val, self.K_gtsam)
                self.median_init_reproj.append(median_init_errs)

                # 2) Errors after
                total_opt_err = graph.error(ba_val)
                mean_opt_err = total_opt_err / n_keyframes
                self.mean_opt_err.append(mean_opt_err)
                _, _, median_opt_errs = reproj_error_px(graph, ba_val, self.K_gtsam)
                self.median_opt_reproj.append(median_opt_errs)

                Lf, Rf = framewise_reproj_errors(graph, ba_val, self.bundle_frames[i], self.K_gtsam)
                self.errs_L_by_frame.append(Lf)
                self.errs_R_by_frame.append(Rf)

        return BAResults(
            KF_indices=list(self.KF_indices),
            bundle_frames=[list(fr) for fr in self.bundle_frames],
            mean_init_err=list(self.mean_init_err),
            mean_opt_err=list(self.mean_opt_err),
            median_init_reproj=list(self.median_init_reproj),
            median_opt_reproj=list(self.median_opt_reproj),
            errs_L_by_frame=[list(e) for e in self.errs_L_by_frame],
            errs_R_by_frame=[list(e) for e in self.errs_R_by_frame],
            graphs=list(graphs),
            init_values=list(values),
            ba_values=ba_values
        )

    def get_kf_indices(self):
        return self.KF_indices