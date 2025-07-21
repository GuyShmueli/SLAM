"""

"""
from collections import defaultdict
import gtsam
import numpy as np
from slam.frontend.io.camera_model import CameraModel
from slam.backend.optimizers.graph_utils import (bundles_split_Rt, Rt_c2w_gtsam, invert_pose,
                                                 extract_intrinsic_param, single_backproject_gtsam,
                                                 Rts_to_homogenous)

class BundleAdjustment:
    """

    """
    def __init__(self,
                 cam: CameraModel,
                 Rts_w2c,
                 links_by_frame):
        self.cam = cam
        self.K_gtsam = gtsam.Cal3_S2Stereo(extract_intrinsic_param(cam.K, cam.M_right))
        self.pixel_noise = 1.0
        self.Rts_w2c = Rts_w2c
        self.links_by_frame = links_by_frame
        self.KF_indices = None      # to be cached in 'build()'

    def _set_keyframe_criterion(self, t_rel):
        """ Label keyframes by accumulating translation until a ~14 m threshold is exceeded. """
        # All the translation norms between consecutive frames (in meters)
        t_norms = np.linalg.norm(t_rel, axis=1)

        # Mean translation between consecutive frames
        # t_mean = np.mean(t_norms)  # obtained ~0.8 meters

        # We want 5-20 frames in each 'bundle window'
        # ~14 m between keyframe_i to keyframe_i+1 performed well
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
                          links_cache,  # dict fid → {tid:Link}
                          init_poses,  # dict fid → Pose3
                          ):
        """Build a gtsam factor-graph + initial Values for one bundle-adjustment window."""
        obs_by_tid = defaultdict(dict)
        for fid in bundle_fids:  # O(#frames in window)
            for tid, link in links_cache[fid].items():
                obs_by_tid[tid][fid] = link

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        fid0 = bundle_fids[0]
        c0 = gtsam.symbol('c', fid0)
        pose0 = init_poses[fid0]
        yaw_pitch_roll_x_y_z = np.array([np.deg2rad(1), np.deg2rad(1), np.deg2rad(1),
                                         0.05, 0.05, 0.50])  # Z bigger than X,Y
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(yaw_pitch_roll_x_y_z)
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

    def _bundle_window(self, T_w2c, curr_KF):
        """ For frames k0 .. k1  (k0 itself is I),
            Each element maps current camera coords to coords of frame k0. """
        T_window = [np.eye(4, dtype=T_w2c.dtype)]
        running_T = np.eye(4, dtype=T_w2c.dtype)
        if curr_KF == len(self.KF_indices) - 1:
            for fid in range(self.KF_indices[curr_KF] + 1, len(T_w2c) + 1):
                running_T = T_w2c[fid - 1] @ running_T
                T_window.append(invert_pose(running_T))
        else:
            for fid in range(self.KF_indices[curr_KF] + 1, self.KF_indices[curr_KF + 1] + 1):
                running_T = T_w2c[fid - 1] @ running_T
                T_window.append(invert_pose(running_T))
        return T_window


    def build_windows(self):
        """
        Partition the trajectory into consecutive bundle‑adjustment windows
        (each rooted at a key‑frame), convert every frame’s pose to its local
        window‑anchor coordinates, wrap the poses into 'gtsam.Pose3' objects,
        and construct a 'gtsam.NonlinearFactorGraph' plus matching initial
        'Values' container for each window.
        """       
        t_w2c = self.Rts_w2c[:, :, 3]
        T_w2c = Rts_to_homogenous(self.Rts_w2c)

        fid_to_KF = self._set_keyframe_criterion(t_w2c)
        KF_indices = list(fid_to_KF.keys())
        self.KF_indices = KF_indices        # cache 'KF_indices' for other functions
        T_bundles = [self._bundle_window(T_w2c, kf_i) for kf_i, _ in enumerate(KF_indices)]

        bundle_frames = [list(range(KF_indices[kf_i], KF_indices[kf_i + 1] + 1))
                         for kf_i, _ in enumerate(KF_indices[:-1])]
        bundle_frames.append(list(range(KF_indices[-1], len(T_w2c))))

        R_cam_to_window, t_cam_to_window = bundles_split_Rt(T_bundles)
        camera_poses_bundles = [Rt_c2w_gtsam(R, t, self.K_gtsam) for R, t in zip(R_cam_to_window, t_cam_to_window)]
        poses_bundles = [pose for cam, pose in camera_poses_bundles]
        fid_to_pose_bundles = []
        for frames, poses in zip(bundle_frames, poses_bundles):
            fid_to_pose_bundles.append({fid: pose for (fid, pose) in zip(frames, poses)})

        # Create graph & value objects for each bundle window
        graphs, values = [], []
        for frames, init_map in zip(bundle_frames, fid_to_pose_bundles):
            g, v = self.make_bundle_graph(frames, self.links_by_frame, init_map)
            graphs.append(g)
            values.append(v)
        return graphs, values


    def optimize_bundles(self, graphs, values):
        """ Perform a Levenberg-Marquardt optimization
            to each bundle-adjustment window. """
        # Optimize each bundle window
        result_bundles = [gtsam.LevenbergMarquardtOptimizer(graph, val).optimize()
                          for graph, val in zip(graphs, values)]
        return result_bundles


    def extract_rel_pose_nextKF_to_currKF(self, bundle_frames, result_bundles):
        """ Extract relative pose  KF_i <-- KF_{i+1}  from each window.
            Note that all the frames in between live in KF_i coordinates, so they
            can be turned into global poses later with a single composition step. """
        rel_poses = []
        for frames, values in zip(bundle_frames, result_bundles):
            next_kf = frames[-1]  # last frame of this window (=first of next window)
            rel_pose = values.atPose3(gtsam.symbol('c', next_kf))  # pose of nextKF in currK coordinates
            rel_poses.append(rel_pose)
        return rel_poses


    def rel_window_to_global_poses(self, rel_poses):
        """ Take relative pose  KF_i <-- KF_{i+1}  from each window &
            chain these relative to absolute poses in the *global* (KF0) frame
            into T_global = T_(global<-KF1) T_(KF1<-KF2)... T_(KFi<-KFi+1) """
        global_poses = {self.KF_indices[0]: gtsam.Pose3()}  # identity
        for kf_curr, rel_pose in zip(self.KF_indices[:-1], rel_poses):
            prev = global_poses[kf_curr]  # T_(global<-KF_curr)
            kf_next = self.KF_indices[self.KF_indices.index(kf_curr) + 1]
            global_poses[kf_next] = prev.compose(rel_pose)  # T_(global<-KF_next)
        return global_poses


    def gather_landmarks_global(self, bundle_frames, result_bundles, global_poses):
        """ Return every landmark Point3 expressed in global (KF0) coords. """
        pts = []
        for frames, values in zip(bundle_frames, result_bundles):
            kf0 = frames[0]
            T_glob_curr_win = global_poses[kf0]  # T_(global<-window)
            for key in values.keys():  # for each frame,
                if gtsam.Symbol(key).chr() == ord('q'):
                    p_win = values.atPoint3(key)
                    p_glob = T_glob_curr_win.transformFrom(p_win)
                    pts.append(p_glob)
        return np.asarray(pts)  # (N,3)


    def get_kf_indices(self):
        return self.KF_indices