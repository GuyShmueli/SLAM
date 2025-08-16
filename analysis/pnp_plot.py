from slam.utils.graph_utils import homogenous
from slam.frontend.geometry.pnp import PnP
import slam.utils.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import cv2


class PnP_Plotter:
    def __init__(self,
                 cam,
                 poses_gt,
                 db):
        self.cam = cam
        self.poses_gt = poses_gt
        self.db = db


    def triangulate_track_using_last_frame_ftrs(self):
        """Triangulate a single track by using its last-frame stereo pair."""
        tracks_equal_to_j = self.db.tracks_equal_to(self.j)

        self.tid_k = tracks_equal_to_j[self.k]
        self.fids_corr_tid_k = self.db.frames(self.tid_k)
        self.fid_last = self.fids_corr_tid_k[-1]

        track_link = self.db.link(self.fid_last, self.tid_k)
        pts_L_last = track_link.left_keypoint().reshape(2, 1)  # 2x1
        pts_R_last = track_link.right_keypoint().reshape(2, 1)  # 2x1

        P_left = self.cam.K @ self.poses_gt[self.fid_last]
        pose_fid_last_hom = homogenous(self.poses_gt[self.fid_last])
        P_right = self.cam.K @ (self.cam.M_right @ pose_fid_last_hom)

        Xh = cv2.triangulatePoints(P_left, P_right, pts_L_last, pts_R_last)  # 4xN
        X = (Xh[:3] / Xh[3]).T  # Nx3
        return X


    def compute_reproj_errs(self, X):
        """Return per-frame reprojection errors (left & right) for a given 3-D point."""
        reproj_errs_L = []  # cache reprojection errors
        reproj_errs_R = []
        for fid in self.fids_corr_tid_k:
            # --- projected left & right pixels ---
            Rt_L_fid = self.poses_gt[fid]
            P_L_fid = self.cam.K @ Rt_L_fid
            proj_pts_L = PnP.project(X, P_L_fid)

            Rt_hom_L_fid = homogenous(Rt_L_fid)
            P_R_fid = self.cam.K @ (self.cam.M_right @ Rt_hom_L_fid)
            proj_pts_R = PnP.project(X, P_R_fid)

            # --- tracked left & right pixels ---
            link_fid_corr_tid_i = self.db.link(fid, self.tid_k)
            tracked_pts_L, tracked_pts_R = link_fid_corr_tid_i.left_keypoint(), \
                link_fid_corr_tid_i.right_keypoint()  # shape (2,)

            tracked_pts_L = np.array(tracked_pts_L).reshape(1, 2)
            tracked_pts_R = np.array(tracked_pts_R).reshape(1, 2)

            # --- computing left & right reprojection error ---
            err_L_fid = np.linalg.norm(proj_pts_L - tracked_pts_L)
            reproj_errs_L.append(err_L_fid)

            err_R_fid = np.linalg.norm(proj_pts_R - tracked_pts_R)
            reproj_errs_R.append(err_R_fid)

        return reproj_errs_L, reproj_errs_R


    def reproj_err_over_track_len(self,
                                       j=40,  # number of frames correspond to the track
                                       k=10,  # The chosen track_id within 'tracks_equal_to_j'
                                       ):
        """After PnP, Plot reprojection error versus distance from the reference frame."""
        self.j = j
        self.k = k
        X = self.triangulate_track_using_last_frame_ftrs()
        reproj_errs_L, reproj_errs_R = self.compute_reproj_errs(X)
        plt.figure(figsize=(16, 6))
        dist_from_reference = [abs(self.fid_last - fid) for fid in self.fids_corr_tid_k]
        plt.plot(dist_from_reference, reproj_errs_L, label='Left')
        plt.plot(dist_from_reference, reproj_errs_R, label='Right')
        plt.xlabel("Distance from reference frame (|fid – fid_last|)")
        plt.ylabel("PnP Reprojection error (L$_2$ norm in pixels)")
        plt.title("PnP reprojection error vs. distance from reference")
        plt.legend(); plt.grid(True); plt.show()


    def abs_position_errs(self, T_est):
        # compute position errors (meters)
        errs = utils.compute_position_errs(T_est, self.poses_gt)
        N = len(self.poses_gt)
        x = np.arange(N)
        plt.figure(figsize=(12, 6))
        plt.plot(x, errs['ex'], '--', label='x')
        plt.plot(x, errs['ey'], '--', label='y')
        plt.plot(x, errs['ez'], '--', label='z')
        plt.plot(x, errs['epos'], '--', label='norm')
        plt.xlabel('frame'); plt.ylabel(f'PnP abs position error [m]')
        plt.title(f'Frame Absolute Position Error (PnP)')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


    def rel_pnp_error_over_subsections(self,
                                       T_est_abs,                 # (N,3,4) world->camera from PnP
                                       seq_lengths=(100, 400, 800),
                                       title_prefix='PnP'):
        """
        KITTI-style 'error per meter' over sliding windows.
        Makes two figures:
          1) Translation error [%] vs starting frame, for each window length
          2) Rotation error [deg/m] vs starting frame, for each window length
        Returns: dict {L: {'starts', 'trans_pct', 'rot_degpm', 'avg_trans_pct', 'avg_rot_degpm'}}
        """
        gt_abs = self.poses_gt
        N = len(gt_abs)

        # ground-truth camera centers and cumulative distance
        C_gt = utils.centers_from_Rts(gt_abs[:N])  # (N,3)
        step = np.linalg.norm(C_gt[1:] - C_gt[:-1], axis=1)  # (N-1,)
        prefix = np.concatenate([[0.0], np.cumsum(step)])    # prefix[i] = dist(0->i)

        ends = np.arange(N)
        results = {}
        for L in seq_lengths:
            if L < 1 or L >= N:
                continue

            starts = np.maximum(0, ends - (L - 1))
            # endpoint relatives for estimate and GT
            T_est_rel = utils.relatives_from_abs_w2c_endpoints(T_est_abs[:N], starts, ends)  # (M,4,4)
            T_gt_rel = utils.relatives_from_abs_w2c_endpoints(gt_abs[:N], starts, ends)  # (M,4,4)

            # --- error pose E = inv(T_gt_rel) @ T_est_rel ---
            R_gt, t_gt = T_gt_rel[:, :3, :3], T_gt_rel[:, :3, 3]
            R_gt_T = R_gt.transpose(0, 2, 1)
            t_est = T_est_rel[:, :3, 3]
            # rotation error
            ang_deg = utils.rot_err_deg(T_est_rel, T_gt_rel)
            # translation error in 'a' coords
            t_err = np.einsum('nij,nj->ni', R_gt_T, (t_est - t_gt))
            loc_err = np.linalg.norm(t_err, axis=1)  # meters

            # denominator: total GT distance inside [a,b]
            dist = prefix[ends] - prefix[starts]     # meters

            # normalize
            with np.errstate(divide='ignore', invalid='ignore'):
                trans_pct  = 100.0 * (loc_err / dist)        # %
                rot_degpm  = ang_deg / dist                  # deg/m

            # averages: only count full-length windows (e >= L-1)
            mask_full = ends >= (L - 1)
            avg_trans = float(np.nanmean(trans_pct[mask_full]))
            avg_rot = float(np.nanmean(rot_degpm[mask_full]))

            results[L] = dict(ends=ends,
                              trans_pct=trans_pct,
                              rot_degpm=rot_degpm,
                              avg_trans_pct=avg_trans,
                              avg_rot_degpm=avg_rot)

        ## plots
        # Translation %
        plt.figure(figsize=(16, 4))
        for L, d in results.items():
            plt.plot(d['ends'], d['trans_pct'], label=f'{L} (avg {d["avg_trans_pct"]:.2f}%)')
        plt.xlabel('End frame number'); plt.ylabel('Translation error [%]')
        plt.title(f'Relative {title_prefix} translation error over sub-sections')
        plt.grid(True); plt.legend(title='Sequence Length'); plt.tight_layout(); plt.show()

        # Rotation deg/m
        plt.figure(figsize=(16, 4))
        for L, d in results.items():
            plt.plot(d['ends'], d['rot_degpm'], label=f'{L} (avg {d["avg_rot_degpm"]:.3f} deg/m)')
        plt.xlabel('End frame number'); plt.ylabel('Rotation error [deg/m]')
        plt.title(f'Relative {title_prefix} rotation error over sub-sections (by window end)')
        plt.grid(True); plt.legend(title='Sequence Length'); plt.tight_layout(); plt.show()

