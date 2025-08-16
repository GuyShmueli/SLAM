# slam/analysis/optimizers_analysis.py
import numpy as np
import gtsam
import matplotlib.pyplot as plt
import slam.utils.utils as utils
from slam.utils.graph_utils import values_keyframes_to_w2c


class OptimizersPlot:
    """
    Base plotter for pose/trajectory metrics shared by BA and LC.
    - Works on keyframe (KF) subsets
    - Accepts gtsam.Values (camera->world) and converts to world->camera (KITTI-style)
    """
    def __init__(self,
                 vals, KF_indices,
                 poses_gt, Rts_abs_all,
                 optimizer_label: str   # either "BA" or "LC"
                 ):
        self.KF_indices = KF_indices
        self.M = len(self.KF_indices)
        self.poses_gt = poses_gt                     # (N,3,4) world->camera for all frames
        self.T_gt_w2c = poses_gt[self.KF_indices]    # (M,3,4) GT at KFs
        self.optimizer_label = optimizer_label
        self.T_est_w2c = values_keyframes_to_w2c(vals, self.M)
        self.Rts_abs_all = Rts_abs_all


    # --- API ---
    ## absolute (with respect to frame 0)
    def abs_position_errs(self):
        errs = utils.compute_position_errs(self.T_est_w2c, self.T_gt_w2c)
        x = np.asarray(self.KF_indices)
        plt.figure(figsize=(12, 6))
        plt.plot(x, errs['ex'], '--', label='x')
        plt.plot(x, errs['ey'], '--', label='y')
        plt.plot(x, errs['ez'], '--', label='z')
        plt.plot(x, errs['epos'], '--', label='norm')
        plt.xlabel('frame'); plt.ylabel(f'{self.optimizer_label} abs position error [m]')
        plt.title(f'Frame Absolute Position Error ({self.optimizer_label})')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


    ## relative (adjacent KFs)
    @staticmethod
    def rel_position_err(rel_pnp_poses,
                         rel_pre_lc_poses,
                         rel_post_lc_poses,
                         rel_gt_poses,
                         KF_indices):
        M = len(KF_indices)
        x = np.arange(M-1)
        fig, ax = plt.subplots(figsize=(12, 6))
        e_pnp = utils.rel_position_err(rel_pnp_poses, rel_gt_poses)
        e_ba = utils.rel_position_err(rel_pre_lc_poses, rel_gt_poses)
        e_lc = utils.rel_position_err(rel_post_lc_poses, rel_gt_poses)
        ax.plot(x, e_pnp, linestyle='--', marker='o', markevery=8, linewidth=1.6, label='PnP')
        ax.plot(x, e_ba, linestyle='-.', marker='s', markevery=8, linewidth=1.6, label='BA')
        ax.plot(x, e_lc, linestyle=':', marker='^', markevery=8, linewidth=1.6, label='LC')

        ax.set_xlabel('keyframe'); ax.set_ylabel('rel position error [m]')
        ax.set_title('Keyframe Relative Position Error (PnP vs BA vs LC)')
        ax.grid(True); ax.legend(loc='upper right')
        plt.tight_layout(); plt.show()


    @staticmethod
    def rel_rotation_err(rel_pnp_poses,
                         rel_pre_lc_poses,
                         rel_post_lc_poses,
                         rel_gt_poses,
                         KF_indices):
        M = len(KF_indices)
        x = np.arange(M-1)
        fig, ax = plt.subplots(figsize=(12, 6))
        e_pnp = utils.rot_err_deg(rel_pnp_poses, rel_gt_poses)
        e_ba = utils.rot_err_deg(rel_pre_lc_poses, rel_gt_poses)
        e_lc = utils.rot_err_deg(rel_post_lc_poses, rel_gt_poses)
        ax.plot(x, e_pnp, linestyle='--', marker='o', markevery=8, linewidth=1.6, label='PnP')
        ax.plot(x, e_ba, linestyle='-.', marker='s', markevery=8, linewidth=1.6, label='BA')
        ax.plot(x, e_lc, linestyle=':', marker='^', markevery=8, linewidth=1.6, label='LC')

        ax.set_xlabel('keyframe'); ax.set_ylabel('rel orientation error [deg]')
        ax.set_title('Keyframe Relative Orientation Error (PnP vs BA vs LC)')
        ax.grid(True); ax.legend(loc='upper right')
        plt.tight_layout(); plt.show()


    @staticmethod
    def abs_rotation_errs(pnp_poses,
                          pre_lc_vals,
                          post_lc_vals,
                          poses_gt,
                          KF_indices):
        M = len(KF_indices)
        pre_lc_poses = values_keyframes_to_w2c(pre_lc_vals, M)
        post_lc_poses = values_keyframes_to_w2c(post_lc_vals, M)
        # Slice GT and PnP to the same keyframe indices
        gt_kf = poses_gt[KF_indices]
        pnp_kf = pnp_poses[KF_indices]

        x = np.asarray(KF_indices)
        fig, ax = plt.subplots(figsize=(12, 6))
        e_pnp = utils.rot_err_deg(pnp_kf, gt_kf)
        e_ba = utils.rot_err_deg(pre_lc_poses, gt_kf)
        e_lc = utils.rot_err_deg(post_lc_poses, gt_kf)
        ax.plot(x, e_pnp, linestyle='--', marker='o', markevery=8, linewidth=1.6, label='PnP')
        ax.plot(x, e_ba, linestyle='-.', marker='s', markevery=8, linewidth=1.6, label='BA')
        ax.plot(x, e_lc, linestyle=':', marker='^', markevery=8, linewidth=1.6, label='LC')

        ax.set_xlabel('keyframe'); ax.set_ylabel('abs orientation error [deg]')
        ax.set_title('Keyframe Absolute Orientation Error (PnP vs BA vs LC)')
        ax.grid(True); ax.legend(loc='upper right')
        plt.tight_layout(); plt.show()


    def _det_size(self, marginals, kfs,
                  conf_mult: float, is_trans: bool):
        sizes = []
        for k in kfs:
            Sigma6 = marginals.marginalCovariance(gtsam.symbol('c', k))
            if is_trans:
                Sigma = Sigma6[3:, 3:]   # 3x3 translation cov (m^2)
            else:
                Sigma = Sigma6[:3, :3]   # 3x3 rotation cov (rad^2)
            s = (np.linalg.det(Sigma)) ** (1.0 / 6.0)  # 1-sigma linear scale
            if not is_trans:
                s = np.rad2deg(s)
            sizes.append(conf_mult * s)  # scale to ~99.7% radius
        return np.asarray(sizes)


    def location_uncertainty(self,
        pose_graph_pre, pre_vals,
        pose_graph, pose_vals,
        conf_mult=3.75,   # desired Mahalanobis-radius, 3.75 is 99.7% in 3 DoF
        ):
        pre_marginals  = gtsam.Marginals(pose_graph_pre, pre_vals)
        post_marginals = gtsam.Marginals(pose_graph, pose_vals)

        M = len(self.KF_indices)
        x = np.arange(M)
        pre  = self._det_size(pre_marginals,  range(M), conf_mult, is_trans=True)
        post = self._det_size(post_marginals, range(M), conf_mult, is_trans=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x, pre,  label='without loop closures', color='C1')
        ax.plot(x, post, label='with loop closures', color='C0')
        ax.set_xlabel('Keyframe index')
        ax.set_ylabel('Location uncertainty size  [m]')
        ax.set_title(r'Location uncertainty vs keyframe (metric $\det(\Sigma_R)^{1/6}$)')
        ax.grid(True); ax.legend(); plt.tight_layout(); plt.show()


    def angle_uncertainty(self,
        pose_graph_pre, pre_vals,
        pose_graph, pose_vals,
        conf_mult=3.75,     # desired Mahalanobis-radius, 3.75 is 99.7% in 3 DoF
        ):

        pre_marginals  = gtsam.Marginals(pose_graph_pre, pre_vals)
        post_marginals = gtsam.Marginals(pose_graph, pose_vals)

        M = len(self.KF_indices)
        x = np.arange(M)
        pre_deg  = self._det_size(pre_marginals,  range(M), conf_mult, is_trans=False)
        post_deg = self._det_size(post_marginals, range(M), conf_mult, is_trans=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x, pre_deg,  label='without loop closures', color='C1')
        ax.plot(x, post_deg, label='with loop closures', color='C0')
        ax.set_xlabel('Keyframe index')
        ax.set_ylabel('Angle uncertainty size  [deg]')
        ax.set_title(r'Angle uncertainty vs keyframe (metric $\det(\Sigma_R)^{1/6}$)')
        ax.grid(True); ax.legend(); plt.tight_layout(); plt.show()


    def uncertainty_score(self,
                               pose_graph_pre, pre_vals,
                               pose_graph, pose_vals,
                               lc=None,
                               ref_rot_deg=1.0,           # normalization: 1deg (in radians internally)
                               ref_trans_m=0.10,          # normalization: 10 cm
                               conf_mult=3.75,            # ~99.7% radius
                               use_log10=True):
        """
        One scalar uncertainty score per keyframe, BA vs LC, plus loop-closure markers.
        The score compresses the full 6x6 marginal covariance into a single number.
        """
        x = np.array(self.KF_indices)
        s_pre  = self._marginal_scores(pose_graph_pre, pre_vals,
                                       ref_rot_deg, ref_trans_m, conf_mult)
        s_post = self._marginal_scores(pose_graph, pose_vals,
                                       ref_rot_deg, ref_trans_m, conf_mult)

        if use_log10:
            y_pre, y_post  = np.log10(s_pre), np.log10(s_post)
            ylab = 'uncertainty score per frame (log10)'
            base = y_pre[0]
            y_pre -= base
            y_post -= base
        else:
            y_pre, y_post = s_pre, s_post
            ylab = 'uncertainty score per frame'

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(x, y_pre,  label='uncertainty score BA')
        ax.plot(x, y_post, label='uncertainty score LC')

        curr_kfs = [d['curr_kf'] for d in lc.get_pair_stats()]  # loop "end" (current KF)
        cand_kfs = [d['cand_kf'] for d in lc.get_pair_stats()]  # loop "start" (candidate KF)
        # map KF -> frame id
        x_curr = np.array([self.KF_indices[k] for k in curr_kfs], dtype=int)
        x_cand = np.array([self.KF_indices[k] for k in cand_kfs], dtype=int)
        x_marks = np.r_[x_curr, x_cand]

        # put markers exactly on the zero baseline
        y_marks = np.zeros_like(x_marks, dtype=float)
        ax.scatter(x_marks, y_marks, s=22, marker='o', zorder=3, label='Loop Closure Location')

        ax.set_xlabel('KeyFrame Index'); ax.set_ylabel(ylab)
        ax.set_title(f'uncertainty score per frame (BA and LC{"- log" if use_log10 else ""})')
        ax.grid(True); ax.legend(); plt.tight_layout(); plt.show()


    def _marginal_scores(self,
                         pose_graph, values,
                         ref_rot_deg: float = 1.0,   # normalization for rotation
                         ref_trans_m: float = 0.10,  # normalization for translation
                         conf_mult: float = 3.75,    # multi-sigma visual scaling
                         ):
        """One scalar uncertainty score per keyframe (combines rot+trans via unit normalization)."""
        M = len(self.KF_indices)
        marg = gtsam.Marginals(pose_graph, values)
        ref_rot_rad = np.deg2rad(ref_rot_deg)

        Qinv = np.diag([1.0 / ref_rot_rad] * 3 + [1.0 / ref_trans_m] * 3)
        scores = np.zeros(M, dtype=float)
        for k in range(M):
            Sigma6 = marg.marginalCovariance(gtsam.symbol('c', k))
            S = Qinv @ Sigma6 @ Qinv.T      # dimensionless 6x6
            s = float(np.linalg.det(S))
            scores[k] = conf_mult * s
        return scores


    # relative over subsections
    def rel_bundle_error_over_subsections(self,
                                          seq_lengths=(100, 400, 800)):
        """
        For each end KF and length L (in frames), pick the closest start KF.
        Compute T_{start<-end} for estimate & GT (world->camera). Plot:
          - translation error % = 100 * ||t_err|| / GT distance (meters)
          - rotation error deg/m = angle(R_gt^T R_est) / GT distance
        """
        # 1) absolutes at KFs
        T_est_abs_kf = self.T_est_w2c  # (M,3,4)
        T_gt_abs_kf = self.T_gt_w2c

        # 2) cumulative GT distance over ALL frames
        C_gt = utils.centers_from_Rts(self.poses_gt)  # (N,3)
        step = np.linalg.norm(C_gt[1:] - C_gt[:-1], axis=1)
        prefix = np.concatenate([[0.0], np.cumsum(step)])  # dist(0->f)

        frames_kf = np.asarray(self.KF_indices, dtype=int)
        ends = np.arange(self.M, dtype=int)  # evaluate at every KF
        x = frames_kf

        results = {}
        for L in seq_lengths:
            if L < 1: continue
            targets = np.maximum(0, frames_kf[ends] - (L - 1))  # desired start frame-id

            i = np.searchsorted(frames_kf, targets, side='left')
            s1 = np.clip(i - 1, 0, ends)    # floor
            s2 = np.minimum(i, ends)        # ceil
            d1 = np.abs(frames_kf[s1] - targets)
            d2 = np.abs(frames_kf[s2] - targets)
            starts = np.where(d1 <= d2, s1, s2).astype(int)

            T_est_rel = utils.relatives_from_abs_w2c_endpoints(T_est_abs_kf, starts, ends)
            T_gt_rel  = utils.relatives_from_abs_w2c_endpoints(T_gt_abs_kf,  starts, ends)

            Rgt, tgt  = T_gt_rel[:, :3, :3], T_gt_rel[:, :3, 3]
            Rest, test = T_est_rel[:, :3, :3], T_est_rel[:, :3, 3]
            R_err = np.einsum('nij,njk->nik', np.transpose(Rgt, (0,2,1)), Rest)
            tr = np.trace(R_err, axis1=1, axis2=2)
            ang_deg = np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))

            t_err = np.einsum('nij,nj->ni', np.transpose(Rgt, (0,2,1)), (test - tgt))
            loc_err = np.linalg.norm(t_err, axis=1)

            d = prefix[frames_kf[ends]] - prefix[frames_kf[starts]]
            with np.errstate(divide='ignore', invalid='ignore'):
                trans_pct = 100.0 * (loc_err / d)
                rot_degpm = ang_deg / d

            valid = np.isfinite(trans_pct) & np.isfinite(rot_degpm) & (d > 0)
            results[L] = dict(
                x=x,
                trans_pct=trans_pct,
                rot_degpm=rot_degpm,
                avg_trans_pct=float(np.nanmean(trans_pct[valid])) if valid.any() else float('nan'),
                avg_rot_degpm=float(np.nanmean(rot_degpm[valid])) if valid.any() else float('nan'),
            )

        ## Plots
        plt.figure(figsize=(16, 4))
        for L, dct in results.items():
            plt.plot(dct['x'], dct['trans_pct'], label=f'{L} (avg {dct["avg_trans_pct"]:.2f}%)')
        plt.xlabel('End frame'); plt.ylabel('Translation error [%]')
        plt.title(f'Relative {self.optimizer_label} translation error over sub-sections (KFs)')
        plt.grid(True); plt.legend(title='Seq length'); plt.tight_layout(); plt.show()

        plt.figure(figsize=(16, 4))
        for L, dct in results.items():
            plt.plot(dct['x'], dct['rot_degpm'], label=f'{L} (avg {dct["avg_rot_degpm"]:.3f} deg/m)')
        plt.xlabel('End frame'); plt.ylabel('Rotation error [deg/m]')
        plt.title(f'Relative {self.optimizer_label} rotation error over sub-sections (KFs)')
        plt.grid(True); plt.legend(title='Seq length'); plt.tight_layout(); plt.show()
