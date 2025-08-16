# slam/utils/utils.py
import numpy as np
import gtsam
import cv2
import matplotlib.pyplot as plt


def array_from_values(vals, kf_indices):
    """Return an (N, 3) array of camera centres in KF order."""
    rows = []
    for k in kf_indices:
        t = vals.atPose3(gtsam.symbol('c', k)).translation()
        rows.append(t)      # Point3  ->  [x, y, z]
    return np.vstack(rows)              # shape (N, 3)


def relative_chain_to_absolute(Rts_rel):
    """
    Rts_rel : (N,3,4)   consecutive   T_{i,i-1}   (cam_i <- cam_{i-1})
    returns : (N,3,4)   cumulative   T_{i,0}     (cam_i <- world)
    """
    Ts_abs = np.empty_like(Rts_rel)
    T_cum  = np.eye(4)                          # world = cam_0
    for i, Rt in enumerate(Rts_rel):
        T_step = np.eye(4)
        T_step[:3, :] = Rt
        T_cum = T_step @ T_cum                  # compose
        Ts_abs[i, :, :] = T_cum[:3, :]          # back to 3x4
    return Ts_abs


def centers_from_Rts(Rts_w2c):
    """
    Rts_w2c: (N,3,4) array of world->camera poses [R|t]
    returns (N,3) array of camera centers in world-coords
    """
    R = Rts_w2c[:, :3, :3]  # (N,3,3)
    t = Rts_w2c[:, :3, 3]  # (N,3)
    # C_i = - R_i^T @ t_i  for each i
    return -np.einsum('nij,nj->ni', R.transpose(0, 2, 1), t)


def chain_rel_to_abs_w2c(rel_Rts):
    """
    rel_Rts[k-1] is 3x4 [R|t] that maps coords in frame (k-1) to camera k:
        x_k = R_rel * x_{k-1} + t_rel     (world->cam form)
    Return T_est[k] = 3x4 world->cam for a single world = frame-0.
    """
    N = rel_Rts.shape[0] + 1
    T_est = np.zeros((N, 3, 4))
    T_est[0, :3, :3] = np.eye(3)          # T_{w0->c0} = I
    for k in range(1, N):
        Rr, tr = rel_Rts[k-1][:, :3], rel_Rts[k-1][:, 3]
        Rp, tp = T_est[k-1, :3, :3],  T_est[k-1, :, 3]
        Rk = Rr @ Rp
        tk = Rr @ tp + tr
        T_est[k, :3, :3] = Rk
        T_est[k, :,  3] = tk
    return T_est    # (N,3,4)


def rot_err_deg(T_est, T_gt):
    """
    Compute the per-frame orientation error (degrees) between
    estimated and ground-truth poses:
    R_err = R_gt.T @ R_est
    r_err_Rodrigues = theta_err * u
    Delta_theta = ||r_err_Rodrigues||
    """
    R_est = T_est[:, :3, :3]
    R_gt = T_gt[:, :3, :3]
    R_err = R_gt.transpose(0, 2, 1) @ R_est
    ang = np.array([np.linalg.norm(cv2.Rodrigues(R)[0]) for R in R_err])
    return np.degrees(ang)


def rel_position_err(T_rel, T_gt_rel):
    t_gt = T_gt_rel[:, :3, 3]
    t_rel = T_rel[:, :3, 3]
    rel_pos_err = np.linalg.norm(t_gt - t_rel, axis=1)
    return rel_pos_err


def compute_position_errs(T_est_w2c, T_gt_w2c):
    """
    Compute the absolute position error between the estimated and ground-truth.
    """
    # position errors
    C_est = centers_from_Rts(T_est_w2c)
    C_gt = centers_from_Rts(T_gt_w2c)
    dC = C_est - C_gt
    ex, ey, ez = np.abs(dC[:, 0]), np.abs(dC[:, 1]), np.abs(dC[:, 2])
    epos = np.linalg.norm(dC, axis=1)
    return dict(ex=ex, ey=ey, ez=ez, epos=epos)


def to_hom(T34):
    """(N,3,4)->(N,4,4)."""
    T = np.zeros((T34.shape[0], 4, 4), dtype=T34.dtype)
    T[:, :3, :4] = T34
    T[:, 3, 3] = 1.0
    return T


def invert_T(T):
    """Batch inverse of world->camera transforms (N,4,4)."""
    R = T[:, :3, :3]
    t = T[:, :3,  3]
    RT = np.transpose(R, (0,2,1))
    Tout = np.zeros_like(T)
    Tout[:, :3, :3] = RT
    Tout[:, :3,  3] = -np.einsum('nij,nj->ni', RT, t)
    Tout[:,  3,  3] = 1.0
    return Tout


def relatives_from_abs_w2c_endpoints(T_abs, starts, ends):
    """
    Generic endpoint relatives:
      returns T_{starts<-ends} = T_abs[starts] @ inv(T_abs[ends])  (world->camera).
    """
    Th = to_hom(T_abs)
    return Th[starts] @ invert_T(Th[ends])


def relatives_from_absolutes_w2c_adjacent(abs_poses, KF_indices=None):
    """
    Takes (N,3,4) absolutes, world->cam (T_{k<-0})
    Returns: (N-1,4,4) or (M-1,4,4) relatives T_{k<-k+1} in homogeneous form.
    Each pose 'k' in 'abs_poses' is 'T_(k<-0)', so if we want 'T_(k<-k+1)':
    T_(k<-k+1) = T_(k<-0) @ T_(0<-k+1), which is:
    T_(k<-0) @ T_(k+1<-0)^(-1)
    """
    T_hom = to_hom(abs_poses)
    if KF_indices is None:
        T0, T1 = T_hom[:-1], T_hom[1:]
    else:
        i0 = np.asarray(KF_indices[:-1])
        i1 = np.asarray(KF_indices[1:])
        T0, T1 = T_hom[i0], T_hom[i1]
    return T0 @ invert_T(T1)


def plot_keyframe_trajectories(
    KF_indices,
    poses_gt_all,
    pre_ba_poses_all,      # before BA (same as PnP)
    post_ba_vals,     # after BA (same as before LC)
    post_lc_vals,     # after LC
    title='Trajectory overlay',
    label_pre='PnP',
    label_mid='after BA',
    label_post=f'after LC'):
    """
    Overlay GT, pre‑BA (PnP), post‑BA/pre‑LC, and post‑LC
    trajectories in X–Z, with RMSE in the legend.
    """
    # --- align to keyframes ---
    poses_gt = poses_gt_all[KF_indices]
    gt_xyz = centers_from_Rts(poses_gt)
    pre_ba_poses = pre_ba_poses_all[KF_indices]

    # ---- assemble xyz arrays ----
    pre_xyz = centers_from_Rts(pre_ba_poses)
    mid_xyz = array_from_values(post_ba_vals, range(len(KF_indices)))
    post_xyz = array_from_values(post_lc_vals, range(len(KF_indices)))

    # ---- compute per‑keyframe RMSE to GT ----
    def rmse(a,b):
        return np.sqrt(np.mean(np.linalg.norm(a - b, axis=1)**2))
    err_pre = rmse(pre_xyz, gt_xyz)
    err_mid = rmse(mid_xyz, gt_xyz)
    err_post = rmse(post_xyz,gt_xyz)

    # ---- align all to start at origin ----
    gt_al = gt_xyz - gt_xyz[0]
    pre_al = pre_xyz - pre_xyz[0]
    mid_al = mid_xyz - mid_xyz[0]
    post_al = post_xyz - post_xyz[0]

    # ---- plotting ----
    fig, ax = plt.subplots(figsize=(14,7))

    ax.plot(gt_al[:,0], gt_al[:,2],'k--', lw=2, label='ground-truth')
    ax.plot(pre_al[:,0], pre_al[:,2], 'r-', lw=1.5,
            label=f'{label_pre} (RMSE {err_pre:.2f} m)')
    ax.plot(mid_al[:,0], mid_al[:,2], 'b-', lw=1.5,
            label=f'{label_mid} (RMSE {err_mid:.2f} m)')
    ax.plot(post_al[:,0], post_al[:,2], 'g-', lw=1.5,
            label=f'{label_post} (RMSE {err_post:.2f} m)')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Z [m]  (forward)'); ax.set_title(title)
    ax.grid(True, ls=':'); ax.legend(loc='best')
    plt.tight_layout(); plt.show()
