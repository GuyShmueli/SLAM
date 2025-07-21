from stage4_tracking_via_database import *
import gtsam
from gtsam.utils import plot as gtsam_plot
import pickle
from collections import defaultdict


# --- Magic constants ---
DATA_PATH = r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\sequences\05\\"
POSES_PATH = r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\poses\05.txt"
DB_PATH = "kitti05_tracks"
EST_POSES_PATH = r'C:\Users\dhtan\SLAM\SLAM-GShmueli\kitti05_relposes.pkl'


def extract_intrinsic_param(K, M_R):
    """Return (fx, fy, skew, cx, cy, baseline) extracted from KITTI stereo intrinsics."""
    fx, fy = K[0,0], K[1,1]     # focal length in x & y (pixels)
    s  = K[0,1]                 # skew (0)
    cx, cy = K[0,2], K[1,2]     # principal point x & y (pixels)
    b = -M_R[0,3]               # baseline (meters)
    return fx, fy, s, cx, cy, b


def compute_Rt_c2w(relative_Rts):
    """Integrate a list of relative world-to-camera 3×4 transforms and return the
    absolute camera-to-world rotation (R_c2w) and translation (t_c2w) for each frame."""
    T = np.zeros((len(relative_Rts), 4, 4), dtype=relative_Rts[0].dtype)
    T[:, :3, :4] = relative_Rts
    T[:, 3, 3] = 1

    pose_w2c = np.eye(4, dtype=T.dtype)
    poses_w2c = [pose_w2c[:3]]

    for T_rel in T:
        pose_w2c = T_rel @ pose_w2c
        poses_w2c.append(pose_w2c[:3])

    poses_w2c = np.asarray(poses_w2c)  # (N+1, 3, 4)
    R_w2c = poses_w2c[:, :, :3]
    t_w2c = poses_w2c[:, :, 3]

    R_c2w = R_w2c.transpose(0, 2, 1)  # R_c2w = R_w2c^T
    t_c2w = -np.einsum('nij,nj->ni', R_c2w, t_w2c)  # t_c2w = -R_w2c^T * t_w2c

    return R_c2w, t_c2w


def Rt_c2w_gtsam(R_cw, t_cw, K_gtsam):
    """Wrap  cam -> world  (R, t) pairs into gtsam.Pose3 objects and matching gtsam.StereoCamera list."""
    cameras = []
    poses = []
    for R, t in zip(R_cw, t_cw):
        rot  = gtsam.Rot3(R)              # wrap the 3×3 into Rot3
        trans = gtsam.Point3(t)           # wrap the 3-vector
        pose = gtsam.Pose3(rot, trans)    # Pose3 = (rotation, translation)
        cam  = gtsam.StereoCamera(pose, K_gtsam)
        poses.append(pose)
        cameras.append(cam)
    return cameras, poses


def single_backproject_gtsam(cam, z):
    """Back-project a stereo measurement z to a 3D world point using gtsam.StereoCamera."""
    # Buffers for the Jacobians (shape 3×6 and 3×3)
    # 'order=F' -> f_contiguous -> data is laid out column-by-column
    H_pose = np.zeros((3, 6), dtype=np.float64, order='F')  # 3×6
    H_meas = np.zeros((3, 3), dtype=np.float64, order='F')  # 3×3

    # Triangulation
    p_xyz = cam.backproject2(z, H_pose, H_meas)  # ndarray (3,)
    return p_xyz


def compute_projections(p_xyz, cameras):
    """Project a 3D world point through each gtsam.StereoCamera.
    return list of (uL,uR,v)."""
    # 1×3 column vector, f_contiguous
    p_world = np.array(p_xyz.reshape(3, 1), order='F')

    # Jacobian buffers
    H_pose = np.zeros((3, 6), order='F')
    H_point = np.zeros((3, 3), order='F')

    projections = []
    for cam in cameras:
        z_hat = cam.project2(p_world, H_pose, H_point)
        projections.append((z_hat.uL(), z_hat.uR(), z_hat.v()))
    return projections


def compute_reproj_errs(projections, links):
    """Compute L2 reprojection error for each (projection, measurement) pair in a track."""
    diffs = []
    for proj, link in zip(projections, links):
        xL_curr, xR_curr, y_curr = link.x_left, link.x_right, link.y
        z_stereo_curr = np.array([xL_curr, xR_curr, y_curr])
        diff = z_stereo_curr - proj
        diffs.append(diff)

    reproj_errs = np.linalg.norm(diffs, axis=1)  # shape (N,)
    return reproj_errs


def plot_reproj_err_over_track_len(frame_ids, reproj_errs):
    """Scatter-plot reprojection error versus frame distance from the reference frame."""
    fig, ax = plt.subplots()
    fid_last = frame_ids[-1]
    x = [abs(fid_last - fid) for fid in frame_ids]
    y = reproj_errs
    ax.plot(x, y, marker='o')
    ax.set_xlabel('Distance from reference frame (|fid - fid_last|)')
    ax.set_ylabel('Reprojection error  (pixels, L$_2$ norm)')
    ax.set_title('Stereo reprojection error along track')
    ax.grid(True)
    plt.show()


def make_stereo_track_graph(p_xyz, poses, links, K_gtsam, sigma_px=1.0):
    """
    graph  : gtsam.NonlinearFactorGraph
    values : gtsam.Values  (initial estimates)
    """
    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # --- landmark ---
    q0_key = gtsam.symbol('q', 0)
    values.insert(q0_key, gtsam.Point3(p_xyz))

    # --- noise model ---
    noise = gtsam.noiseModel.Isotropic.Sigma(3, sigma_px)    # identity 3x3 matrix

    # --- per-frame pose + stereo factor ---
    for i, (pose_mat, link) in enumerate(zip(poses, links)):
        pose_key = gtsam.symbol('c', i)
        values.insert(pose_key, gtsam.Pose3(pose_mat))

        z = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)

        factor = gtsam.GenericStereoFactor3D(z, noise, pose_key, q0_key, K_gtsam)
        graph.add(factor)

    return graph, values


def factor_errors(graph, values):
    """ Returns array with error per factor: 1/2 r^T  Σ^-1  r """
    errs = []
    for idx in range(graph.size()):
        factor = graph.at(idx)
        errs.append(factor.error(values))
    return np.asarray(errs)


def plot_factor_err_over_track_len(frame_ids, factor_errs):
    plt.figure()
    fid_last = frame_ids[-1]
    x = [abs(fid_last - fid) for fid in frame_ids]
    y = factor_errs
    plt.plot(x, y, marker='o')
    plt.xlabel('Distance from reference frame (|fid - fid_last|)')
    plt.ylabel(r'Factor error  $\frac{1}{2}\,r^T \Sigma^{-1} r$')
    plt.title('Stereo factor error along track')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_factor_err_vs_reproj_err(reproj_errs, factor_errs,
                                  adjust_errs: bool, sigma_px=1.0):
    """ Plot factor error over reprojection error.
        If the covariance is proportional to the identity matrix,
        these errors can simply be achieved from one another.
        That is, if adjust_errs, we adjust reproj_errs to align with reproj_errs.
     """
    plt.figure()
    if adjust_errs:
        x = 0.5*(reproj_errs**2) / sigma_px**2
        x_label = r'Adjusted reprojection error  $\frac{1}{2\sigma^{2}}\,r^{\top}r$'
        title = 'ADJUSTED Factor Error vs Reprojection Error'
    else:
        x = reproj_errs
        x_label = r'Reprojection error $\sqrt{r^Tr}$'
        title = 'Factor Error vs Reprojection Error'
    y = factor_errs
    y_label = r'Factor error  $\frac{1}{2}\,r^T \Sigma^{-1} r$'
    plt.plot(x, y, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def Rts_to_homogenous(Rts_wc):
    """Convert an array of 3×4  world -> cam  matrices into 4×4 homogeneous transforms."""
    T_wc = np.zeros((len(Rts_wc), 4, 4), dtype=Rts_wc[0].dtype)
    T_wc[:, :3, :4] = Rts_wc
    T_wc[:, 3, 3] = 1
    return T_wc


def set_keyframe_criterion(t_rel):
    """Label keyframes by accumulating translation until a ~14 m threshold is exceeded."""
    # All the translation norms between consecutive frames (in meters)
    t_norms = np.linalg.norm(t_rel, axis=1)

    # Mean translation between consecutive frames
    t_mean = np.mean(t_norms)   # obtained ~0.8 meters

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

    return fid_to_KF


def invert_pose(T):
    """Return the inverse of a 4×4 homogeneous pose matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t
    return T_inv


def bundle_window(T_w2c, KF_indices, curr_KF):
    """ For frames k0 … k1  (k0 itself is I),
        Each element maps current camera coords to coords of frame k0. """
    T_window = [np.eye(4, dtype=T_w2c.dtype)]
    running_T = np.eye(4, dtype=T_w2c.dtype)
    if curr_KF == len(KF_indices) - 1:
        for fid in range(KF_indices[curr_KF] + 1, len(T_w2c) + 1):
            running_T = T_w2c[fid - 1] @ running_T
            T_window.append(invert_pose(running_T))
    else:
        for fid in range(KF_indices[curr_KF] + 1, KF_indices[curr_KF + 1] + 1):
            running_T = T_w2c[fid - 1] @ running_T
            T_window.append(invert_pose(running_T))

    return T_window


def compute_total_and_relative_errs(graph, values):
    """Return total graph error plus mean and per-factor errors for given Values."""
    # 1) total error
    total_err = graph.error(values)

    # 2) relative errors & mean
    rel_errs = factor_errors(graph, values)
    avg_rel_errs = np.mean(rel_errs)

    return total_err, avg_rel_errs, rel_errs


def _links_by_frame(db):
    """ Re-indexes the whole database from (frame, track) --> Link dict
    to frame --> {track: Link} dict.
    After this one pass you all links of any frame can be grabbed in O(1). """
    links_by_frame = defaultdict(dict)
    for (fid, tid), link in db.linkId_to_link.items():
        links_by_frame[fid][tid] = link
    return links_by_frame


def make_bundle_graph(bundle_fids,
                      links_cache,      # dict fid → {tid:Link}
                      init_poses,       # dict fid → Pose3
                      K_gtsam,
                      sigma_px=1.0):
    """Build a gtsam factor-graph + initial Values for one bundle-adjustment window."""
    obs_by_tid = defaultdict(dict)
    for fid in bundle_fids:                      # O(#frames in window)
        for tid, link in links_cache[fid].items():
            obs_by_tid[tid][fid] = link

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    fid0 = bundle_fids[0]
    c0 = gtsam.symbol('c', fid0)
    pose0 = init_poses[fid0]
    yaw_pitch_roll_x_y_z = np.array([np.deg2rad(1), np.deg2rad(1), np.deg2rad(1),
                   0.05, 0.05, 0.50])   # Z bigger than X,Y
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(yaw_pitch_roll_x_y_z)
    graph.add(gtsam.PriorFactorPose3(
              c0, pose0, prior_noise))
    values.insert(c0, pose0)

    pose_used = {fid0}
    meas_noise = gtsam.noiseModel.Isotropic.Sigma(3, sigma_px)

    for tid, obs in obs_by_tid.items():
        if len(obs) < 2:            # need ≥2 frames for BA
            continue
        qk = gtsam.symbol('q', tid)
        fid_ref = max(obs)          # last obs – purely a heuristic
        link = obs[fid_ref]

        cam_ref = gtsam.StereoCamera(init_poses[fid_ref], K_gtsam)
        z_ref = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
        p_xyz = single_backproject_gtsam(cam_ref, z_ref)
        values.insert(qk, gtsam.Point3(p_xyz))

        for fid, lnk in obs.items():
            ck = gtsam.symbol('c', fid)
            pose_used.add(fid)
            z = gtsam.StereoPoint2(lnk.x_left, lnk.x_right, lnk.y)
            graph.add(gtsam.GenericStereoFactor3D(
                      z, meas_noise, ck, qk, K_gtsam))

    for fid in pose_used:
        ck = gtsam.symbol('c', fid)
        if not values.exists(ck):
            values.insert(ck, init_poses[fid])

    return graph, values


def compute_z_proj_z_meas(tid, # the argmax track id
                          fid, # the argmax frame id
                          graph,
                          values,
                          K_gtsam,
                          max_err_idx):
    """Return (z_proj, z_meas) for the factor with the largest pre-BA error."""
    # 1) recreate keys
    q_key = gtsam.symbol('q', tid)
    c_key = gtsam.symbol('c', fid)

    # 2) retrieve the argmax guesses for q and c
    q_init = values.atPoint3(q_key)
    pose0 = values.atPose3(c_key)
    cam0 = gtsam.StereoCamera(pose0, K_gtsam)

    # 3) use initial guess for relevant q and relevant cam to project
    z_proj = compute_projections(q_init, [cam0])[0]

    # 4) take z_meas from the factor graph
    z_meas = graph.at(max_err_idx).measured()

    return z_proj, z_meas


def two_pts_from_z(z):
    if not isinstance(z, gtsam.gtsam.StereoPoint2):
        z = gtsam.StereoPoint2(z[0], z[1], z[2])
    pt_L = z.uL(), z.v()
    pt_R = z.uR(), z.v()
    # pt_L = z[0], z[2]
    # pt_R = z[1], z[2]
    return pt_L, pt_R


def draw_proj_meas_z_LR(z_proj,
                        z_meas,
                        margin,  # how many pixels around points
                        pair_idx=0,
                        data_path=DATA_PATH):
    # 1) load left & right images
    imgL, imgR = read_images(0, DATA_PATH)

    # 2) unpack left meas, right meas, left proj, right proj points
    ptL_meas, ptR_meas = two_pts_from_z(z_meas)
    ptL_proj, ptR_proj = two_pts_from_z(z_proj)

    # 3) collect into lists
    imgs = [imgL, imgR]
    pts_meas = [ptL_meas, ptR_meas]
    pts_proj = [ptL_proj, ptR_proj]
    titles = ['Left Zoom', 'Right Zoom']

    # 4) plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    margin = margin

    for ax, img, meas, proj, title in zip(axes, imgs, pts_meas, pts_proj, titles):
        ax.imshow(img, cmap='gray')
        ax.scatter(meas[0], meas[1], c='red', marker='o', s=100, label='measured')
        ax.scatter(proj[0], proj[1], c='lime', marker='x', s=100, label='projected')
        ax.set_title(title)
        ax.axis('off')

        # zoom in around both points by 'margin'
        xs = [meas[0], proj[0]]
        ys = [meas[1], proj[1]]
        x0 = max(min(xs) - margin, 0)
        x1 = min(max(xs) + margin, img.shape[1])
        y0 = max(min(ys) - margin, 0)
        y1 = min(max(ys) + margin, img.shape[0])

        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


def compute_proj_meas_distance(z_proj, z_meas):
    """Return Euclidean pixel gaps between projected and measured points in left & right images."""
    pt_L_proj, pt_R_proj = two_pts_from_z(z_proj)
    pt_L_meas, pt_R_meas = two_pts_from_z(z_meas)

    proj_meas_dist_L = np.linalg.norm(np.array(pt_L_proj) - np.array(pt_L_meas))
    proj_meas_dist_R = np.linalg.norm(np.array(pt_R_proj) - np.array(pt_R_meas))

    return proj_meas_dist_L, proj_meas_dist_R


def build_filtered_values(vals, z_min, z_max):
    """
    Return a new gtsam.Values that contains
      I. every camera ('c' keys) unchanged, and
      II. only those landmarks ('q' keys) whose Z is in [z_min , z_max].
    """
    out = gtsam.Values()
    for key in vals.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord('c'):  # keep ALL cameras
            out.insert(key, vals.atPose3(key))
        elif sym.chr() == ord('q'):  # candidate landmark
            p = np.asarray(vals.atPoint3(key), dtype=float)
            if z_min <= p[2] <= z_max:  # Z test
                out.insert(key, vals.atPoint3(key))
    return out


def plot_bundle1_poses_landmarks(result_bundle1, z_min, z_max, is_above_view):
    """  3D/2D plot – cameras + landmarks on one plot """
    # 1) filter z values (to see trajectory-landmarks relation in a good scale)
    vals_filt = build_filtered_values(result_bundle1, z_min=z_min, z_max=z_max)

    # 2) plot cameras (trajectory)
    fignum = 1
    gtsam_plot.plot_trajectory(fignum=fignum,
                               values=vals_filt,
                               scale=1)

    # 3) plot *filtered* landmarks
    gtsam_plot.plot_3d_points(fignum=fignum,
                              values=vals_filt,
                              linespec="r.")

    # 4) equal axis scale & perhaps top-down view
    gtsam.utils.plot.set_axes_equal(fignum)
    fig = plt.figure(fignum)
    ax = fig.axes[0]
    if is_above_view:
        ax.view_init(elev=0, azim=-90)  # XZ (top-down) view
        ax.set_proj_type('ortho')
        ax.set_title(f"Top-down (XZ) view – cameras + landmarks    ({z_min} < Z < {z_max} m)")
    else:
        ax.set_title(f"Bundle-1  – cameras + landmarks ({z_min} < Z < {z_max} m)")
    plt.tight_layout()
    plt.show()


def convert_T_into_Rt(T_bundles):
    """Split each 4×4 pose in every bundle window into separate rotation and translation arrays."""
    R_cam_to_window = [np.empty((len(window), 3, 3)) for window in T_bundles]
    t_cam_to_window = [np.empty((len(window), 3)) for window in T_bundles]

    for w, window in enumerate(T_bundles):
        for f, T in enumerate(window):
            R_cam_to_window[w][f] = T[:3, :3]  # 3×3 rotation
            t_cam_to_window[w][f] = T[:3, 3]  # 3-vector translation

    return R_cam_to_window, t_cam_to_window


def plot_keyframe_localization_error_over_time(KF_indices, err_pre, err_post):
    """Plot per-keyframe Euclidean localisation error before and after BA."""
    fig, ax = plt.subplots(figsize=(6.8, 3))
    ax.plot(KF_indices, err_pre,  'r-o', ms=4, label='before BA')
    ax.plot(KF_indices, err_post, 'b-o', ms=4, label='after  BA')
    ax.set_xlabel('Keyframe index')
    ax.set_ylabel('Position error  [m]')
    ax.set_title('Per-keyframe localisation error')
    ax.grid(True, ls=':')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_trajectory_overlay(gt_xyz, pre_xyz, post_xyz,
                            err_pre, err_post):
    """Overlay ground-truth, pre-BA, and post-BA keyframe trajectories in the X-Z plane."""
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.plot(gt_xyz [:,0], gt_xyz [:,2], 'k--',  lw=2, label='ground-truth')
    ax.plot(pre_xyz [:,0], pre_xyz [:,2],  'r-',  lw=1.5,
            label=f'before BA (RMSE {np.sqrt(np.mean(err_pre**2)):.2f} m)')
    ax.plot(post_xyz[:,0], post_xyz[:,2], 'b-',  lw=1.5,
            label=f'after BA (RMSE {np.sqrt(np.mean(err_post**2)):.2f} m)')
    ax.scatter(gt_xyz[0,0],   gt_xyz[0,2],   c='g', s=60, label='start')
    ax.scatter(gt_xyz[-1,0],  gt_xyz[-1,2],  c='m', s=60, label='end')
    ax.set_aspect('equal')
    ax.set_xlabel('X  [m]')
    ax.set_ylabel('Z  [m]  (forward)')
    ax.set_title('KITTI seq. 05 – keyframe trajectories')
    ax.grid(True, ls=':')
    ax.legend(loc='best')
    plt.tight_layout()


def extract_rel_pose_nextKF_to_currKF(bundle_frames, result_bundles):
    """ Extract relative pose  KF_i <-- KF_{i+1}  from each window.
        Note that all the frames in between live in KF_i coordinates, so they
        can be turned into global poses later with a single composition step. """
    rel_poses = []
    for frames, values in zip(bundle_frames, result_bundles):
        next_kf = frames[-1]   # last frame of this window (=first of next window)
        rel_pose = values.atPose3(gtsam.symbol('c', next_kf))   # pose of nextKF in currK coordinates
        rel_poses.append(rel_pose)
    return rel_poses


def rel_window_to_global_poses(KF_indices, rel_poses):
    """ Take relative pose  KF_i <-- KF_{i+1}  from each window &
        chain these relative to absolute poses in the *global* (KF0) frame
        into T_global = T_(global<-KF1) T_(KF1<-KF2)... T_(KFi<-KFi+1) """
    global_poses = {KF_indices[0]: gtsam.Pose3()}  # identity
    for kf_curr, rel_pose in zip(KF_indices[:-1], rel_poses):
        prev = global_poses[kf_curr]                    # T_(global<-KF_curr)
        kf_next = KF_indices[KF_indices.index(kf_curr) + 1]
        global_poses[kf_next] = prev.compose(rel_pose)  # T_(global<-KF_next)
    return global_poses


def gather_landmarks_global(bundle_frames, result_bundles, global_poses):
    """ Return every landmark Point3 expressed in global (KF0) coords. """
    pts = []
    for frames, values in zip(bundle_frames, result_bundles):
        kf0 = frames[0]
        T_glob_curr_win = global_poses[kf0]   # T_(global<-window)
        for key in values.keys(): # for each frame,
            if gtsam.Symbol(key).chr() == ord('q'):
                p_win = values.atPoint3(key)
                p_glob = T_glob_curr_win.transformFrom(p_win)
                pts.append(p_glob)
    return np.asarray(pts)   # (N,3)


def plot_topdown_scene_cropped(pts_xyz,         # all landmarks (world coords)
                               kf_xyz,          # all keyframe centers (world coords)
                               x_lim, z_lim,    # for cropping landmark outliers
                               title="KITTI seq. 05 – top-down (cropped view to ignore outliers)"):
    """Plot X-Z top-down view of landmarks and key-frame centres, cropped by x_lim/z_lim."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pts_xyz[:, 0], pts_xyz[:, 2],
               c='grey', s=3, alpha=0.4, label='3D points')
    ax.plot   (kf_xyz[:, 0], kf_xyz[:, 2],
               'b-o', ms=5, lw=1.2, label='keyframes')

    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if z_lim is not None:
        ax.set_ylim(*z_lim)      # Z is vertical axis in this 2D plot

    ax.set_aspect('equal')
    ax.set_xlabel('X  [m]')
    ax.set_ylabel('Z  [m]  (forward)')
    ax.set_title(title)
    ax.grid(True, ls=':')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def main(data_path=DATA_PATH,
          poses_path=POSES_PATH,
          db_path=DB_PATH,
          est_poses_path=EST_POSES_PATH):
    """
    Run an entire Bundle-Adjustment workflow:

    • Load stereo calibration, feature-tracking DB, and the PnP trajectory.
    • Task 1 – pick one 10-frame track, triangulate its landmark, project it
      back to all frames, and plot reprojection / factor errors.
    • Task 2 – define the first key-frame window, build a factor graph,
      perform local bundle adjustment, and visualise the worst residual before
      and after optimization.
    • Task 3 – slide the BA window across all key-frames, optimize each
      window, chain the relative poses into a global trajectory, compare it to
      KITTI ground truth, and generate summary plots (top-down map,
      trajectory overlay, per-KF error).
    """
    # --- Setting useful structures ---
    # camera matrices
    K, M_L, M_R  = read_cameras(data_path)
    P_L, P_R = K @ M_L, K @ M_R
    K_gtsam = gtsam.Cal3_S2Stereo(extract_intrinsic_param(K, M_R))

    # initialize & load DB
    db = TrackingDB()
    db.load(db_path)

    # upload world-to-camera poses computed during PnP
    with open(est_poses_path, 'rb') as f:
        Rts_wc = pickle.load(f)
    Rts_wc = np.asarray(Rts_wc)

    ### Task 5.1 ###
    print("--- Task 5.1 ---")
    ## Triangulate a 3d point in global coordinates from the last frame
    ## of the track and project this point to all the frames of the track

    # Pick some track of length 10
    tracks_equal_to_10 = db.tracks_equal_to(10)
    i = 0
    track_i_len10 = tracks_equal_to_10[i]
    frames_corr_tracki = db.frames(track_i_len10)
    print(f"The 10 frames corresponding to 'track_id' = {track_i_len10} are:\n{frames_corr_tracki}")
    links_track_i = list(db.track(track_i_len10).values())
    fid_last = frames_corr_tracki[-1]
    link_last = links_track_i[fid_last]

    # Switch to camera-to-world frame
    R_cw_track_i, t_cw_track_i = compute_Rt_c2w(Rts_wc[frames_corr_tracki[:-1]])
    cameras_track_i, poses_track_i = Rt_c2w_gtsam(R_cw_track_i, t_cw_track_i, K_gtsam)

    # The last-frame's left & right pixels corresponding to that landmark
    z_last = gtsam.StereoPoint2(link_last.x_left,
                                link_last.x_right,
                                link_last.y)

    last_cam = cameras_track_i[-1]
    p_world = single_backproject_gtsam(last_cam, z_last)  # 3D world-point

    # Projections of the track's 'p_world' onto each one of the 10 frames
    projections = compute_projections(p_world, cameras_track_i)

    reproj_errs = compute_reproj_errs(projections, links_track_i)
    plot_reproj_err_over_track_len(frames_corr_tracki, reproj_errs)

    graph, values = make_stereo_track_graph(p_world, poses_track_i, links_track_i, K_gtsam)
    factor_errs = factor_errors(graph, values)  # errors before optimization
    plot_factor_err_over_track_len(frames_corr_tracki, factor_errs)

    plot_factor_err_vs_reproj_err(reproj_errs, factor_errs, adjust_errs=False)
    plot_factor_err_vs_reproj_err(reproj_errs, factor_errs, adjust_errs=True)


    ### Task 5.3 ###
    print("\n--- Task 5.3 ---")
    ## Perform local Bundle Adjustment on a small window consisting of consecutive frames.
    ## Each bundle ‘window’ starts and ends in special frames we call keyframes.

    # Break the 3x4 matrices to separated R & t
    R_wc, t_wc = Rts_wc[:, :, :3], Rts_wc[:, :, 3]
    # Convert to a homogenous transformation
    T_wc = Rts_to_homogenous(Rts_wc)

    fid_to_KF = set_keyframe_criterion(t_wc)
    KF_indices = list(fid_to_KF.keys())

    # First-window
    frame_bundle1 = list(range(KF_indices[0], KF_indices[1] + 1))
    R_cw_bundle1, t_cw_bundle1 = compute_Rt_c2w(Rts_wc[frame_bundle1[:-1]])

    cameras_bundle1, poses_bundle1 = Rt_c2w_gtsam(R_cw_bundle1, t_cw_bundle1, K_gtsam)
    fid_to_pose_bundle1 = {fid: pose for (fid, pose) in zip(frame_bundle1, poses_bundle1)}

    links_by_frame = _links_by_frame(db)

    graph_bundle1, values_bundle1 = make_bundle_graph(frame_bundle1[:frame_bundle1[-1]],
                                             links_by_frame,  # dict: track_id -> {fid: Link, ...}
                                             fid_to_pose_bundle1,  # dict: fid -> gtsam.Pose3  (PnP result)
                                             K_gtsam)

    # printing how many factors there are in the graph
    print("total number of factors in this graph :", graph_bundle1.size())

    # computing & printing tot & avg errors before optimization
    total_err_pre, avg_rel_err_pre, rel_errs_pre =\
                compute_total_and_relative_errs(graph_bundle1, values_bundle1)
    print(f"The total factor graph error BEFORE optimization is {total_err_pre:.2f}")
    print(f"The average factor graph error BEFORE optimization is {avg_rel_err_pre:.2f}")

    # performing graph optimization
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph_bundle1, values_bundle1)
    result_bundle1 = optimizer.optimize()
    total_err_post, avg_rel_err_post, rel_errs_post = \
              compute_total_and_relative_errs(graph_bundle1, result_bundle1)

    # computing & printing tot & avg errors after optimization
    print(f"The total factor graph error AFTER optimization is {total_err_post:.2f}")
    print(f"The average factor graph error AFTER optimization is {avg_rel_err_post:.2f}")

    # picking the projection factor with the largest initial error
    max_err_idx_pre = rel_errs_pre.argmax()
    print(graph_bundle1.at(max_err_idx_pre))    # {c0, q548}
    print(f"The factor's largest initial error before optimization is {rel_errs_pre.max():.2f}")

    # z_proj_pre is the projected pixel(s) BEFORE optimization
    z_proj_pre, z_meas = compute_z_proj_z_meas(548, 0, graph_bundle1,
                                               values_bundle1, K_gtsam, max_err_idx_pre)

    print(f"The projected z BEFORE optimization: {z_proj_pre}")
    print(f"The corresponding measured z: {z_meas}")

    # presenting the L & R projections on both images, along with the measurement
    draw_proj_meas_z_LR(z_proj_pre, z_meas, margin=150)

    # z_proj_post is the projected pixel(s) AFTER optimization
    z_proj_post, _ = compute_z_proj_z_meas(548, 0, graph_bundle1,
                                           result_bundle1, K_gtsam, max_err_idx_pre)
    print(f"Largest-error factor AFTER optimization: {rel_errs_post[max_err_idx_pre]:.2f}")
    print(f"The same projected z AFTER optimization: {z_proj_post}")
    draw_proj_meas_z_LR(z_proj_post, z_meas, margin=150)

    dist_L_pre, dist_R_pre = compute_proj_meas_distance(z_proj_pre, z_meas)
    print(f"Reprojection error BEFORE optimization over the left image: {dist_L_pre:.2f}")
    print(f"Reprojection error BEFORE optimization over the right image: {dist_R_pre:.2f}")


    dist_L_post, dist_R_post = compute_proj_meas_distance(z_proj_post, z_meas)
    print(f"Reprojection error AFTER optimization over the left image: {dist_L_post:.2f}")
    print(f"Reprojection error AFTER optimization over the right image: {dist_R_post:.2f}")

    # plotting resulting positions of the first bundle as a 3D graph & top-down graph
    plot_bundle1_poses_landmarks(result_bundle1, z_min=0, z_max=20, is_above_view=False)
    plot_bundle1_poses_landmarks(result_bundle1, z_min=0, z_max=20, is_above_view=True)


    ### 5.4 ###
    print("\n--- Task 5.4 ---")

    ## Split the movie into bundle windows, create a graph and optimize each separately
    T_bundles = [bundle_window(T_wc, KF_indices, kf_i) for kf_i, _ in enumerate(KF_indices)]

    bundle_frames = [list(range(KF_indices[kf_i], KF_indices[kf_i + 1] + 1))
                     for kf_i, _ in enumerate(KF_indices[:-1])]
    bundle_frames.append(list(range(KF_indices[-1], len(T_wc))))

    R_cam_to_window, t_cam_to_window = convert_T_into_Rt(T_bundles)
    camera_poses_bundles = [Rt_c2w_gtsam(R, t, K_gtsam) for R, t in zip(R_cam_to_window, t_cam_to_window)]
    poses_bundles = [pose for cam, pose in camera_poses_bundles]
    fid_to_pose_bundles = []
    for frames, poses in zip(bundle_frames, poses_bundles):
        fid_to_pose_bundles.append({fid: pose for (fid, pose) in zip(frames, poses)})

    # Create graph & value objects for each bundle window
    graphs, values = [], []
    for frames, init_map in zip(bundle_frames, fid_to_pose_bundles):
        g, v = make_bundle_graph(frames, links_by_frame, init_map, K_gtsam)
        graphs.append(g)
        values.append(v)

    # Optimize each bundle window
    result_bundles = [gtsam.LevenbergMarquardtOptimizer(graph, val).optimize()
                             for graph, val in zip(graphs, values)]

    # Save - run just once
    with open("all_bundles.pkl", "wb") as f:
        pickle.dump((KF_indices, bundle_frames, graphs, result_bundles), f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    ## Pose of first frame in LAST window
    # Computing inter-keyframe poses in relative & global coordinates, computing landmark positions
    rel_poses = extract_rel_pose_nextKF_to_currKF(bundle_frames, result_bundles)
    global_poses = rel_window_to_global_poses(KF_indices, rel_poses)
    global_landmarks = gather_landmarks_global(bundle_frames, result_bundles, global_poses)

    last_bundle = result_bundles[-1]  # gtsam.Values
    first_fid_last = bundle_frames[-1][0]
    c0_last_key = gtsam.symbol('c', first_fid_last)
    # (last) window coordinates
    pose_c0_last = last_bundle.atPose3(c0_last_key)   # window coordinates
    print(f"Optimized position of first frame in last window (window coordinates):  \
    {np.round(pose_c0_last.translation(), 8)}")
    # (first) global coordinates
    pose_c0_last_global = global_poses[first_fid_last]    # global <- window
    print(f"Optimized position of first frame in last window (global coordinates):  \
    {np.round(pose_c0_last_global.translation(), 2)}")

    ## Anchoring factor final error
    bundle_idx = 0   # we care about the first window
    graph_idx = graphs[bundle_idx]
    anchor_fac = graph_idx.at(0)  # first factor inserted is of 'PriorFactorPose3' kind
    anchor_err = anchor_fac.error(result_bundles[bundle_idx])  # the residual: r=Log(inv(T_prior) T_estimate)
    print(f'Anchoring factor final error = {anchor_err:.8f}')

    # For indexing
    KF_indices = np.asarray(KF_indices)

    # Keyframe camera centers BEFORE BA (world frame)
    pre_xyz = compute_estimated_centers(Rts_wc)[KF_indices]    # (M,3)

    # Keyframe camera centers AFTER BA (world frame)
    post_xyz = np.stack([global_poses[k].translation() for k in KF_indices])   # (M,3)

    # Keyframe camera centers KITTI ground-truth (world frame)
    poses_gt = np.loadtxt(poses_path).reshape(-1, 3, 4)    # (N,3,4)
    R_gt = poses_gt[:, :, :3]    # (N,3,3)  world -> cam rotation
    t_gt = poses_gt[:, :, 3]     # (N,3)    world -> cam translation
    gt_xyz_all = -np.einsum('nij,nj->ni',       # camera center in world coords
                      R_gt.transpose(0, 2, 1), t_gt)   # (N,3)
    gt_xyz = gt_xyz_all[KF_indices]   # (M,3)  match pre & post

    # Error metrics
    err_pre  = np.linalg.norm(pre_xyz - gt_xyz, axis=1)   # per-KF Euclidean
    err_post = np.linalg.norm(post_xyz - gt_xyz, axis=1)

    ## Plot - A view from above (2d) of the scene, with all keyframes & 3D points
    plot_topdown_scene_cropped(global_landmarks,
                               post_xyz,
                               x_lim=(-300,  300),
                               z_lim=(-200,  400))

    ## Plot – Trajectories overlay
    plot_trajectory_overlay(gt_xyz, pre_xyz, post_xyz,
                            err_pre, err_post)

    ## Plot – Per-keyframe localization error
    plot_keyframe_localization_error_over_time(KF_indices, err_pre, err_post)


if __name__ == '__main__':
    main()
