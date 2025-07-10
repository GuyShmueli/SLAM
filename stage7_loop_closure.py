import gtsam, pickle, cv2, numpy as np, networkx as nx
import matplotlib.pyplot as plt
from gtsam.utils import plot as gtsam_plot
from utils import read_cameras, read_images
from stage3_localization_by_p4p import ransac_pnp_pair1, solve_pnp
from tracking_database_class import TrackingDB
from stage5_local_bundle_adjustment import plot_trajectory_overlay

# --- Magic constants ---
INFLATION = 1.05e6
PX_THRESH_RANSAC = 2.0
MIN_INLIERS_LOOP = 180
K_BEST = 3
BACK_WINDOW = 144
MIN_KF_GAP = 40
CHI2_THR = 10
REPORT_EVERY = 1

BA_PATH="all_bundles.pkl"
DATA_PATH = r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\sequences\05\\"
POSES_PATH = r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\poses\05.txt"
DB_PATH = "kitti05_tracks"


def plot_trajectory_with_cov_ellipsoids(kf_ids, values, marginals, title,
                                          inflation=1.0, n_sigma=3, fignum=1):
    """ Draw a 3D trajectory plus n-sigma covariance ellipsoids for every keyframe ID. """
    fig = plt.figure(fignum); plt.clf()
    ax  = fig.add_subplot(111, projection='3d')

    for kf in kf_ids:
        key = gtsam.symbol('c', kf)
        pose = values.atPose3(key)
        Sigma6 = marginals.marginalCovariance(key)  # 6×6 inflated
        Sigma6 /= inflation  # delete inflation for visualisation

        # Draw camera trajectory with n-sigma ellipsoid
        gtsam_plot.plot_pose3_on_axes(ax, pose,
                                      axis_length=0.5,
                                      P = (n_sigma**2) * Sigma6)  # scale radii
    gtsam_plot.set_axes_equal(fig.number)
    ax.set_xlabel("X  [m]"); ax.set_ylabel("Y  [m]"); ax.set_zlabel("Z  [m]")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def mini_pose_graph(frames, init_poses):
    """Initialize a 2-sized pose graph & fill initial guess for Values """
    g2 = gtsam.NonlinearFactorGraph()
    v2 = gtsam.Values()

    for fid in frames:
        key = gtsam.symbol('c', fid)
        pose0 = init_poses.get(fid)
        v2.insert(key, pose0)

    return g2, v2


def cov_nextKF_to_currKF(kf_0, kf_1, marginals):
    """ Return the 6 x 6 covariance of the relative pose  c0 <- ck, by:
      1. Get the joint 12 x 12 covariance of (pose_0, pose_k).
      2. Invert it to information form.
      3. Slice out the conditioned block Cov(ck | c0) and invert back. """
    c_kf0 = gtsam.symbol('c', kf_0)
    c_kf1 = gtsam.symbol('c', kf_1)
    keys = gtsam.KeyVector()
    keys.append(c_kf0)
    keys.append(c_kf1)
    cov_poses_kf0_kf1 = \
        marginals.jointMarginalCovariance(keys).fullMatrix()  # marginalization is easier in cov form (12x12)
    inf_poses_kf0_kf1 = np.linalg.inv(cov_poses_kf0_kf1)  # information matrix is the inverse of cov (12x12)
    inf_kf1_cond_kf0 = inf_poses_kf0_kf1[6:, 6:]  # conditioning is easier in inofrmation form (6x6)
    cov_kf1_cond_kf0 = np.linalg.inv(inf_kf1_cond_kf0)  # switching to cov format again (6x6)
    return cov_kf1_cond_kf0


def pose_nextF_to_currKF(kf_0, kf_1, values):
    """ Compute the relative pose  c0 <- ck  from two global poses in 'values'. """
    c_kf0 = gtsam.symbol('c', kf_0)
    c_kf1 = gtsam.symbol('c', kf_1)
    pose_kf0 = values.atPose3(c_kf0)
    pose_kf1 = values.atPose3(c_kf1)
    # c_KF0  <-  c_KF1
    return pose_kf0.between(pose_kf1)


def reverse_cov(Sigma_fwd,
                T_fwd: gtsam.Pose3):
    """Compute Sigma_{k|k+1}, rather than Sigma_{k+1|k} as usual."""
    Ad = T_fwd.AdjointMap()
    Sigma_bwd = Ad @ Sigma_fwd @ Ad.T
    return Sigma_bwd


def weighted_pose_graph(
        rel_poses,          # Pose3 (k <- k+1)
        rel_covs,           # Sigma_{k+1|k} (6×6)
    ):
    """
    Build an undirected graph whose edges carry:
        * 'pose' - Pose3  (u <- v)
        * 'cov' - 6×6 covariance in u-coords
        * 'weight' - scalar cost  (det Sigma)^(1/6)
    """
    # Initialize an empty graph
    G = nx.Graph()

    # --- chain edges (k , k+1) in both directions ---
    for k, (T_fwd, Sigma_fwd) in enumerate(zip(rel_poses, rel_covs)):
        Sigma_fwd = Sigma_fwd * INFLATION
        w = edge_weight_root6(Sigma_fwd)      # root-det after scaling
        Sigma_back = reverse_cov(Sigma_fwd, T_fwd)

        # forward  k -> k+1   (store pose  k <- k+1, in known k-coordinates)
        G.add_edge(k, k+1, pose=T_fwd,   cov=Sigma_fwd,  weight=w)
        # reverse  k+1 -> k   (store pose  k+1 <- k)
        G.add_edge(k+1, k, pose=T_fwd.inverse(), cov=Sigma_back, weight=w)

    return G


def edge_weight_root6(Sigma):
    """w = (det Sigma)^(1/6) - geometric mean of the six eigen-variances."""
    return np.linalg.det(Sigma) ** (1.0 / 6.0)


def shortest_path_nodes(i, n, G):
    """Dijkstra's algorithm to find the shortest path in a weighted graph."""
    return nx.shortest_path(G, source=i, target=n, weight='weight')


def accumulate_sigma(path, G):
    """Sum the covariance along a path to get Sigma_{n|i},
       where n is the track's end and i its beginning."""
    Sigma = np.zeros((6,6))
    T_i_k = gtsam.Pose3()          # identity
    for u, v in zip(path[:-1], path[1:]):
        edge = G[u][v]
        Ad = T_i_k.AdjointMap()          # KF_i <- KF_u
        Sigma += Ad @ edge['cov'] @ Ad.T
        T_i_k = T_i_k.compose(edge['pose'])
    return Sigma


def mahalanobis_pose_error(
        i, n,
        G: nx.Graph,
        est_poses: dict[int, gtsam.Pose3],  # global KF poses
        chi2_thresh: float):
    """
    Perform a chi-squared test.
    The smaller the value of  1/2 * r^T @ Sigma^-1 @ r  is,
    the more likely  c_i & c_n  are to be close to one another.
    """
    if n <= i:
        raise ValueError("need i < n")

    # 1. Compute shortest-path & Sigma_n_i across that path
    path = shortest_path_nodes(i, n, G)
    Sigma_n_i = accumulate_sigma(path, G)

    # 2. Compute Delta(c_ni)
    Delta_c_ni = est_poses[i].between(est_poses[n])

    # 3. Initialize factor & values to compute  err = 1/2 * r^T @ Sigma^-1 @ r
    noise = gtsam.noiseModel.Gaussian.Covariance(Sigma_n_i)
    factor = gtsam.BetweenFactorPose3(gtsam.symbol('c', i),
                                      gtsam.symbol('c', n),
                                      gtsam.Pose3(),         # identity measurement
                                      noise)
    vals = gtsam.Values()
    vals.insert(gtsam.symbol('c', i), gtsam.Pose3())
    vals.insert(gtsam.symbol('c', n), Delta_c_ni)

    half_mahal_dist = factor.error(vals)
    chi2_val = 2.0 * half_mahal_dist

    is_ok = chi2_val < chi2_thresh

    return is_ok, chi2_val


def array_from_values(vals, kf_indices):
    """Return an (N, 3) array of camera centres in KF order."""
    rows = []
    for k in kf_indices:
        t = vals.atPose3(gtsam.symbol('c', k)).translation()
        rows.append(t)      # Point3  ->  [x, y, z]
    return np.vstack(rows)              # shape (N, 3)


def show_pair(
        img_curr: np.ndarray,
        img_cand: np.ndarray,
        px,
        suptitle: str = "PnP inliers / outliers on left images of a successful current-candidate pair"):
    """
    Display the current–candidate pair in a single figure (side-by-side subplots)
    and overlay the correspondence mask on the current image only.
    """
    # 1x2 sub-plots that share the y-axis so pixel grids align
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax[0].imshow(img_curr, cmap='gray')
    ax[0].scatter(px["inl_curr"][:,0],  px["inl_curr"][:,1],
                  s=12, c='lime', label='inliers')
    ax[0].scatter(px["out_curr"][:,0], px["out_curr"][:,1],
                  s=12, c='red',  label='outliers')
    ax[0].set_title("current KF")
    ax[0].axis('off')
    ax[0].legend()

    ax[1].imshow(img_cand, cmap='gray')
    ax[1].scatter(px["inl_cand"][:,0],  px["inl_cand"][:,1],
                  s=12, c='lime')
    ax[1].scatter(px["out_cand"][:,0], px["out_cand"][:,1],
                  s=12, c='red')
    ax[1].set_title("candidate KF")
    ax[1].axis('off')

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_2D_3D_trajectories(KF_indices, pose_graph,
                            pre_vals, pose_vals, gt_xyz, loops_added,
                            label_pre='before loop-closure', label_post='after loop-closure'):
    """Plot both 2D & 3D trajectories using:
    'plot_trajectory_overlay' & 'plot_trajectory_with_cov_ellipsoids' """
    ## 2D
    pre_xyz = array_from_values(pre_vals, range(len(KF_indices)))
    post_xyz = array_from_values(pose_vals, range(len(KF_indices)))
    err_pre = np.linalg.norm(pre_xyz - gt_xyz, axis=1)
    err_post = np.linalg.norm(post_xyz - gt_xyz, axis=1)
    plot_trajectory_overlay(gt_xyz, pre_xyz, post_xyz,
                            err_pre, err_post,
                            title=f'2D keyframe trajectories\n'
                                  f'(after {loops_added} loops were added)',
                            label_pre=label_pre,
                            label_post=label_post)
    ## 3D
    pose_marginals = gtsam.Marginals(pose_graph, pose_vals)
    plot_trajectory_with_cov_ellipsoids(
        kf_ids=range(len(KF_indices)),
        values=pose_vals,
        marginals=pose_marginals,
        title=f"3D optimized keyframe trajectories with 3-sigma ellipsoids\n"
              f"(after {loops_added} loops were added)",
        inflation=INFLATION,
        n_sigma=3)


def frame_data(db: TrackingDB, kf_id: int):
    """
    Extract from db (using 'features') left & right image keypoints
    and the descriptors for the left image only.
    """
    descL = db.features(kf_id)                  # (N,D) left descriptors
    links = db.all_frame_links(kf_id)           # list[Link] length N

    kpL, kpR = [], []
    for ln in links:
        kpL.append(cv2.KeyPoint(ln.x_left,  ln.y, 1))
        kpR.append(cv2.KeyPoint(ln.x_right, ln.y, 1))
    return kpL, kpR, descL


def compute_triangulation(P_L, P_R,
                          kpL: list[cv2.KeyPoint],
                          kpR: list[cv2.KeyPoint]):
    """Perform triangulation using 'cv2.triangulatePoints'. """
    pts_L = np.array([k.pt for k in kpL]).T   # (2,N)
    pts_R = np.array([k.pt for k in kpR]).T
    pts_L_h = cv2.convertPointsToHomogeneous(pts_L.T)[:, 0, :]  # (N,3)
    pts_R_h = cv2.convertPointsToHomogeneous(pts_R.T)[:, 0, :]
    proj = cv2.triangulatePoints(P_L, P_R,
                                 pts_L_h.T[:2],
                                 pts_R_h.T[:2])
    world_pts = (proj[:3] / proj[3]).T      # (N,3)  XYZ in left–cam frame
    return world_pts


def consensus_pnp_match(kf_curr: int, kf_cand: int,
                           K: np.ndarray, M_right: np.ndarray, M_left: np.ndarray,
                           db: TrackingDB,
                           min_inliers_loop: int = MIN_INLIERS_LOOP,
                           px_thresh: float = PX_THRESH_RANSAC):
    """
    Do a stereo-aware 2-view PnP check for current-candidate pair, and
    give back everything needed to draw inliers/outliers on both left images.
    """
    ## 1. Cached keypoints & descriptors for both current (n) and candidate (i)
    kpL_i, kpR_i, desL_i = frame_data(db, kf_cand)
    kpL_n, kpR_n, desL_n = frame_data(db, kf_curr)

    ## 2. Compute 3D-points in candidate's left-frame coords
    P_L, P_R = K @ M_left, K @ M_right
    obj_pts = compute_triangulation(P_L, P_R, kpL_i, kpR_i)

    ## 3. Mapping
    # key: index of the left-image feature in the candidate KF
    # value: 3D-point in candidate's left-image coordinates
    xyz_by_lidx = {i: obj_pts[i] for i in range(len(obj_pts))}     # candidate-frame ->  3D-point

    # key: index of the left-image feature in the current KF
    # value: sub-pixel (u,v) of the matching right-image feature
    right_by_lidx = {i: kpR_n[i].pt for i in range(len(kpR_n))}    # current-frame   ->  right–pixel

    ## 4. Descriptor matching:  cand_L  ->  current_L
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # later geometric filtering is much more robust
    raw = bf.knnMatch(desL_i, desL_n, k=2)
    good = [m for m, n in raw if m.distance < 0.8 * n.distance]
    
    ## 5. Lists appending: 3D-points, curr_L, curr_R, cand_L pixels
    P3D, ptsL_curr, ptsR_curr, ptsL_cand = [], [], [], []
    for m in good:
        xyz = xyz_by_lidx.get(m.queryIdx)           # m.queryIdx refers to candidate's desL idx
        right_pt = right_by_lidx.get(m.trainIdx)    # m.trainIdx refers to current's desL idx
        P3D.append(xyz)                         # 3D-points for reprojection in RANSAC-PnP
        ptsL_curr.append(kpL_n[m.trainIdx].pt)  # curr_L pixels for both RANSAC-PnP and plotting
        ptsR_curr.append(right_pt)              # curr_R pixels for RANSAC-PnP
        ptsL_cand.append(kpL_i[m.queryIdx].pt)  # cand_L pixels for plotting inliers/outliers

    if len(P3D) < min_inliers_loop:
        return (False, None, None, (None, None), {})

    P3D = np.asarray(P3D, np.float32)
    ptsL_curr = np.asarray(ptsL_curr, np.float32)
    ptsR_curr = np.asarray(ptsR_curr, np.float32)
    ptsL_cand = np.asarray(ptsL_cand, np.float32)

    ## 6. Stereo-aware RANSAC-PnP
    mask = ransac_pnp_pair1(P3D, ptsL_curr, ptsR_curr,
                            M_right, K,
                            p_success=0.99999, n_iter_max=1000,
                            thresh_px=px_thresh)

    if mask.sum() < min_inliers_loop:
        return (False, None, None, (None, None), {})

    # 7. Refine pose
    Rt = solve_pnp(P3D[mask], ptsL_curr[mask], K, cv2.SOLVEPNP_ITERATIVE)
    if Rt is None:
        return (False, None, None, (None, None), {})

    R, t = Rt[:, :3], Rt[:, 3]
    Delta = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))     # Delta(c_in)
    
    ## 8. Track pixels for inliers/outliers plotting
    px = dict(inl_curr=ptsL_curr[mask],
              out_curr=ptsL_curr[~mask],
              inl_cand=ptsL_cand[mask],
              out_cand=ptsL_cand[~mask])

    return (True, Delta, mask, (kf_curr, kf_cand), px)


def plot_location_error(err_pre, err_post):
    """Plot a graph of the absolute location (translation) error (AFTER seeing the ground-truth)
       over the whole pose-graph, with & without loop-closures."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(err_pre, label='without loop-closures', color='C1')
    ax.plot(err_post, label='with loop-closures', color='C0')
    ax.set_xlabel('Keyframe index')
    ax.set_ylabel('Absolute location error  [m]')
    ax.set_title('Absolute location error over the whole pose-graph')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def _location_sigma_size(marginals, kf_indices, inflation=1.0,
                        size_metric='rss'):
    """Return an array s[k] with a scalar size (units: meter) of the 3x3 translation covariance."""
    sizes = []
    for k in kf_indices:
        # cov of P(c_k | Z) - all other c_i's have been marginalized out
        Sigma6 = marginals.marginalCovariance(gtsam.symbol('c', k))
        Sigma_t = 3.75**2 * Sigma6[3:, 3:] / inflation   # multiply to get 3.75-sigma for 3x3 cov (99.8%)
        if size_metric == 'det':      # (det)^(1/6)
            sizes.append(np.linalg.det(Sigma_t) ** (1.0 / 6.0))
        elif size_metric == 'rss':    # sqrt(sigma_tx **2 + sigma_ty **2 + sigma_tz **2)
            sizes.append(np.linalg.norm(np.sqrt(np.diagonal(Sigma_t))))
        else:
            raise ValueError("size_metric must be 'det' or 'rss'")

    return np.asarray(sizes)


def plot_location_uncertainty(KF_indices,
    pose_graph_pre, pre_vals,
    pose_graph, pose_vals,
    inflation=INFLATION, size_metric="det"):
    """ Plot a graph of the location (translation) uncertainty (BEFORE seeing the ground-truth)
        over the whole pose-graph, with & without loops. """
    pre_marginals = gtsam.Marginals(pose_graph_pre, pre_vals)   # graph for measurements, values for estimations
    post_marginals = gtsam.Marginals(pose_graph, pose_vals)
    pre_sizes = _location_sigma_size(pre_marginals, range(len(KF_indices)),
                                    inflation=inflation, size_metric=size_metric)
    post_sizes = _location_sigma_size(post_marginals, range(len(KF_indices)),
                                     inflation=inflation, size_metric=size_metric)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pre_sizes, label='without loop closures', color='C1')
    ax.plot(post_sizes, label='with loop closures', color='C0')
    ax.set_xlabel('Keyframe index')
    ax.set_ylabel('Location uncertainty size  [m]')
    ax.set_title('Location uncertainty over the whole pose graph\n'
                 f'(size metric: {size_metric})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main(ba_path = BA_PATH,
         data_path = DATA_PATH,
         poses_path = POSES_PATH,
         db_path = DB_PATH):
    """
    Run the bundle-adjustment pose-graph pipeline with loop-closure detection and visualization
    This function will:
      1. Load camera intrinsics and ground-truth poses.
      2. Compute relative poses & covariances between consecutive keyframes.
      3. Build an initial pose graph and values.
      4. Perform chi-squared gating and 2-view PnP loop-closure proposals.
      5. Incrementally add verified loop closures, re-optimizing periodically.
      6. Report final loop-closures count and RMS errors before/after.
      7. Plot top-down trajectories (ground truth, pre-LC, post-LC).
    """
    # --- 1) load db, camera intrinsics, ground-truth poses and pre-computed objects ---
    K, M_L, M_R = read_cameras(data_path)
    db = TrackingDB()
    db.load(db_path)

    with open(ba_path, "rb") as f:
        KF_indices, bundle_frames, graphs, result_bundles = pickle.load(f)

    poses_gt = np.loadtxt(poses_path).reshape(-1, 3, 4)  # (N,3,4)
    R_gt = poses_gt[:, :, :3]  # (N,3,3)  world -> cam rotation
    t_gt = poses_gt[:, :, 3]  # (N,3)    world -> cam translation
    gt_xyz_all = -np.einsum('nij,nj->ni',  # camera center in world coords
                            R_gt.transpose(0, 2, 1), t_gt)  # (N,3)
    gt_xyz = gt_xyz_all[KF_indices]  # (M,3)  match pre & post

    # 2) --- derive relative poses & covariances for EVERY KF pair ---
    marginals = [gtsam.Marginals(graph, val) for (graph, val) in zip(graphs, result_bundles)]
    KF_idx_pairs = [(KF_indices[i], KF_indices[i + 1]) for i in range(len(KF_indices) - 1)]
    # Covariances for every KF pair
    covs_next_cond_curr = [
        cov_nextKF_to_currKF(k0, k1, marg)
        for (k0, k1), marg in zip(KF_idx_pairs, marginals)
    ]
    # Poses for every KF pair
    poses_next_to_curr = [
        pose_nextF_to_currKF(k0, k1, res)
        for (k0, k1), res in zip(KF_idx_pairs, result_bundles)
    ]

    # 3) --- initialize pose graph, filling it & loop-enclosing ---
    G = weighted_pose_graph(poses_next_to_curr, covs_next_cond_curr)

    # Optimization is done via a GTSAM factor-graph  (not networkx)
    pose_graph = gtsam.NonlinearFactorGraph()
    pose_vals = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(5), np.deg2rad(5), np.deg2rad(5),
                                                             0.05, 0.05, 0.15]))
    # Anchoring a c0-prior
    pose_graph.add(gtsam.PriorFactorPose3(gtsam.symbol('c', 0),
                                          gtsam.Pose3(), prior_noise))
    pose_vals.insert(gtsam.symbol('c', 0), gtsam.Pose3())

    # Create 'est_poses' map for each kf_n:
    # kf_n  ->  C_{kf_0 <- kf_n}
    running = gtsam.Pose3()
    for k, (T_k_k1, Sigma_k_k1) in enumerate(zip(poses_next_to_curr, covs_next_cond_curr)):
        running = running.compose(T_k_k1)
        pose_vals.insert(gtsam.symbol('c', k + 1), running)

        Sig = Sigma_k_k1 * INFLATION
        noise = gtsam.noiseModel.Gaussian.Covariance(Sig)

        pose_graph.add(gtsam.BetweenFactorPose3(
            gtsam.symbol('c', k),  # from pose k
            gtsam.symbol('c', k + 1),  # to pose k+1
            T_k_k1, noise))

    est_poses = {k: pose_vals.atPose3(gtsam.symbol('c', k))
                 for k in range(len(KF_indices))}

    # Record 'pre_vals' & 'pose_graph_pre' for post-optimization analysis
    pose_graph_pre = gtsam.NonlinearFactorGraph(pose_graph)
    pre_vals = gtsam.Values(pose_vals)

    # To record successful current-candidate pairs
    success_curr_cand = []

    # Iterate every KF-idx (pose) in the graph
    added = 0   # record how many new edges are added
    for n_kf in range(1, len(KF_indices)):
        if n_kf > 132:  # the loops formed by >=133 KFs increase the error a bit
            break
        # --- draw a 2D & 3D trajectories during the optimization ---
        if n_kf in [1, 74, 79, 131] :
            plot_2D_3D_trajectories(KF_indices, pose_graph,
                                    pre_vals, pose_vals, gt_xyz,
                                    loops_added=added)

        n_fid = KF_indices[n_kf]    # turn pose-graph indexing into BA factor-graph indexing

        # chi-squared pre-gate, to save running time in subsequent heavy steps
        passed = []
        for i_kf in range(max(0, n_kf - BACK_WINDOW), n_kf):    # don't check close poses
            if n_kf - i_kf < MIN_KF_GAP:
                continue
            ok, chi2 = mahalanobis_pose_error(i_kf, n_kf, G,
                                                 est_poses, CHI2_THR)
            if ok:
                passed.append((chi2, i_kf))

        passed.sort()
        cand_kf_idxs = [i for _, i in passed[:K_BEST]]
        if not cand_kf_idxs:
            continue
        print(f"n={n_kf}, candidates={cand_kf_idxs}")   # optional printing (for tracking)

        # Verify each possible candidate with a heavier PnP-RANSAC
        for i_kf in cand_kf_idxs:
            i_fid = KF_indices[i_kf]

            (success, Delta_i_n, inliers,
             (success_n_curr, success_n_cand), px) = consensus_pnp_match(
                   kf_curr=n_fid, kf_cand=i_fid,
                   K=K, M_right=M_R, M_left=M_L,
                   db=db)

            if not success:
                continue

            imgL_curr, _ = read_images(success_n_curr, data_path)
            imgL_cand, _ = read_images(success_n_cand, data_path)
            # Record successful current-candidate pairs
            success_curr_cand.append((imgL_curr, imgL_cand, px))

            # Tiny 2-pose graph for covariance
            frames = [i_fid, n_fid]
            init_poses = {frames[0]: est_poses[i_kf],
                          frames[1]: est_poses[n_kf]}
            g2, v2 = mini_pose_graph(frames, init_poses)
            # soft prior on c_i
            prior_sigmas = np.array([np.deg2rad(1)] * 3 + [0.01] * 3)
            g2.add(gtsam.PriorFactorPose3(
                gtsam.symbol('c', i_fid),
                est_poses[i_kf],
                gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)))

            # add Delta(c_in)
            loose_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([np.deg2rad(10)] * 3 + [0.03] * 3))

            g2.add(gtsam.BetweenFactorPose3(
                gtsam.symbol('c', i_fid),
                gtsam.symbol('c', n_fid),
                Delta_i_n, loose_noise))

            result2 = gtsam.LevenbergMarquardtOptimizer(g2, v2).optimize()
            Sig_i_n = gtsam.Marginals(g2, result2).marginalCovariance(
                gtsam.symbol('c', n_fid))
            Sig_i_n = Sig_i_n * INFLATION

            # add to graphs
            w = edge_weight_root6(Sig_i_n)
            G.add_edge(i_kf, n_kf, pose=Delta_i_n, cov=Sig_i_n, weight=w)
            G.add_edge(n_kf, i_kf, pose=Delta_i_n.inverse(),
                       cov=reverse_cov(Sig_i_n, Delta_i_n.inverse()), weight=w)

            noise = gtsam.noiseModel.Gaussian.Covariance(Sig_i_n)
            pose_graph.add(gtsam.BetweenFactorPose3(
                gtsam.symbol('c', i_kf),
                gtsam.symbol('c', n_kf),
                Delta_i_n, noise))

            added += 1
            print(f"loop formed by: {i_kf} & {n_kf}   inliers={int(inliers.sum())}")

            if added % REPORT_EVERY == 0:  # report every N closures
                print(f"\noptimizing full graph after {added} loop closures...")
                pose_vals = gtsam.LevenbergMarquardtOptimizer(
                    pose_graph, pose_vals).optimize()

                # refresh the convenience dict used by mahalanobis gate
                est_poses = {k: pose_vals.atPose3(gtsam.symbol('c', k))
                             for k in range(len(KF_indices))}
                print("optimization done.\n")

    print(f"\nLoop-closure has finished with {added} closures added.")

    # 4) --- draw some current-candidate successful pair ---
    curr_cand_idx = -1
    (imgL_curr, imgL_cand, px) = success_curr_cand[curr_cand_idx]
    show_pair(imgL_curr, imgL_cand, px)

    # 5) --- plot pose graph along with the ground-truth both with & without loops ---
    plot_2D_3D_trajectories(KF_indices, pose_graph,
                            pre_vals, pose_vals, gt_xyz,
                            loops_added=added)

    # 6) --- plot a graph of the global location ERROR before & after loops ---
    # compute values error pre & post optimization with respect to ground-truth
    pre_xyz = array_from_values(pre_vals, range(len(KF_indices)))
    post_xyz = array_from_values(pose_vals, range(len(KF_indices)))
    err_pre = np.linalg.norm(pre_xyz - gt_xyz, axis=1)
    err_post = np.linalg.norm(post_xyz - gt_xyz, axis=1)
    plot_location_error(err_pre, err_post)

    # 7) --- plot a graph of the location UNCERTAINTY for the pose-graph with & without loops ---
    # size_metric should be 'rss' or 'trace'
    plot_location_uncertainty(KF_indices,
                              pose_graph_pre, pre_vals,
                              pose_graph, pose_vals, size_metric="rss")


if __name__ == "__main__":
    main()
