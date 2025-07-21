import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import read_images, compute_kps_descs_matches, read_cameras
from stage2_mapping_by_triangulation import \
    create_inlier_outlier_matches, compute_triangulation_opencv, plot_3d_cloudpoint

# --- Magic constants ---
MAX_FRAME = 2_600
DATA_PATH = r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\sequences\05\\"
POSES_PATH = r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\poses\05.txt"
AKAZE_THRESH = 0.0005


def solve_pnp(obj_pts, img_pts, K_mat, flag):
    """ Perform a PnP algorithm. """

    def _rotation_vec_to_mat(R_vec, t_vec):
        """ A helper that takes a vectoric rotation and translation
        and turns into an extrinsic matrix [R|t]. """
        R_mat, _ = cv2.Rodrigues(R_vec)
        return np.hstack((R_mat, t_vec))

    is_ok, R_vec, t_vec = cv2.solvePnP(obj_pts, img_pts, K_mat, None, flags=flag)
    extrinsic_mat = None
    if is_ok:
        extrinsic_mat = _rotation_vec_to_mat(R_vec, t_vec)

    return extrinsic_mat


def project(Xw, P):
    """ Project a world-point onto an image with camera matrix P. """
    Xw_h = np.column_stack([Xw, np.ones(len(Xw))])      # N×4
    pix  = (P @ Xw_h.T).T
    return pix[:, :2] / pix[:, 2:3]                     # N×2


def match_inter_pairs(des_L0, des_L1, cross_check=True):
    """ Match between the left image in previous-frame to the
    left image in current-frame. """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, cross_check)
    matches01 = bf.match(des_L0, des_L1)
    matches01 = sorted(matches01, key=lambda x: x.distance)
    return matches01


def create_L_R_pixels(matches01, intra_matches, intra_inlier_matches_indices, kps_L, kps_R, pair_0_or_1):
    """ Within the stereo pair, take just those matches that correspond
    to the obtained inter-pair matches. """

    def _convert_left_to_right(intra_matches):
        """ A helper that creates a dictionary mapping the left
        image's descriptor indices to the right ones."""
        L_to_R = {m.queryIdx: m.trainIdx for m in intra_matches}
        return L_to_R

    # 1) build a map: left1_kp_idx  ->  right1_kp_idx
    L_to_R = _convert_left_to_right(intra_matches)

    # 2) recover original left-1 kp index for each frame-to-frame match
    pix_L = []  # left image pixels list
    pix_R = []  # right image pixels list

    for m in matches01:
        # L0 indices that were both used to create 01
        if pair_0_or_1 == 0:
            L_index = intra_inlier_matches_indices[m.queryIdx]
        # L1
        elif pair_0_or_1 == 1:
            L_index = intra_inlier_matches_indices[m.trainIdx]
        else:
            raise ValueError("pair_0_or_1 should be either 0 or 1")

        # left pixel
        pix_L.append(kps_L[L_index].pt)

        # right pixel
        R_idx = L_to_R[L_index]
        pix_R.append(kps_R[R_idx].pt)

    pix_L = np.asarray(pix_L, dtype=np.float32)
    pix_R = np.asarray(pix_R, dtype=np.float32)

    return pix_L, pix_R


def supporters_pair(meas_L, proj_L, meas_R, proj_R, thresh=2.0):
    """ Create a mask, containing 'True' for error lower than thresh for
      both left-right images. """

    def _compute_err(meas_pts, proj_pts):
        """ Compute the error between the measured pixels
        and the projected pixels. """
        err = np.linalg.norm(meas_pts - proj_pts, axis=1)  # pixel distance
        return err

    err_L = _compute_err(meas_L, proj_L)
    err_R = _compute_err(meas_R, proj_R)
    return (err_L <= thresh) & (err_R <= thresh)


def ransac_pnp_pair1(obj_pts,
                     meas_L1, meas_R1,
                     M_right, K,
                     p_success   = 0.9999,   # desired global confidence
                     n_iter_max  = 1500,     # hard cap
                     early_inliers = None,  # stop when reached
                     thresh_px   = 2.0):
    """ Adaptive RANSAC-PnP: update the required iteration count
    as the inlier ratio estimate improves. """
    N          = len(obj_pts)
    m          = 4                           # minimal sample size (AP3P)
    best_mask  = None
    best_count = 0
    iter_done  = 0
    M_right_h = np.vstack([M_right, [0, 0, 0, 1]])     # cache once

    # until we know the inlier ratio, assume the worst -> run n_iter_max
    n_iter_required = n_iter_max

    rng = np.random.default_rng(12345)

    while iter_done < n_iter_required and iter_done < n_iter_max:
        iter_done += 1

        idx4  = rng.choice(N, m, replace=False)
        Rt_L1 = solve_pnp(obj_pts[idx4], meas_L1[idx4],
                          K, cv2.SOLVEPNP_AP3P)
        if Rt_L1 is None:
            continue

        # --- reprojection ---
        P_L1    = K @ Rt_L1
        proj_L1 = project(obj_pts, P_L1)

        Rt_L1_h = np.vstack([Rt_L1, [0, 0, 0, 1]])
        Rt_R1   = (M_right_h @ Rt_L1_h)[:3]
        P_R1    = K @ Rt_R1
        proj_R1 = project(obj_pts, P_R1)

        mask = supporters_pair(meas_L1, proj_L1,
                                meas_R1, proj_R1,
                                thresh_px)
        inliers = mask.sum()
        if inliers > best_count:
            best_count, best_mask = inliers, mask

            # --- adaptive update of required iterations ---
            inlier_ratio = best_count / N
            if inlier_ratio > 0:                     # avoid log(0)
                prob_hit_all_inliers = inlier_ratio**m
                # formula:  k ≥ log(1-p) / log(1 - w^m)
                n_iter_required = math.ceil(
                    math.log(1 - p_success) /
                    math.log(1 - prob_hit_all_inliers)
                )

        if early_inliers is not None and best_count >= early_inliers:
            break

    return best_mask


def _compute_4_cameras_rel_position(Rt_right, Rt_left1):
    """ A helper that computes the relative position of the 4 cameras,
    based on random 4 world-points. """

    def _cam_center(Rt):
        """ A helper to compute a camera's center in world-coordinates. """
        R, t = Rt[:, :3], Rt[:, 3]
        return -R.T @ t

    C_left0 = np.zeros(3)
    C_right0 = _cam_center(Rt_right)

    C_left1 = _cam_center(Rt_left1)
    R1 = Rt_left1[:, :3]
    C_right1 = C_left1 + R1.T @ C_right0

    centers = np.vstack([C_left0, C_right0, C_left1, C_right1])  # 4 × 3
    return centers

def plot_4_cameras_rel_position(Rt_right, Rt_left1):
    """ Plot the relative positions of 4 camera in top view (X-Z plane). """
    centers = _compute_4_cameras_rel_position(Rt_right, Rt_left1)
    X, Z = centers[:, 0], centers[:, 2]

    labels = ['left\u2080', 'right\u2080', 'left\u2081', 'right\u2081']
    colors = ['b', 'r', 'g', 'm']

    fig, ax = plt.subplots()
    ax.scatter(X, Z, c=colors, s=60)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (X[i] + 0.05, Z[i] + 0.05))

    pad = 0.2
    ax.set_xlim(X.min() - pad, X.max() + pad)
    ax.set_ylim(Z.min() - pad, Z.max() + pad)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Top-view: relative positions of the four cameras')
    plt.show()

def project_pair0(world_pts, P_L0, P_R0):
    """ A helper that projects world-points onto pair-0. """
    left0_proj = project(world_pts, P_L0)
    right0_proj = project(world_pts, P_R0)
    return left0_proj, right0_proj


def project_pair1(world_pts, K, Rt_left1, M_R):
    """ A helper that projects world-points onto pair-1. """
    left1_proj = project(world_pts, K @ Rt_left1)
    Rt_left1_hom = np.vstack([Rt_left1, [0, 0, 0, 1]])
    M_R_hom = np.vstack([M_R, [0, 0, 0, 1]])
    # compose world→left-1 with (left-1→right-1 baseline)
    Rt_right1 = (M_R_hom @ Rt_left1_hom)[:3, :]  # 3×4  extrinsic
    right1_proj = project(world_pts, K @ Rt_right1)
    return left1_proj, right1_proj


def show_matches(img, pts, support_mask, title, sub_idx, nrows=1, ncols=2):
    """ Plot supporter & non-supporter matches on an image-pair. """
    plt.subplot(nrows, ncols, sub_idx)
    plt.imshow(img, cmap='gray')

    # non-supporters (yellow x)
    regular_mask = ~support_mask
    plt.scatter(pts[regular_mask, 0], pts[regular_mask, 1],
                s=12, marker='x',  c='yellow', label='non-supporter')

    # supporters (red o)
    plt.scatter(pts[support_mask, 0],  pts[support_mask, 1],
                s=12, marker='o',  c='red',    label='supporter')

    plt.title(title)
    plt.axis('off')
    plt.legend(loc='lower right', fontsize=8)


def compute_rel_poses_entire_movie(data_path, max_frame,
                                   K, P_L, P_R, M_R,
                                   akaze_thresh):
    """ Estimate the relative pose R|t of every frame k  (k = 0 … max_frame-1)
    with respect to frame k-1. Utilizes caching to save running time. """
    # --- 1. frame-0 pre-processing (acts as “prev”) ---
    img_L_prev, img_R_prev = read_images(0, data_path)

    kp_L_prev, des_L_prev, kp_R_prev, _, matches_prev = \
        compute_kps_descs_matches(img_L_prev, img_R_prev, threshold=akaze_thresh)

    inlier_matches_prev, _ = \
        create_inlier_outlier_matches(matches_prev, kp_L_prev, kp_R_prev)

    inlier_idx_L_prev = [m.queryIdx for m in inlier_matches_prev]
    inlier_des_L_prev = des_L_prev[inlier_idx_L_prev]

    # triangulated cloud lives in *frame-0* left-camera axes
    X_prev = compute_triangulation_opencv(P_L, P_R,
                                          kp_L_prev, kp_R_prev,
                                          inlier_matches_prev)

    # container for all relative poses
    relative_Rts = []

    # --- 2. main loop (frames 1 … max_frame-1) ---
    for idx in range(1, max_frame):
        # --- current stereo pair ---
        img_L, img_R = read_images(idx, data_path)
        kp_L, des_L, kp_R, _, matches = \
            compute_kps_descs_matches(img_L, img_R, threshold=akaze_thresh)

        inlier_matches, _ = create_inlier_outlier_matches(matches, kp_L, kp_R)
        inlier_idx_L = [m.queryIdx for m in inlier_matches]
        inlier_des_L = des_L[inlier_idx_L]

        # --- inter-frame (prev <-> current) matching ---
        matches01 = match_inter_pairs(inlier_des_L_prev, inlier_des_L)
        matches01_idx_prev = [m.queryIdx for m in matches01]  # indices in prev slice

        # 3-D correspondences come from previous frame cloud
        X_corr = X_prev[matches01_idx_prev]

        pix_L_curr, pix_R_curr = create_L_R_pixels(matches01, inlier_matches,
                                                      inlier_idx_L,
                                                      kp_L, kp_R, 1)

        # --- robust PnP on correspondences ---
        inlier_mask = ransac_pnp_pair1(X_corr,
                                       pix_L_curr, pix_R_curr,
                                       M_R, K)

        Rt_prev_to_curr = solve_pnp(X_corr[inlier_mask],
                                    pix_L_curr[inlier_mask],
                                    K,
                                    cv2.SOLVEPNP_ITERATIVE)

        relative_Rts.append(Rt_prev_to_curr)

        # --- prepare cache for next loop ---
        X_prev = compute_triangulation_opencv(P_L, P_R,
                                              kp_L, kp_R,
                                              inlier_matches)
        inlier_des_L_prev = inlier_des_L

    return relative_Rts


def plot_pair1_and_transformed_pair0_from_above(world_pts0, world_pts1, Rt_left1):
    """ Plotting a point cloud from above, representing world-points computed from:
    I. pair1 triangulation.
    II. pair0 triangulation followed by Rt_left1 transformation. """
    world_pts0_h = np.hstack((world_pts0, np.ones((world_pts0.shape[0], 1))))  # N×4
    world_pts0_in_left1 = (Rt_left1 @ world_pts0_h.T).T  # N×3

    plt.figure(figsize=(7, 7))
    plt.scatter(world_pts0_in_left1[:, 0], world_pts0_in_left1[:, 2],   # only X, Z
                s=6, label='pair 0 -> frame 1', alpha=0.6, c='tab:orange')
    plt.scatter(world_pts1[:, 0], world_pts1[:, 2],
                s=6, label='pair 1 (native)', alpha=0.6, c='tab:blue')

    ax = plt.gca()  # current Axes
    ax.set_xlim(-40, 40)  # x limits first …
    ax.set_ylim(0, 100)  # … and y limits
    ax.set_aspect('equal', adjustable='box')  # keep 1:1 scale, don’t change limits

    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)  [forward]')
    ax.set_title('Alignment of two stereo point clouds in left$_1$ frame')
    ax.grid(True, linestyle='--', linewidth=0.4)
    ax.legend()
    plt.tight_layout()  # optional: nicer spacing
    plt.show()

def compute_estimated_centers(relative_Rts):
    """ Compute each camera's center (for each frame)
    in global coordinates, by:
    I. Promote each [R_k | t_k] to a 4 × 4 homogeneous matrix T_k.
    II. Chain the matrices to obtain the cumulative world-to-camera
       transform for each frame.
    III. Split the cumulative transform back into its rotation R_k and
       translation t_k parts.
    IV. Computing each camera k center in world-coords as C_w = -R.T * t. """
    T = np.zeros((len(relative_Rts), 4, 4), dtype=relative_Rts[0].dtype)
    T[:, :3, :4] = relative_Rts
    T[:, 3, 3] = 1

    pose_w2c = np.eye(4, dtype=T.dtype)
    poses_w2c = [pose_w2c[:3]]

    for T_rel in T:
        pose_w2c = T_rel @ pose_w2c
        poses_w2c.append(pose_w2c[:3])

    poses_w2c = np.asarray(poses_w2c)  # (N+1, 3, 4)
    R_est = poses_w2c[:, :, :3]
    t_est = poses_w2c[:, :, 3]
    # Equivalent to: C_est = -(R_est.transpose(0, 2, 1) @ t_est[..., None]).squeeze(-1)
    C_est = -np.einsum('nij,nj->ni',                    # C_w = -R.T * t
                       R_est.transpose(0, 2, 1), t_est)
    return C_est

def plot_true_and_estimated_traj(relative_Rts, poses_path):
    """ Plotting the ground truth trajectory along the estimated trajectory. """
    C_est = compute_estimated_centers(relative_Rts)
    poses_gt = np.loadtxt(
        poses_path
    ).reshape(-1, 3, 4)

    R_gt = poses_gt[:, :, :3]
    t_gt = poses_gt[:, :, 3]
    C_gt = -np.einsum('nij,nj->ni', R_gt.transpose(0, 2, 1), t_gt)

    x_est, y_est, z_est = C_est.T
    x_gt, y_gt, z_gt = C_gt.T

    x_est -= x_est[0]
    z_est -= z_est[0]
    x_gt -= x_gt[0]
    z_gt -= z_gt[0]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(x_est, z_est, '-', lw=1.5, label='estimated')  # blue line
    ax.plot(x_gt, z_gt, '.', ms=3, label='ground truth')  # orange dots

    # highlight start/end of the estimated track
    ax.scatter(x_est[0], z_est[0], s=60, c='red', label='start', zorder=3)
    ax.scatter(x_est[-1], z_est[-1], s=60, c='lime', label='end', zorder=3)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title('KITTI sequence 05 – estimated vs. ground truth')
    ax.grid(True, ls=':', alpha=.6)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def main(
        data_path = DATA_PATH,
        poses_path = POSES_PATH,
        max_frame = MAX_FRAME,
        akaze_thresh = AKAZE_THRESH
):
    """ End-to-end PnP pipeline for SLAM Exercise 3.
        Steps:
            I.  Load stereo camera intrinsics/extrinsics.
            II.  Read the first two stereo pairs, detect AKAZE features, and
                match intra-pair correspondences that satisfy the epipolar
                (v_R = v_L) constraint.
            III.  Triangulate 3-D points for both pairs and visualise the point clouds.
            IV.  Build inter-frame matches, solve PnP (AP3P -> EPNP -> RANSAC) to
                obtain the relative pose between frame 0 and frame 1, and plot
                the four-view geometry.
            V.  Iterate over the whole sequence (`max_frame`) to accumulate all
                relative poses and compare the estimated trajectory with KITTI
                ground truth. """
    # --- I) intrinsic, extrinsic & camera matrices ---
    K, M_L, M_R  = read_cameras(data_path)
    P_L, P_R     = K @ M_L, K @ M_R

    # --- II) read, detect & match pair0 and pair1 images ---
    left0_img, right0_img = read_images(0, data_path)
    left1_img, right1_img = read_images(1, data_path)

    left0_kp, left0_des, right0_kp, _, matches0 = compute_kps_descs_matches(left0_img, right0_img,
                                                                            threshold=akaze_thresh)
    left1_kp, left1_des, right1_kp, _, matches1 = compute_kps_descs_matches(left1_img, right1_img,
                                                                            threshold=akaze_thresh)

    # detect inliers based on intra-pair stereo-condition (v_R = v_L)
    inlier_matches0, _ = create_inlier_outlier_matches(matches0, left0_kp, right0_kp)
    inlier_matches1, _ = create_inlier_outlier_matches(matches1, left1_kp, right1_kp)

    # --- III) compute triangulation based on the intra-inliers for both pair0 and pair1 ---
    pair0_X = compute_triangulation_opencv(P_L, P_R, left0_kp, right0_kp, inlier_matches0)
    pair1_X = compute_triangulation_opencv(P_L, P_R, left1_kp, right1_kp, inlier_matches1)

    # question 3.1 - plot 3D point clouds
    plot_3d_cloudpoint(pair0_X, "orange", "OpenCV", "point cloud for pair$_0$")
    plot_3d_cloudpoint(pair1_X, "blue", "OpenCV", "point cloud for pair$_1$")

    # question 3.3 - apply PNP between pair0 world-points & their corr pair1 pixels
    # 1. collect the stereo-inlier indices once
    inlier_matches0_indices = [m.queryIdx for m in inlier_matches0]
    inlier_matches1_indices = [m.queryIdx for m in inlier_matches1]

    # 2. slice the original descriptor matrices
    inlier_left0_des = left0_des[inlier_matches0_indices]
    inlier_left1_des = left1_des[inlier_matches1_indices]

    # 3. match just inlier descriptors in pair0 and inlier descriptors in pair1
    matches01 = match_inter_pairs(inlier_left0_des, inlier_left1_des)

    # 4. collect left0 indices that were matched with left1 & slice pair0_X
    matches01_indices0 = [m.queryIdx for m in matches01]
    pair0_X_corr1 = pair0_X[matches01_indices0]

    # 5. obtain the pixels in left1, right1 corresponding to computed pair0-worldpoints
    left1_pix, right1_pix = create_L_R_pixels(matches01, inlier_matches1,
                                                inlier_matches1_indices, left1_kp, right1_kp, 1)

    # 6. compute [R|t] between frame0 to frame1, using 4 points
    Rt_left1 = solve_pnp(pair0_X_corr1[100:104], left1_pix[100:104],
                         K, cv2.SOLVEPNP_AP3P)

    plot_4_cameras_rel_position(M_R, Rt_left1)

    # question 3.4 - draw the two panels presenting supporters, before RANSAC
    left0_pix, right0_pix = create_L_R_pixels(matches01, inlier_matches0,
                                              inlier_matches0_indices, left0_kp, right0_kp, 0)

    left0_proj, right0_proj = project_pair0(pair0_X_corr1, P_L, P_R)
    left1_proj, right1_proj = project_pair1(pair0_X_corr1, K, Rt_left1, M_R)

    supporters_pair0 = supporters_pair(left0_pix, left0_proj, right0_pix, right0_proj)
    supporters_pair1 = supporters_pair(left1_pix, left1_proj, right1_pix, right1_proj)
    support_mask01 = supporters_pair0 & supporters_pair1

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    show_matches(left0_img, left0_pix, support_mask01,
                  r'left$_0$', sub_idx=1)
    show_matches(left1_img, left1_pix, support_mask01,
                  r'left$_1$', sub_idx=2)
    fig.suptitle('Frame-to-frame matches and 4-view supporters',
                 fontsize=15, y=0.8)
    plt.show()

    # question 3.5 - plotting pair1 and transformed-pair0 from above
    # & plotting frame-to-frame supporters after RANSAC
    ransac_support_mask01 = ransac_pnp_pair1(pair0_X_corr1, left1_pix, right1_pix, M_R, K)
    Rt_left1_refined = solve_pnp(pair0_X_corr1[ransac_support_mask01],
                                 left1_pix[ransac_support_mask01],
                                 K, cv2.SOLVEPNP_ITERATIVE)
    plot_pair1_and_transformed_pair0_from_above(pair0_X, pair1_X, Rt_left1_refined)

    fig1 = plt.figure(figsize=(12, 6), constrained_layout=True)
    show_matches(left0_img, left0_pix, ransac_support_mask01,
                  r'left$_0$', sub_idx=1)
    show_matches(left1_img, left1_pix, ransac_support_mask01,
                  r'left$_1$', sub_idx=2)
    fig1.suptitle('Frame-to-frame supporters after RANSAC',
                 fontsize=15, y=0.8)
    plt.show()

    # --- V) Iterate over the whole sequence to accumulate all relative poses
    # question 3.6 - compare the estimated trajectory with KITTI to ground truth
    rel_Rts = compute_rel_poses_entire_movie(data_path, max_frame,
                                             K, P_L, P_R, M_R,
                                             akaze_thresh)
    plot_true_and_estimated_traj(rel_Rts, poses_path)


if __name__ == '__main__':
    main()
