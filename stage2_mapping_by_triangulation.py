import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utils import read_images, compute_kps_descs_matches, read_cameras

DATA_PATH = r'C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\sequences\05\\'

def create_deviations_list(matches, kp1, kp2):
    """Vertical pixel gaps |v₂−v₁| for every correspondence."""
    v_diff_list = []
    for m in matches:
        # m.queryIdx is the index of the kp in img1
        # m.trainIdx is the index of the kp in img2

        # (u1, v1) in image 1:
        u1, v1 = kp1[m.queryIdx].pt

        # (u2, v2) in image 2:
        u2, v2 = kp2[m.trainIdx].pt

        v_diff_list.append(int(np.abs(v2 - v1)))
    return v_diff_list

def create_inlier_outlier_matches(matches, kp1, kp2, pixel_thresh=2):
    """Classify matches by whether the gap |v₂−v₁| ≤ pixel_thresh."""
    inlier_matches = []
    outlier_matches = []
    for m in matches:
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt

        if int(np.abs(v2 - v1)) <= pixel_thresh:
            inlier_matches.append(m)
        else:
            outlier_matches.append(m)

    return inlier_matches, outlier_matches

def create_deviations_histogram(v_diff_list, figsize=(8,6), bins=50):
    plt.figure(figsize=figsize)
    plt.hist(v_diff_list, bins=bins)
    plt.xlabel('Deviation from rectified stereo pattern')
    plt.ylabel('Number of matches')
    plt.title('Y-axis Deviations Histogram')
    plt.show()


def print_deviated_matches_percentage(v_diff_list, pixel_thresh=2):
    v_diff_array = np.array(v_diff_list)

    exceed_thresh_array = v_diff_array[v_diff_array > pixel_thresh]

    percentage = 100 * (len(exceed_thresh_array) / len(v_diff_array))

    print(f"The deviated matches percentage is: {percentage:.2f}")


### Helpers for draw_inlier_outlier_both_images ###
def _extract_matched_keypoints(matches, kps_left, kps_right):
    """Return key‑points from *matches* for the left and right image."""
    return ([kps_left[m.queryIdx]  for m in matches],
            [kps_right[m.trainIdx] for m in matches])


def draw_inlier_outlier_both_images(img_left, kps_left, img_right, kps_right, inlier_matches, outlier_matches):
    """Display two stacked images with inliers (orange) / outliers (cyan)."""
    def _draw_keypoints_bgr(img, inlier_kps, outlier_kps):
        img = cv2.drawKeypoints(img, inlier_kps, None, (0, 165, 255))  # orange
        return cv2.drawKeypoints(img, outlier_kps, img, (255, 255, 0),  # cyan
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    in_l,  in_r  = _extract_matched_keypoints(inlier_matches,  kps_left, kps_right)
    out_l, out_r = _extract_matched_keypoints(outlier_matches, kps_left, kps_right)

    left  = cv2.cvtColor(_draw_keypoints_bgr(img_left,  in_l,  out_l), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(_draw_keypoints_bgr(img_right, in_r, out_r), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16,8))
    plt.subplot(2, 1, 1); plt.imshow(left);  plt.axis("off")
    plt.title("Matches in images 1 & 2 \n Showing inliers in orange and outliers in cyan \n")
    plt.subplot(2, 1, 2); plt.imshow(right); plt.axis("off")
    plt.tight_layout(); plt.show()


def triangulate_matches_batched(P_left, P_right, kp1, kp2, matches):
    """
    Compute the world 3D points corresponding to pairs of pixels, manually (least squares).
    """

    def _build_single_camera_A_matrix(x_pix, y_pix, camera_matrix):
        """
        Build the 2×4 linear block for ONE camera / ONE point.
        camera_matrix : (3,4)
        x_pix, y_pix  : scalars (pixel coordinates)
        """
        P1, P2, P3 = camera_matrix[0], camera_matrix[1], camera_matrix[2]
        first_row = (y_pix * P3) - P2
        second_row = P1 - (x_pix * P3)
        return np.vstack([first_row, second_row])  # (2,4)

    def _build_A_matrix(A1, A2):
        """Stack the two cameras’ 2×4 blocks → 4×4 matrix. Still ONE point"""
        return np.vstack([A1, A2])  # (4,4)

    # Create a 3D-tensor, the first dim is the number of points
    N = len(matches)
    A_3d = np.empty((N, 4, 4), dtype=float)

    for k, m in enumerate(matches):
        uL, vL = kp1[m.queryIdx].pt
        uR, vR = kp2[m.trainIdx].pt

        A_left = _build_single_camera_A_matrix(uL, vL, P_left)
        A_right = _build_single_camera_A_matrix(uR, vR, P_right)

        A_for_point_k = _build_A_matrix(A_left, A_right)

        A_3d[k] = A_for_point_k

    _, _, Vt = np.linalg.svd(A_3d)

    # For each point, we take the last row of Vt (which is equivalent to last col of V)
    Xh_opt = Vt[:, -1, :]  # (N,4)

    # Transforming from P^3 back to R^3, equivalent to:
    # X_opt = Xh_opt[:, :3] / Xh_opt[:, 3:4]  # (N,3)
    X_opt = np.squeeze(cv2.convertPointsFromHomogeneous(Xh_opt))

    return X_opt

def compute_triangulation_opencv(P_left, P_right, kp1, kp2, matches):
    """Compute the world 3D points corresponding to pairs of pixels, using OpenCV triangulation function."""
    # 2×N matrices:  [[u0 u1 … uN-1],
    #                 [v0 v1 … vN-1]]
    pts_L = np.array([kp1[m.queryIdx].pt for m in matches],
                     dtype=np.float32).T
    pts_R = np.array([kp2[m.trainIdx].pt for m in matches],
                     dtype=np.float32).T

    # OpenCV returns homogeneous coords (4×N)
    Xh_cv = cv2.triangulatePoints(P_left, P_right, pts_L, pts_R)  # 4×N
    X_cv = (Xh_cv[:3] / Xh_cv[3]).T  # N×3  in left-cam frame

    return X_cv

def compare_triangulations(X_custom, X_cv):
    """Compare OpenCV with my SVD solution."""
    dists = np.linalg.norm(X_custom - X_cv, axis=1)
    print(f"Median L2 distance (custom vs OpenCV): {np.median(dists):.8f}")


def plot_3d_cloudpoint(pts, color, label, title, elev=15, azim=45,
        x_min=0, x_max=200, y_min=-30, y_max=30, z_min=-30, z_max=30):
    """
    3D visualization of the cloud point created by reconstructing all of 3D-points.
    """
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # ---  swap X <--> Z ---
    xs = pts[:, 2]  # what used to be Z
    ys = pts[:, 1]  # keep Y
    zs = pts[:, 0]  # what used to be X
    ax.scatter(xs, ys, zs, s=6, color=color, label=label)

    ax.set_xlabel("Z (m)")   # because xs are actually z
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("X (m)")   # because zs are actually x

    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(x_min, x_max)   # really z-range
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)   # really x-range
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


### Helpers for compute_erroneous_locations ###
def _count_erroneous_world_locations(X, min_thresh=0):
    count = 0
    for row in X:
        if row[2] < min_thresh:
            count += 1
    return count

def _compute_erroneous_locations_single_image(idx):
    img1, img2 = read_images(idx, DATA_PATH)
    kp1, _, kp2, _, matches = compute_kps_descs_matches(img1, img2)

    inlier_matches, _ = create_inlier_outlier_matches(matches, kp1, kp2)

    K, M1, M2 = read_cameras(DATA_PATH)
    P1, P2 = K @ M1, K @ M2

    X_cv = compute_triangulation_opencv(P1, P2, kp1, kp2, inlier_matches)

    erroneous_locations = _count_erroneous_world_locations(X_cv)

    return erroneous_locations


def compute_erroneous_locations(low_thresh, high_thresh, data_path):
    """Compute the number of erroneous locations for a given range."""
    rng = np.arange(low_thresh, high_thresh)
    erroneous_locations_list = []
    for i in rng:
        erroneous_locations_list.append((int(i), _compute_erroneous_locations_single_image(i)))
    max_tuple = max(erroneous_locations_list, key=lambda t: t[1])
    idx_max, err_locations_max = max_tuple
    print(f"In {low_thresh}-{high_thresh} range, the image that contains the most erroneous locations "
          f"is at idx {idx_max} and has {err_locations_max} erroneous locations.")
    return erroneous_locations_list

def main(idx=0, low_thresh=0, high_thresh=50,
         data_path=DATA_PATH):
    """End‑to‑end stereo pipeline: match -> check -> triangulate -> visualise."""
    # 1) detect + match
    img1, img2 = read_images(idx, data_path)
    kp1, _, kp2, _, matches = compute_kps_descs_matches(img1, img2)

    # 2) rectification sanity‑check (vertical gap stats)
    v_diff_list = create_deviations_list(matches, kp1, kp2)
    create_deviations_histogram(v_diff_list)
    print_deviated_matches_percentage(v_diff_list)

    # 3) classify matches, overlay on images
    inlier_matches, outlier_matches = create_inlier_outlier_matches(matches, kp1, kp2)
    draw_inlier_outlier_both_images(img1, kp1, img2, kp2, inlier_matches, outlier_matches)

    # 4) projection matrices  P = K [R|t]
    K, M1, M2 = read_cameras(data_path)
    P1, P2 = K @ M1, K @ M2

    # 5) triangulate via our SVD solver and via OpenCV
    X_svd = triangulate_matches_batched(P1, P2, kp1, kp2, inlier_matches)
    X_cv  = compute_triangulation_opencv (P1, P2, kp1, kp2, inlier_matches)
    compare_triangulations(X_svd, X_cv)

    # 6) 3‑D scatter
    plot_3d_cloudpoint(X_svd, color='blue', label='SVD (custom)', title='Custom Triangulated point clouds')
    plot_3d_cloudpoint(X_cv, color='orange', label='cv2.triangulatePoints', title='OpenCV Triangulated point clouds')

    # 7) check, among a specified range,
    lst = compute_erroneous_locations(low_thresh, high_thresh, data_path)

if __name__ == '__main__':
    main()