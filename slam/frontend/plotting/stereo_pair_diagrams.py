"""
slam/frontend/plotting]/stereo_pair_diagrams.py

Objective:
    Given matches, kpL, kpR, desL, desR
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from slam.frontend.vision.descriptor_matcher import DescriptorMatcher


def create_deviations_list(matches, kpL, kpR):
    """Vertical pixel gaps |v₂−v₁| for every correspondence."""
    v_diff_list = []
    for m in matches:
        # m.queryIdx is the index of the kp in img1
        # m.trainIdx is the index of the kp in img2
        # (uL, vL) in image 1:
        uL, vL = kpL[m.queryIdx].pt
        # (uR, vR) in image 2:
        uR, vR = kpR[m.trainIdx].pt
        v_diff_list.append(int(np.abs(vL - vR)))
    return v_diff_list


def create_deviations_histogram(v_diff_list, figsize=(8,6), bins=50):
    """ Plot a histogram presenting the deviation from a rectified stereo pattern. """
    plt.figure(figsize=figsize)
    plt.hist(v_diff_list, bins=bins)
    plt.xlabel('Deviation from rectified stereo pattern')
    plt.ylabel('Number of matches')
    plt.title('Y-axis Deviations Histogram')
    plt.show()


def print_deviated_matches_percentage(v_diff_list, pixel_thresh=2):
    """ Print the deviated matches percentage. """
    v_diff_array = np.array(v_diff_list)
    exceed_thresh_array = v_diff_array[v_diff_array > pixel_thresh]
    percentage = 100 * (len(exceed_thresh_array) / len(v_diff_array))
    print(f"The deviated matches percentage is: {percentage:.2f}")


def _draw_keypoints_bgr(img, inlier_kps, outlier_kps):
    img = cv2.drawKeypoints(img, inlier_kps, None, (0, 165, 255))    # orange
    return cv2.drawKeypoints(img, outlier_kps, img, (255, 255, 0),            # cyan
                             cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)


def draw_inlier_outlier_both_images(img_left, kps_left, img_right, kps_right,
                                    inlier_matches, outlier_matches):
    """Display two stacked images with inliers (orange) / outliers (cyan)."""
    in_l, in_r = DescriptorMatcher.extract_matched_keypoints(inlier_matches,  kps_left, kps_right)
    out_l, out_r = DescriptorMatcher.extract_matched_keypoints(outlier_matches, kps_left, kps_right)

    left  = cv2.cvtColor(_draw_keypoints_bgr(img_left,  in_l,  out_l), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(_draw_keypoints_bgr(img_right, in_r, out_r), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16,8))
    plt.subplot(2, 1, 1); plt.imshow(left);  plt.axis("off")
    plt.title("Matches in images 1 & 2 \n Showing inliers in orange and outliers in cyan \n")
    plt.subplot(2, 1, 2); plt.imshow(right); plt.axis("off")
    plt.tight_layout(); plt.show()


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


