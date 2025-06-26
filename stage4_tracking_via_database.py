import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import read_images, compute_kps_descs_matches, read_cameras
from stage2_mapping_by_triangulation import create_inlier_outlier_matches, compute_triangulation_opencv
from stage3_localization_by_p4p import *
from tracking_database import TrackingDB
import matplotlib.patches as patches

# --- Magic constants ---
MAX_FRAME = 2_600
DATA_PATH = r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\sequences\05\\"
POSES_PATH = r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\poses\05.txt"
AKAZE_THRESH = 0.0005
DISPARITY_MIN = 1.0

def stereo_inliers_with_disparity(matchesLR, kpL, kpR, d_min):
    """Return only those matches whose (uL-uR) ≥ d_min and vL==vR."""
    good = []
    for m in matchesLR:
        (uL,vL) = kpL[m.queryIdx].pt
        (uR,vR) = kpR[m.trainIdx].pt
        if abs(vL - vR) < 1.0 and (uL - uR) >= d_min:   # rectified ⇒ v match
            good.append(m)
    return good


def fill_database(data_path, max_frame, akaze_thresh,
                  K, P_L, P_R, M_R):
    """ Build a TrackingDB over frames [0 .. max_frame-1] using AKAZE + stereo + RANSAC‐PnP. """
    db = TrackingDB()

    # 1) Frame 0: do stereo detection + inlier‐filter + triangulation
    img_L_prev, img_R_prev = read_images(0, data_path)

    # Detect AKAZE keypoints + descriptors on left & right images of frame 0
    kp_L_prev, des_L_prev, kp_R_prev, _, matches_LR_prev = \
        compute_kps_descs_matches(img_L_prev, img_R_prev, threshold=akaze_thresh)

    # # Filter only the stereo‐inliers (v_L = v_R)
    # inliers_LR_prev, _ = create_inlier_outlier_matches(matches_LR_prev, kp_L_prev, kp_R_prev)

    inliers_LR_prev = stereo_inliers_with_disparity(
        matches_LR_prev, kp_L_prev, kp_R_prev, DISPARITY_MIN)

    # Triangulate all those stereo‐inliers to get a 3D cloud, in exactly the same order:
    # X_prev_unsorted[i]  <-  3D point for inliers_LR_prev[i]
    X_prev_unsorted = compute_triangulation_opencv(
        P_L, P_R,
        kp_L_prev, kp_R_prev,
        inliers_LR_prev
    )  # shape = (N_prev × 3), in the same order as inliers_LR_prev

    # Use create_links() to build (inlier_des_L_prev, links_prev) in exactly the same order as inliers_LR_prev:
    inlier_des_L_prev, links_prev = TrackingDB.create_links(
        des_L_prev,          # full descriptor array for left image 0
        kp_L_prev,           # keypoints on left image 0
        kp_R_prev,           # keypoints on right image 0
        inliers_LR_prev,     # list[cv2.DMatch] of just the stereo‐inliers
        inliers=None,        # since we only passed the inliers themselves, no extra mask is needed
        keep_match_order=True
    )
    # Now: inlier_des_L_prev[i]  <-->  links_prev[i]  <-->  X_prev_unsorted[i]

    # We can treat X_prev_unsorted as “X_prev” because it’s already aligned with inlier_des_L_prev
    X_prev = X_prev_unsorted.copy()  # shape = (N_prev × 3)

    # Add frame-0 into the DB (no “matches_to_previous” because it’s the first frame)
    db.add_frame(links_prev, inlier_des_L_prev, None)

    # 2) Loop over frames 1 … max_frame-1
    for idx in range(1, max_frame):
        # 2a. Load frame‐idx’s stereo pair, detect and compute descriptors
        img_L, img_R = read_images(idx, data_path)
        kp_L, des_L, kp_R, _, matches_LR = \
            compute_kps_descs_matches(img_L, img_R, threshold=akaze_thresh)

        # # Filter stereo‐inliers (left vs. right)
        # inliers_LR, _ = create_inlier_outlier_matches(
        #     matches_LR, kp_L, kp_R
        # )
        inliers_LR = stereo_inliers_with_disparity(
            matches_LR, kp_L, kp_R, DISPARITY_MIN)

        # 2b. Triangulate all those stereo‐inliers to get 3D points for this frame:
        X_curr_unsorted = compute_triangulation_opencv(
            P_L, P_R,
            kp_L, kp_R,
            inliers_LR
        )  # shape = (N_curr × 3), in the same order as inliers_LR

        # 2c. Build (inlier_des_L, links_curr) in exactly the same “inlier” order:
        inlier_des_L, links_curr = TrackingDB.create_links(
            des_L,            # full descriptor array for left image of frame idx
            kp_L,             # keypoints on left image idx
            kp_R,             # keypoints on right image idx
            inliers_LR,       # list[cv2.DMatch] for stereo‐inliers only
            inliers=None,
            keep_match_order=True
        )
        # Now X_curr_unsorted[i] corresponds exactly to inlier_des_L[i] <--> links_curr[i].

        X_curr = X_curr_unsorted.copy()

        # 2d. Match “previous‐frame inliers” <--> “current‐frame inliers” on the left descriptors:
        matches01 = match_inter_pairs(
            inlier_des_L_prev,   # descriptors from frame‐(idx-1) after stereo‐filter
            inlier_des_L,        # descriptors from frame‐idx after stereo‐filter
            cross_check=False    # we’ll rely on RANSAC next to prune bad matches
        )

        # Build an array of 3D points corresponding to each match’s “previous‐frame” end:
        matches01_idx_prev = [m.queryIdx for m in matches01]
        X_corr = X_prev[matches01_idx_prev]
        # shape = (len(matches01) × 3), aligned with `matches01`

        # 2e. Build “current‐frame” 2D pixel arrays for those same matches (needed by RANSAC‐PnP):
        # We need the left and right‐pixel coordinates in frame idx for each match01.
        inlier_idx_L = [m.queryIdx for m in inliers_LR]  # indices of left‐keypoints that survived stereo
        pix_L_curr, pix_R_curr = create_L_R_pixels(
            matches01,
            inliers_LR,
            inlier_idx_L,
            kp_L, kp_R,
            pair_0_or_1=1
        )
        # pix_L_curr.shape == (len(matches01) × 2), pix_R_curr likewise.

        # 2f. Run RANSAC‐PnP to filter out 3D -> 2D outliers:
        inlier_mask = ransac_pnp_pair1(
            X_corr,
            pix_L_curr,
            pix_R_curr,
            M_R,
            K
        )

        # 2g. Add frame‐idx into the DB.
        db.add_frame(links_curr, inlier_des_L, matches01, inlier_mask)

        # 2h. Shift “current” -> “previous” for next iteration:
        X_prev = X_curr
        inlier_des_L_prev = inlier_des_L

    # 3) After the loop, save the DB to “kitti05_tracks.pkl”
    db.serialize("kitti05_tracks")


def tracks_number(db: TrackingDB) -> int:
    """Return the count of tracks that span at least two frames."""
    track_ids = [
        tid for tid in db.all_tracks()
        if len(db.frames(tid)) > 1
    ]
    return len(track_ids)


def frames_number(db: TrackingDB) -> int:
    """Return the total number of frames stored in the database."""
    frame_ids = [
        fid for fid in db.all_frames()
    ]
    return len(frame_ids)


def max_track_len(db: TrackingDB) -> int:
    """
    Return the maximum track length (in #frames) in this TrackingDB.
    """
    return max(db.track_length(tid) for tid in db.all_tracks())

def min_track_len(db: TrackingDB) -> int:
    """
    Return the minimum track length (in #frames) in this TrackingDB.
    """
    return min(db.track_length(tid) for tid in db.all_tracks())

def mean_track_len(db: TrackingDB) -> float:
    """
    Return the mean track length (in #frames) in this TrackingDB.
    """
    track_lens = [db.track_length(tid) for tid in db.all_tracks()]
    mean_len = float(np.mean(track_lens))
    return round(mean_len, 3)


def mean_links_per_frame(db: TrackingDB, frames_num: int) -> float:
    """Return the average number of stereo inlier links per frame."""
    mean = db.link_num() / frames_num
    return round(mean, 3)


def plot_track_frames(
    images,
    pixels_tid_i,     # list of (x,y) coordinates for each frame
    frame_ids,        # list of frame IDs (for titles)
    tid_i,            # track ID, used in each subplot title
    patch_size=20,    #  size of the zoom‐window
    width_ratios=(3, 1)  # how wide full vs. zoom columns should be
):
    """ For a given track, show each frame side‐by‐side with a small zoom‐in patch. """
    half = patch_size // 2
    n = len(images)
    figsize = (14, 3.5 * n)

    # Create one row per image, with two columns (full image / zoom)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios}
    )

    for (ax_full, ax_zoom), img, (x, y), fid in zip(axes, images, pixels_tid_i, frame_ids):
        # --- left subplot: show the full image with a red square around the keypoint ---
        ax_full.imshow(img, cmap="gray")
        ax_full.set_title(f"track {tid_i} – frame {fid}")
        ax_full.axis("off")

        # Draw a red rectangle of size patch_size×patch_size around (x,y)
        rect = patches.Rectangle(
            (x - half, y - half),
            patch_size,
            patch_size,
            edgecolor="r",
            facecolor="none",
            linewidth=2
        )
        ax_full.add_patch(rect)

        # --- right subplot: extract and show the zoom patch ---
        h, w = img.shape
        # clamp x-half to [0, w - patch_size]; similarly for y
        x0 = int(np.clip(x - half, 0, w - patch_size))
        y0 = int(np.clip(y - half, 0, h - patch_size))

        # slice out exactly (patch_size x patch_size)
        patch = img[y0 : y0 + patch_size, x0 : x0 + patch_size]

        ax_zoom.imshow(patch, cmap="gray", interpolation="nearest")
        ax_zoom.axis("off")
        # mark the exact center of that patch (to see the original keypoint)
        ax_zoom.plot(half, half, "xr", ms=8, mew=3)

    plt.show()


def compute_connectivity(db):
    """Return a list whose k-th element is the number of tracks that
       appear in both frame k and frame k + 1 (i.e. survive the transition)."""
    num_outgoing_tracks = []
    for fid in range(db.frame_num() - 1):
        active_prev = set(db.tracks(fid))
        active_curr = set(db.tracks(fid+1))
        outgoing = active_prev.intersection(active_curr)
        num_outgoing = len(outgoing)
        num_outgoing_tracks.append(num_outgoing)
    return num_outgoing_tracks


def plot_connectivity(db, num_outgoing_tracks):
    """Plot the per-frame connectivity curve produced by *compute_connectivity*,
        with a horizontal line marking the global mean."""
    plt.figure(figsize=(10, 4))
    plt.plot(db.all_frames()[:-1], num_outgoing_tracks)
    mean_val = np.mean(num_outgoing_tracks)
    plt.axhline(mean_val, color="tab:green", linewidth=2)
    plt.xlabel("Frame ID")
    plt.ylabel("Tracks continuing to next frame")
    plt.title("Per‐Frame Connectivity (Outgoing Tracks)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_inliers_per_frame_percentage(db):
    """For each consecutive frame pair (k, k+1) compute the percentage of
        tracks in frame k that are still present in frame k+1."""
    ratios = []
    for fid in range(db.frame_num() - 1):
        tids_in_fid = set(db.tracks(fid))
        tids_in_fid1 = set(db.tracks(fid+1))
        common = len(tids_in_fid & tids_in_fid1)  # frame-to-frame inliers
        all = len(tids_in_fid) or 1
        ratios.append(common / all)
    return ratios


def plot_inliers_per_frame_percentage(db, inliers_per_frame_percentage):
    """Visualise the percentages from 'compute_inliers_per_frame_percentage'
    as a line plot (0 – 100 %), including a green line for the average value."""
    inliers_per_frame_percentage = np.array(inliers_per_frame_percentage) * 100
    plt.figure(figsize=(12, 4))
    plt.plot(db.all_frames()[:-1], inliers_per_frame_percentage, lw=1.2)
    mean_val = np.mean(inliers_per_frame_percentage)
    plt.axhline(mean_val, color="tab:green", linewidth=2)
    plt.xlabel("Frame  k")
    plt.ylabel("Inliers surviving to  k+1 (%)")
    plt.title("Per-frame RANSAC-PnP inlier percentage")
    plt.ylim(0, 100)
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_tracks_len_histogram(db):
    """Draw a histogram (log-scaled y-axis) of track lengths,
       i.e. how many frames each track lasts – starting from length 2."""
    # collect track lengths
    track_lens = np.array([db.track_length(tid) for tid in db.all_tracks()])
    # histogram on log-scaled y-axis
    plt.figure(figsize=(8, 5))
    plt.hist(track_lens, bins=np.arange(2, 52))
    plt.yscale('log')
    plt.xlabel('Track length')
    plt.ylabel('Track #')
    plt.title('Track length histogram')
    plt.tight_layout()
    plt.show()


def triangulate_track_using_last_frame_ftrs(j,  # number of frames correspond to the track
                                            k,  # The chosen track_id within 'tracks_equal_to_j'
                                            db, K, poses_gt, M_R):
    """Triangulate a single track by using its last-frame stereo pair."""
    tracks_equal_to_j = db.tracks_equal_to(j)

    tid_k = tracks_equal_to_j[k]
    fids_corr_tid_k = db.frames(tid_k)
    fid_last = fids_corr_tid_k[-1]

    track_link = db.link(fid_last, tid_k)
    pts_L_last = track_link.left_keypoint().reshape(2, 1)  # 2×1
    pts_R_last = track_link.right_keypoint().reshape(2, 1)  # 2×1

    P_left = K @ poses_gt[fid_last]

    pose_fid_last_hom = np.vstack([poses_gt[fid_last], np.array([0, 0, 0, 1])])
    P_right = K @ (M_R @ pose_fid_last_hom)

    Xh = cv2.triangulatePoints(P_left, P_right, pts_L_last, pts_R_last)  # 4×N
    X = (Xh[:3] / Xh[3]).T  # N×3

    return X, tid_k, fids_corr_tid_k, fid_last


def compute_reproj_errs(X, db, frame_ids, tid, poses, K, M_right):
    """Return per-frame reprojection errors (left & right) for a given 3-D point."""
    errs_L = []
    errs_R = []
    for fid in frame_ids:
        # --- projected left & right pixels ---
        Rt_L_fid = poses[fid]
        P_L_fid = K @ Rt_L_fid
        proj_pts_L = project(X, P_L_fid)

        Rt_hom_L_fid = np.vstack([Rt_L_fid, np.array([0, 0, 0, 1])])
        P_R_fid = K @ (M_right @ Rt_hom_L_fid)
        proj_pts_R = project(X, P_R_fid)

        # --- tracked left & right pixels ---
        link_fid_corr_tid_i = db.link(fid, tid)
        tracked_pts_L, tracked_pts_R = link_fid_corr_tid_i.left_keypoint(), \
            link_fid_corr_tid_i.right_keypoint()  # shape (2,)

        tracked_pts_L = np.array(tracked_pts_L).reshape(1, 2)
        tracked_pts_R = np.array(tracked_pts_R).reshape(1, 2)

        # --- computing left & right reprojection error ---
        err_L_fid = np.linalg.norm(proj_pts_L - tracked_pts_L)
        errs_L.append(err_L_fid)

        err_R_fid = np.linalg.norm(proj_pts_R - tracked_pts_R)
        errs_R.append(err_R_fid)

    return errs_L, errs_R


def plot_reproj_err_over_track_len(fid_last, frame_ids, errs_L, errs_R):
    """Plot reprojection error versus distance from the reference frame."""
    plt.figure(figsize=(8,6))
    dist_from_reference = [abs(fid_last - fid) for fid in frame_ids]
    plt.plot(dist_from_reference, errs_L, label='Left')
    plt.plot(dist_from_reference, errs_R, label='Right')
    plt.xlabel("Distance from reference frame (|fid – fid_last|)")
    plt.ylabel("Reprojection error (L$_2$ norm in pixels)")
    plt.title("Reprojection error vs. distance from reference")
    plt.legend()
    plt.grid(True)
    plt.show()


def main(data_path=DATA_PATH,
         db_path="kitti05_tracks",
         poses_path=POSES_PATH):
    """End-to-end demo / evaluation script:
    1. Load intrinsics (*K*) and stereo rig geometry (*M_L*, *M_R*).
    2. Compute or load a pre-computed *TrackingDB*.
    3. Print basic statistics (track count, frame count, track-length stats,
       mean links per frame).
    4. Produce several diagnostic visualisations:
       – keypoint trajectory across 6 frames
       – frame-to-frame connectivity curve
       – RANSAC-PnP inlier ratio per frame
       – histogram of track lengths
       – reprojection-error curve for a chosen track.
    """

    K, M_L, M_R = read_cameras(data_path)

    # --- Run just one time to fill the database ---
    P_L, P_R = K @ M_L, K @ M_R
    fill_database(data_path, MAX_FRAME, AKAZE_THRESH,
                  K, P_L, P_R, M_R)

    # --- Initialize & load DB ---
    db = TrackingDB()
    db.load(db_path)

    # --- 4.2: statistics ---
    tracks_num = tracks_number(db)
    print(f"Found {tracks_num} non‐trivial tracks.")

    frames_num = frames_number(db)
    print(f"Found {frames_num} frames.")

    max_track_length = max_track_len(db)
    min_track_length = min_track_len(db)
    mean_track_length = mean_track_len(db)
    print(f"Max track length is {max_track_length}.")
    print(f"Min track length is {min_track_length}.")
    print(f"Mean track length is {mean_track_length}.")

    mean_links = mean_links_per_frame(db, frames_num)
    print(f"Mean links per frame is {mean_links}.")

    # --- 4.3: Plotting track features over 6 frames---
    # picking a (fixed) six–frame track, 'tid_i'
    i = 0
    tid_i = db.tracks_equal_to(6)[i]
    frame_ids = db.frames(tid_i)
    links_tid_i = [db.link(fid, tid_i) for fid in frame_ids]
    pixels_tid_i = [link.left_keypoint() for link in links_tid_i]

    imgs_names = ['{:06d}.png'.format(fid) for fid in frame_ids]
    images = [cv2.imread(data_path + 'image_0\\' + img_name, cv2.IMREAD_GRAYSCALE)
              for img_name in imgs_names]

    plot_track_frames(images, pixels_tid_i, frame_ids, tid_i)

    # --- 4.4: Plotting a connectivity graph ---
    num_outgoing_tracks = compute_connectivity(db)
    plot_connectivity(db, num_outgoing_tracks)

    # --- 4.5: Plotting a graph of the percentage of inliers per frame ---
    inliers_per_frame_percentage = compute_inliers_per_frame_percentage(db)
    plot_inliers_per_frame_percentage(db, inliers_per_frame_percentage)

    # --- 4.6: Plotting a track length histogram graph ---
    plot_tracks_len_histogram(db)

    # --- 4.7: Plotting a graph of the reprojection error size over the track’s frames
    poses_gt = np.loadtxt(poses_path).reshape(-1, 3, 4)
    X, tid_k, fids_corr_tid_k, fid_last = triangulate_track_using_last_frame_ftrs(40, 0, db, K, poses_gt, M_R)
    reproj_errs_L, reproj_errs_R = compute_reproj_errs(X, db, fids_corr_tid_k, tid_k, poses_gt, K, M_R)
    plot_reproj_err_over_track_len(fid_last, fids_corr_tid_k, reproj_errs_L, reproj_errs_R)


if __name__ == '__main__':
    main()