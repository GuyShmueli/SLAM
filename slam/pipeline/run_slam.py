import argparse
from pathlib import Path
import numpy as np
import gtsam
from slam.backend.tracking_database import TrackingDB
from slam.frontend.io.camera_model import CameraModel
from slam.frontend.io.image_sequence import ImageSequence
from slam.backend.tracking_pipeline import TrackingPipeline
from slam.backend.optimizers.bundle_adjustment import BundleAdjustment
from slam.backend.optimizers.loop_closure import LoopClosure
import matplotlib.pyplot as plt

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        help="KITTI sequence folder",
                        default=Path(r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\sequences\05"))
    parser.add_argument("--poses-path",
                        help="Ground‑truth poses file",
                        default=Path(r"C:\Users\dhtan\Desktop\VAN_ex\dataset\2025_dataset\poses\05.txt"))
    parser.add_argument("--out-path", default="kitti05_tracks")
    parser.add_argument("--max-frames", type=int, default=2_600)
    parser.add_argument("--akaze", type=float, default=5e-3)  # 5e^-3
    parser.add_argument("--disp-min", type=float, default=2.0)
    parser.add_argument("--pix-thresh", type=float, default=0.8)
    parser.add_argument("--analyze-and-plot", type=bool, default=False)
    parser.add_argument("--stereo-cross-check", type=bool, default=True)
    parser.add_argument("--frame-cross-check", type=bool, default=False)  # rely on RANSAC-PnP test
    parser.add_argument("--force-rebuild", type=bool, default=False)  # build DB even if already exists
    return parser.parse_args()

def plot_trajectory_overlay(gt_xyz, pre_xyz, post_xyz,
                            err_pre, err_post,
                            title='KITTI seq. 05 - keyframe trajectories',
                            label_pre = 'before BA',
                            label_post = 'after BA',):
    """Overlay ground-truth, pre-optimization & post-optimization
    keyframe trajectories in the X-Z plane."""
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.plot(gt_xyz [:,0], gt_xyz [:,2], 'k--',  lw=2, label='ground-truth')
    ax.plot(pre_xyz [:,0], pre_xyz [:,2],  'r-',  lw=1.5,
            label=label_pre + f' (RMSE {np.sqrt(np.mean(err_pre**2)):.2f} m)')
    ax.plot(post_xyz[:,0], post_xyz[:,2], 'b-',  lw=1.5,
            label=label_post + f' (RMSE {np.sqrt(np.mean(err_post**2)):.2f} m)')
    ax.scatter(gt_xyz[0,0], gt_xyz[0,2], c='g', s=60, label='start')
    ax.scatter(gt_xyz[-1,0], gt_xyz[-1,2], c='m', s=60, label='end')
    ax.set_aspect('equal')
    ax.set_xlabel('X  [m]')
    ax.set_ylabel('Z  [m]  (forward)')
    ax.set_title(title)
    ax.grid(True, ls=':')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

def array_from_values(vals, kf_indices):
    """Return an (N, 3) array of camera centres in KF order."""
    rows = []
    for k in kf_indices:
        t = vals.atPose3(gtsam.symbol('c', k)).translation()
        rows.append(t)      # Point3  ->  [x, y, z]
    return np.vstack(rows)              # shape (N, 3)

def plot_2D_trajectories(KF_indices, pose_graph,
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

def main():
    # 0) Parse arguments
    args = parse_cli()

    # if args.analyze_and_plot:

    # 1) Load camera intrinsics and extrinsics:
    calib_file = Path(args.data) / "calib.txt"
    cam = CameraModel.from_kitti(calib_file)

    # 2)
    img_seq = ImageSequence(
    base_path=Path(args.data),
    left_dir="image_0",
    right_dir="image_1",
    extension=".png"
    )

    # 2) Build pixels-DB:
    db_path = Path(args.out_path)



    # only pixels that survive the stereo‑pair test enter the DB (via TrackingDB.create_links()),
    # and of those, only those that also survive the RANSAC‑PnP inlier check get promoted into tracks
    tracking_pipeline = TrackingPipeline(
        cam = cam,
        img_seq = img_seq,
        akaze_thresh=args.akaze,
        disparity_min=args.disp_min,
        pixel_thresh=args.pix_thresh,
        stereo_cross_check=args.stereo_cross_check,
        frame_cross_check=args.frame_cross_check
    )
    # Both builds DB and return world-to-camera poses
    tracking_pipeline.build(max_frame=args.max_frames,
                                      out_path=Path(args.out_path))
    Rts_w2c = tracking_pipeline.get_relative_Rts()

    # 3) Load DB
    db = TrackingDB()
    db.load(args.out_path)

    # 4) Create a local-BA graph (gtsam expects 'camera -> world' format):
    ba = BundleAdjustment(cam,
                          Rts_w2c,
                          db.links_by_frame())
    graphs, values = ba.build_windows()

    # 5) Optimize each window (separately)
    opt_vals = ba.optimize_bundles(graphs, values)

    # 6)

    lc = LoopClosure(cam,
                     img_seq,
                     ba.get_kf_indices(),
                     graphs,
                     opt_vals,
                     args.frame_cross_check)
    # Prior sigmas (yaw-pitch-roll-x-y-z)
    prior_sigmas = np.array([np.deg2rad(5), np.deg2rad(5), np.deg2rad(5),
                                                                 0.05, 0.05, 0.15])
    early_terminate = 144   # the loops formed by >=133 KFs increase the error a bit
    (pose_graph_pre, pre_vals), (pose_graph, pose_vals) = lc.run_loop_closure(
                        db,
                        prior_sigmas,
                        early_terminate)

    poses_gt = np.loadtxt(args.poses_path).reshape(-1, 3, 4)  # (N,3,4)
    R_gt = poses_gt[:, :, :3]  # (N,3,3)  world -> cam rotation
    t_gt = poses_gt[:, :, 3]  # (N,3)    world -> cam translation
    gt_xyz_all = -np.einsum('nij,nj->ni',  # camera center in world coords
                            R_gt.transpose(0, 2, 1), t_gt)  # (N,3)
    gt_xyz = gt_xyz_all[ba.get_kf_indices()]  # (M,3)  match pre & post


    plot_2D_trajectories(ba.get_kf_indices(), pose_graph,
                            pre_vals, pose_vals, gt_xyz,
                            loops_added=10)

if __name__ == '__main__':
    main()