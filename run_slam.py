# slam/run_slam.py
import argparse
from pathlib import Path
import numpy as np
import pickle
from slam.backend.tracking.database import TrackingDB
from slam.backend.tracking.fill_database import TrackingPipeline
from slam.frontend.io.camera_model import CameraModel
from slam.frontend.io.image_sequence import ImageSequence
from slam.backend.optimizers.bundle_adjustment import BundleAdjustment
from slam.backend.optimizers.loop_closure import LoopClosure
from slam.analysis.ba_plot import BA_Plotter
from slam.analysis.lc_plot import LC_Plotter
from slam.analysis.pnp_plot import PnP_Plotter
from slam.analysis.optimizers_analysis import OptimizersPlot
import slam.utils.utils as utils
from slam.utils.graph_utils import vals_to_rel_arrays


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
    parser.add_argument("--akaze", type=float, default=2e-3)
    parser.add_argument("--disp-min", type=float, default=1.2)
    parser.add_argument("--pix-thresh", type=float, default=4.0)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--stereo-cross-check", type=bool, default=False)
    parser.add_argument("--frame-cross-check", type=bool, default=False)  # rely on RANSAC-PnP test
    parser.add_argument("--build-db", type=bool, default=False)           # build DB even if already exists
    parser.add_argument('--feature-type', default='akaze', choices=['akaze', 'superpoint'])
    parser.add_argument('--matcher-type', default='bf', choices=['bf', 'superglue'])
    return parser.parse_args()


def main():
    # 0) Parse arguments
    args = parse_cli()

    # 1) Load camera intrinsics and extrinsics:
    calib_file = Path(args.data) / "calib.txt"
    cam = CameraModel.from_kitti(calib_file)
    poses_gt = np.loadtxt(args.poses_path).reshape(-1, 3, 4)        # (k <- 0),   (N,3,4)

    # 2) Create ImageSequence object to handle stereo pairs
    img_seq = ImageSequence(
    base_path=Path(args.data),
    left_dir="image_0",
    right_dir="image_1",
    extension=".png"
    )

    # 3) A. Build pixels-DB:
    # only pixels that survive the stereo‑pair test enter the DB (via TrackingDB.create_links()),
    # and of those, only those that also survive the RANSAC‑PnP inlier check get promoted into tracks
    tracking_pipeline = TrackingPipeline(
        cam = cam,
        img_seq = img_seq,
        akaze_thresh=args.akaze,
        disparity_min=args.disp_min,
        pixel_thresh=args.pix_thresh,
        stereo_cross_check=args.stereo_cross_check,
        frame_cross_check=args.frame_cross_check,
        feature_type=args.feature_type,
        matcher_type=args.matcher_type,
        plot=args.plot
    )
    if args.build_db:
        # Build DB and return
        tracking_pipeline.build(max_frame=args.max_frames,
                                          out_path=Path(args.out_path))
        Rt_curr_to_next = tracking_pipeline.get_relative_Rts()
    else:
        with open("relative_Rts", "rb") as f:
            Rt_curr_to_next = pickle.load(f)

    # 3) B. Load DB
    db = TrackingDB()
    db.load(args.out_path)

    # 4) A. Create a local-BA graph (gtsam expects 'camera -> world' format):
    ba = BundleAdjustment(cam, Rt_curr_to_next,
                          db.links_by_frame(), plot=args.plot)

    graphs, values = ba.build_windows()

    # 4) B. Optimize each window (separately)
    res_ba = ba.optimize_bundles(graphs, values)

    # 5) Create a pose-graph and perform LC optimization.
    KF_indices = ba.get_kf_indices()  # array of length 144
    lc = LoopClosure(cam, img_seq, KF_indices,
                     graphs, res_ba.ba_values,
                     args.frame_cross_check)

    (pose_graph_pre, pre_vals), (pose_graph, pose_vals) = lc.run_loop_closure(db)

    # world-to-camera poses
    Rts_abs_all = utils.chain_rel_to_abs_w2c(Rt_curr_to_next)  # (N,3,4)
    # 6) Optional: Performance Analysis
    if args.plot:
        # Compute PnP / BA / LC / GT rel poses
        rel_pnp_poses = utils.relatives_from_absolutes_w2c_adjacent(Rts_abs_all, KF_indices)
        rel_pre_lc_poses = vals_to_rel_arrays(pre_vals)
        rel_post_lc_poses = vals_to_rel_arrays(pose_vals)  # (M-1,4,4)
        rel_gt_poses = utils.relatives_from_absolutes_w2c_adjacent(poses_gt, KF_indices)  # (k <- k+1), (M-1,4,4)

        # --- PnP Performance Analysis ---
        pnp_plot = PnP_Plotter(cam, poses_gt, db)
        pnp_plot.reproj_err_over_track_len()
        pnp_plot.abs_position_errs(Rts_abs_all)  # plot reprojection errors over chosen track

        # --- Bundle-Adjustment Performance Analysis ---
        ba_plot = BA_Plotter(res_ba, poses_gt, Rts_abs_all)
        ba_plot.mean_error_per_bundle()
        ba_plot.median_error_per_bundle()
        ba_plot.reproj_vs_distance(window_idx=96)
        ba_plot.abs_position_errs()

        # --- Loop-Closure Performance Analysis ---
        lc_plot = LC_Plotter(pose_vals, KF_indices, poses_gt, Rts_abs_all)
        lc_plot.abs_position_errs()
        lc_plot.lc_success_stats(lc.get_pair_stats())

        ## Uncertainties
        lc_plot.location_uncertainty(pose_graph_pre, pre_vals,
                                  pose_graph, pose_vals, conf_mult=1.0)

        lc_plot.angle_uncertainty(pose_graph_pre, pre_vals,
                                  pose_graph, pose_vals, conf_mult=1.0)

        lc_plot.uncertainty_score(pose_graph_pre, pre_vals,
                                  pose_graph, pose_vals, lc=lc,
                                  ref_rot_deg=1.0, ref_trans_m=0.10,
                                  conf_mult=1.0, use_log10=True)

        # --- Shared graphs comparing PnP, BA, LC on the same figure ---
        OptimizersPlot.abs_rotation_errs(Rts_abs_all, pre_vals,
                                         pose_vals, poses_gt,
                                         KF_indices)

        OptimizersPlot.rel_position_err(rel_pnp_poses, rel_pre_lc_poses,
                                        rel_post_lc_poses,  rel_gt_poses,
                                        KF_indices)

        OptimizersPlot.rel_rotation_err(rel_pnp_poses, rel_pre_lc_poses,
                                        rel_post_lc_poses, rel_gt_poses,
                                        KF_indices)

    # --- Final Trajectories for each Algorithm ---
    utils.plot_keyframe_trajectories(KF_indices, poses_gt, Rts_abs_all,
                                     pre_vals, pose_vals,
                                     label_post=f"after LC ({lc.get_loops()} loops)")


if __name__ == '__main__':
    main()