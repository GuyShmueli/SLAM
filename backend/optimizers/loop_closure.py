# slam/backend/optimizers/loop_closure.py
from slam.utils.graph_utils import reverse_cov, cov_nextKF_to_currKF, pose_nextKF_to_currKF
from slam.frontend.geometry.pnp import PnP
from slam.frontend.io.camera_model import CameraModel
from slam.backend.tracking.database import TrackingDB
from slam.frontend.io.image_sequence import ImageSequence
from slam.frontend.vision.descriptor_matcher import DescriptorMatcher
import gtsam
import networkx as nx
import numpy as np
import cv2

class LoopClosure:
    """

    """
    def __init__(self,
                 cam: CameraModel,
                 img_seq: ImageSequence,
                 KF_indices,
                 graphs,
                 ba_vals,
                 frame_cross_check,
                 inflation = 1.0e2,        # to get reasonable chi-squared values
                 min_inliers_loop = 60,    # minimal number of inliers to consider 2 frames as a loop
                 k_best = 3,               # take 'k_best' candidates after chi-squared-test
                 back_window = 144,        # how far looking back to detect loop candidates
                 min_kf_gap = 5,           # don't examine candidates which their gap is less than 'min_kf_gap'
                 chi2_thresh = 15.0,       # don't examine candidates that chi-squared-test > 'chi2_tresh'
                 report_every = 1          # report loop-closure every 'report_every' iterations
                 ):
        self.pnp = PnP(cam, 1.5)
        self.img_seq = img_seq
        self.KF_indices = KF_indices
        self.graphs = graphs
        self.ba_vals = ba_vals      # (keyframe <- camera)

        self.frame_cross_check = frame_cross_check
        self.inflation = inflation
        self.min_inliers_loop = min_inliers_loop
        self.k_best = k_best
        self.back_window = back_window
        self.min_kf_gap = min_kf_gap
        self.chi2_thresh = chi2_thresh
        self.report_every = report_every

        self.loops = 0            # cache number of loops
        self.lc_pair_stats = []   # <- (curr_kf, cand_kf, curr_frame, cand_frame, #matches, #inliers, inlier%)


    def mini_pose_graph(self, frames, init_poses):
        """Initialize a 2-sized pose graph & fill initial guess for Values """
        g2 = gtsam.NonlinearFactorGraph()
        v2 = gtsam.Values()

        for fid in frames:
            key = gtsam.symbol('c', fid)
            pose0 = init_poses.get(fid)
            v2.insert(key, pose0)

        return g2, v2


    def edge_weight_root6(self, Sigma):
        """w = (det Sigma)^(1/6) - geometric mean of the six eigen-variances."""
        return np.linalg.det(Sigma) ** (1.0 / 6.0)


    def weighted_pose_graph(self,
                            rel_poses,    # Pose3 (k <- k+1)
                            rel_covs,     # Sigma_{k+1|k} (6×6)
                            ):
        """
        Build an undirected graph whose edges carry:
            * 'pose' - Pose3  (u <- v)
            * 'cov' - 6x6 covariance in u-coords
            * 'weight' - scalar cost  (det Sigma)^(1/6)
        """
        # Initialize an empty graph
        G = nx.DiGraph()

        # --- chain edges (k , k+1) in both directions ---
        for k, (T_fwd, Sigma_fwd) in enumerate(zip(rel_poses, rel_covs)):
            Sigma_fwd = Sigma_fwd * self.inflation
            w = self.edge_weight_root6(Sigma_fwd)      # root-det after scaling
            T_inv, Sigma_bwd = reverse_cov(Sigma_fwd, T_fwd)

            # forward  k -> k+1   (store pose  k <- k+1, in known k-coordinates)
            G.add_edge(k, k+1, pose=T_fwd,   cov=Sigma_fwd,  weight=w)
            # reverse  k+1 -> k   (store pose  k+1 <- k)
            G.add_edge(k+1, k, pose=T_inv, cov=Sigma_bwd, weight=w)

        return G


    def shortest_path_nodes(self, i, n, G):
        """Dijkstra's algorithm to find the shortest path in a weighted graph."""
        return nx.shortest_path(G, source=i, target=n, weight='weight')


    def accumulate_sigma(self, path, G):
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


    def mahalanobis_pose_error(self,
            i, n,
            G: nx.Graph,
            est_poses: dict[int, gtsam.Pose3],  # global KF poses
            ):
        """
        Perform a chi-squared test.
        The smaller the value of  1/2 * r^T @ Sigma^-1 @ r  is,
        the more likely  c_i & c_n  are to be close to one another.
        """
        if n <= i:
            raise ValueError("need i < n")

        # 1. Compute shortest-path & Sigma_n_i across that path
        path = self.shortest_path_nodes(i, n, G)
        Sigma_n_i = self.accumulate_sigma(path, G)

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

        is_ok = chi2_val < self.chi2_thresh
        # print(chi2_val)
        return is_ok, chi2_val


    def frame_data(self, db: TrackingDB, kf_id: int):
        """
        Extract from db (using 'features') left & right image keypoints
        and the descriptors for the left image only.
        """
        descL = db.features(kf_id)  # (N,D) left descriptors
        links = db.all_frame_links(kf_id)  # list[Link] length N

        kpL, kpR = [], []
        for ln in links:
            kpL.append(cv2.KeyPoint(ln.x_left, ln.y, 1))
            kpR.append(cv2.KeyPoint(ln.x_right, ln.y, 1))
        return kpL, kpR, descL


    def consensus_pnp_match(self, kf_curr: int, kf_cand: int,
                            db: TrackingDB):
        """
        Do a stereo-aware 2-view PnP check for current-candidate pair, and
        give back everything needed to draw inliers/outliers on both left images.
        """
        ## 1. Cached keypoints & descriptors for both current (n) and candidate (i)
        kpL_i, kpR_i, desL_i = self.frame_data(db, kf_cand)
        kpL_n, kpR_n, desL_n = self.frame_data(db, kf_curr)

        ## 2. Compute 3D-points in candidate's left-frame coords
        # kpL_i, kpR_i are already stereo-pair because they were cached in the db
        obj_pts = self.compute_triangulation(
                         self.pnp.P_left, self.pnp.P_right, kpL_i, kpR_i)

        ## 3. Mapping
        # key: index of the left-image feature in the candidate KF
        # value: 3D-point in candidate's left-image coordinates
        xyz_by_lidx = {i: obj_pts[i] for i in range(len(obj_pts))}  # candidate-frame ->  3D-point

        # key: index of the left-image feature in the current KF
        # value: sub-pixel (u,v) of the matching right-image feature
        right_by_lidx = {i: kpR_n[i].pt for i in range(len(kpR_n))}  # current-frame   ->  right–pixel

        ## 4. Descriptor matching:  cand_L  ->  current_L
        bf = DescriptorMatcher(cross_check=self.frame_cross_check)
        # raw = bf.knn_match(desL_i, desL_n, k=2)
        # good, _ = bf.ratio_test(raw, ratio=0.99)
        matches = bf.match(desL_i, desL_n)
        ## 5. Lists appending: 3D-points, curr_L, curr_R, cand_L pixels
        P3D, ptsL_curr, ptsR_curr, ptsL_cand = [], [], [], []
        for m in matches:
            xyz = xyz_by_lidx.get(m.queryIdx)  # m.queryIdx refers to candidate's desL idx
            right_pt = right_by_lidx.get(m.trainIdx)  # m.trainIdx refers to current's desL idx
            P3D.append(xyz)  # 3D-points for reprojection in RANSAC-PnP
            ptsL_curr.append(kpL_n[m.trainIdx].pt)  # curr_L pixels for both RANSAC-PnP and plotting
            ptsR_curr.append(right_pt)  # curr_R pixels for RANSAC-PnP
            ptsL_cand.append(kpL_i[m.queryIdx].pt)  # cand_L pixels for plotting inliers/outliers

        if len(P3D) < self.min_inliers_loop:
            return (False, None, None, (None, None), {})

        P3D = np.asarray(P3D, np.float32)
        ptsL_curr = np.asarray(ptsL_curr, np.float32)
        ptsR_curr = np.asarray(ptsR_curr, np.float32)
        ptsL_cand = np.asarray(ptsL_cand, np.float32)

        ## 6. Stereo-aware RANSAC-PnP
        mask = self.pnp.ransac_pnp_pair1(P3D, ptsL_curr, ptsR_curr,
                                p_success=0.99999, n_iter_max=1500)

        if mask.sum() < self.min_inliers_loop:
            return (False, None, None, (None, None), {})

        # 7. Refine pose
        Rt = self.pnp.solve_pnp(P3D[mask], ptsL_curr[mask], cv2.SOLVEPNP_ITERATIVE)
        if Rt is None:
            return (False, None, None, (None, None), {})

        R, t = Rt[:, :3], Rt[:, 3]
        Delta = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))  # Delta(c_in)
        Delta_inv = Delta.inverse()
        ## 8. Track pixels for inliers/outliers plotting
        px = dict(inl_curr=ptsL_curr[mask],
                  out_curr=ptsL_curr[~mask],
                  inl_cand=ptsL_cand[mask],
                  out_cand=ptsL_cand[~mask])

        return (True, Delta_inv, mask, (kf_curr, kf_cand), px)

    def compute_triangulation(self, P_L, P_R,
                              kpL: list[cv2.KeyPoint],
                              kpR: list[cv2.KeyPoint]):
        """Perform triangulation using 'cv2.triangulatePoints'. """
        pts_L = np.array([k.pt for k in kpL]).T  # (2,N)
        pts_R = np.array([k.pt for k in kpR]).T
        pts_L_h = cv2.convertPointsToHomogeneous(pts_L.T)[:, 0, :]  # (N,3)
        pts_R_h = cv2.convertPointsToHomogeneous(pts_R.T)[:, 0, :]
        proj = cv2.triangulatePoints(P_L, P_R,
                                     pts_L_h.T[:2],
                                     pts_R_h.T[:2])
        world_pts = (proj[:3] / proj[3]).T  # (N,3)  XYZ in left–cam frame
        return world_pts


    def run_loop_closure(self,
                         db: TrackingDB):
        # 1) --- derive relative poses & covariances for EVERY KF pair ---
        marginals = [
                    gtsam.Marginals(graph, val)
                    for (graph, val) in zip(self.graphs, self.ba_vals)
                    ]

        KF_idx_pairs = [
                        (self.KF_indices[i], self.KF_indices[i + 1])
                        for i in range(len(self.KF_indices) - 1)
                       ]

        # Relative poses for every KF pair
        poses_next_to_curr = [
                              pose_nextKF_to_currKF(k0, k1, res)
                              for (k0, k1), res in zip(KF_idx_pairs, self.ba_vals)
                             ]

        # Conditional covariances for every KF pair
        covs_next_cond_curr = [
                               cov_nextKF_to_currKF(k0, k1, marg, rel_pose)
                               for (k0, k1), marg, rel_pose in zip(KF_idx_pairs, marginals, poses_next_to_curr)
                              ]

        # 2) --- initialize pose graph, filling it & loop-enclosing ---
        G = self.weighted_pose_graph(poses_next_to_curr, covs_next_cond_curr)

        # 3) --- optimization: done via a GTSAM factor-graph (not networkx) ---
        pose_graph = gtsam.NonlinearFactorGraph()

        # Prior sigmas (yaw-pitch-roll-x-y-z)
        prior_sigmas = np.array([np.deg2rad(5), np.deg2rad(5), np.deg2rad(5),
                                 0.05, 0.05, 0.15])
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
        pose_graph.add(gtsam.PriorFactorPose3(gtsam.symbol('c', 0),
                                              gtsam.Pose3(), prior_noise))

        pose_vals = gtsam.Values()
        pose_vals.insert(gtsam.symbol('c', 0), gtsam.Pose3())  # c0 = I

        running = gtsam.Pose3()   # cumulative pose  C_{0 <- k}
        est_poses = {0: running}  # will be used by mahalanobis gate

        # Fill the initial pose-graph
        for k, (T_k_k1, Sigma_k_k1_w) in enumerate(zip(poses_next_to_curr,
                                                       covs_next_cond_curr)):

            noise = gtsam.noiseModel.Gaussian.Covariance(Sigma_k_k1_w)

            # Add the between-factor  (k , k+1)
            pose_graph.add(gtsam.BetweenFactorPose3(
                gtsam.symbol('c', k),
                gtsam.symbol('c', k + 1),
                T_k_k1, noise))

            # Extend the running pose & Values / est_poses
            running = running.compose(T_k_k1)  # C_{0 <- k+1}
            pose_vals.insert(gtsam.symbol('c', k + 1), running)
            est_poses[k + 1] = running

        # Record 'pre_vals' & 'pose_graph_pre' for post-optimization analysis
        pose_graph_pre = gtsam.NonlinearFactorGraph(pose_graph)
        pre_vals = gtsam.Values(pose_vals)

        # To record successful current-candidate pairs
        success_curr_cand = []

        # Iterate every KF-idx (pose) in the graph
        for n_kf in range(1, len(self.KF_indices)):
            n_fid = self.KF_indices[n_kf]    # turn pose-graph indexing into BA factor-graph indexing

            # chi-squared pre-gate, to save running time in subsequent heavy steps
            passed = []
            for i_kf in range(max(0, n_kf - self.back_window), n_kf):    # don't check close poses
                if n_kf - i_kf < self.min_kf_gap:
                    continue
                ok, chi2 = self.mahalanobis_pose_error(i_kf, n_kf, G,
                                                     est_poses)
                if ok:
                    passed.append((chi2, i_kf))

            passed.sort()
            cand_kf_idxs = [i for _, i in passed[:self.k_best]]
            if not cand_kf_idxs:
                continue
            print(f"n={n_kf}, candidates={cand_kf_idxs}")   # optional printing (for tracking)

            # Verify each possible candidate with a heavier PnP-RANSAC
            for i_kf in cand_kf_idxs:
                i_fid = self.KF_indices[i_kf]

                (success, Delta_i_n, inliers,
                 (success_n_curr, success_n_cand), px) = self.consensus_pnp_match(
                       kf_curr=n_fid, kf_cand=i_fid, db=db)
                if not success:
                    continue

                # stats for this successful pair (for later analysis)
                num_matches = int(inliers.size)    # correspondences given to RANSAC
                num_inliers = int(inliers.sum())   # inliers from the mask
                inlier_pct = 100.0 * num_inliers / max(1, num_matches)
                self.lc_pair_stats.append({
                    'curr_kf': n_kf,
                    'cand_kf': i_kf,
                    'curr_frame': n_fid,
                    'cand_frame': i_fid,
                    'num_matches': num_matches,
                    'num_inliers': num_inliers,
                    'inlier_pct': inlier_pct,
                })

                imgL_curr = self.img_seq[success_n_curr]
                imgL_cand = self.img_seq[success_n_cand]
                # Record successful current-candidate pairs
                success_curr_cand.append((imgL_curr, imgL_cand, px))

                # Tiny 2-pose graph for covariance
                frames = [i_fid, n_fid]
                init_poses = {frames[0]: est_poses[i_kf],
                              frames[1]: est_poses[n_kf]}

                g2, v2 = self.mini_pose_graph(frames, init_poses)

                key_i = gtsam.symbol('c', i_fid)
                key_n = gtsam.symbol('c', n_fid)

                # Fix i exactly -> conditioning on i
                g2.add(gtsam.NonlinearEqualityPose3(key_i, est_poses[i_kf]))

                # Add whatever produced the relative measurement Delta_{i->n}
                between_noise = gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([
                        np.deg2rad(1.0), np.deg2rad(1.0), np.deg2rad(3.0),  # rot sigmas [rad]
                        0.30, 0.15, 0.40  # trans sigmas [m]
                    ])
                )
                g2.add(gtsam.BetweenFactorPose3(key_i, key_n, Delta_i_n, between_noise))

                # Optimize mini-graph
                result2 = gtsam.LevenbergMarquardtOptimizer(g2, v2).optimize()

                # Read Sigma_{n|i}
                margs2 = gtsam.Marginals(g2, result2)
                Sig_i_n_body = margs2.marginalCovariance(key_n)

                inflated_Sig = Sig_i_n_body * self.inflation
                # add to graphs
                w = self.edge_weight_root6(inflated_Sig)
                G.add_edge(i_kf, n_kf, pose=Delta_i_n, cov=inflated_Sig, weight=w)

                Delta_i_n_inv, cov_bwd = reverse_cov(inflated_Sig, Delta_i_n)
                G.add_edge(n_kf, i_kf,
                           pose=Delta_i_n_inv,
                           cov=cov_bwd,
                           weight=w)

                noise = gtsam.noiseModel.Gaussian.Covariance(Sig_i_n_body)
                pose_graph.add(gtsam.BetweenFactorPose3(
                    gtsam.symbol('c', i_kf),
                    gtsam.symbol('c', n_kf),
                    Delta_i_n, noise))

                self.loops += 1
                print(f"loop formed by: {i_kf} & {n_kf}   inliers={int(inliers.sum())}")

                if self.loops % self.report_every == 0:  # report every N closures
                    print(f"\noptimizing full graph after {self.loops} loop closures...")
                    pose_vals = gtsam.LevenbergMarquardtOptimizer(      # define new 'pose_vals' an edge added
                        pose_graph, pose_vals).optimize()
                    # Refresh the convenience dict used by mahalanobis gate
                    est_poses = {k: pose_vals.atPose3(gtsam.symbol('c', k))
                                 for k in range(len(self.KF_indices))}
                    print("optimization done.\n")

        print(f"\nLoop-closure has finished with {self.loops} closures added.")

        return (pose_graph_pre, pre_vals), (pose_graph, pose_vals)    # (before), (after)


    def get_loops(self):
        """ Return the number of loop closures detected. """
        return self.loops


    def get_pair_stats(self):
        return list(self.lc_pair_stats)