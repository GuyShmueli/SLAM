"""

"""
from slam.backend.optimizers.graph_utils import reverse_cov
from slam.frontend.geometry.pnp import PnP
from slam.frontend.geometry.triangulation import Triangulation
from slam.frontend.io.camera_model import CameraModel
from slam.backend.tracking_database import TrackingDB
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
                 opt_vals,
                 frame_cross_check,
                 inflation = 1.05e6,       # to get reasonable chi-squared values
                 pixel_thresh = 2.0,       # as in PnP class, for RANSAC
                 disparity_min = 1.0,
                 min_inliers_loop = 180,   # minimal number of inliers to consider 2 frames as a loop
                 k_best = 3,               # take 'k_best' candidates after chi-squared-test
                 back_window = 144,        # how far looking back to detect loop candidates
                 min_kf_gap = 40,          # don't examine candidates which their gap is less than 'min_kf_gap'
                 chi2_thresh = 10,         # don't examine candidates that chi-squared-test > 'chi2_tresh'
                 report_every = 1          # report loop-closure every 'report_every' iterations
                 ):
        self.pnp = PnP(cam, pixel_thresh)
        self.triang = Triangulation(cam, disparity_min, pixel_thresh)
        self.img_seq = img_seq
        self.KF_indices = KF_indices
        self.graphs = graphs
        self.opt_vals = opt_vals

        self.frame_cross_check = frame_cross_check
        self.inflation = inflation
        self.pixel_thresh = pixel_thresh
        self.min_inliers_loop = min_inliers_loop
        self.k_best = k_best
        self.back_window = back_window
        self.min_kf_gap = min_kf_gap
        self.chi2_thresh = chi2_thresh
        self.report_every = report_every


    def mini_pose_graph(self, frames, init_poses):
        """Initialize a 2-sized pose graph & fill initial guess for Values """
        g2 = gtsam.NonlinearFactorGraph()
        v2 = gtsam.Values()

        for fid in frames:
            key = gtsam.symbol('c', fid)
            pose0 = init_poses.get(fid)
            v2.insert(key, pose0)

        return g2, v2

    def cov_nextKF_to_currKF(self, kf_0, kf_1, marginals):
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


    def pose_nextF_to_currKF(self, kf_0, kf_1, values):
        """ Compute the relative pose  c0 <- ck  from two global poses in 'values'. """
        c_kf0 = gtsam.symbol('c', kf_0)
        c_kf1 = gtsam.symbol('c', kf_1)
        pose_kf0 = values.atPose3(c_kf0)
        pose_kf1 = values.atPose3(c_kf1)
        # c_KF0  <-  c_KF1
        return pose_kf0.between(pose_kf1)


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
            * 'cov' - 6×6 covariance in u-coords
            * 'weight' - scalar cost  (det Sigma)^(1/6)
        """
        # Initialize an empty graph
        G = nx.Graph()

        # --- chain edges (k , k+1) in both directions ---
        for k, (T_fwd, Sigma_fwd) in enumerate(zip(rel_poses, rel_covs)):
            Sigma_fwd = Sigma_fwd * self.inflation
            w = self.edge_weight_root6(Sigma_fwd)      # root-det after scaling
            Sigma_back = reverse_cov(Sigma_fwd, T_fwd)

            # forward  k -> k+1   (store pose  k <- k+1, in known k-coordinates)
            G.add_edge(k, k+1, pose=T_fwd,   cov=Sigma_fwd,  weight=w)
            # reverse  k+1 -> k   (store pose  k+1 <- k)
            G.add_edge(k+1, k, pose=T_fwd.inverse(), cov=Sigma_back, weight=w)

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
            chi2_thresh: float):
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

        is_ok = chi2_val < chi2_thresh

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
                         self.triang.P_left, self.triang.P_right, kpL_i, kpR_i)

        ## 3. Mapping
        # key: index of the left-image feature in the candidate KF
        # value: 3D-point in candidate's left-image coordinates
        xyz_by_lidx = {i: obj_pts[i] for i in range(len(obj_pts))}  # candidate-frame ->  3D-point

        # key: index of the left-image feature in the current KF
        # value: sub-pixel (u,v) of the matching right-image feature
        right_by_lidx = {i: kpR_n[i].pt for i in range(len(kpR_n))}  # current-frame   ->  right–pixel

        ## 4. Descriptor matching:  cand_L  ->  current_L
        bf = DescriptorMatcher(cross_check=self.frame_cross_check)
        raw = bf.knn_match(desL_i, desL_n, k=2)
        good, _ = bf.ratio_test(raw, ratio=0.8)

        ## 5. Lists appending: 3D-points, curr_L, curr_R, cand_L pixels
        P3D, ptsL_curr, ptsR_curr, ptsL_cand = [], [], [], []
        for m in good:
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
                                p_success=0.99999, n_iter_max=1000)

        if mask.sum() < self.min_inliers_loop:
            return (False, None, None, (None, None), {})

        # 7. Refine pose
        Rt = self.pnp.solve_pnp(P3D[mask], ptsL_curr[mask], cv2.SOLVEPNP_ITERATIVE)
        if Rt is None:
            return (False, None, None, (None, None), {})

        R, t = Rt[:, :3], Rt[:, 3]
        Delta = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))  # Delta(c_in)

        ## 8. Track pixels for inliers/outliers plotting
        px = dict(inl_curr=ptsL_curr[mask],
                  out_curr=ptsL_curr[~mask],
                  inl_cand=ptsL_cand[mask],
                  out_cand=ptsL_cand[~mask])

        return (True, Delta, mask, (kf_curr, kf_cand), px)

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
                         db: TrackingDB,
                         prior_sigmas,
                         early_terminate):
        # 2) --- derive relative poses & covariances for EVERY KF pair ---
        marginals = [
                    gtsam.Marginals(graph, val)
                    for (graph, val) in zip(self.graphs, self.opt_vals)
                    ]

        KF_idx_pairs = [
                        (self.KF_indices[i], self.KF_indices[i + 1])
                        for i in range(len(self.KF_indices) - 1)
                       ]

        # Covariances for every KF pair
        covs_next_cond_curr = [
                               self.cov_nextKF_to_currKF(k0, k1, marg)
                               for (k0, k1), marg in zip(KF_idx_pairs, marginals)
                              ]
        # Poses for every KF pair
        poses_next_to_curr = [
                              self.pose_nextF_to_currKF(k0, k1, res)
                              for (k0, k1), res in zip(KF_idx_pairs, self.opt_vals)
                             ]

        # 3) --- initialize pose graph, filling it & loop-enclosing ---
        G = self.weighted_pose_graph(poses_next_to_curr, covs_next_cond_curr)

        # Optimization is done via a GTSAM factor-graph  (not networkx)
        pose_graph = gtsam.NonlinearFactorGraph()
        pose_vals = gtsam.Values()

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
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

            Sig = Sigma_k_k1 * self.inflation
            noise = gtsam.noiseModel.Gaussian.Covariance(Sig)

            pose_graph.add(gtsam.BetweenFactorPose3(
                gtsam.symbol('c', k),  # from pose k
                gtsam.symbol('c', k + 1),  # to pose k+1
                T_k_k1, noise))

        est_poses = {k: pose_vals.atPose3(gtsam.symbol('c', k))
                     for k in range(len(self.KF_indices))}

        # Record 'pre_vals' & 'pose_graph_pre' for post-optimization analysis
        pose_graph_pre = gtsam.NonlinearFactorGraph(pose_graph)
        pre_vals = gtsam.Values(pose_vals)

        # To record successful current-candidate pairs
        success_curr_cand = []

        # Iterate every KF-idx (pose) in the graph
        added = 0   # record how many new edges are added
        for n_kf in range(1, len(self.KF_indices)):
            if n_kf > early_terminate:
                break

            n_fid = self.KF_indices[n_kf]    # turn pose-graph indexing into BA factor-graph indexing

            # chi-squared pre-gate, to save running time in subsequent heavy steps
            passed = []
            for i_kf in range(max(0, n_kf - self.back_window), n_kf):    # don't check close poses
                if n_kf - i_kf < self.min_kf_gap:
                    continue
                ok, chi2 = self.mahalanobis_pose_error(i_kf, n_kf, G,
                                                     est_poses, self.chi2_thresh)
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

                imgL_curr = self.img_seq[success_n_curr]
                imgL_cand = self.img_seq[success_n_cand]
                # Record successful current-candidate pairs
                success_curr_cand.append((imgL_curr, imgL_cand, px))

                # Tiny 2-pose graph for covariance
                frames = [i_fid, n_fid]
                init_poses = {frames[0]: est_poses[i_kf],
                              frames[1]: est_poses[n_kf]}
                g2, v2 = self.mini_pose_graph(frames, init_poses)
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
                Sig_i_n = Sig_i_n * self.inflation

                # add to graphs
                w = self.edge_weight_root6(Sig_i_n)
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

                if added % self.report_every == 0:  # report every N closures
                    print(f"\noptimizing full graph after {added} loop closures...")
                    pose_vals = gtsam.LevenbergMarquardtOptimizer(
                        pose_graph, pose_vals).optimize()

                    # refresh the convenience dict used by mahalanobis gate
                    est_poses = {k: pose_vals.atPose3(gtsam.symbol('c', k))
                                 for k in range(len(self.KF_indices))}
                    print("optimization done.\n")

        print(f"\nLoop-closure has finished with {added} closures added.")

        return (pose_graph_pre, pre_vals), (pose_graph, pose_vals)    # (before), (after)