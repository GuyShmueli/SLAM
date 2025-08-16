"""

"""
import numpy as np
import gtsam

def compute_err(meas_pts, proj_pts):
    """ Compute the error between the measured pixels
    and the projected pixels. """
    err = np.linalg.norm(meas_pts - proj_pts, axis=1)  # pixel distance
    return err


def extract_intrinsic_param(K, M_right):
    """
    Return (fx, fy, skew, cx, cy, baseline) extracted from intrinsic-matrix.
    Used for providing parameters to gtsam.Cal3_S2Stereo().
    """
    fx, fy = K[0, 0], K[1, 1]  # focal length in x & y (pixels)
    s = K[0, 1]  # skew (0)
    cx, cy = K[0, 2], K[1, 2]  # principal point x & y (pixels)
    b = abs(M_right[0, 3])  # baseline (meters)
    return fx, fy, s, cx, cy, b


def homogenous(Rt):
    """ Make a (3x4) matrix homogenous (4x4). """
    T = np.vstack([Rt, [0, 0, 0, 1]])
    return T


def non_homogenous(T):
    """ Make a (4x4) homogenous matrix into (3x4). """
    Rt = T[:3, :]  # (3x4)
    return Rt


def split_R_t(Rt):
    """ Given a non-homogenous Rt or homogenous T, return (R,t) """
    R = Rt[:3, :3]
    t = Rt[:3, 3]
    return R, t


def invert_pose(T):
    """ Return the inverse of a (4x4) homogeneous pose matrix. """
    R, t = split_R_t(T)
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t
    return T_inv


def bundles_split_Rt(T_bundles):
    """ Split each (4x4) pose in every bundle window into separate rotation and translation arrays. """
    R_cam_to_window = [np.empty((len(window), 3, 3)) for window in T_bundles]
    t_cam_to_window = [np.empty((len(window), 3)) for window in T_bundles]

    for w, window in enumerate(T_bundles):
        for f, T in enumerate(window):
            R_cam_to_window[w][f], t_cam_to_window[w][f] = split_R_t(T)

    return R_cam_to_window, t_cam_to_window


def single_backproject_gtsam(cam, z):
    """Back-project a stereo measurement z to a 3D world point using gtsam.StereoCamera."""
    # Buffers for the Jacobians (shape 3x6 and 3x3)
    # 'order=F' -> f_contiguous -> data is laid out column-by-column
    H_pose = np.zeros((3, 6), dtype=np.float64, order='F')      # 3x6
    H_meas = np.zeros((3, 3), dtype=np.float64, order='F')      # 3x3

    # Triangulation
    p_xyz = cam.backproject2(z, H_pose, H_meas)  # ndarray (3,)
    return p_xyz


def Rts_to_homogenous(Rts_wc):
    """Convert an array of 3x4  world -> cam  matrices into 4x4 homogeneous transforms."""
    T_wc = np.zeros((len(Rts_wc), 4, 4), dtype=Rts_wc[0].dtype)
    T_wc[:, :3, :4] = Rts_wc
    T_wc[:, 3, 3] = 1
    return T_wc


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


def reverse_cov(Sigma_fwd,
                T_fwd: gtsam.Pose3):
    """Compute Sigma_{k|k+1}, rather than Sigma_{k+1|k} as usual."""
    T_inv = T_fwd.inverse()
    Ad = T_inv.AdjointMap()
    Sigma_bwd = Ad @ Sigma_fwd @ Ad.T
    return T_inv, Sigma_bwd


def vals_to_rel_arrays(vals):
    """ Given a 'Values' object, return a numpy array of the
    homogenous relative poses C_{k<-k+1} """
    T_list = []
    for k in range(vals.size() - 1):
        Tk  = vals.atPose3(gtsam.symbol('c', k))     # C_{0<-k}
        Tk1 = vals.atPose3(gtsam.symbol('c', k+1))   # C_{0<-k+1}
        Trel = Tk.between(Tk1)                       # C_{k<-k+1}

        R = Trel.rotation().matrix()           # (3,3)
        t = Trel.translation()                 # (3,)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        T_list.append(T)
    return np.array(T_list)    # shape (M-1,4,4)


def cov_nextKF_to_currKF(kf_0, kf_1, marginals, rel_pose):
    """
    Return Cov( ξ_{0←1} ), i.e. the 6×6 covariance of the relative pose c0 <- c1,
    expressed in the c0 frame.
    """
    c_kf0 = gtsam.symbol('c', kf_0)
    c_kf1 = gtsam.symbol('c', kf_1)

    # --- 1. joint covariance -----------------------------
    keys = gtsam.KeyVector([c_kf0, c_kf1])
    Sigma_joint = marginals.jointMarginalCovariance(keys).fullMatrix()  # 12×12

    # --- 2. conditional cov  Σ_{1|0} ---------------------
    Lambda_joint = np.linalg.inv(Sigma_joint)
    Sigma_1_cond_0 = np.linalg.inv(Lambda_joint[6:, 6:])  # 6×6

    # --- 3. adjoint to c0 frame --------------------------
    # T_0_1 = values.atPose3(c_kf0).between(values.atPose3(c_kf1))
    Adj_0_1 = gtsam.Pose3.AdjointMap(rel_pose)  # 6×6
    Sigma_rel = Adj_0_1 @ Sigma_1_cond_0 @ Adj_0_1.T  # 6×6

    return Sigma_rel

def pose_nextKF_to_currKF(kf_0, kf_1, values):
    """ Compute the relative pose  c0 <- ck  from two global poses in 'values'. """
    c_kf0 = gtsam.symbol('c', kf_0)
    c_kf1 = gtsam.symbol('c', kf_1)
    pose_kf0 = values.atPose3(c_kf0)
    pose_kf1 = values.atPose3(c_kf1)
    # c_KF0  <-  c_KF1
    return pose_kf0.between(pose_kf1)


def rel_to_vals(poses_next_to_curr):
    """ Compute gtsam.Values object (in global-coords) from relative poses. """
    pose_vals = gtsam.Values()
    pose_vals.insert(gtsam.symbol('c', 0), gtsam.Pose3())  # c0 = I
    running = gtsam.Pose3()  # cumulative pose  C_{0 <- k}
    for k, T_k_k1 in enumerate(poses_next_to_curr):
        running = running.compose(T_k_k1)  # C_{0 <- k+1}
        pose_vals.insert(gtsam.symbol('c', k + 1), running)
    return pose_vals


def values_keyframes_to_w2c(vals: gtsam.Values, num_kf: int):
    """
    Extract (num_kf,3,4) world->camera poses for nodes c0..c{num_kf-1}
    from a gtsam.Values that stores Pose3 in camera->world form.
    """
    T = np.zeros((num_kf, 3, 4))
    for k in range(num_kf):
        P = vals.atPose3(gtsam.symbol('c', k))  # Pose3 (camera->world)
        Rcw = P.rotation().matrix()
        tcw = np.array(P.translation())
        Rwc = Rcw.T
        twc = -Rwc @ tcw
        T[k, :3, :3] = Rwc
        T[k, :, 3] = twc
    return T
