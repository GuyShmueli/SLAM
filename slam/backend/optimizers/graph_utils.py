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
    Ad = T_fwd.AdjointMap()
    Sigma_bwd = Ad @ Sigma_fwd @ Ad.T
    return Sigma_bwd
