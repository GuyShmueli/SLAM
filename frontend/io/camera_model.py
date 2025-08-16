"""
SLAM/frontend/io/camera_model.py
"""
from pathlib import Path
import numpy as np

class CameraModel:
    """
    Represents a stereo camera rig. Supports construction from explicit matrices or
    from a KITTI-style calibration file (calib.txt with P0/P1 lines).

    ---
    Attributes:
    K           (3×3) intrinsic matrix
    M_left      (3×4) left-camera extrinsic [R|t] in normalized coords
    M_right     (3×4) right-camera extrinsic
    """
    def __init__(self,
                 K: np.ndarray,
                 M_left: np.ndarray,
                 M_right: np.ndarray):
        """
        Initialize from explicit intrinsic and extrinsic matrices.

        ---
        Args:
            K       3x3 intrinsic matrix
            M_left  3x4 left-camera extrinsic [R|t] in left-camera coords
            M_right 3x4 right-camera extrinsic [R|t]
        """
        self.K = K
        self.M_left = M_left
        self.M_right = M_right
        # Precompute projection matrices
        self.P_left = self.K @ self.M_left
        self.P_right = self.K @ self.M_right


    @classmethod
    def from_kitti(
        cls,
        calib_txt: Path
    ) -> "CameraModel":
        """
        Build a CameraModel by parsing a KITTI-style calib.txt file.
        Expects first two lines:
            P0: fx  0  cx  tx  0  fy  cy  ty  0  0  1  0  (12 values)
            P1: ... same format for right camera

        Args:
            calib_txt: path to calib.txt
        Returns:
            CameraModel instance
        """
        path = Path(calib_txt)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(path, 'r') as f:
            l1 = f.readline().split()[1:]  # skip first token
            l2 = f.readline().split()[1:]  # move cursor to next line

        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)

        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)

        k = m1[:, :3]  # intrinsic matrix
        m1 = np.linalg.inv(k) @ m1  # extrinsic (left)
        m2 = np.linalg.inv(k) @ m2  # extrinsic (right)

        return cls(k, m1, m2)