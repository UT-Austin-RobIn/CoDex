from typing import List, Union
from scipy.spatial.transform import Rotation as R
import numpy as np

def quat_to_rot(q: Union[List[float], np.ndarray], wxyz: bool = True) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        q (Union[List[float], np.ndarray]): 4-element quaternion array [w, x, y, z] or [x, y, z, w].
        wxyz (bool, optional): If True, the quaternion is in [w, x, y, z] format, otherwise [x, y, z, w]. Defaults to True.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    if wxyz:
        q = [q[1], q[2], q[3], q[0]]
    r = R.from_quat(q)
    return r.as_matrix()

def quat_pos_to_pose_mat(quat: Union[List[float], np.ndarray], pos: Union[List[float], np.ndarray], wxyz: bool = True) -> np.ndarray:
    """
    Convert quaternion and position to 4x4 pose matrix.

    Args:
        quat (Union[List[float], np.ndarray]): 4-element quaternion array [w, x, y, z] or [x, y, z, w].
        pos (Union[List[float], np.ndarray]): 3-element position array [x, y, z].
        wxyz (bool, optional): If True, the quaternion is in [w, x, y, z] format, otherwise [x, y, z, w]. Defaults to True.

    Returns:
        np.ndarray: 4x4 pose matrix.
    """
    rot = quat_to_rot(quat, wxyz=wxyz)
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose