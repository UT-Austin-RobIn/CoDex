from typing import Union
import numpy as np
import torch
from codex.utils import utils_rotation as transforms
from mani_skill.utils.structs.pose import Pose


from codex.grasp_gen.load_grasp import load_grasp_config  # noqa: F401  (re-exported)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Transform points using a transformation matrix.
    
    Args:
        points: Points to be transformed (shape: [n, 3])
        transform: Transformation matrix (shape: [b, 4, 4])
        
    Returns:
        Transformed points (shape: [n, b, 3])
    """
    assert len(points.shape) == 2 and points.shape[1] == 3, "Points should be of shape [n, 3]"
    transform = convert_to_pose_mat(transform)
    points = points.to(transform.device)
    n_points = points.shape[0]
    points_h = torch.cat([points, torch.ones(n_points, 1).to(points.device)], dim=-1)
    transformed_points_h = transform @ points_h.T  # [b, 4, n]
    transformed_points_h = transformed_points_h.permute(2, 0, 1)  # [n, b, 4]
    transformed_points_h = transformed_points_h[..., :3] / transformed_points_h[..., 3:]  # [n, b, 3]
    
    return transformed_points_h

def compute_hand_pose(hand_rel_pose: Pose, object_pose: Pose, offset: Union[np.ndarray, torch.tensor]=None) -> torch.Tensor:
    """Compute the world-space hand pose from a relative hand pose and the object pose.

    Args:
        hand_rel_pose: Hand pose relative to the object, as a Pose, tensor, or ndarray.
        object_pose: Current world-space object pose.
        offset (optional): Position offset to subtract from the hand position.

    Returns:
        hand_pose_mat (torch.Tensor): World-space hand pose as a [b, 4, 4] transformation matrix.
    """
    if offset is None:
        offset = torch.zeros(3, dtype=torch.float32).to(object_pose.device)
    if isinstance(offset, np.ndarray):
        offset = torch.from_numpy(offset).float().to(object_pose.device)

    object_pose_mat = convert_to_pose_mat(object_pose)
    hand_rel_pose_mat = convert_to_pose_mat(hand_rel_pose, object_pose_mat.device)
    hand_pose_mat = object_pose_mat @ hand_rel_pose_mat

    # apply the offset
    hand_pose_mat[:, :3, 3] -= offset

    return hand_pose_mat

def mat_to_pos_euler(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert a transformation matrix to position and Euler angles.
    
    Args:
        mat: Transformation matrix (shape: [n, 4, 4])
    """
    pos = mat[..., :3, 3]
    rot = mat[..., :3, :3].clone()
    # PyTorch3D's XYZ conversion applies asin to R[0,2]. A finite float32
    # rotation can round one epsilon outside [-1, 1] near pitch +/-90 deg;
    # clamp only that inverse-trig input, without mutating the physical pose.
    rot[..., 0, 2] = rot[..., 0, 2].clamp(-1.0, 1.0)
    euler = transforms.matrix_to_euler_angles(rot, convention='XYZ')
    results = torch.cat([pos, euler], dim=-1)
    return results

def pos_euler_to_mat(pos_euler: torch.Tensor) -> torch.Tensor:
    """
    Convert position and Euler angles to a transformation matrix.
    """
    pos, euler = pos_euler[..., :3], pos_euler[..., 3:]
    assert pos.shape[1] == 3 and euler.shape[1] == 3, "Position and Euler angles should be of shape [n, 3]"
    rot = transforms.euler_angles_to_matrix(euler, convention='XYZ')
    # clone(): expand() returns a view sharing one storage; in-place writes
    # below would otherwise alias every batch element.
    mat = torch.eye(4).unsqueeze(0).expand(pos.shape[0], -1, -1).clone().to(pos.device)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot
    return mat

def mat_to_pos_quat(mat: torch.Tensor, xyzw=True, concat_result=False) -> torch.Tensor:
    """
    Convert a transformation matrix to position and quaternion.
    
    Args:
        mat: Transformation matrix (shape: [n, 4, 4])
    """
    pos = mat[..., :3, 3]
    rot = mat[..., :3, :3]
    quat = transforms.matrix_to_quaternion(rot)
    if not xyzw:
        quat = quat[..., [3, 0, 1, 2]]  # Convert to wxyz

    if concat_result:
        return torch.cat([pos, quat], dim=-1)
    else:
        return pos, quat

def convert_to_pose_mat(pose: Union[Pose, torch.Tensor, np.ndarray], device=None) -> torch.Tensor:
    """Convert a Pose, numpy array, or tensor to a float32 [b, 4, 4] transformation matrix on `device`."""
    if isinstance(pose, Pose):
        pose_mat = pose.to_transformation_matrix()
    elif isinstance(pose, np.ndarray):
        pose_mat = torch.tensor(pose).float()
    elif isinstance(pose, torch.Tensor):
        pose_mat = pose.float()

    if device is not None:
        pose_mat = pose_mat.to(device)
    return pose_mat


def reorder_finger_config(finger_config: np.ndarray, mode, is_leap=True) -> np.ndarray:
    """
    Reorder the finger configuration to match the target order.
    
    Args:
        finger_config: Original finger configuration (shape: [16])
        mode: "name2tree" or "tree2name" to specify the reordering direction.
        
    Returns:
        Reordered finger configuration (shape: [16])
    """

    if mode not in ("tree2names", "names2tree"):
        raise ValueError(f"Invalid mode: {mode}. Expected 'tree2names' or 'names2tree'.")
    # The articulation orders joints level by level across fingers (joint 0
    # of every finger, then joint 1, ...); grasp files store per-finger blocks
    # [axl, mcp, pip, dip] (thumb [cmc, axl, mcp, ipl]). The permutation is an
    # involution, so both directions share it.
    target_order = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

    return finger_config[..., target_order]
