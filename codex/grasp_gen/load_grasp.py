"""Helpers for loading hand grasp configuration files and metadata."""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional, Tuple

import numpy as np

from codex.utils.utils_transform import quat_pos_to_pose_mat

def load_grasp_config(
    grasp_path: str,
    joint_only: bool = False,
    offset: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[list[int]]]:
    """Load a grasp text file and its metadata.

    Args:
        grasp_path: Path to the grasp txt file (length 16 or 23).
        joint_only: If True, return only the joint vector without composing a pose.
        offset: Translation applied to the Cartesian portion when joint_only is False.

    Returns:
        joint_config: 16-dof hand joint vector (names order).
        hand_pose_mat: 4x4 pose matrix (None when `joint_only` is True or pose absent).
        actuation_indices: Optional list describing actuated fingers from metadata.
    """
    if not os.path.exists(grasp_path):
        raise FileNotFoundError(f"Grasp file not found: {grasp_path}")
    
    # Load the metadata (grasp_<i>_metadata.json next to the grasp file)
    grasp_name = os.path.basename(grasp_path)
    grasp_name = grasp_name.removesuffix(".txt").removesuffix("_optimized")
    meta_path = os.path.join(os.path.dirname(grasp_path), f"{grasp_name}_metadata.json")
    if not os.path.exists(meta_path):
        warnings.warn(f"Metadata file not found: {meta_path}")
        actuation_indices = None
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            grasp_meta = json.load(f)
        actuation_indices = grasp_meta.get('actuation_finger_indices')

    # Load the grasp data
    grasp_data = np.loadtxt(grasp_path)
    if len(grasp_data) == 16:
        return grasp_data, None, actuation_indices
    elif len(grasp_data) == 23:
        hand_pose = grasp_data[:7]
        joint_config = grasp_data[7:]

        if joint_only:
            return joint_config, None, actuation_indices
        else:
            assert offset is not None, "Offset is required to lower the hand pose in z"
            hand_pose[4:] -= offset
            hand_pose_mat = quat_pos_to_pose_mat(hand_pose[:4], hand_pose[4:])

            return joint_config, hand_pose_mat, actuation_indices
    else:
        raise ValueError(f"Invalid grasp pose length: {len(grasp_data)}. Expected 16 or 23 values.")