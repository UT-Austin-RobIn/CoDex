import os
from copy import deepcopy
from typing import List

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose, Pose

from codex.utils.utils_path import PROJECT_ROOT_DIR
from codex.codex_train.agents.load_grasp_qpos import mat_to_pos_quat


# Pad centers in each distal link's raw frame, matching the grasp solver.
LEAP_PAD_OFFSETS_RAW = {
    "leap_rh_if_ds": (-0.010, -0.032, 0.015),
    "leap_rh_mf_ds": (-0.010, -0.032, 0.015),
    "leap_rh_rf_ds": (-0.010, -0.032, 0.015),
    "leap_rh_th_ds": (-0.009, -0.035, -0.011),
}


def leap_pad_pose_matrix(raw_pose: torch.Tensor, link_name: str) -> torch.Tensor:
    """Return the distal pad pose; its +X axis is the solver's -raw-X press axis."""
    pad_transform = torch.eye(4, dtype=raw_pose.dtype, device=raw_pose.device)
    pad_transform[:3, :3] = raw_pose.new_tensor(
        [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
    )  # Ry(pi) @ Rx(pi/2), preserving the existing tip orientation.
    pad_transform[:3, 3] = raw_pose.new_tensor(LEAP_PAD_OFFSETS_RAW[link_name])
    return raw_pose @ pad_transform


@register_agent()
class FloatingLeapHandRight(BaseAgent):
    uid = "floating_leap_hand_right"
    load_multiple_collisions = True
    disable_self_collisions = True

    urdf_path = os.path.join(PROJECT_ROOT_DIR, "assets", "models", "leap_rh", "floating_leap.urdf")
    # Friction is set per environment by the task (hand/object pair friction,
    # configs/primitive.py DOMAIN_RANDOMIZATION). No per-link materials here:
    # a URDF-config material on the fingertip links ("tip") left the first
    # GPU sub-scene's table/object contact without friction, so its object
    # slid off on its own.
    urdf_config = dict(
        material=dict(static_friction=0.7, dynamic_friction=0.7, restitution=0.0),
    )
    keyframes = dict(
        palm_up=Keyframe(
            qpos=np.zeros(22),  # 6 for floating + 16 for leap hand
            pose=sapien.Pose([0, 0, 0.5], q=[0.707, 0, -0.707, 0]),
        ),
        palm_down=Keyframe(
            qpos=np.zeros(22),
            pose=sapien.Pose([0, 0, 0.5], q=[0.707, 0, 0.707, 0]),
        ),
        palm_forward=Keyframe(
            qpos=np.zeros(22),
            pose=sapien.Pose([0, 0, 0.5], q=[1, 0, 0, 0]),
        ),
    )
    
    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]
    
    def __init__(self, *args, **kwargs):
        """Initialize the floating Leap hand with joint names, controller parameters, and link names."""
        # Controller joint order = the grasp-file "names" order (per finger:
        # axl, mcp, pip, dip; thumb: cmc, axl, mcp, ipl). The env computes
        # finger deltas and observations in this order, so the controller must
        # consume the action segment in the same order. (A previous version
        # listed mcp before axl - the URDF document order - which routed each
        # commanded axl/mcp delta to the other joint on the index/middle/ring
        # fingers; the thumb's sub-order coincides in both conventions.)
        self.finger_joint_names = [
            "leap_rh_if_axl",
            "leap_rh_if_mcp",
            "leap_rh_if_pip",
            "leap_rh_if_dip",
            "leap_rh_mf_axl",
            "leap_rh_mf_mcp",
            "leap_rh_mf_pip",
            "leap_rh_mf_dip",
            "leap_rh_rf_axl",
            "leap_rh_rf_mcp",
            "leap_rh_rf_pip",
            "leap_rh_rf_dip",
            "leap_rh_th_cmc",
            "leap_rh_th_axl",
            "leap_rh_th_mcp",
            "leap_rh_th_ipl",
        ]
        
        # Controller parameters - match Allegro
        self.joint_stiffness = 1e3
        self.joint_damping = 1e1
        self.joint_force_limit = 0.93
        
        # Order: index, middle, ring, thumb
        self.tip_link_names = [
            "leap_rh_if_ds",
            "leap_rh_mf_ds",
            "leap_rh_rf_ds",
            "leap_rh_th_ds",
        ]

        # Finger links in same order as Allegro (thumb, index, middle, ring)
        self.finger_link_names = [
            # Index links  
            "leap_rh_if_bs", "leap_rh_if_px", "leap_rh_if_md", "leap_rh_if_ds",
            # Middle links
            "leap_rh_mf_bs", "leap_rh_mf_px", "leap_rh_mf_md", "leap_rh_mf_ds",
            # Ring links
            "leap_rh_rf_bs", "leap_rh_rf_px", "leap_rh_rf_md", "leap_rh_rf_ds",
            # Thumb links
            "leap_rh_th_mp", "leap_rh_th_bs", "leap_rh_th_px", "leap_rh_th_ds",
        ]
        
        self.palm_link_names = [
            "leap_rh_palm_FROGGERSAMPLE",
        ]

        self.palm_link_name = "base_link"
        super().__init__(*args, **kwargs)

    def _after_init(self):
        """Cache frequently accessed links after robot initialization."""
        self.tip_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.palm_link: sapien.Entity = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )
        self.finger_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.finger_link_names
        )
        self.hand_links: List[sapien.Entity] = self.finger_links + sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.palm_link_names
        )

    @property
    def _controller_configs(self):
        """Return all available controller configurations for floating base and finger joints."""
        # Root (floating base) controllers - match Allegro
        base_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=None,
            upper=None,
            stiffness=3e3,
            damping=1e2,
            force_limit=700,
            normalize_action=False,
        )
        
        base_pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=3e3,
            damping=1e2,
            force_limit=700,
            use_delta=True,
        )

        # Hand joint controllers
        finger_pd_joint_pos = PDJointPosControllerConfig(
            self.finger_joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        
        finger_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.finger_joint_names,
            -0.1,
            0.1,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
        )
        
        finger_pd_joint_target_delta_pos = deepcopy(finger_pd_joint_delta_pos)
        finger_pd_joint_target_delta_pos.use_target = True

        # Create the controller configurations dictionary
        return dict(
            pd_joint_delta_pos=dict(
                base=base_pd_joint_delta_pos, 
                finger=finger_pd_joint_delta_pos
            ),
            pd_joint_pos=dict(
                base=base_pd_joint_pos, 
                finger=finger_pd_joint_pos
            ),
            pd_joint_target_delta_pos=dict(
                base=base_pd_joint_delta_pos,
                finger=finger_pd_joint_target_delta_pos
            ),
        )

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        tip_poses = self.tip_poses
        tip_poses_vec = [vectorize_pose(pose, device=self.device) for pose in tip_poses]
        tip_poses_vec = torch.stack(tip_poses_vec, dim=-2)

        palm_pose_vec = vectorize_pose(self.palm_pose, device=self.device)

        obs.update(
            {
                "palm_pose": palm_pose_vec,
                "tip_poses": tip_poses_vec.reshape(-1, len(self.tip_links) * 7),
            }
        )

        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = []
        for link_name, link in zip(self.tip_link_names, self.tip_links):
            raw_pose = link.pose.to_transformation_matrix()
            transformed_pose = leap_pad_pose_matrix(raw_pose, link_name)
            p, q = mat_to_pos_quat(transformed_pose)
            tip_poses.append(Pose.create_from_pq(p, q))
        return tip_poses

    @property
    def palm_pose(self):
        """
        Get the palm pose for leap hand
        """
        return self.palm_link.pose
    
    @property
    def finger_link_poses(self):
        """
        Get the poses of the finger links in the order matching Allegro's convention.
        This is used by environments for contact detection.
        """
        # Return poses in the same format as Allegro: list of poses
        return [link.pose for link in self.finger_links]
