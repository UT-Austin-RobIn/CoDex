from typing import Literal, Union
import torch
import numpy as np
import gymnasium as gym
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.registration import register_env
from codex.codex_train.envs.pick_mesh import PickMeshEnv
from mani_skill.utils.geometry import rotation_conversions as p3d
from codex.codex_train.agents.load_grasp_qpos import (
    mat_to_pos_euler, pos_euler_to_mat, reorder_finger_config
)


def interpolate_pose(A, B, t):
    """Poses [B,4,4] a fraction t of the way from A to B (linear position, slerp rotation)."""
    qa, qb = p3d.matrix_to_quaternion(A[:, :3, :3]), p3d.matrix_to_quaternion(B[:, :3, :3])
    dot = (qa * qb).sum(-1, keepdim=True)
    qb = torch.where(dot < 0, -qb, qb)
    theta = torch.acos(dot.abs().clamp(max=1.0))
    sin = torch.sin(theta)
    wa = torch.where(sin > 1e-6, torch.sin((1 - t) * theta) / sin.clamp(min=1e-6), torch.full_like(sin, 1 - t))
    wb = torch.where(sin > 1e-6, torch.sin(t * theta) / sin.clamp(min=1e-6), torch.full_like(sin, t))
    out = A.clone()
    out[:, :3, :3] = p3d.quaternion_to_matrix(torch.nn.functional.normalize(wa * qa + wb * qb, dim=-1))
    out[:, :3, 3] = (1 - t) * A[:, :3, 3] + t * B[:, :3, 3]
    return out


@register_env("PrimitiveGraspEnv-v1")
class PrimitiveGraspEnv(PickMeshEnv):
    """
    Single-step environment that executes a complete grasp-and-lift motion primitive.
    
    Directly extends PickMeshEnv instead of using gym.Wrapper.
    """
    
    def __init__(
        self, 
        *args,
        pregrasp_offset=0.1,
        lift_height=0.3,
        num_approach_steps=100,
        num_grasp_steps=150,
        num_lift_steps=100,
        num_actuate_steps=60,
        approach_step_size=0.2,
        grasp_step_size=0.2,
        lift_step_size=0.2,
        pose_est_noise=0.01,
        movement_noise=0.05,
        finger_noise=0.03,
        waypoint_mode=False,
        delta_pos_scale=0.04,
        delta_rot_scale=0.1,
        delta_finger_scale=0.2,
        action_mode: Literal["joints_only", "grasp_only", "all"] = "joints_only",
        goal_pose=None,
        **kwargs
    ):
        # Initialize the base class (PickMeshEnv)
        super().__init__(*args, **kwargs)
        
        # Store parameters
        self.pregrasp_offset = pregrasp_offset
        self.lift_height = lift_height
        self.num_approach_steps = num_approach_steps
        self.num_grasp_steps = num_grasp_steps
        self.num_lift_steps = num_lift_steps
        self.num_actuate_steps = num_actuate_steps
        self.approach_step_size = approach_step_size
        self.grasp_step_size = grasp_step_size
        self.lift_step_size = lift_step_size
        self.pose_est_noise = pose_est_noise
        self.movement_noise = movement_noise
        self.finger_noise = finger_noise
        self.waypoint_mode = waypoint_mode

        self.delta_pos_scale = delta_pos_scale
        self.delta_rot_scale = delta_rot_scale
        self.delta_finger_scale = delta_finger_scale
        # Evaluation only: carry the functional object to a goal pose instead
        # of lifting it straight up. goal_pose (4x4) is the object's goal
        # relative to its pose at the start of the episode (see goal_pose_from_scene in eval.py).
        self.goal_pose = None if goal_pose is None else torch.as_tensor(
            np.asarray(goal_pose, dtype=np.float32), device=self.device)

        # Override the action and observation spaces for the primitive interface
        finger_action_dim = 16  # Number of finger joints
        
        # Store original spaces first
        self._original_action_space = self.action_space
        self._original_observation_space = self.observation_space
        
        # Define new action space (two sets of finger joint deltas)
        self.action_mode = action_mode
        if action_mode == "joints_only":
            action_dim = 2 * finger_action_dim    # Two delta configs
        elif action_mode == "grasp_only":
            action_dim = 6 + finger_action_dim      # Hand pose + finger joints
        elif action_mode == "all":
            action_dim = 6 + 2 * finger_action_dim
        else:
            raise ValueError(f"Invalid action_mode: {action_mode}. Choose 'joints_only' or 'all'.")
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_envs, action_dim,),
            dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Define new observation space for the flattened observations
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, 23,),  # 16 + 6 + 1
            dtype=np.float32
        )
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),  # 16 + 6 + 1
            dtype=np.float32
        )
        
    # ==================== Environment Interface Functions ====================
        
    def _prepare_observation(self):
        """Prepare the observation as a flat tensor"""
        finger_joints, hand_rel_pose_mat, act_idx = self.loaded_grasp
        hand_rel_pose_tensor = mat_to_pos_euler(hand_rel_pose_mat)

        flat_obs = torch.cat([
            finger_joints.to(self.device),
            hand_rel_pose_tensor.to(self.device),
            act_idx[..., :1].to(self.device),
        ], dim=-1)
        
        return flat_obs
    
    def _maybe_render(self):
        """Render the environment if needed"""
        self.update_sites()
        if self.render_mode == "human":
            self.render()
    
    def compute_target_poses_mat(self, hand_rel_pose_mat):
        """
        Compute target poses as transformation matrices.
        
        Args:
            hand_rel_pose_mat: Hand pose matrix relative to object
            
        Returns:
            target_pose_mat: Target hand pose matrix
            precontact_pose_mat: Pre-contact pose matrix
            preprecontact_pose_mat: Initial approach pose matrix
        """
        # Calculate hand pose matrix from grasp config and object pose
        target_pose_mat = self.compute_grasp_pose(hand_rel_pose_mat)
        
        # Calculate pre-contact position (offset along hand's x-axis by pregrasp_offset)
        precontact_pose_mat = target_pose_mat.clone()
        precontact_pose_mat[:, :3, 3] -= target_pose_mat[:, :3, 0] * self.pregrasp_offset
        
        # Calculate initial approach position (above pre-contact pose)
        preprecontact_pose_mat = precontact_pose_mat.clone()
        z_offset = torch.zeros_like(preprecontact_pose_mat[:, :3, 3])
        z_offset[:, 2] = 0.25
        preprecontact_pose_mat[:, :3, 3] += z_offset
        
        return target_pose_mat, precontact_pose_mat, preprecontact_pose_mat
    
    # The functional object counts as held while it stays within this
    # distance of where it sat in the hand after the lift.
    HELD_TOL = 0.05

    def object_in_hand(self):
        """The functional object's pose in the palm frame (4x4 per env)."""
        return (torch.linalg.inv(self.get_hand_pose().to_transformation_matrix())
                @ self.get_object_pose().to_transformation_matrix())

    def goal_object_pose(self, init_obj_pose):
        """World goal pose of the functional object for each env: goal_pose is
        relative to the object's pose at the start (see goal_pose_from_scene in eval.py)."""
        return init_obj_pose.to_transformation_matrix() @ self.goal_pose.to(self.device)

    def move_hand(self, 
                  target_pose_mat=None, 
                  target_finger_joints=None,
                  finger_joint_order: Literal["names", "tree"] = "names",
                  num_steps=40, 
                  hand_step_size=0.2,
                  finger_step_size=0.2,
                  movement_noise=0,
                  finger_noise=0,
                  teleport=False,
                  waypoint_mode=False,
                  early_stop=True,
                  step_callback=None):
        """
        Move the hand to a target pose and/or close the hand to a target finger configuration.
        
        Args:
            target_pose_mat: Target transformation matrix for the hand pose (optional)
            target_finger_joints: Target joint positions for fingers (optional)
            finger_joint_order: Order of finger joints ("names" or "tree")
            num_steps: Number of steps for the motion
            hand_step_size: Maximum step size per step for hand position/orientation
            finger_step_size: Maximum step size per step for finger joints
            teleport: If True, teleport to pose instead of stepping
            early_stop: If True, stop early when close to target
        """
        # Handle teleport case
        if teleport:
            if target_pose_mat is not None:
                self.set_hand_qpos(pose=target_pose_mat)
            if target_finger_joints is not None:
                self.set_hand_qpos(finger_joints=target_finger_joints, finger_joint_order=finger_joint_order)
            if self.gpu_sim_enabled:
                self.scene.px.gpu_apply_articulation_qpos()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()
            self._maybe_render()
            return
        
        current_qpos = self.get_hand_qpos()

        # Convert target pose matrix to position and Euler angles if provided
        target_pos_euler = current_qpos[:, :6]
        if target_pose_mat is not None:
            target_pos_euler = mat_to_pos_euler(target_pose_mat)
        
        # Handle finger joints target if provided
        if target_finger_joints is not None:
            target_finger_joints = target_finger_joints.clone()
            # Ensure target joints are in "names" order if they're in "tree" order
            if finger_joint_order == "tree":
                target_finger_joints = reorder_finger_config(target_finger_joints, mode="tree2names", is_leap=self.is_leap)
        else:
            target_finger_joints = current_qpos[:, 6:]  # read finger joints are in "tree" order
            target_finger_joints = reorder_finger_config(target_finger_joints, mode="tree2names", is_leap=self.is_leap)  # convert to "names" order

        if waypoint_mode:
            # Get current joint positions in "names" order
            current_joints = self.get_finger_joints(order="names")

            # Vectorized interpolation between current and target finger joints
            # Create interpolation weights for all steps at once
            interp_weights = torch.linspace(0, 1, num_steps + 1, device=self.device)[1:]  # [num_steps]
            interp_weights = interp_weights.unsqueeze(0).unsqueeze(-1)  # [1, num_steps, 1]
            
            # Expand current and target joints for broadcasting
            current_expanded = current_joints.unsqueeze(1)  # [n_envs, 1, joint_dim]
            target_expanded = target_finger_joints.unsqueeze(1)  # [n_envs, 1, joint_dim]
            
            # Compute all waypoints at once: [n_envs, num_steps, joint_dim]
            waypoints = current_expanded + (target_expanded - current_expanded) * interp_weights
            
        # Perform the motion over multiple steps
        for i in range(num_steps):
            # Create action
            n_envs = self.num_envs
            action_dim = self._original_action_space.shape[-1]
            action = torch.zeros((n_envs, action_dim), device=self.device)
            
            pose_close_enough = True
            fingers_close_enough = True
            
            # ---------------
            # Hand Movement
            # ---------------
            # Get current hand pose
            current_pose = self.get_hand_pose()
            current_mat = current_pose.to_transformation_matrix()
            current_pos_euler = mat_to_pos_euler(current_mat)
            
            # Calculate position and orientation delta
            delta_pos_euler = target_pos_euler - current_pos_euler
            delta_pos, delta_euler = delta_pos_euler[:, :3], delta_pos_euler[:, 3:]
            
            # Wrap around the angles to [-pi, pi]
            while torch.any(abs(delta_euler) > np.pi):
                delta_euler = torch.where(delta_euler > np.pi, delta_euler - 2 * np.pi, delta_euler)
                delta_euler = torch.where(delta_euler < -np.pi, delta_euler + 2 * np.pi, delta_euler)
            
            # Limit step size using scaling method
            delta_pos_norm = torch.norm(delta_pos, dim=-1)
            scale = torch.ones_like(delta_pos_norm)
            scale = torch.where(delta_pos_norm > hand_step_size, 
                            hand_step_size / (delta_pos_norm + 1e-9), 
                            scale)
            
            # Set position and orientation deltas in the action
            action[:, :3] = delta_pos * scale.unsqueeze(-1)
            action[:, 3:6] = delta_euler
            if movement_noise > 0:
                action[:, 3:6] += torch.randn_like(action[:, 3:6]) * movement_noise * 10
                action[:, :3] += torch.randn_like(action[:, :3]) * movement_noise
            
            # Check if we're close enough to target pose for early stopping
            pose_close_enough = torch.all(abs(delta_pos) < 0.005) and torch.all(abs(delta_euler) < 0.01)
            
            # ---------------
            # Finger Movement
            # ---------------
            # Get current joint positions in "names" order
            current_joints = self.get_finger_joints(order="names")
            
            # Calculate joint deltas
                
            joint_delta = target_finger_joints - current_joints
            
            if not waypoint_mode:
                # Limit step size using scaling
                scale = torch.ones_like(joint_delta)
                scale = torch.where(abs(joint_delta) > finger_step_size, 
                                finger_step_size / (abs(joint_delta) + 1e-9), 
                                scale)
                
                # Apply scaling to limit movement speed
                joint_delta = joint_delta * scale
                
                # Set joint deltas for the finger joints (indices 6 and above)
                action[:, 6:] = joint_delta + torch.randn_like(joint_delta) * finger_noise
            else:
                waypoint_joints = waypoints[:, i, :]
                self.set_hand_qpos(finger_joints=waypoint_joints, finger_joint_order="names")

            # Check if we're close enough to target finger positions for early stopping
            fingers_close_enough = torch.all(abs(joint_delta) < 0.005)
            
            # ---------------
            # Apply Action
            # ---------------
            # Early stop if the deltas are small and early_stop is enabled
            if early_stop and pose_close_enough and fingers_close_enough:
                break
            
            # Take step in environment - directly call _step_action
            self._step_action(action)
            if step_callback is not None:
                step_callback()
            self._maybe_render()

    def reset(self, **kwargs):
        """Reset the environment and return the flat observation"""
        # Reset the base environment using parent's reset method
        obs, info = super().reset(**kwargs)
        
        # Prepare observation as tensor
        tensor_obs = self._prepare_observation()
        
        return tensor_obs, info
    
    def step(self, action):
        """
        Execute the full grasp-lift motion primitive in a single call.
        
        Args:
            action: numpy array or tensor of shape (2*finger_joints,)
                   First half: delta for pregrasp position
                   Second half: delta for final grasp position
        """
        # Convert action to tensor if needed
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        
        # Extract the two delta components
        finger_joint_dim = 16
        if self.action_mode in ["joints_only", "all"]:
            grasp_delta = action[:, :finger_joint_dim] * self.delta_finger_scale
            # Centered at half-curl and bounded to [0, 1]: with the raw channel a
            # near-zero action approaches with fully extended fingers (which sweep
            # the object over during the final approach) and negative values
            # command a curl beyond full extension.
            pregrasp_perc = torch.clamp(
                0.5 + 0.5 * action[:, finger_joint_dim: 2 * finger_joint_dim],
                0.0, 1.0)
            hand_delta = action[:, 2 * finger_joint_dim:]
        elif self.action_mode == "grasp_only":
            grasp_delta = action[:, :finger_joint_dim] * self.delta_finger_scale
            pregrasp_perc = torch.ones_like(grasp_delta) * 0.2
            hand_delta = action[:, finger_joint_dim:]

        if hand_delta.shape[1] == 0:
            hand_delta = torch.zeros((hand_delta.shape[0], 6), device=self.device)
        hand_delta[:, :3] *= self.delta_pos_scale  # Scale the position delta
        hand_delta[:, 3:] *= self.delta_rot_scale  # Scale the orientation delta
        pose_est_noise = torch.randn_like(hand_delta) * self.pose_est_noise
        pose_est_noise[:, 3:] *= 5.    # Scale the orientation noise
        hand_delta_mat = pos_euler_to_mat(hand_delta + pose_est_noise)
        
        # Load the needed info
        init_obj_pose = self.get_object_pose()
        loaded_finger_joints, hand_rel_pose_mat, act_idx = self.loaded_grasp
        loaded_finger_joints = loaded_finger_joints.to(self.device).float()
        hand_rel_pose_mat = hand_rel_pose_mat.to(self.device).float()
        act_idx = act_idx.to(self.device)
        
        # Calculate pregrasp and final grasp finger joints
        # These are in "name" order as self.grasp_joints is in "name" order
        
        grasp_finger_joints = loaded_finger_joints + grasp_delta
        pregrasp_finger_joints = loaded_finger_joints * pregrasp_perc
        hand_rel_pose_mat = hand_rel_pose_mat @ hand_delta_mat

        # Squeeze the fingers
        finger_mask = {
            0: [2, 3], 1: [6, 7], 2: [10, 11], 3: [14, 15],
        }
        act_finger = int(act_idx.reshape(-1)[0])
        grasp_mask = []
        for finger_idx in range(4):
            if finger_idx == act_finger:
                continue
            grasp_mask += finger_mask[finger_idx]
        # Close the non-actuation fingers a bit further (+ = flexion for LEAP).
        grasp_fixed_delta = torch.zeros_like(grasp_finger_joints)
        grasp_fixed_delta[:, grasp_mask] = 0.2
        grasp_finger_joints += grasp_fixed_delta

        # Grasp-move-actuate sequencing: the actuation finger is held at its
        # pregrasp standoff through the close and the lift — it takes no part
        # in establishing or stabilizing the hold — and is driven to its
        # press target only in the actuate phase at the end. Pressing (or
        # not) must not change the object's pose in the hand; the support
        # wrap alone carries it.
        act_joint_idx = list(range(4 * act_finger, 4 * act_finger + 4))
        hold_finger_joints = grasp_finger_joints.clone()
        hold_finger_joints[:, act_joint_idx] = pregrasp_finger_joints[:, act_joint_idx]

        # Compute target poses for approach sequence (as matrices)
        target_pose_mat, precontact_pose_mat, preprecontact_pose_mat = self.compute_target_poses_mat(hand_rel_pose_mat)
        lifted_pose_mat = target_pose_mat.clone()
        lifted_pose_mat[:, 2, 3] += self.lift_height  # Add lift height to z-coordinate
        
        # Initialize with target position and pregrasp finger joints (teleport)
        self.move_hand(target_pose_mat=preprecontact_pose_mat,
                       target_finger_joints=pregrasp_finger_joints,
                       teleport=True)

        # Move to pre-contact position and then to grasp position. This leg must
        # be PD-controlled (no teleport): teleporting the hand into the precontact
        # pose materializes it next to the object and the contact solver ejects
        # whatever it overlaps.
        self.move_hand(
            target_pose_mat=precontact_pose_mat,
            target_finger_joints=pregrasp_finger_joints,
            num_steps=self.num_approach_steps // 2,
            hand_step_size=self.approach_step_size * 3,
            waypoint_mode=self.waypoint_mode,
        )
        self.move_hand(
            target_pose_mat=target_pose_mat,
            target_finger_joints=pregrasp_finger_joints,
            num_steps=self.num_approach_steps,
            hand_step_size=self.approach_step_size,
            finger_noise=self.finger_noise,
            waypoint_mode=self.waypoint_mode,
        )
        self.move_hand(
            target_pose_mat=target_pose_mat,
            target_finger_joints=hold_finger_joints,
            num_steps=self.num_grasp_steps,
            hand_step_size=self.grasp_step_size,
            finger_step_size=self.grasp_step_size,
            finger_noise=self.finger_noise,
            waypoint_mode=self.waypoint_mode,
        )
        if self.rotation_locked:
            # 1. record the locked actor’s current world pose
            old_pose = self.get_object_pose()
            p, q = old_pose.p, old_pose.q

            # 2. remove the locked actor from the scene and reposition it
            _p = p + torch.ones_like(p) * 0.5
            new_pose = Pose.create_from_pq(_p, q)  # create a new pose
            self.obj_locked.set_pose(new_pose)  # move the free actor to a new position
            # self._hidden_objects.append(self.obj)  # add the free actor to the hidden objects

            # 3. show & reposition the free actor
            self.obj_free.set_pose(old_pose)
            self.obj = self.obj_free

            # 4. disable future swaps
            self.rotation_locked = False

        # Record actual joints
        actual_finger_joints = self.get_finger_joints(order="names")
        pre_lifted_obj_pose = self.get_object_pose()
        
        # Move to lifted position (actuation finger still at standoff)
        self.move_hand(
            target_pose_mat=lifted_pose_mat,
            target_finger_joints=hold_finger_joints,
            num_steps=int(0.2 * self.num_lift_steps),
            hand_step_size=self.lift_step_size,
            finger_step_size=self.lift_step_size,
            waypoint_mode=self.waypoint_mode,
            early_stop=False,
        )
        pre_lifted_palm_dist = self.compute_palm_dist()
        pre_lifted_act_dist, _, _ = self.compute_actuation_dist(act_idx_mode=act_idx)
        goal_obj_pose = None
        if self.goal_pose is None:
            self.move_hand(
                target_pose_mat=lifted_pose_mat,
                target_finger_joints=hold_finger_joints,
                num_steps=int(0.8 * self.num_lift_steps),
                hand_step_size=self.lift_step_size,
                finger_step_size=self.lift_step_size,
                movement_noise=self.movement_noise,
                waypoint_mode=self.waypoint_mode,
                early_stop=False,
            )
        else:
            # Carry from the lifted pose to the goal through interpolated
            # waypoints (the hand controller applies a rotation target in one
            # step, so large turns are split up), then settle there.
            goal_obj_pose = self.goal_object_pose(init_obj_pose)
            in_hand = self.object_in_hand()          # where the lifted functional object sits in the hand
            goal_hand = goal_obj_pose @ torch.linalg.inv(init_obj_pose.to_transformation_matrix()) @ target_pose_mat
            n_way = 8
            for k in range(1, n_way + 1):
                self.move_hand(
                    target_pose_mat=interpolate_pose(lifted_pose_mat, goal_hand, k / n_way),
                    target_finger_joints=hold_finger_joints,
                    num_steps=max(1, int(0.8 * self.num_lift_steps) // n_way),
                    hand_step_size=self.lift_step_size,
                    finger_step_size=self.lift_step_size,
                    movement_noise=self.movement_noise,
                    waypoint_mode=self.waypoint_mode,
                    early_stop=False,
                )
            # Place the functional object, not the hand: the object can settle
            # in the hand while it closes, so correct the hand by the object's
            # measured pose error (as a system tracking the object would).
            # Only while the object is still in the hand: a dropped object's
            # error says nothing about where the hand should go.
            lifted_pose_mat = goal_hand
            for _ in range(3):
                self.move_hand(
                    target_pose_mat=lifted_pose_mat,
                    target_finger_joints=hold_finger_joints,
                    num_steps=int(0.2 * self.num_lift_steps),
                    hand_step_size=self.lift_step_size,
                    finger_step_size=self.lift_step_size,
                    waypoint_mode=self.waypoint_mode,
                    early_stop=False,
                )
                error = goal_obj_pose @ torch.linalg.inv(self.get_object_pose().to_transformation_matrix())
                held = (self.object_in_hand()[:, :3, 3] - in_hand[:, :3, 3]).norm(dim=-1) < self.HELD_TOL
                error[~held] = torch.eye(4, device=self.device)
                lifted_pose_mat = error @ lifted_pose_mat

        # Collision is checked HERE — at the settled hold, before the press —
        # so it asks "do any non-actuation fingers block the press path?",
        # not "are wrap fingers inside the cylinder after the press happened?".
        # Checking post-press is a false positive: wrap fingers that are
        # geometrically in the trigger/button zone when the hold is established
        # are correctly there; measuring them after the actuation finger has
        # also entered confounds hold-collision with press-arrival.
        pre_press_collision = self.compute_actuation_collision()

        # Actuate phase: with the hold settled, drive the actuation finger to
        # its press target (solver pose + the policy's delta).
        press_hold_steps = getattr(self, "press_hold_steps", 0)
        if press_hold_steps:
            self._begin_press_hold(act_idx, pre_lifted_palm_dist)
        self.move_hand(
            target_pose_mat=lifted_pose_mat,
            target_finger_joints=grasp_finger_joints,
            num_steps=self.num_actuate_steps,
            hand_step_size=self.lift_step_size,
            finger_step_size=self.grasp_step_size,
            waypoint_mode=self.waypoint_mode,
            early_stop=False,
            step_callback=self._sample_press_step if press_hold_steps else None,
        )
        final_palm_dist = self.compute_palm_dist()
        final_act_dist, final_act_dir_diff, _ = self.compute_actuation_dist(act_idx_mode=act_idx)
        
        # Get observation
        obs = self._prepare_observation()

        # Calculate reward
        final_obj_pose = self.get_object_pose()
        info = {
            'hand_rel_pose_mat': hand_rel_pose_mat,
            'pregrasp_finger_joints': pregrasp_finger_joints,
            'grasp_finger_joints': grasp_finger_joints,
            'actual_grasp_finger_joints': actual_finger_joints,
            'act_idx': act_idx,
            'init_obj_pose': init_obj_pose,
            'pre_lifted_obj_pose': pre_lifted_obj_pose,
            'pre_lifted_palm_dist': pre_lifted_palm_dist,
            'pre_lifted_act_dist': pre_lifted_act_dist,
            'final_palm_dist': final_palm_dist,
            'final_act_dist': final_act_dist,
            'final_act_dir_diff': final_act_dir_diff,
            'final_obj_pose': final_obj_pose,
            'pre_press_collision': pre_press_collision,
        }
        if goal_obj_pose is not None:
            info['goal_obj_pose'] = goal_obj_pose
        reward, info = self.compute_reward(obs=obs, info=info)

        # Set reward to zero if NaN
        nan_mask = torch.isnan(reward)
        if torch.any(nan_mask):
            print(f"Warning: NaN reward in {int(nan_mask.sum())} environments; set to 0")
            reward[nan_mask] = 0

        # Check termination
        terminated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)  # Always terminate after one step
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, obs, info: dict):
        # 1. Calculate stability component: change in hand-object distance
        palm_dist_change = torch.norm(info["final_palm_dist"] - info["pre_lifted_palm_dist"], dim=-1)
        act_dist_change = torch.norm(info["final_act_dist"] - info["pre_lifted_act_dist"], dim=-1)
        dist_change = palm_dist_change + act_dist_change
        stable_reward = torch.exp(-dist_change * 50.0)
        stable_success = dist_change < 0.02
        
        # 2. Calculate lift success component. Only upward motion is rewarded
        # (abs() would reward knocking the object over; see the force env).
        height_diff = info["final_obj_pose"].p[:, 2] - info["init_obj_pose"].p[:, 2]
        height_reward = torch.clamp(height_diff / 0.1, min=0.0, max=1.0)
        lift_success = height_diff > 0.1

        # 3. Compute the actuation contact reward
        act_dist = torch.norm(info["final_act_dist"], dim=-1)
        act_dir_dist = info["final_act_dir_diff"] * 0.01
        act_reward = torch.exp(-act_dist * 100.0) * 0.5
        act_reward = torch.where(
            act_dist < 0.04, 
            act_reward + torch.exp(-act_dir_dist * 100.0) * 0.5,
            act_reward)
        act_success = (act_dist < 0.02) & (act_dir_dist < 0.01)

        # 4. Actuation collision reward — use the pre-press snapshot computed
        # in step() before the actuate phase (same rationale as force env).
        est_collision = info["pre_press_collision"]
        collision_free_reward = (~est_collision).float()

        # 5. Upright gate (same semantics as force env).
        pre_lifted_obj_z = info["pre_lifted_obj_pose"].to_transformation_matrix()[:, 2, 2]
        upright_success = pre_lifted_obj_z > 0.5

        # Combine rewards
        reward = stable_reward + stable_success + \
                 height_reward + lift_success + \
                 act_reward + collision_free_reward
        reward /= 6.0
        # Check if the object is completely away from the hand
        invalid_mask = dist_change > 0.15
        reward[invalid_mask] = 0

        info.update({
            "success": lift_success & stable_success & act_success & upright_success,
            "reward": reward,
            "lift_reward": height_reward,
            "lift_success": lift_success,
            "stable_reward": stable_reward,
            "dist_change": dist_change,
            "stable_success": stable_success,
            "act_reward": act_reward,
            "act_success": act_success,
            "act_dist": act_dist,
            "act_dir_dist": act_dir_dist,
            "est_collision": est_collision,
        })

        return reward, info
    
    def compute_palm_dist(self):
        palm_pose = self.get_hand_pose()
        act_pos = self.get_actuation_pos()
        palm_dist = palm_pose.p - act_pos
        return palm_dist
    
    def compute_actuation_collision(self, dist_threshold=0.04, normal_distance_thereshold=0.03):
        tip_poses = self.get_finger_link_poses()
        act_pos = self.get_actuation_pos()  # [n_envs, 3]
        act_dir = self.get_actuation_dir()  # [n_envs, 3]
        ref_pos = act_pos - act_dir * 0.01  # [n_envs, 3]

        tip_to_ref = [tip_pose.p - ref_pos for tip_pose in tip_poses]
        tip_to_ref = torch.stack(tip_to_ref, dim=0)   # [n_links, n_envs, 3]

        # Compute the distance along/normal to the actuation direction
        tip_dists_along_act = torch.sum(tip_to_ref * act_dir.unsqueeze(0), dim=-1)  # [n_links, n_envs]
        tip_to_ref_normal = torch.cross(tip_to_ref, act_dir.unsqueeze(0), dim=-1)  # [n_links, n_envs, 3]
        tip_dists_normal = torch.norm(tip_to_ref_normal, dim=-1)  # [n_links, n_envs]

        # Compute the estimated collision
        est_collision = (tip_dists_along_act < dist_threshold) & (tip_dists_along_act > 0.015)
        est_collision &= (tip_dists_normal < normal_distance_thereshold)
        est_collision = est_collision.any(dim=0)  # [n_envs]

        return est_collision
    
    def compute_actuation_dist(self, act_idx_mode: Union[Literal["min", "loaded"], torch.tensor] = "min"):
        tip_poses = self.get_tip_poses()
        act_pos = self.get_actuation_pos()
        act_dir = self.get_actuation_dir()

        # LEAP tip poses are solver pad centers. Allegro retains its legacy
        # 1 cm surface-center reference until its frames are calibrated.
        target_pos = act_pos if self.is_leap else act_pos - act_dir * 0.01
        tip_dists = [tip_pose.p - target_pos for tip_pose in tip_poses]
        tip_dists = torch.stack(tip_dists, dim=0)  # [n_tips, n_envs, 3]
        tip_dist_norms = tip_dists.norm(dim=-1)

        if isinstance(act_idx_mode, torch.Tensor):
            tip_idx = act_idx_mode.view(-1)
        else:
            if act_idx_mode == "min":
                tip_idx = torch.argmin(tip_dist_norms, dim=0)
            else:
                raise NotImplementedError
            
        pos_dist = tip_dists[tip_idx, torch.arange(self.num_envs), :]

        tip_dirs = [tip_pose.to_transformation_matrix()[:, :3, 0] for tip_pose in tip_poses]
        tip_dirs = torch.stack(tip_dirs, dim=0)  # [n_tips, n_envs, 3]
        tip_dir = tip_dirs[tip_idx, torch.arange(self.num_envs), :]
        dir_diff = torch.sum(tip_dir * act_dir, dim=-1)
        # Roundoff in two unit directions can put their dot one epsilon
        # outside [-1, 1]; keep the physical directions unchanged.
        dir_diff = torch.acos(dir_diff.clamp(-1.0, 1.0))
        return pos_dist, dir_diff, tip_idx
