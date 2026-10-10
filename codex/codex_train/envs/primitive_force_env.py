import math
import torch
from mani_skill.utils.registration import register_env
from codex.codex_train.envs.primitive_env import PrimitiveGraspEnv
# Imported for its @register_agent side effect (registers the LEAP hand uid).
from codex.codex_train.agents.leap_rh.floating_leap_right import (
    FloatingLeapHandRight, LEAP_PAD_OFFSETS_RAW, leap_pad_pose_matrix,
)
from codex.codex_train.envs.stability_frame_math import palm_actuation_vector


def press_hold_step_count(seconds, control_freq, num_actuate_steps):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("press_hold_seconds must be finite and non-negative")
    if seconds > num_actuate_steps / control_freq:
        raise ValueError("Press hold window exceeds num_actuate_steps")
    return math.ceil(seconds * control_freq)


class PressHoldWindow:
    """GPU-resident summary of the final actuation ticks, without per-env traces."""

    def __init__(self, num_envs, device, steps, control_freq):
        self.steps = steps
        self.control_freq = control_freq
        self.count = 0
        self.valid_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.final_run = torch.zeros_like(self.valid_count)
        self.force_sum = torch.zeros(num_envs, device=device)
        self.min_force = torch.full((num_envs,), float("inf"), device=device)
        self.last_force = torch.zeros(num_envs, device=device)

    def update(self, distance, selected_force, palm_drift):
        if self.count >= self.steps:
            raise RuntimeError("Press hold window has more samples than configured steps")
        near = distance < 0.03
        stable = palm_drift < 0.04
        valid = near & stable & (selected_force > 0.6)
        self.valid_count += valid.int()
        self.final_run = torch.where(valid, self.final_run + 1, 0)
        self.force_sum += torch.clamp(selected_force, 0.0, 1.0) * (near & stable)
        self.min_force = torch.minimum(self.min_force, selected_force)
        self.last_force = selected_force
        self.count += 1

    def metrics(self):
        if self.count != self.steps:
            raise RuntimeError(f"Press hold window collected {self.count}/{self.steps} steps")
        return {
            "held_press_success": self.final_run == self.steps,
            "press_contact_fraction": self.valid_count.float() / self.steps,
            "press_duration_s": self.final_run.float() / self.control_freq,
            "press_selected_force": self.last_force,
            "press_min_selected_force": self.min_force,
            "press_force_reward": self.force_sum / self.steps,
        }

@register_env("PrimitiveGraspForceEnv-v1")
class PrimitiveGraspForceEnv(PrimitiveGraspEnv):
    """
    A subclass of GraspLiftEnv that adds a utility method for computing the net projected contact force
    on the object that lies within a specified spatial threshold of the actuation point.
    """
    SUPPORTED_ROBOTS = ["floating_leap_hand_right", "floating_allegro_hand_right"]
    GOAL_POS_TOL = 0.05       # m: functional object within 5 cm of its goal pose ...
    GOAL_ROT_TOL_DEG = 30.0   # ... and within 30 degrees
    
    def __init__(self, *args, contact_threshold: float = 0.025,
                 press_hold_seconds: float = 0.0,
                 stability_frame: str = "world", **kwargs):
        """
        Args:
            contact_threshold: Distance (in meters) below which a contact's position is considered to 
                               be at the actuation point (default: 0.02 m, i.e. 2 cm).
        """
        if stability_frame not in ("world", "hand"):
            raise ValueError("stability_frame must be 'world' or 'hand'")
        self.stability_frame = stability_frame
        super().__init__(*args, **kwargs)

        self.contact_threshold = contact_threshold
        self.gpu_enabled = getattr(self.unwrapped.scene, "gpu_sim_enabled", False)
        self.press_hold_seconds = press_hold_seconds
        self.press_hold_steps = press_hold_step_count(
            press_hold_seconds, self.control_freq, self.num_actuate_steps)
        self._press_hold = None

    def compute_palm_dist(self):
        """Palm-to-actuation vector in legacy world or rotation-invariant hand coordinates."""
        if self.stability_frame == "world":
            return super().compute_palm_dist()
        palm_pose = self.get_hand_pose()
        palm_rotation = palm_pose.to_transformation_matrix()[:, :3, :3]
        return palm_actuation_vector(palm_pose.p, self.get_actuation_pos(), palm_rotation, "hand")

    def _begin_press_hold(self, act_idx, pre_lifted_palm_dist):
        self._press_act_idx = act_idx.long().reshape(-1)
        self._press_pre_lifted_palm_dist = pre_lifted_palm_dist
        self._press_actuation_tick = 0
        self._press_hold = PressHoldWindow(
            self.num_envs, self.device, self.press_hold_steps, self.control_freq)

    def _sample_press_step(self):
        self._press_actuation_tick += 1
        if self._press_actuation_tick <= self.num_actuate_steps - self.press_hold_steps:
            return
        rows = torch.arange(self.num_envs, device=self.device)
        act_pos, act_dir = self.get_actuation_pos(), self.get_actuation_dir()
        pad_positions = torch.stack([pose.p for pose in self.get_tip_poses()])
        distance = torch.linalg.vector_norm(pad_positions[self._press_act_idx, rows] - act_pos, dim=-1)
        # Query only the four distal links; proximal/palm force cannot earn hold credit.
        tip_forces = torch.stack([
            self.scene.get_pairwise_contact_forces(link, self.obj)
            for link in self.agent.tip_links
        ])
        projected_force = -(tip_forces[self._press_act_idx, rows] * act_dir).sum(dim=-1)
        tip_near = distance < self.contact_threshold
        selected_force = torch.where(tip_near, projected_force, 0.0)
        palm_drift = torch.linalg.vector_norm(
            self.compute_palm_dist() - self._press_pre_lifted_palm_dist, dim=-1)
        self._press_hold.update(distance, selected_force, palm_drift)

    def get_hand_contacts(self):
        """
        Returns:
            all_forces: A tensor of shape (num_links, n_env, 3) containing the contact forces on each link.
            all_link_poses: A tensor of shape (num_links, n_env, 4, 4) containing the poses of each link.
        """
        all_forces = []
        all_link_poses = []
        for hand_link in self.agent.hand_links:
            forces = self.scene.get_pairwise_contact_forces(hand_link, self.obj)
            link_pose = hand_link.pose.to_transformation_matrix()
            if isinstance(self.agent, FloatingLeapHandRight) and hand_link.name in LEAP_PAD_OFFSETS_RAW:
                # Only distal links have pad frames; keep proximal and palm origins raw.
                link_pose = leap_pad_pose_matrix(link_pose, hand_link.name)
            all_forces.append(forces)
            all_link_poses.append(link_pose)
        all_link_poses = torch.stack(all_link_poses, dim=0)
        all_forces = torch.stack(all_forces, dim=0)
        return all_forces, all_link_poses
    
    def compute_reward(self, obs, info: dict):
        """Compute the multi-component reward for force-based grasping and lifting."""
        # 1. Calculate stability component: the object must hold steady
        # relative to the PALM through the lift. The actuation finger's
        # displacement is deliberately excluded — descending from hover to
        # the actuation point is task progress, and counting it as slip
        # makes any late press an automatic stability failure (the policy
        # then learns to hover just outside contact).
        palm_dist_change = torch.norm(info["final_palm_dist"] - info["pre_lifted_palm_dist"], dim=-1)
        dist_change = palm_dist_change
        stable_reward = torch.exp(-dist_change * 50.0)
        stable_success = dist_change < 0.04
        
        # 2. Calculate lift success component
        height_diff = info["final_obj_pose"].p[:, 2] - info["init_obj_pose"].p[:, 2]
        goal_metrics = {}
        if "goal_obj_pose" in info:
            # Carried to a goal pose: progress is closeness to the goal, not height.
            final = info["final_obj_pose"].to_transformation_matrix()
            goal = info["goal_obj_pose"]
            pos_err = torch.norm(final[:, :3, 3] - goal[:, :3, 3], dim=-1)
            cos = ((final[:, :3, :3].transpose(1, 2) @ goal[:, :3, :3]).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
            rot_err = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))
            goal_success = (pos_err < self.GOAL_POS_TOL) & (rot_err < self.GOAL_ROT_TOL_DEG)
            goal_metrics = {"goal_success": goal_success, "goal_pos_err": pos_err, "goal_rot_err": rot_err}
            height_reward = (0.5 * torch.exp(-pos_err / self.GOAL_POS_TOL)
                             + 0.5 * torch.exp(-rot_err / self.GOAL_ROT_TOL_DEG))
            lift_success = goal_success
        else:
            height_reward = torch.clamp(height_diff / 0.1, min=0.0, max=1.0)
            lift_success = height_diff > 0.1

        # 3. Compute the actuation contact reward. The sharp exp(-100 d) term
        # is numerically zero beyond ~4 cm; a wider-horizon term keeps the
        # gradient toward the actuation point alive at decimeter range
        # (near-field total kept at 0.5).
        act_dist = torch.norm(info["final_act_dist"], dim=-1)
        act_dir_dist = info["final_act_dir_diff"] * 0.01
        alignment_reward = (torch.exp(-act_dist * 100.0) * 0.35
                            + torch.exp(-act_dist * 20.0) * 0.15)
        alignment_reward = torch.where(
            act_dist < 0.04,
            alignment_reward + torch.exp(-act_dir_dist * 100.0) * 0.5,
            alignment_reward)
        # Success requires proximity only; whether the press actuates the
        # mechanism is verified by the force PROJECTED along the annotated
        # press direction (below). The fingertip-pad orientation term stays
        # in the shaping as a style guide but does not veto success — an
        # over-the-top press with the pad angled sideways still depresses
        # the button.
        alignment_success = act_dist < 0.03

        # 4. Actuation collision reward — use the pre-press snapshot computed in
        # the primitive before the actuate phase; checking post-press confounds
        # wrap-finger positions (legitimately in the trigger zone at hold time)
        # with press-path obstruction.
        est_collision = info["pre_press_collision"]
        collision_free_reward = (~est_collision).float()

        # 5. Upright: the object must not be knocked over at the start of the
        # lift. The gate uses the same threshold (0.5 = cos 60°) for all
        # objects — it is a tolerance on the whole task formulation, not a
        # per-object property. upright_reward contributes both as a penalty
        # gradient (×0.3 unconditional, so doing nothing to avoid tipping is
        # never the optimal strategy) and as a lift-gated reward (×0.7, so
        # the main payoff requires also lifting the object upright). Together
        # they scale to the same weight as a single gated term while supplying
        # gradient from the very first rollout.
        pre_lifted_obj_z = info["pre_lifted_obj_pose"].to_transformation_matrix()[:, 2, 2]
        upright_reward = torch.exp(-(1 - pre_lifted_obj_z) * 5.0)
        upright_success = pre_lifted_obj_z > 0.5

        # 6. Compute the force reward
        all_forces, all_link_poses = self.get_hand_contacts()
        act_pos, act_dir = self.get_actuation_pos(), self.get_actuation_dir()
        link_to_act_dist = torch.norm(
            all_link_poses[:, :, :3, 3] - act_pos, dim=-1)  # [num_links, n_env]
        mask = (link_to_act_dist < self.contact_threshold)  # [num_links, n_env]
        combined_forces = torch.sum(all_forces * mask[:, :, None], dim=0) # [n_env, 3]
        act_force = torch.sum(-combined_forces * act_dir, dim=-1)
        act_force_reward = torch.clamp(act_force, min=0.0, max=1.0)
        act_force_reward = torch.where(
            (act_dist < 0.04) & (act_dir_dist < 0.1),
            act_force_reward,
            0)
        legacy_act_success = alignment_success & (act_force_reward > 0.6)
        act_success = legacy_act_success
        force_reward_for_training = act_force_reward
        press_metrics = None
        if self.press_hold_steps:
            press_metrics = self._press_hold.metrics()
            # The stable objective credits only the selected distal tip over
            # the full final window; other-link endpoint force remains diagnostic.
            force_reward_for_training = press_metrics["press_force_reward"]
            act_success = (press_metrics["held_press_success"]
                           & alignment_success & (act_dir_dist < 0.1))

        # 7. A lift only counts as successful with the actuation finger at
        # its point — holding the object high with the finger elsewhere is
        # not the task, and an unconditional lift bonus is exactly what made
        # recruiting the actuation finger into the wrap profitable. The
        # dense height/alignment shaping stays unconditional so learning can
        # still bootstrap through sloppy first lifts.
        lift_success = lift_success & (act_dist < 0.04)

        # Combine rewards.
        # With a goal pose, lift_success above is goal_success: the object must be
        # held at the goal (instead of lifted 10 cm) while the press is held.
        success = lift_success & stable_success & act_success & upright_success & ~est_collision
        reward = stable_reward * lift_success + stable_success * lift_success + \
                 height_reward + lift_success + \
                 upright_reward * lift_success + \
                 2 * alignment_reward + alignment_success + \
                 force_reward_for_training + collision_free_reward * alignment_success + \
                 success.float() * 5
        reward /= 15.0

        # Check if the object is completely away from the hand
        invalid_mask = dist_change > 0.15
        reward[invalid_mask] = 0

        info.update({
            "success": success,
            "reward": reward,
            "lift_reward": height_reward,
            "lift_success": lift_success,
            "stable_reward": stable_reward,
            "stable_success": stable_success,
            'upright_reward': upright_reward,
            "act_reward": alignment_reward + force_reward_for_training,
            "act_success": act_success,
            "act_dist": act_dist,
            "act_force": act_force,
            "act_dir_dist": act_dir_dist,
            "upright_success": upright_success,
            "est_collision": est_collision,
            **goal_metrics,
        })
        if press_metrics is not None:
            info.update(press_metrics)
            info["legacy_act_success"] = legacy_act_success

        return reward, info
