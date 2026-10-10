"""CPU checks for the opt-in final-window selected-tip press gate."""

from types import SimpleNamespace
import unittest

import torch

from codex.codex_train.envs.primitive_force_env import (
    PressHoldWindow, PrimitiveGraspForceEnv, press_hold_step_count,
)


class PressHoldTests(unittest.TestCase):
    def test_duration_validation_and_legacy_default(self):
        self.assertEqual(press_hold_step_count(0, 20, 60), 0)
        self.assertEqual(press_hold_step_count(1.0, 20, 60), 20)
        self.assertEqual(press_hold_step_count(.101, 20, 60), 3)
        for seconds in (-1, float("nan"), float("inf"), 3.1):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                press_hold_step_count(seconds, 20, 60)

    def test_full_final_window_required_and_reward_uses_selected_force(self):
        def run(forces, distances=None, drifts=None):
            window = PressHoldWindow(1, "cpu", len(forces), 20)
            distances = distances or [.02] * len(forces)
            drifts = drifts or [.01] * len(forces)
            for force, distance, drift in zip(forces, distances, drifts):
                window.update(torch.tensor([distance]), torch.tensor([force]), torch.tensor([drift]))
            return window.metrics()

        full = run([.8] * 20)
        self.assertTrue(full["held_press_success"].item())
        self.assertAlmostEqual(full["press_contact_fraction"].item(), 1.0)
        self.assertAlmostEqual(full["press_duration_s"].item(), 1.0)
        self.assertAlmostEqual(full["press_force_reward"].item(), .8, places=6)
        self.assertAlmostEqual(full["press_min_selected_force"].item(), .8, places=6)

        transient = run([0.] * 19 + [.8])
        self.assertFalse(transient["held_press_success"].item())
        self.assertAlmostEqual(transient["press_duration_s"].item(), .05)
        self.assertAlmostEqual(transient["press_contact_fraction"].item(), .05)

        dropped = run([.8] * 10 + [0.] + [.8] * 9)
        self.assertFalse(dropped["held_press_success"].item())
        self.assertAlmostEqual(dropped["press_duration_s"].item(), .45)
        unstable = run([.8] * 20, drifts=[.01] * 19 + [.05])
        self.assertFalse(unstable["held_press_success"].item())
        too_far = run([.8] * 20, distances=[.02] * 19 + [.031])
        self.assertFalse(too_far["held_press_success"].item())

        incomplete = PressHoldWindow(1, "cpu", 20, 20)
        incomplete.update(torch.tensor([.02]), torch.tensor([.8]), torch.tensor([.01]))
        with self.assertRaises(RuntimeError):
            incomplete.metrics()

    def test_other_link_force_cannot_satisfy_selected_tip(self):
        zero = torch.zeros(1, 3)
        tips = [SimpleNamespace(p=zero) for _ in range(4)]
        # Canonical tip order is independent of hand_links layout (including Allegro).
        tip_links = [object() for _ in range(4)]
        other_link = object()
        force_by_link = {link: zero for link in tip_links}
        force_by_link[other_link] = torch.tensor([[-100., 0., 0.]])
        env = SimpleNamespace(
            num_envs=1, device="cpu", num_actuate_steps=2, press_hold_steps=2,
            contact_threshold=.025, _press_act_idx=torch.tensor([2]),
            _press_actuation_tick=0, _press_hold=PressHoldWindow(1, "cpu", 2, 20),
            _press_pre_lifted_palm_dist=zero,
            agent=SimpleNamespace(tip_links=tip_links, hand_links=[other_link] + tip_links[::-1]),
            scene=SimpleNamespace(get_pairwise_contact_forces=lambda link, obj: force_by_link[link]),
            obj=object(),
            get_actuation_pos=lambda: zero, get_actuation_dir=lambda: torch.tensor([[1., 0., 0.]]),
            get_tip_poses=lambda: tips,
            compute_palm_dist=lambda: zero,
        )
        PrimitiveGraspForceEnv._sample_press_step(env)
        PrimitiveGraspForceEnv._sample_press_step(env)
        metrics = env._press_hold.metrics()
        self.assertFalse(metrics["held_press_success"].item())
        self.assertEqual(metrics["press_selected_force"].item(), 0.0)
        self.assertEqual(metrics["press_force_reward"].item(), 0.0)

        force_by_link[tip_links[2]] = torch.tensor([[-.8, 0., 0.]])
        env._press_hold = PressHoldWindow(1, "cpu", 2, 20)
        env._press_actuation_tick = 0
        PrimitiveGraspForceEnv._sample_press_step(env)
        PrimitiveGraspForceEnv._sample_press_step(env)
        self.assertTrue(env._press_hold.metrics()["held_press_success"].item())

    def test_legacy_endpoint_reward_and_stable_hold_reward_are_separate(self):
        zero = torch.zeros(1, 3)
        upright = torch.eye(4).unsqueeze(0)
        legacy_forces = torch.zeros(1, 1, 3)
        legacy_poses = upright.unsqueeze(0)
        info = {
            "final_palm_dist": zero, "pre_lifted_palm_dist": zero,
            "init_obj_pose": SimpleNamespace(p=zero),
            "final_obj_pose": SimpleNamespace(p=torch.tensor([[0., 0., .2]])),
            "pre_lifted_obj_pose": SimpleNamespace(to_transformation_matrix=lambda: upright),
            "final_act_dist": zero, "final_act_dir_diff": torch.zeros(1),
            "pre_press_collision": torch.tensor([False]),
        }
        env = SimpleNamespace(
            contact_threshold=.025, press_hold_steps=0,
            get_hand_contacts=lambda: (legacy_forces, legacy_poses),
            get_actuation_pos=lambda: zero,
            get_actuation_dir=lambda: torch.tensor([[1., 0., 0.]]),
        )
        legacy_reward, legacy_info = PrimitiveGraspForceEnv.compute_reward(env, None, info.copy())
        self.assertFalse(legacy_info["act_success"].item())
        self.assertNotIn("held_press_success", legacy_info)
        self.assertAlmostEqual(legacy_reward.item(), 9 / 15, places=6)

        hold = PressHoldWindow(1, "cpu", 20, 20)
        for _ in range(20):
            hold.update(torch.tensor([.02]), torch.tensor([.8]), torch.tensor([0.]))
        env.press_hold_steps = 20
        env._press_hold = hold
        hold_reward, hold_info = PrimitiveGraspForceEnv.compute_reward(env, None, info.copy())
        self.assertTrue(hold_info["act_success"].item())
        self.assertTrue(hold_info["success"].item())
        self.assertFalse(hold_info["legacy_act_success"].item())
        self.assertAlmostEqual(hold_info["act_force"].item(), 0.0)
        self.assertAlmostEqual(hold_reward.item(), (9 + .8 + 5) / 15, places=6)


if __name__ == "__main__":
    unittest.main()
