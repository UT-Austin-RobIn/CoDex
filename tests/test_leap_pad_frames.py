"""CPU checks for solver-aligned LEAP pad measurements."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from codex.codex_train.agents.leap_rh.floating_leap_right import (
    FloatingLeapHandRight, LEAP_PAD_OFFSETS_RAW, leap_pad_pose_matrix,
)
from codex.codex_train.envs.primitive_env import PrimitiveGraspEnv
from codex.codex_train.envs import primitive_force_env as force_module


def raw_pose():
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[0, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    pose[1, :3, :3] = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    pose[:, :3, 3] = torch.tensor([[.1, -.2, .3], [-.3, .4, .2]])
    return pose


class PadFrameTests(unittest.TestCase):
    def test_all_distal_pad_frames_match_solver_points_and_direction(self):
        raw = raw_pose()
        solver_offsets = {
            "leap_rh_if_ds": (-.010, -.032, .015),
            "leap_rh_mf_ds": (-.010, -.032, .015),
            "leap_rh_rf_ds": (-.010, -.032, .015),
            "leap_rh_th_ds": (-.009, -.035, -.011),
        }
        self.assertEqual(LEAP_PAD_OFFSETS_RAW, solver_offsets)
        for name, offset in solver_offsets.items():
            with self.subTest(name=name):
                pad = leap_pad_pose_matrix(raw, name)
                expected = raw[:, :3, 3] + raw[:, :3, :3] @ raw.new_tensor(offset)
                torch.testing.assert_close(pad[:, :3, 3], expected)
                torch.testing.assert_close(pad[:, :3, 0], -raw[:, :3, 0])
        self.assertNotEqual(LEAP_PAD_OFFSETS_RAW["leap_rh_th_ds"], LEAP_PAD_OFFSETS_RAW["leap_rh_rf_ds"])

    def test_agent_tip_poses_use_named_pad_frames(self):
        raw = raw_pose()
        links = [SimpleNamespace(pose=SimpleNamespace(to_transformation_matrix=lambda: raw))
                 for _ in LEAP_PAD_OFFSETS_RAW]
        agent = SimpleNamespace(tip_link_names=list(LEAP_PAD_OFFSETS_RAW), tip_links=links)
        tips = FloatingLeapHandRight.tip_poses.fget(agent)
        for tip, name in zip(tips, LEAP_PAD_OFFSETS_RAW):
            torch.testing.assert_close(tip.p, leap_pad_pose_matrix(raw, name)[:, :3, 3])

    def test_force_locations_adjust_only_named_leap_distal_links(self):
        raw = raw_pose()
        names = ["leap_rh_if_ds", "leap_rh_th_ds", "leap_rh_if_md", "base_link"]
        links = [SimpleNamespace(name=name, pose=SimpleNamespace(to_transformation_matrix=lambda: raw))
                 for name in names]

        class FakeLeap:
            def __init__(self):
                self.hand_links = links

        scene = SimpleNamespace(get_pairwise_contact_forces=lambda link, obj: torch.ones(2, 3))
        fake_env = SimpleNamespace(agent=FakeLeap(), scene=scene, obj=object())
        with mock.patch.object(force_module, "FloatingLeapHandRight", FakeLeap):
            forces, poses = force_module.PrimitiveGraspForceEnv.get_hand_contacts(fake_env)
        self.assertEqual(forces.shape, (4, 2, 3))
        for i, name in enumerate(names):
            expected = leap_pad_pose_matrix(raw, name) if name in LEAP_PAD_OFFSETS_RAW else raw
            torch.testing.assert_close(poses[i], expected)

        fake_env.agent = SimpleNamespace(hand_links=links)
        _, allegro_poses = force_module.PrimitiveGraspForceEnv.get_hand_contacts(fake_env)
        torch.testing.assert_close(allegro_poses, raw.repeat(4, 1, 1, 1))

    def test_leap_actuation_distance_uses_pad_and_allegro_keeps_shift(self):
        raw = raw_pose()[:1]
        pad = leap_pad_pose_matrix(raw, "leap_rh_if_ds")
        tip = SimpleNamespace(p=pad[:, :3, 3], to_transformation_matrix=lambda: pad)
        act_dir = pad[:, :3, 0]
        fake_env = SimpleNamespace(get_tip_poses=lambda: [tip],
                                   get_actuation_pos=lambda: tip.p,
                                   get_actuation_dir=lambda: act_dir,
                                   is_leap=True, num_envs=1)
        distance, angle, index = PrimitiveGraspEnv.compute_actuation_dist(fake_env)
        torch.testing.assert_close(distance, torch.zeros_like(distance))
        torch.testing.assert_close(angle, torch.zeros_like(angle))
        self.assertEqual(index.item(), 0)
        fake_env.is_leap = False
        distance, _, _ = PrimitiveGraspEnv.compute_actuation_dist(fake_env)
        torch.testing.assert_close(distance, act_dir * .01)


if __name__ == "__main__":
    unittest.main()
