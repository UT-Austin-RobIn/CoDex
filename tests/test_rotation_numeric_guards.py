"""Float32 roundoff at inverse-trig boundaries must not poison controls."""

from types import SimpleNamespace
import unittest

import torch

from codex.codex_train.agents.load_grasp_qpos import mat_to_pos_euler, pos_euler_to_mat
from codex.codex_train.envs.primitive_env import PrimitiveGraspEnv


class RotationNumericGuards(unittest.TestCase):
    def test_xyz_euler_roundoff_clamps_only_conversion_input(self):
        eps = torch.finfo(torch.float32).eps
        for sign in (-1., 1.):
            matrix = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            matrix[:, 0, 2] = sign * (1. + eps)
            matrix[:, :3, 3] = torch.tensor([.12, -.03, .45])
            original = matrix.clone()
            result = mat_to_pos_euler(matrix)
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(torch.equal(matrix, original))
            self.assertTrue(torch.equal(result[:, :3], matrix[:, :3, 3]))

        ordinary = torch.tensor([[.12, -.03, .45, .2, -.3, .4]])
        ordinary_matrix = pos_euler_to_mat(ordinary)
        self.assertTrue(torch.allclose(mat_to_pos_euler(ordinary_matrix), ordinary, atol=1e-5))

        invalid = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        invalid[:, 0, 2] = float("nan")
        self.assertTrue(torch.isnan(mat_to_pos_euler(invalid)).any())
        self.assertTrue(torch.isnan(invalid[:, 0, 2]).all())

    def test_tip_direction_acos_roundoff_is_finite_without_mutating_pose(self):
        eps = torch.finfo(torch.float32).eps
        for sign in (-1., 1.):
            matrix = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            matrix[:, 0, 0] = sign * (1. + eps)
            original = matrix.clone()
            pose = SimpleNamespace(p=torch.zeros((1, 3)), to_transformation_matrix=lambda: matrix)
            env = SimpleNamespace(
                get_tip_poses=lambda: [pose] * 4,
                get_actuation_pos=lambda: torch.zeros((1, 3)),
                get_actuation_dir=lambda: torch.tensor([[1., 0., 0.]]),
                is_leap=True, num_envs=1,
            )
            _, direction_difference, _ = PrimitiveGraspEnv.compute_actuation_dist(
                env, act_idx_mode=torch.tensor([0]))
            self.assertTrue(torch.isfinite(direction_difference).all())
            self.assertTrue(torch.equal(matrix, original))


if __name__ == "__main__":
    unittest.main()
