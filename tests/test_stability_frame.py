"""Regression tests for optional palm-relative stability coordinates."""
import unittest

import torch

from codex.codex_train.envs.stability_frame_math import palm_actuation_vector


class PalmVectorTests(unittest.TestCase):
    def test_hand_frame_ignores_global_rigid_rotation(self):
        """A globally rotated, rigidly held object must not gain apparent slip."""
        identity = torch.eye(3).unsqueeze(0)
        quarter_turn = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        before = palm_actuation_vector(torch.tensor([[.2, 0., 0.]]), torch.zeros(1, 3), identity, "hand")
        after = palm_actuation_vector(torch.tensor([[0., .2, 0.]]), torch.zeros(1, 3), quarter_turn, "hand")
        self.assertTrue(torch.allclose(before, torch.tensor([[.2, 0., 0.]]), atol=1e-7))
        self.assertTrue(torch.allclose(after, before, atol=1e-7))


    def test_hand_frame_detects_actual_relative_slip(self):
        """Moving the actuation point 5 cm within the same hand exceeds the 4 cm gate."""
        identity = torch.eye(3).unsqueeze(0)
        before = palm_actuation_vector(torch.tensor([[.2, 0., 0.]]), torch.zeros(1, 3), identity, "hand")
        after = palm_actuation_vector(torch.tensor([[.2, 0., 0.]]), torch.tensor([[.05, 0., 0.]]), identity, "hand")
        self.assertAlmostEqual(torch.linalg.vector_norm(after - before, dim=-1).item(), .05, places=6)


    def test_world_mode_preserves_saved_gate_coordinates(self):
        rotation = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        result = palm_actuation_vector(torch.tensor([[0., .2, 0.]]), torch.zeros(1, 3), rotation, "world")
        self.assertTrue(torch.allclose(result, torch.tensor([[0., .2, 0.]]), atol=1e-7))


    def test_unknown_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stability_frame"):
            palm_actuation_vector(torch.zeros(1, 3), torch.zeros(1, 3), torch.eye(3).unsqueeze(0), "camera")
