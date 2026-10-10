import numpy as np
import torch
import unittest

from codex.grasp_gen.batched_grasp_solver import (
    BatchedSDF, FC_LSTAR_MIN_MARGIN, force_closure_margin_passes,
    split_force_closure_contacts, solve_min_weight_lp_batch,
)


class ForceClosureContactTests(unittest.TestCase):
    def test_final_fc_requires_positive_margin_above_lp_noise(self):
        self.assertEqual(FC_LSTAR_MIN_MARGIN, 1e-6)
        values = torch.tensor([-1., -1e-8, 0., 1.4e-8, 1.8e-8,
                               FC_LSTAR_MIN_MARGIN, 1.1e-6, .0094686863])
        self.assertEqual(force_closure_margin_passes(values).tolist(),
                         [False, False, False, False, False, False, True, True])

    def test_noncontact_palm_cannot_rescue_three_support_contacts(self):
        # The first 12 primitive wrenches alone cannot balance +X. Four
        # phantom palm wrenches at -3X make the full LP positive if included.
        support_w = np.zeros((1, 6, 12))
        support_w[:, 0, :] = 1.0
        for dim in range(1, 6):
            support_w[:, dim, 2 * (dim - 1)] = 1.0
            support_w[:, dim, 2 * (dim - 1) + 1] = -1.0
        palm_w = np.zeros((1, 6, 4))
        palm_w[:, 0, :] = -3.0
        self.assertGreater(solve_min_weight_lp_batch(np.concatenate((support_w, palm_w), axis=2))[0][0], 0)
        self.assertLessEqual(solve_min_weight_lp_batch(support_w)[0][0], 0)

        support_pts = torch.zeros((2, 3, 3))
        palm_pts = torch.ones((2, 3))
        gap_m = torch.tensor([0.050, 0.001])
        cases = split_force_closure_contacts(support_pts, palm_pts, gap_m, torch.ones(2, dtype=torch.bool))
        no_palm = next((mask, pts) for mask, pts, included in cases if not included)
        with_palm = next((mask, pts) for mask, pts, included in cases if included)
        self.assertEqual(no_palm[0].tolist(), [True, False])
        self.assertEqual(no_palm[1].shape[1], 3)
        self.assertEqual(with_palm[0].tolist(), [False, True])
        self.assertEqual(with_palm[1].shape[1], 4)

    def test_palm_gap_uses_object_surface_even_when_floor_grid_is_near(self):
        sdf = BatchedSDF.__new__(BatchedSDF)
        sdf.sdf_tensor = torch.full((1, 1, 2, 2, 2), -0.001)
        sdf.object_sdf_tensor = torch.full((1, 1, 2, 2, 2), 0.050)
        sdf.bbox_min_t = torch.zeros(3)
        sdf.range_t = torch.ones(3)
        point = torch.full((1, 1, 3), 0.5)
        self.assertTrue(torch.allclose(sdf.query(point, include_clearance=False), torch.tensor([[-0.001]])))
        self.assertTrue(torch.allclose(sdf.query(point, include_clearance=False, include_floor=False), torch.tensor([[0.050]])))

    def test_add_floor_preserves_object_only_grid(self):
        sdf = BatchedSDF.__new__(BatchedSDF)
        sdf.sdf_tensor = torch.full((1, 1, 2, 2, 2), 0.050)
        sdf.object_sdf_tensor = sdf.sdf_tensor
        sdf.bbox_min = np.zeros(3)
        sdf.bbox_max = np.ones(3)
        sdf.bbox_min_t = torch.zeros(3)
        sdf.range_t = torch.ones(3)
        sdf.add_floor(0.5)
        point = torch.tensor([[[0.5, 0.5, 0.0]]])
        self.assertLess(sdf.query(point, include_clearance=False).item(), 0)
        self.assertAlmostEqual(sdf.query(point, include_clearance=False, include_floor=False).item(), 0.050)


if __name__ == "__main__":
    unittest.main()
