"""The min-weight LP must implement free alpha/l with explicit solve status."""

import unittest

import numpy as np
import torch

from codex.grasp_gen.batched_grasp_solver import (
    FC_LSTAR_MIN_MARGIN,
    force_closure_margin_passes,
    min_weight_gradient_batch,
    solve_min_weight_lp_batch,
)


def outside_wrench_matrix(width):
    rng = np.random.default_rng(7)
    wrench = rng.normal(size=(6, 12))
    wrench[0] = np.linspace(1 - width / 2, 1 + width / 2, 12)
    assert np.linalg.matrix_rank(wrench) == 6
    return wrench


class MinWeightLpBoundsTest(unittest.TestCase):
    def test_finite_negative_margin_has_gradient_even_below_old_sentinel(self):
        wrench = outside_wrench_matrix(.2)
        lstar, alpha, lamb, nu, status = solve_min_weight_lp_batch(
            wrench[None], return_status=True)
        self.assertTrue(status[0])
        self.assertLess(lstar[0], -1.0)
        self.assertAlmostEqual(alpha[0].sum(), 1.0, places=7)
        np.testing.assert_allclose(wrench @ alpha[0], 0, atol=1e-7)

        gradient = min_weight_gradient_batch(
            wrench[None], lstar, alpha, lamb, nu,
            device='cpu', valid_mask=status)[0].numpy()
        delta = 1e-5
        positive, negative = wrench.copy(), wrench.copy()
        positive[0, 0] += delta
        negative[0, 0] -= delta
        lp_positive = solve_min_weight_lp_batch(positive[None])[0][0]
        lp_negative = solve_min_weight_lp_batch(negative[None])[0][0]
        finite_difference = (lp_positive - lp_negative) / (2 * delta)
        self.assertAlmostEqual(float(gradient[0, 0]), finite_difference, places=4)
        self.assertGreater(abs(finite_difference), 1.0)
        self.assertFalse(force_closure_margin_passes(torch.tensor(lstar))[0])

    def test_positive_margin_and_default_four_result_api(self):
        wrench = np.column_stack((np.eye(6), -np.eye(6)))
        four = solve_min_weight_lp_batch(wrench[None])
        self.assertEqual(len(four), 4)
        lstar, _, _, _, status = solve_min_weight_lp_batch(
            wrench[None], return_status=True)
        self.assertTrue(status[0])
        self.assertAlmostEqual(lstar[0], 1 / 12, places=10)
        self.assertEqual(lstar[0], four[0][0])
        self.assertGreater(lstar[0], FC_LSTAR_MIN_MARGIN)

    def test_true_affine_failure_and_masked_rows_have_no_gradient(self):
        wrench = outside_wrench_matrix(.8)
        wrench[0] = 1.0  # contradicts W alpha=0, sum alpha=1
        batch = np.stack((wrench, outside_wrench_matrix(.2)))
        lstar, alpha, lamb, nu, status = solve_min_weight_lp_batch(
            batch, return_status=True)
        self.assertFalse(status[0])
        self.assertTrue(status[1])
        gradient = min_weight_gradient_batch(
            batch, lstar, alpha, lamb, nu,
            device='cpu', valid_mask=status).numpy()
        self.assertTrue(np.all(gradient[0] == 0))
        self.assertGreater(np.linalg.norm(gradient[1]), 0)
        with self.assertRaisesRegex(ValueError, 'success mask required'):
            min_weight_gradient_batch(batch, lstar, alpha, lamb, nu, device='cpu')

        _, _, _, _, masked_status = solve_min_weight_lp_batch(
            batch, env_mask=np.array([False, True]), return_status=True)
        np.testing.assert_array_equal(masked_status, np.array([False, True]))


if __name__ == '__main__':
    unittest.main()
