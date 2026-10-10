"""Primitive-aware grasp synthesis and simulated selection (CPU checks)."""

import unittest

import numpy as np
import torch
import trimesh

from codex.grasp_gen import primitive_grasp as pg
from codex.grasp_gen.sim_screen import screen_score, successive_halving


def _box_parts():
    # Two overlapping convex boxes (a union), resting on z=0.
    a = trimesh.creation.box(extents=(0.06, 0.06, 0.16))
    a.apply_translation((0, 0, 0.08))
    b = trimesh.creation.box(extents=(0.10, 0.04, 0.04))
    b.apply_translation((0.03, 0, 0.14))
    return [a, b]


class PartUnionSDFTest(unittest.TestCase):
    def test_exact_outside_and_negative_inside(self):
        sdf = pg.PartUnionSDF(_box_parts(), resolution=96, device="cpu")
        pts = torch.tensor([[[0.0, 0.0, 0.08],       # inside box a
                             [0.075, 0.0, 0.14],     # inside box b only
                             [0.0, 0.08, 0.08]]])    # 5 cm outside box a
        d = sdf(pts)[0]
        self.assertLess(d[0].item(), 0)
        self.assertLess(d[1].item(), 0)
        self.assertAlmostEqual(d[2].item(), 0.05, delta=0.004)


class SynthesizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        act_pos = np.array([0.0, 0.0, 0.16])      # top face of box a
        act_dir = np.array([0.0, 0.0, -1.0])      # press down, into the object
        cls.syn = pg.PrimitiveGraspSynthesizer(_box_parts(), act_pos, act_dir, device="cpu", seed=3)

    def test_anchored_seeds_put_the_actuation_pad_on_the_keypoint(self):
        B = 16
        pos, R, q, fa = self.syn.sample(B, max_tilt=0.0)
        T = torch.eye(4).repeat(B, 1, 1)
        T[:, :3, :3], T[:, :3, 3] = R, pos
        pads, axes = self.syn.hand.pads(self.syn.hand.fk(q), T)
        rows = torch.arange(B)
        target = self.syn.a + pg.ACT_PRELOAD * self.syn.d
        torch.testing.assert_close(pads[rows, fa], target.expand(B, 3), atol=1e-4, rtol=0)
        torch.testing.assert_close(axes[rows, fa], self.syn.d.expand(B, 3), atol=1e-4, rtol=0)

    def test_energy_is_finite_and_differentiable(self):
        B = 8
        pos, R, q, fa = self.syn.sample(B)
        r6 = torch.cat([R[:, :, 0], R[:, :, 1]], -1).requires_grad_(True)
        q = q.clone().requires_grad_(True)
        pos = pos.clone().requires_grad_(True)
        e, terms = self.syn.energy(pos, r6, q, fa, torch.ones(B), torch.full((B,), 0.04))
        self.assertTrue(torch.isfinite(e).all())
        e.sum().backward()
        for g in (pos.grad, r6.grad, q.grad):
            self.assertTrue(torch.isfinite(g).all())
        self.assertIn("preload", terms)
        self.assertIn("approach", terms)

    def test_finger_gaps_detect_a_cross_over(self):
        H = self.syn.hand
        q = torch.zeros(3, 16)
        q[1:, 8] = 1.2                   # ring finger flexed at the MCP ...
        q[1, 9], q[2, 9] = 1.0, -1.0     # ... and spread fully either way
        gaps = H.finger_gaps(H.fk(q))
        self.assertGreater(gaps[0].min().item(), pg.FINGER_GAP)   # open hand keeps its order
        self.assertLess(min(gaps[1].min().item(), gaps[2].min().item()), 0.0)

    def test_grasp_vectors_follow_the_file_convention(self):
        res = self.syn.optimize(B=8, oversample=1, steps=2, log_every=0)
        vec = pg.to_grasp_vectors(res, [0])[0]
        self.assertEqual(vec.shape, (23,))
        self.assertAlmostEqual(float(np.linalg.norm(vec[:4])), 1.0, places=5)
        ok, gates = self.syn.gates(res)
        self.assertEqual(ok.shape, (8,))
        self.assertIn("preload", gates)
        self.assertIn("order", gates)


class SuccessiveHalvingTest(unittest.TestCase):
    def test_keeps_the_best_and_pools_rollouts(self):
        quality = np.linspace(0, 1, 10)

        class FakeScreener:
            calls = []

            def screen(self, grasps, fingers, reps, seed, rollouts):
                self.calls.append((len(grasps), reps))
                out = []
                for k, g in enumerate(grasps):
                    rollouts[k].extend([{"q": float(g[0])}] * reps)
                    s = float(g[0])
                    out.append(dict(success=s, lift_success=s, raw_lift=s, stable_success=s,
                                    upright_success=1.0, act_dist=0.01, est_collision=0.0,
                                    n=len(rollouts[k])))
                return out

        scr = FakeScreener()
        order, stats = successive_halving(scr, [np.array([x]) for x in quality], [0] * 10,
                                          schedule=((None, 2), (4, 4), (2, 8)), log=lambda m: None)
        self.assertEqual(scr.calls, [(10, 2), (4, 4), (2, 8)])
        self.assertEqual(order, [9, 8])
        self.assertEqual(stats[9]["n"], 14)                  # 2 + 4 + 8 pooled rollouts
        self.assertGreater(screen_score(stats[9]), screen_score(stats[8]))


if __name__ == "__main__":
    unittest.main()
