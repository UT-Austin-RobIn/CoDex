"""Evaluation with the stage-2 goal: goal transfer to the simulator, the carry, outcomes."""

import json
import os
import tempfile
import unittest

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from codex.codex_train import eval as ev
from codex.codex_train.envs.primitive_env import interpolate_pose


def pose(rotvec=(0, 0, 0), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


class GoalFromSceneTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.scene = self.dir.name
        self.mesh = os.path.join(self.scene, "mesh")
        os.makedirs(self.mesh)
        trimesh.creation.box(extents=(0.04, 0.04, 0.10)).export(os.path.join(self.mesh, "mesh.glb"))
        self.spawn = pose(t=(0, 0, 0.05))            # the simulator stands the (centered) box on the table
        # Captured lying on its side (turned 90 degrees about x), 2 cm high, heading 30 degrees.
        self.captured = pose(t=(0.4, 0.1, 0.02))
        self.captured[:3, :3] = (Rotation.from_euler("z", 30, degrees=True) * Rotation.from_euler("x", 90, degrees=True)).as_matrix()

    def tearDown(self):
        self.dir.cleanup()

    def write(self, method, goal):
        d = os.path.join(self.scene, "goal_gen", method)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "best_candidate.json"), "w") as f:
            json.dump({"transform_world": goal.tolist(),
                       "relative_transform": (np.linalg.inv(self.captured) @ goal).tolist()}, f)

    def test_goal_keeps_its_orientation_and_height_from_the_standing_spawn(self):
        goal = pose((0, 0, np.radians(45)), (0.6, 0.3, 0.25)) @ self.captured
        goal[2, 3] = 0.25
        self.write("vlm_cem_dir", goal)
        G, info = ev.goal_pose_from_scene(self.scene, self.mesh)
        at_goal = self.spawn @ G                      # where the simulator carries it
        np.testing.assert_allclose(at_goal[:3, :3], goal[:3, :3], atol=1e-9)
        self.assertAlmostEqual(at_goal[2, 3], 0.25)
        np.testing.assert_allclose(at_goal[:2, 3], goal[:2, 3] - self.captured[:2, 3], atol=1e-9)

    def test_goal_inside_the_table_is_an_error(self):
        into = self.captured.copy()
        into[:3, 3] = [0.5, 0.1, 0.0]                  # on its side at height 0: 2 cm into the table
        self.write("vlm_cem_dir", into)
        with self.assertRaisesRegex(ValueError, "into the table"):
            ev.goal_pose_from_scene(self.scene, self.mesh)

    def test_method_choice(self):
        self.write("vlm_cem", pose(t=(0.1, 0.0, 0.2)) @ self.captured)
        self.assertTrue(ev.goal_pose_from_scene(self.scene, self.mesh)[1]["source"].endswith("vlm_cem/best_candidate.json"))
        self.write("vlm_cem_dir", pose(t=(0.1, 0.0, 0.3)) @ self.captured)
        self.assertIn("vlm_cem_dir", ev.goal_pose_from_scene(self.scene, self.mesh)[1]["source"])
        self.assertIn("/vlm_cem/", ev.goal_pose_from_scene(self.scene, self.mesh, "vlm_cem")[1]["source"])

    def test_without_goal_and_old_results(self):
        G, why = ev.goal_pose_from_scene(self.scene, self.mesh)
        self.assertIsNone(G)
        self.assertIn("no goal", why)
        os.makedirs(os.path.join(self.scene, "goal_gen", "vlm_cem_dir"))
        with open(os.path.join(self.scene, "goal_gen", "vlm_cem_dir", "best_candidate.json"), "w") as f:
            json.dump({"dist": 0.1, "yaw": 0.0}, f)
        with self.assertRaises(ValueError):
            ev.goal_pose_from_scene(self.scene, self.mesh)


class CarryTest(unittest.TestCase):
    def test_interpolated_waypoints(self):
        A = torch.eye(4)[None]
        B = torch.as_tensor(pose((0, 0, np.pi / 2), (0.2, 0.0, 0.1)), dtype=torch.float32)[None]
        np.testing.assert_allclose(interpolate_pose(A, B, 0.0), A, atol=1e-6)
        np.testing.assert_allclose(interpolate_pose(A, B, 1.0), B, atol=1e-6)
        mid = interpolate_pose(A, B, 0.5)[0].numpy()
        self.assertAlmostEqual(Rotation.from_matrix(mid[:3, :3]).magnitude(), np.pi / 4, places=5)
        np.testing.assert_allclose(mid[:3, 3], [0.1, 0.0, 0.05], atol=1e-6)


class OutcomeTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(ev.outcome({"success": True}), "success")
        self.assertEqual(ev.outcome({"success": False, "goal_success": True}), "press_missed")
        self.assertEqual(ev.outcome({"success": False, "goal_success": False, "object_height_change_m": 0.3}),
                         "not_held")
        self.assertEqual(ev.outcome({"success": False, "object_height_change_m": 0.2}), "press_missed")



class GoalSearchViewsTest(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        C = np.eye(4)
        C[:3, :3] = Rotation.from_euler("x", -120, degrees=True).as_matrix()   # looking down at the table
        C[:3, 3] = [0.0, -0.6, 0.6]
        self.scene = SimpleNamespace(cam_extrinsics=C)
        self.effect = np.array([0.3, 0.2, 0.05])

    def test_turned_view_keeps_the_effect_point_in_place_and_top_looks_down(self):
        from codex.goal_gen.synthesize_3d_goal import candidate_views
        camera, side, top = candidate_views(self.scene, self.effect)
        pix = lambda C: (np.linalg.inv(C) @ np.append(self.effect, 1))[:3]
        np.testing.assert_allclose(pix(side), pix(camera), atol=1e-9)          # same image position
        self.assertAlmostEqual(side[2, 3], camera[2, 3])                         # same height
        np.testing.assert_allclose(top[:3, 2], [0, 0, -1])                       # straight down
        np.testing.assert_allclose(top[:2, 3], self.effect[:2])                  # above the effect point
        self.assertAlmostEqual(np.linalg.det(top[:3, :3]), 1.0)

    def test_effect_point_is_lifted_on_the_target(self):
        from types import SimpleNamespace
        from codex.goal_gen.synthesize_3d_goal import lift_effect_point
        h, w = 40, 60
        points = np.zeros((h, w, 3))
        points[..., 2] = 1.0                        # table at 1 m
        points[10:20, 20:30, 2] = 0.9               # the block is 10 cm closer
        block = np.zeros((h, w), bool); block[10:20, 20:30] = True
        scene = SimpleNamespace(points=points, objects=[SimpleNamespace(name="bottle", mask=np.zeros((h, w))),
                                                      SimpleNamespace(name="block", mask=block)])
        self.assertAlmostEqual(lift_effect_point(scene, (25, 15), "block")[2], 0.9)
        self.assertAlmostEqual(lift_effect_point(scene, (5, 35), "block")[2], 0.9)   # off the mask: snaps onto it

    def test_scene_creation_rejects_empty_or_table_masks(self):
        from codex.scene.create_scene import check_masks
        table = np.zeros((40, 60), bool); table[20:] = True
        block = np.zeros((40, 60), bool); block[10:18, 20:30] = True
        check_masks({"block": block}, table)
        with self.assertRaisesRegex(RuntimeError, "nothing"):
            check_masks({"block": np.zeros_like(table)}, table)
        with self.assertRaisesRegex(RuntimeError, "essentially the table"):
            check_masks({"block": table.copy()}, table)


if __name__ == "__main__":
    unittest.main()
