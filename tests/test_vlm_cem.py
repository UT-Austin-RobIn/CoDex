"""Contracts for the full 6DOF goal search and its stage 2 entry point."""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from codex.goal_gen import vlm_cem as cem
from codex.goal_gen import synthesize_3d_goal as synthesis


class FullGoalTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(tempfile.gettempdir())
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        keypoints = self.root / "keypoints"
        keypoints.mkdir()
        (keypoints / "fn_pt_mesh.json").write_text("[0.02, 0.03, 0.04]")
        # The full method must not require an actuation direction.
        (keypoints / "fn_key_pts.json").write_text(json.dumps({
            "fn_pts": [[25, 50]], "target_pts": [[60, 40]],
        }))
        self.base = np.eye(4)
        np.savetxt(self.root / "pose.txt", self.base)
        self.mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
        self.target_point = np.array([0.4, 0.2, 0.6])
        target = SimpleNamespace(name="target", mask=np.ones((8, 8), bool),
                                 get_pcd=lambda: SimpleNamespace(points=np.array([[1., 2., 3.]])))
        self.scene = SimpleNamespace(
            image=Image.new("RGB", (8, 8)), cam_extrinsics=np.eye(4),
            points=np.zeros((8, 8, 3)), objects=[SimpleNamespace(name="bottle"), target],
        )
        self.queries = []
        self.renders = []
        self.critic_override = None
        self.session_kwargs = []

    def query(self, images, prompt, **kwargs):
        self.queries.append((len(images), prompt))
        if "Please propose" in prompt:
            return json.dumps({str(i): {
                "yaw": 10 * i, "pitch": 2 * i, "roll": 3 * i,
                "tx": 0.01 * i, "ty": -0.02 * i, "tz": 0.03 * i,
            } for i in (1, 2)})
        if self.critic_override:
            return json.dumps(self.critic_override(images, len(self.queries)))
        return json.dumps({
            "candidates": {str(image.info["id"]): {
            "score": image.info["id"] / 10,
            "description": "judged from renders",
            } for image in images},
            "best_id": max(image.info["id"] for image in images),
            "suitable": False, "stop": False,
            "next_random_count": 2, "exploration_radius_m": 0.1,
            "explanation": "Move toward the target after inspecting clearance.",
        })

    def render(self, mesh, pose, scene, candidate_id, **kwargs):
        self.renders.append((candidate_id, kwargs["multi_view"],
                             kwargs["scene_representation"], kwargs.get("axes_origin")))
        image = Image.new("RGB", (8, 8))
        image.info["id"] = candidate_id
        return image

    def make_session(self, *args, **kwargs):
        self.session_kwargs.append(kwargs)
        return SimpleNamespace(query=self.query)

    def run_goal(self, views=3, scene_representation="points", iters=2):
        cem._candidate_id = 0
        random.seed(37)
        with contextlib.ExitStack() as stack:
            for name, value in {
                "load_scene_meta": lambda _: {"task": "use bottle", "functional_object": "bottle", "target_object": "target"},
                "load_mesh": lambda _: (self.mesh, {}),
                "lift_effect_point": lambda *args, **kwargs: self.target_point,
                "render_candidate": self.render,
                "VLMInterface": lambda: None,
                "VLMSession": self.make_session,
            }.items():
                stack.enter_context(patch.object(cem, name, value))
            stack.enter_context(patch.object(cem.ImageScene, "load", return_value=self.scene))
            stack.enter_context(patch.object(cem.o3d.visualization, "draw_geometries",
                                             side_effect=AssertionError("headless run opened a window")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            cem.generate_3d_goal(scene_dir=str(self.root), iters=iters, cands=4, views=views,
                                 scene_representation=scene_representation)
        return json.loads((self.root / "goal_gen" / "vlm_cem" / "best_candidate.json").read_text())

    def test_search_uses_three_views_and_independent_offsets_without_direction(self):
        result = self.run_goal()
        self.assertEqual([n for n, _ in self.queries], [4, 4, 5])
        self.assertEqual(len(self.renders), 9)
        self.assertTrue(all(multi for _, multi, _, _ in self.renders))
        self.assertTrue(all(representation == "points" for _, _, representation, _ in self.renders))
        self.assertTrue(all(origin is not None for _, _, _, origin in self.renders[:-1]))
        self.assertIsNone(self.renders[-1][-1])
        self.assertEqual((result["tx"], result["ty"], result["tz"]), (0.02, -0.04, 0.06))
        self.assertEqual(result["description"], "judged from renders")
        self.assertEqual(result["stop_reason"], "no_suitable_candidate")
        self.assertEqual(result["rounds"], 2)
        self.assertTrue(self.session_kwargs[0]["log_enabled"])
        self.assertTrue(self.session_kwargs[0]["log_dir"].endswith("vlm_logs"))
        self.assertTrue((self.root / "goal_gen" / "vlm_cem" / "best_candidate.jpg").is_file())
        self.assertTrue((self.root / "goal_gen" / "vlm_cem" / "best_candidate_axes.jpg").is_file())

    def test_one_view_is_forwarded_to_candidate_and_final_render(self):
        self.run_goal(views=1)
        self.assertTrue(all(not multi for _, multi, _, _ in self.renders))

    def test_surface_representation_is_forwarded(self):
        self.run_goal(scene_representation="surface")
        self.assertTrue(all(representation == "surface" for _, _, representation, _ in self.renders))

    def test_penetrating_candidates_reach_vlm_and_can_be_selected(self):
        self.target_point = np.array([0.4, 0.2, -0.6])
        result = self.run_goal()
        self.assertEqual(len(self.renders), 9)
        self.assertEqual([n for n, _ in self.queries], [4, 4, 5])
        self.assertGreater(result["table_penetration"], 0)
        self.assertGreater(result["score"], 0)

    def test_critic_can_stop_early_on_valid_suitable_choice(self):
        self.critic_override = lambda images, _: {
            "candidates": {str(image.info["id"]): {
                "score": 0.7, "description": "clear approach"
            } for image in images},
            "best_id": 2, "suitable": True, "stop": True,
            "next_random_count": 2, "exploration_radius_m": 0.1,
            "explanation": "Enough clearance and alignment",
        }
        result = self.run_goal(iters=5)
        self.assertEqual(result["rounds"], 1)
        self.assertTrue(result["suitable"])
        self.assertEqual(result["stop_reason"], "suitable")
        self.assertEqual(result["id"], 2)
        self.assertEqual(len(self.renders), 5)
        self.assertEqual(len(self.queries), 1)

    def test_invalid_best_id_cannot_authorize_stop(self):
        self.critic_override = lambda images, _: {
            "candidates": {str(image.info["id"]): {
                "score": 0.8, "description": "visible"
            } for image in images},
            "best_id": 9999, "suitable": True, "stop": True,
            "next_random_count": 2, "exploration_radius_m": 0.1,
            "explanation": "invalid winner",
        }
        result = self.run_goal()
        self.assertEqual(result["rounds"], 2)
        self.assertFalse(result["suitable"])
        self.assertEqual(result["stop_reason"], "no_suitable_candidate")
        self.assertTrue(all(not r["critic_best_id_valid"] for r in result["search_trace"]))

    def test_critic_allocation_is_bounded_and_critique_reaches_proposer(self):
        self.critic_override = lambda images, _: {
            "candidates": {str(image.info["id"]): {
                "score": 0.5, "description": "Move around the bowl rim"
            } for image in images},
            "best_id": max(image.info["id"] for image in images),
            "suitable": False, "stop": False,
            "next_random_count": 999, "exploration_radius_m": 99,
            "explanation": "Try more lateral offsets",
        }
        result = self.run_goal()
        self.assertEqual(result["search_trace"][1]["random_count"], 3)
        self.assertEqual(result["search_trace"][1]["proposed_count"], 1)
        self.assertEqual(result["search_trace"][0]["exploration_radius_m"], 0.5)
        proposal_prompt = self.queries[1][1]
        self.assertIn("Move around the bowl rim", proposal_prompt)
        self.assertIn("Try more lateral offsets", proposal_prompt)
        self.assertIn("8 cm along the displayed red +X axis", proposal_prompt)
        self.assertIn("table_penetration_m", proposal_prompt)
        first = result["search_trace"][0]["candidates"]["1"]
        self.assertIn(f"functional_point_height_m={first['functional_point_height_m']:.4f}",
                      proposal_prompt)
        lift_text = ", ".join(f"{x:.4f}" for x in first["local_delta_for_world_up_5cm"])
        self.assertIn(f"local_delta_for_world_up_5cm=[{lift_text}]", proposal_prompt)
        self.assertIn("ADD all three components", proposal_prompt)
        self.assertIn("Changing angles invalidates", proposal_prompt)
        self.assertIn("blank silhouette is missing scene data, not a second physical object",
                      proposal_prompt)
        self.assertIn("Missing data also does not prove the region empty", proposal_prompt)
        critic_prompt = self.queries[0][1]
        self.assertIn("round 1 of at most 2", critic_prompt)
        self.assertIn("fixed budget of 4", critic_prompt)
        self.assertIn("between 1 and 3", critic_prompt)
        self.assertIn("blank silhouette is missing scene data, not a second physical object",
                      critic_prompt)
        self.assertIn("Continue to fix a specific material issue", critic_prompt)

    def test_initial_batch_balances_registered_nearby_and_global_orientations(self):
        random.seed(37)
        initial = cem._random_candidates(8, generation=0, radius_m=0.2,
                                         initial=True)
        self.assertEqual((initial[0].yaw, initial[0].pitch, initial[0].roll),
                         (0, 0, 0))
        for candidate in initial[1:4]:
            self.assertTrue(all(abs(value) <= 20 for value in
                                (candidate.yaw, candidate.pitch, candidate.roll)))
        self.assertTrue(any(abs(candidate.roll) > 20 for candidate in initial[4:]))
        for index, candidate in enumerate(initial):
            scale = (0.025, 0.08, 0.2)[index % 3]
            self.assertTrue(all(abs(value) <= scale for value in
                                (candidate.tx, candidate.ty, candidate.tz)))
        later = cem._random_candidates(8, generation=1, radius_m=0.2,
                                       initial=False)
        self.assertTrue(any(abs(candidate.roll) > 20 for candidate in later))

    def test_suitable_without_stop_uses_max_rounds_status(self):
        self.critic_override = lambda images, _: {
            "candidates": {str(image.info["id"]): {
                "score": 0.7, "description": "plausible"
            } for image in images},
            "best_id": max(image.info["id"] for image in images),
            "suitable": True, "stop": False,
            "next_random_count": -20, "exploration_radius_m": 0,
            "explanation": "Continue checking",
        }
        result = self.run_goal()
        self.assertEqual(result["stop_reason"], "max_rounds")
        self.assertTrue(result["suitable"])
        self.assertEqual(result["search_trace"][1]["requested_proposed_count"], 3)
        self.assertEqual(result["search_trace"][1]["proposed_count"], 2)
        self.assertEqual(result["search_trace"][1]["random_count"], 2)
        self.assertEqual(result["search_trace"][0]["exploration_radius_m"], 0.02)

    def test_missing_scores_do_not_break_proposal_history_or_claim_suitability(self):
        def critic(images, call_number):
            if call_number == 1:
                return {"candidates": {}, "best_id": 999, "suitable": True,
                        "stop": True, "next_random_count": 2,
                        "exploration_radius_m": 0.1,
                        "explanation": "Scores were unavailable"}
            return {"candidates": {}, "best_id": 999, "suitable": True,
                    "stop": True, "next_random_count": 2,
                    "exploration_radius_m": 0.1,
                    "explanation": "Still unavailable"}
        self.critic_override = critic
        result = self.run_goal()
        self.assertEqual(result["rounds"], 2)
        self.assertFalse(result["suitable"])
        self.assertEqual(result["stop_reason"], "no_suitable_candidate")
        self.assertIn("score=unscored", self.queries[1][1])

    def test_pose_offset_uses_rotated_local_axes(self):
        base = np.eye(4)
        base[:3, :3] = cem.euler_to_matrix(30, 0, 0)
        fn = np.array([0.02, 0.03, 0.04])
        target = np.array([0.4, 0.2, 0.6])
        pose = cem.place_pose(base, fn, 20, -10, 40, 0.08, 0.02, -0.03, target)
        rotation = base[:3, :3] @ cem.euler_to_matrix(20, -10, 40)
        np.testing.assert_allclose(pose[:3, :3], rotation)
        np.testing.assert_allclose((pose @ np.append(fn, 1))[:3],
                                   target + rotation @ [0.08, 0.02, -0.03])

    def test_local_lift_vector_raises_world_functional_point_five_centimeters(self):
        base = np.eye(4)
        base[:3, :3] = cem.euler_to_matrix(30, -12, 7)
        base[:3, 3] = [0.2, -0.1, 0.4]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cem.euler_to_matrix(-20, 9, 13)
        extrinsic[:3, 3] = [-0.3, 0.5, 0.2]
        fn = np.array([0.02, 0.03, 0.04])
        target = np.array([0.4, 0.2, 0.6])
        candidate = cem.Candidate3D(122.5, -74.4, -47, -0.14, 0.013, 0.005)
        height, lift = cem._world_height_and_lift(candidate, base, fn, target, extrinsic)
        initial = cem.place_pose(base, fn, candidate.yaw, candidate.pitch,
                                 candidate.roll, candidate.tx, candidate.ty,
                                 candidate.tz, target)
        raised = cem.place_pose(base, fn, candidate.yaw, candidate.pitch,
                                candidate.roll, candidate.tx + lift[0],
                                candidate.ty + lift[1], candidate.tz + lift[2], target)
        initial_world = (extrinsic @ initial @ np.append(fn, 1))[:3]
        raised_world = (extrinsic @ raised @ np.append(fn, 1))[:3]
        self.assertAlmostEqual(height, initial_world[2])
        np.testing.assert_allclose(raised_world - initial_world, [0, 0, 0.05], atol=1e-12)

    def test_cli_defaults_to_three_views(self):
        with patch.object(cem, "generate_3d_goal") as generate, patch(
            "sys.argv", ["vlm_cem", "--scene-dir", str(self.root)]
        ):
            cem.main()
        self.assertEqual(generate.call_args.kwargs["views"], 3)
        self.assertEqual(generate.call_args.kwargs["scene_representation"], "points")
        self.assertEqual(generate.call_args.kwargs["cands"], 8)

    def test_stage2_entry_point_defaults_to_the_direction_method(self):
        import runpy
        from codex.goal_gen import vlm_cem_dir as cem_dir
        script = str(Path(__file__).resolve().parents[1] / "scripts" / "2_generate_goal.py")
        for argv, used, unused in ((["x", "--scene-dir", "s"], cem_dir, cem),
                                   (["x", "--method", "vlm_cem", "--scene-dir", "s"], cem, cem_dir)):
            with patch.object(used, "main") as run, patch.object(unused, "main") as other, \
                    patch("sys.argv", argv):
                runpy.run_path(script, run_name="__main__")
                run.assert_called_once()
                other.assert_not_called()
                self.assertNotIn("--method", sys.argv)

    def test_synthesis_prefers_scene_local_direction_method(self):
        full = self.root / "goal_gen" / "vlm_cem" / "best_candidate.json"
        direction = self.root / "goal_gen" / "vlm_cem_dir" / "best_candidate.json"
        full.parent.mkdir(parents=True)
        direction.parent.mkdir(parents=True)
        full.write_text("{}")
        direction.write_text("{}")
        self.assertEqual(synthesis._find_best_candidate(str(self.root), None), str(direction))


if __name__ == "__main__":
    unittest.main()
