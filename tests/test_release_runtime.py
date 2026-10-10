"""Stage-5 evaluator: model discovery, grasp resolution and seeding order."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from codex.codex_train import eval as eval_module
from codex.codex_train.configs import primitive


class EvaluatorTests(unittest.TestCase):
    def test_model_discovery_skips_newer_run_without_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            older = scene / "rl_runs" / "older"
            newer = scene / "rl_runs" / "newer"
            best = older / "best_model" / "best_model.zip"
            best.parent.mkdir(parents=True)
            best.touch()
            newer.mkdir()
            os.utime(older, (100, 100))
            os.utime(newer, (200, 200))
            self.assertEqual(eval_module.infer_model_path_from_scene(scene), str(best))

    def test_run_grasp_is_default_and_explicit_grasp_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "grasp_3_optimized.txt").touch()
            run = {"env_config": {"grasp_dir": temporary, "grasp_configs": ["grasp_3_optimized"],
                                  "mesh_dir": temporary, "actuation_dir": temporary, "num_envs": 9}}
            ec = eval_module.build_env_config(run)
            self.assertEqual((ec["grasp_dir"], ec["grasp_configs"]), (temporary, ["grasp_3_optimized"]))
            self.assertNotIn("num_envs", ec)
            chosen = Path(temporary) / "grasps" / "grasp_1_optimized.txt"
            chosen.parent.mkdir()
            chosen.touch()
            ec = eval_module.build_env_config(run, grasp_path=chosen)
            self.assertEqual((ec["grasp_dir"], ec["grasp_configs"]), (str(chosen.parent), ["grasp_1_optimized"]))

    def test_missing_run_paths_fall_back_to_the_scene(self):
        with tempfile.TemporaryDirectory() as scene:
            (Path(scene) / "x.txt").touch()
            run = {"env_config": {"grasp_dir": scene, "grasp_configs": ["x"],
                                  "mesh_dir": "/moved/mesh_coacd", "actuation_dir": "/moved/keypoints"}}
            ec = eval_module.build_env_config(run, scene_dir=scene)
            self.assertEqual(ec["mesh_dir"], str(Path(scene).resolve() / "mesh_coacd"))
            self.assertEqual(ec["actuation_dir"], str(Path(scene).resolve() / "keypoints"))

    def test_zero_action_without_run_uses_the_training_primitive(self):
        with tempfile.TemporaryDirectory() as scene:
            (Path(scene) / "grasps").mkdir()
            (Path(scene) / "grasps" / "grasp_1_optimized.txt").touch()
            ec = eval_module.build_env_config(None, scene)
            for key, value in primitive.PRIMITIVE_ENV_CONFIG.items():
                self.assertEqual(ec[key], value)
            self.assertEqual(ec["press_hold_seconds"], primitive.PRESS_HOLD_SECONDS)
            self.assertEqual(ec["num_actuate_steps"], primitive.NUM_ACTUATE_STEPS)

    def test_seed_precedes_scene_construction(self):
        seed = 47

        def make_env(*args, **kwargs):
            self.assertEqual(np.random.random(), np.random.RandomState(seed).random_sample())
            expected = torch.rand(1, generator=torch.Generator().manual_seed(seed)).item()
            self.assertEqual(torch.rand(1).item(), expected)
            self.assertEqual(kwargs["num_envs"], 3)
            self.assertEqual(kwargs["sim_backend"], "physx_cuda")
            raise RuntimeError("scene constructed")

        np.random.seed(999)
        with mock.patch.object(eval_module.gym, "make", side_effect=make_env):
            with self.assertRaisesRegex(RuntimeError, "scene constructed"):
                eval_module.evaluate({"grasp_configs": ["g"]}, episodes=3, seed=seed, video_episodes=0)

    def test_policy_loads_before_evaluation_and_zero_action_loads_nothing(self):
        events = []
        with tempfile.TemporaryDirectory() as scene:
            run_dir = Path(scene) / "rl_runs" / "run"
            model = run_dir / "best_model" / "best_model.zip"
            model.parent.mkdir(parents=True)
            model.touch()
            (Path(scene) / "grasps").mkdir()
            (Path(scene) / "grasps" / "grasp_1_optimized.txt").touch()
            (Path(scene) / "grasp_1_optimized.txt").touch()
            config = {"env_id": eval_module.ENV_ID, "algorithm": "PPO",
                      "env_config": {"grasp_dir": scene, "grasp_configs": ["grasp_1_optimized"],
                                     "mesh_dir": scene, "actuation_dir": scene}}
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
            fake = mock.Mock()
            summary = {"success": 1.0}

            def load(*args):
                events.append("load")
                return fake

            def run(*args, **kwargs):
                events.append(("evaluate", kwargs["model"] is fake))
                return summary, []

            with mock.patch.object(eval_module, "load_trained_model", side_effect=load), \
                    mock.patch.object(eval_module, "evaluate", side_effect=run):
                eval_module.main(["--scene-dir", scene, "--video-episodes", "0"])
                eval_module.main(["--scene-dir", scene, "--zero-action", "--video-episodes", "0"])
        self.assertEqual(events, ["load", ("evaluate", True), ("evaluate", False)])


if __name__ == "__main__":
    unittest.main()
