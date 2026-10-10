"""Training CLI and RNG setup checks that do not construct a GPU scene."""

import random
import json
from pathlib import Path
import sys
import tempfile
import zipfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from codex.codex_train import train as train_module


class TrainRuntimeTests(unittest.TestCase):
    def test_cli_defaults_match_the_primitive_config(self):
        from codex.codex_train.configs import primitive
        with mock.patch.object(sys, "argv", ["train"]):
            defaults = train_module.parse_args()
        self.assertIsNone(defaults.seed)
        self.assertEqual(defaults.num_actuate_steps, primitive.NUM_ACTUATE_STEPS)
        self.assertEqual(defaults.eval_episodes, 32)
        self.assertEqual(defaults.press_hold_seconds, primitive.PRESS_HOLD_SECONDS)
        # Release training settings: small exploration noise, no entropy bonus,
        # capped PPO updates, and the scene's goal carry.
        self.assertEqual(defaults.log_std_init, -3.0)
        self.assertEqual(defaults.ent_coef, 0.0)
        self.assertEqual(defaults.target_kl, 0.02)
        self.assertEqual(defaults.goal, "scene")
        self.assertIsNone(defaults.stability_frame)
        with mock.patch.object(sys, "argv", ["train", "--seed", "41", "--num-actuate-steps", "80", "--eval-episodes", "48", "--press-hold-seconds", "1.0", "--log-std-init", "-2.0", "--stability-frame", "hand"]):
            chosen = train_module.parse_args()
        self.assertEqual((chosen.seed, chosen.num_actuate_steps, chosen.eval_episodes, chosen.press_hold_seconds), (41, 80, 48, 1.0))
        self.assertEqual(chosen.log_std_init, -2.0)
        self.assertEqual(chosen.stability_frame, "hand")
        self.assertIsNone(defaults.resume_checkpoint)

    def test_resume_preflight_requires_matching_config_and_optimizer(self):
        scratch = Path(tempfile.gettempdir())
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            source = Path(temporary) / "prior"
            (source / "checkpoints").mkdir(parents=True)
            checkpoint = source / "checkpoints/model.zip"
            env_config = {**train_module.P.DOMAIN_RANDOMIZATION, "press_hold_seconds": 1.0,
                          "mesh_dir": "mesh", "grasp_dir": "grasps"}
            with mock.patch.object(sys, "argv", ["train", "--seed", "41", "--press-hold-seconds", "1.0",
                                                 "--resume-checkpoint", str(checkpoint), "--total-timesteps", "8192",
                                                 "--num-envs", "4096"]):
                args = train_module.parse_args()
            fields = ("gamma", "gae_lambda", "ent_coef", "n_steps", "batch_size",
                      "n_epochs", "learning_rate", "target_kl")
            source_config = {key: getattr(args, key) for key in fields}
            source_config.update(algorithm="PPO", env_id=args.env_id, policy=args.policy,
                                 seed=41, num_envs=2048, press_hold_seconds=1.0,
                                 log_std_init=-2.0, env_config=env_config)
            (source / "config.yaml").write_text(yaml.safe_dump(source_config))
            saved = {key: getattr(args, key) for key in fields}
            saved.update(seed=41, use_sde=True, sde_sample_freq=4,
                         num_timesteps=40960, n_envs=2048)
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("data", json.dumps(saved))
                archive.writestr("policy.optimizer.pth", b"optimizer-present")
            metadata = train_module._resume_metadata(args, env_config)
            self.assertEqual(metadata["initial_timesteps"], 40960)
            self.assertEqual(metadata["target_timesteps"], 49152)
            self.assertEqual(metadata["log_std_init"], -2.0)
            self.assertEqual((metadata["old_num_envs"], metadata["new_num_envs"]), (2048, 4096))
            self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
            args.target_kl = 0.8
            with self.assertRaisesRegex(ValueError, "target_kl"):
                train_module._resume_metadata(args, env_config)
            args.target_kl = 0.02
            args.batch_size = 4096
            with self.assertRaisesRegex(ValueError, "batch_size"):
                train_module._resume_metadata(args, env_config)
            args.batch_size = 2048
            with self.assertRaisesRegex(ValueError, "env_config"):
                train_module._resume_metadata(args, {**env_config, "press_hold_seconds": 0.0})
            with self.assertRaisesRegex(ValueError, "env_config"):
                train_module._resume_metadata(args, {**env_config, "hand_friction": (0.5, 1.0)})
            hand_config = {**env_config, "stability_frame": "hand"}
            with self.assertRaisesRegex(ValueError, "env_config"):
                train_module._resume_metadata(args, hand_config)
            source_config["env_config"] = hand_config
            (source / "config.yaml").write_text(yaml.safe_dump(source_config))
            self.assertEqual(train_module._resume_metadata(args, hand_config)["initial_timesteps"], 40960)
            with self.assertRaisesRegex(ValueError, "env_config"):
                train_module._resume_metadata(args, env_config)

    def test_success_metric_uses_entire_vector_batch(self):
        callback = train_module.InfoMetricsLogger(keywords=[])
        logger = mock.Mock()
        callback.model = SimpleNamespace(logger=logger)
        callback.num_timesteps = 2048
        callback.locals = {"infos": [
            {"is_success": np.bool_(index == 0), "episode": {"r": 1.0}}
            for index in range(2048)
        ]}
        with mock.patch.object(train_module.wandb, "log") as wandb_log:
            self.assertTrue(callback._on_step())
        expected = 1 / 2048
        logger.record.assert_called_once_with("rollout/full_batch_success", expected)
        wandb_log.assert_called_once_with({"rollout/full_batch_success": expected}, step=2048)
        self.assertEqual(sum(bool(info["is_success"]) for info in callback.locals["infos"][-100:]), 0)

    def test_seed_precedes_each_scene_and_reaches_wrappers_and_model(self):
        scratch = Path(tempfile.gettempdir())
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            with mock.patch.object(sys, "argv", [
                "train", "--mesh-path", "mesh.glb", "--grasp-path", "grasp_1_optimized.txt",
                "--actuation-dir", "keypoints", "--save-dir", temporary,
                "--seed", "41", "--num-actuate-steps", "80", "--eval-episodes", "32", "--press-hold-seconds", "1.0", "--log-std-init", "-2.0", "--stability-frame", "hand", "--total-timesteps", "1",
                "--no-eval-video",
            ]):
                args = train_module.parse_args()

            events = []

            class FakeVec:
                def __init__(self, name):
                    self.name = name

                def seed(self, value):
                    events.append(("seed", self.name, value))

                def close(self):
                    pass

            class FakeModel:
                def learn(self, **kwargs):
                    events.append(("learn",))

                def save(self, path):
                    pass

            def make_env(env_id, **kwargs):
                is_train = kwargs["num_envs"] == args.num_envs
                name, seed = ("train", args.seed) if is_train else ("eval", args.seed + 10000)
                expected_np = np.random.RandomState(seed).random_sample()
                expected_py = random.Random(seed).random()
                expected_torch = torch.rand(1, generator=torch.Generator().manual_seed(seed)).item()
                self.assertEqual(np.random.random(), expected_np)
                self.assertEqual(random.random(), expected_py)
                self.assertEqual(torch.rand(1).item(), expected_torch)
                self.assertEqual(kwargs["num_actuate_steps"], 80)
                self.assertEqual(kwargs["press_hold_seconds"], 1.0)
                self.assertEqual(kwargs["stability_frame"], "hand")
                events.append(("make", name))
                return SimpleNamespace(name=name)

            def make_model(*model_args, **kwargs):
                self.assertEqual(kwargs["seed"], 41)
                self.assertEqual(kwargs["learning_rate"], args.learning_rate)
                self.assertEqual(kwargs["policy_kwargs"], {"log_std_init": -2.0})
                events.append(("model",))
                return FakeModel()

            callback = lambda *args, **kwargs: object()
            def make_eval_callback(*args, **kwargs):
                self.assertEqual(kwargs["n_eval_episodes"], 32)
                events.append(("eval_callback",))
                return object()
            with mock.patch.object(train_module, "parse_args", return_value=args), \
                 mock.patch.object(train_module.wandb, "init", side_effect=lambda **kw: SimpleNamespace(config=SimpleNamespace(**kw["config"]))), \
                 mock.patch.object(train_module.gym, "make", side_effect=make_env), \
                 mock.patch.object(train_module, "ManiSkillVecMonitor", side_effect=lambda env, **kw: FakeVec(env.name)), \
                 mock.patch.object(train_module, "VecCheckNan", side_effect=lambda env, **kw: env), \
                 mock.patch.object(train_module, "EvalCallback", side_effect=make_eval_callback), \
                 mock.patch.object(train_module, "CheckpointCallback", side_effect=callback), \
                 mock.patch.object(train_module, "TqdmCallback", side_effect=callback), \
                 mock.patch.object(train_module, "WandbCallback", side_effect=callback), \
                 mock.patch.object(train_module, "InfoMetricsLogger", side_effect=callback), \
                 mock.patch.object(train_module, "CallbackList", side_effect=callback), \
                 mock.patch.object(train_module, "PPO", side_effect=make_model):
                train_module.main()

            self.assertEqual(events, [
                ("make", "train"), ("seed", "train", 41),
                ("make", "eval"), ("seed", "eval", 10041), ("eval_callback",),
                ("model",), ("learn",),
            ])
            config_path = next(Path(temporary).glob("*/config.yaml"))
            config = yaml.safe_load(config_path.read_text())
            self.assertEqual((config["seed"], config["eval_seed"]), (41, 10041))
            self.assertEqual(config["eval_episodes"], 32)
            self.assertEqual(config["env_config"]["num_actuate_steps"], 80)
            self.assertEqual(config["press_hold_seconds"], 1.0)
            self.assertEqual(config["log_std_init"], -2.0)
            self.assertEqual(config["env_config"]["press_hold_seconds"], 1.0)
            self.assertEqual(config["env_config"]["stability_frame"], "hand")

    def test_resume_loads_new_env_then_reseeds_and_learns_additional_steps(self):
        scratch = Path(tempfile.gettempdir())
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            with mock.patch.object(sys, "argv", [
                "train", "--mesh-path", "mesh.glb", "--grasp-path", "grasp_1_optimized.txt",
                "--actuation-dir", "keypoints", "--save-dir", temporary,
                "--resume-checkpoint", str(Path(temporary) / "prior/checkpoints/model.zip"),
                "--num-envs", "4096", "--total-timesteps", "8192", "--press-hold-seconds", "1.0", "--log-std-init", "-3.0",
                "--no-eval-video",
            ]):
                args = train_module.parse_args()
            events = []
            resume = {"checkpoint": args.resume_checkpoint, "initial_timesteps": 40960,
                      "target_timesteps": 49152, "old_num_envs": 2048, "new_num_envs": 4096,
                      "checkpoint_sha256": "a" * 64,
                      "log_std_init": -2.0,
                      "source_run_dir": str(Path(temporary) / "prior"),
                      "target_kl": args.target_kl}

            def preflight(parsed, env_config):
                self.assertEqual(env_config["press_hold_seconds"], 1.0)
                parsed.seed = 41  # Adopt the original run's saved seed.
                events.append(("preflight",))
                return resume

            class FakeVec:
                def __init__(self, name):
                    self.name = name

                def seed(self, value):
                    events.append(("seed", self.name, value))

                def close(self):
                    pass

            def make_env(env_id, **kwargs):
                name = "train" if kwargs["num_envs"] == 4096 else "eval"
                expected_seed = 41 if name == "train" else 10041
                self.assertEqual(np.random.random(), np.random.RandomState(expected_seed).random_sample())
                events.append(("make", name))
                return SimpleNamespace(name=name)

            class FakeModel:
                num_timesteps = 40960
                n_envs = 4096
                n_steps = 1
                batch_size = 2048
                target_kl = args.target_kl
                policy = SimpleNamespace(optimizer=SimpleNamespace(state={"step": 1}))
                _stats_window_size = 100
                ep_info_buffer = ["old"]
                ep_success_buffer = ["old"]
                tensorboard_log = "prior/tensorboard"

                def learn(self, **kwargs):
                    self.assert_empty_buffers_and_seeded(kwargs)

                def assert_empty_buffers_and_seeded(self, kwargs):
                    assert kwargs["total_timesteps"] == 8192
                    assert kwargs["reset_num_timesteps"] is False
                    assert len(self.ep_info_buffer) == len(self.ep_success_buffer) == 0
                    assert temporary in self.tensorboard_log
                    assert np.random.random() == np.random.RandomState(41).random_sample()
                    events.append(("learn",))

                def save(self, path):
                    events.append(("save",))

            model = FakeModel()

            def load_checkpoint(path, **kwargs):
                self.assertEqual(path, resume["checkpoint"])
                self.assertEqual(kwargs["env"].name, "train")
                self.assertEqual((kwargs["device"], kwargs["force_reset"]), ("cuda", True))
                self.assertNotIn("policy_kwargs", kwargs)
                events.append(("load",))
                np.random.random()  # Loading advances RNG; main must reseed afterwards.
                return model

            callback = lambda *a, **kw: object()
            with mock.patch.object(train_module, "parse_args", return_value=args), \
                 mock.patch.object(train_module, "_resume_metadata", side_effect=preflight), \
                 mock.patch.object(train_module.wandb, "init", side_effect=lambda **kw: SimpleNamespace(config=SimpleNamespace(**kw["config"]))), \
                 mock.patch.object(train_module.gym, "make", side_effect=make_env), \
                 mock.patch.object(train_module, "ManiSkillVecMonitor", side_effect=lambda env, **kw: FakeVec(env.name)), \
                 mock.patch.object(train_module, "VecCheckNan", side_effect=lambda env, **kw: env), \
                 mock.patch.object(train_module, "EvalCallback", side_effect=callback), \
                 mock.patch.object(train_module, "CheckpointCallback", side_effect=callback), \
                 mock.patch.object(train_module, "TqdmCallback", side_effect=callback), \
                 mock.patch.object(train_module, "WandbCallback", side_effect=callback), \
                 mock.patch.object(train_module, "InfoMetricsLogger", side_effect=callback), \
                 mock.patch.object(train_module, "CallbackList", side_effect=callback), \
                 mock.patch.object(train_module.PPO, "load", side_effect=load_checkpoint):
                train_module.main()

            self.assertEqual(events, [
                ("preflight",), ("make", "train"), ("seed", "train", 41),
                ("make", "eval"), ("seed", "eval", 10041),
                ("load",), ("seed", "train", 41), ("seed", "eval", 10041),
                ("learn",), ("save",),
            ])
            config = yaml.safe_load(next(Path(temporary).glob("*-grasp_1/config.yaml")).read_text())
            self.assertEqual(config["resume"]["target_timesteps"], 49152)
            self.assertEqual(config["seed"], 41)
            self.assertEqual(config["log_std_init"], -2.0)


if __name__ == "__main__":
    unittest.main()
