import os
import argparse
import datetime
import hashlib
import json
import random
import zipfile
from collections import deque
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecCheckNan
import wandb
from wandb.integration.sb3 import WandbCallback
from codex.codex_train.utils.custom_record import CustomRecordEpisode
from codex.codex_train.utils.vecmonitor import ManiSkillVecMonitor
from codex.codex_train.configs import primitive as P
import codex.codex_train.envs  # noqa: F401  (registers the environments)


def _seed_global_rngs(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resume_metadata(args, env_config):
    """Check a PPO checkpoint and its run config before constructing a new scene."""
    checkpoint = Path(args.resume_checkpoint).expanduser().resolve()
    source_run_dir = checkpoint.parent.parent if checkpoint.parent.name in {
        "checkpoints", "best_model", "models"
    } else checkpoint.parent
    source_config_path = source_run_dir / "config.yaml"
    with source_config_path.open() as stream:
        source = yaml.safe_load(stream)
    if args.algorithm.upper() != "PPO" or source.get("algorithm", "").upper() != "PPO":
        raise ValueError("--resume-checkpoint currently supports PPO only")
    if source.get("seed") is None:
        raise ValueError("Resume requires a saved training seed")
    if args.seed is None:
        args.seed = int(source["seed"])
    elif args.seed != source["seed"]:
        raise ValueError("Resume seed differs from saved training seed")
    requested = {
        "env_id": args.env_id, "policy": args.policy, "gamma": args.gamma,
        "gae_lambda": args.gae_lambda, "ent_coef": args.ent_coef,
        "n_steps": args.n_steps, "batch_size": args.batch_size,
        "n_epochs": args.n_epochs, "learning_rate": args.learning_rate,
        "press_hold_seconds": args.press_hold_seconds,
        "target_kl": args.target_kl or None,
    }
    mismatches = [key for key, value in requested.items()
                  if (source.get(key) or None if key == "target_kl" else source.get(key)) != value]
    # Compare the representation written to config.yaml: tuple ranges become lists.
    if source.get("env_config") != yaml.safe_load(yaml.safe_dump(env_config)):
        mismatches.append("env_config")
    old_num_envs = source.get("num_envs")
    if not isinstance(old_num_envs, int) or old_num_envs <= 0:
        mismatches.append("num_envs")
    if mismatches:
        raise ValueError(f"Resume settings differ from saved run: {', '.join(mismatches)}")
    with zipfile.ZipFile(checkpoint) as archive:
        if "policy.optimizer.pth" not in archive.namelist():
            raise ValueError("Resume checkpoint lacks PPO optimizer state")
        saved = json.loads(archive.read("data"))
    checkpoint_keys = ("gamma", "gae_lambda", "ent_coef", "n_steps",
                       "batch_size", "n_epochs", "learning_rate", "target_kl")
    wrong_checkpoint = [key for key in checkpoint_keys if saved.get(key) != requested[key]]
    if saved.get("use_sde") is not True or saved.get("sde_sample_freq") != 4:
        wrong_checkpoint.append("gSDE settings")
    if saved.get("seed") != args.seed:
        wrong_checkpoint.append("seed")
    if saved.get("n_envs") != old_num_envs:
        wrong_checkpoint.append("n_envs")
    if wrong_checkpoint:
        raise ValueError(f"Checkpoint PPO settings differ from requested run: {', '.join(wrong_checkpoint)}")
    initial = int(saved["num_timesteps"])
    if initial < 0 or args.total_timesteps <= 0:
        raise ValueError("Resume requires positive additional timesteps and a valid checkpoint counter")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "source_run_dir": str(source_run_dir),
        "source_config_sha256": _sha256_file(source_config_path),
        "initial_timesteps": initial,
        "additional_timesteps": args.total_timesteps,
        "target_timesteps": initial + args.total_timesteps,
        "old_num_envs": old_num_envs,
        "new_num_envs": args.num_envs,
        "log_std_init": source.get("log_std_init", 0.0),
        "target_kl": requested["target_kl"],
    }

# --- Custom Tqdm Callback for progress bar ---
class TqdmCallback(BaseCallback):
    """Callback that displays a tqdm progress bar during training."""

    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        self.pbar.n = self.model.num_timesteps
        self.pbar.refresh()
        return True

    def _on_training_end(self):
        self.pbar.close()

class InfoMetricsLogger(BaseCallback):
    """Callback that logs per-episode info metrics to WandB and the SB3 logger."""

    def __init__(self, keywords, verbose=0):
        super().__init__(verbose)
        self.keywords = keywords

    def _on_step(self):
        infos = self.locals["infos"]       # list of dicts, one per env
        # SB3's rollout/success_rate retains only its last 100 episodes. This
        # one-step vector task finishes all environments together, so report
        # the complete training batch under a distinct name.
        successes = [bool(inf["is_success"]) for inf in infos if "is_success" in inf]
        if successes:
            full_batch_success = sum(successes) / len(successes)
            self.logger.record("rollout/full_batch_success", full_batch_success)
            wandb.log({"rollout/full_batch_success": full_batch_success}, step=self.num_timesteps)
        for k in self.keywords:
            vals = [inf["episode"][k] for inf in infos
                    if "episode" in inf and k in inf["episode"]]
            if vals:
                self.logger.record(f"rollout/{k}", sum(vals)/len(vals))
                wandb.log({k: sum(vals)/len(vals)}, step=self.num_timesteps)
        return True

def parse_args():
    """
    Parse command line arguments to override default parameters.
    """
    parser = argparse.ArgumentParser(
        description="Training script for grasp lift environment with command line overrides."
    )
    # General training parameters
    parser.add_argument("--env-id", dest="env_id", type=str, default="PrimitiveGraspForceEnv-v1", help="Environment ID to use")
    parser.add_argument("--total-timesteps", dest="total_timesteps", type=int, default=1_000_000, help="Total timesteps for training")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"], help="Algorithm to use (PPO or SAC)")
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Policy type to use")
    parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor gamma")
    parser.add_argument("--gae-lambda", dest="gae_lambda", type=float, default=0.9, help="GAE lambda")
    parser.add_argument("--ent-coef", dest="ent_coef", type=float, default=0.0,
                        help="Entropy coefficient (0: an entropy bonus keeps widening the exploration noise until "
                             "sampled episodes fail and the policy drifts off the grasp)")
    parser.add_argument("--target-kl", dest="target_kl", type=float, default=0.02,
                        help="Stop a PPO update early once the policy moves this far (approximate KL); "
                             "0 disables")
    parser.add_argument("--goal", choices=("scene", "lift"), default="scene",
                        help="scene: carry the object to the scene's stage-2 goal pose before pressing, as in "
                             "evaluation (lift when the scene has no goal); lift: lift it 30 cm straight up")
    parser.add_argument("--goal-method", dest="goal_method", default=None,
                        help="Stage-2 result to use (default vlm_cem_dir, else vlm_cem)")
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=1, help="Number of steps (for PPO)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--n-epochs", dest="n_epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--num-envs", dest="num_envs", type=int, default=2048, help="Number of parallel training environments")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Resume PPO weights and optimizer from a checkpoint; --total-timesteps is additional work")
    parser.add_argument("--seed", type=int, default=None, help="Training seed; evaluation uses seed + 10000. Omit for unseeded runs.")
    parser.add_argument("--eval-freq", dest="eval_freq", type=int, default=10, help="Evaluation frequency (in training rollouts)")
    parser.add_argument("--eval-episodes", type=int, default=32, help="Number of episodes scored per evaluation (default: all 32 evaluation environments)")
    parser.add_argument("--no-eval-video", dest="no_eval_video", action="store_true", default=False, help="Skip video recording during eval (avoids Vulkan contention with training envs)")
    parser.add_argument("--checkpoint-freq", dest="checkpoint_freq", type=int, default=20, help="Checkpoint frequency")
    
    # SAC-specific parameters
    parser.add_argument("--buffer-size", dest="buffer_size", type=int, default=2_000_000, help="Buffer size for SAC")
    parser.add_argument("--learning-starts", dest="learning_starts", type=int, default=10_000, help="Learning starts for SAC")
    parser.add_argument("--train-freq", dest="train_freq", type=int, default=1, help="Train frequency for SAC")
    parser.add_argument("--gradient-steps", dest="gradient_steps", type=int, default=1, help="Gradient steps for SAC")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=3e-4, help="Learning rate for PPO or SAC")
    parser.add_argument("--log-std-init", type=float, default=-3.0,
                        help="Initial PPO gSDE log standard deviation for a fresh model. Small (-3) so that "
                             "sampled episodes still succeed often enough to learn from; resumed models keep "
                             "checkpoint weights")
    
    # Run and logging parameters
    parser.add_argument("--run-group", dest="run_group", type=str, default="codex", help="Weights & Biases run group")
    parser.add_argument("--run-name", dest="run_name", type=str, default=None, help="Weights & Biases run name")

    # Task parameters
    parser.add_argument("--scene-dir", dest="scene_dir", type=str, default=None, help="Path to the scene directory. When provided, mesh-path, grasp-path, and actuation-dir are auto-discovered.")
    parser.add_argument("--mesh-path", dest="mesh_path", type=str, default=None, help="Path to mesh file (e.g. out/scenes/<scene>/mesh/mesh.glb). Overrides --scene-dir.")
    parser.add_argument("--save-dir", dest="save_dir", type=str, default=None, help="Root directory for RL run outputs. Defaults to <scene_dir>/rl_runs/")
    parser.add_argument("--grasp-path", dest="grasp_path", type=str, default=None, help="Path to grasp config file (e.g. <scene_dir>/grasps/grasp_1_optimized.txt). Defaults to the first *_optimized.txt found in <scene_dir>/grasps/.")
    parser.add_argument("--actuation-dir", dest="actuation_dir", type=str, default=None, help="Directory containing fn_key_pts.json. Overrides --scene-dir.")
    parser.add_argument("--standing", action="store_true", help="Randomize the object's yaw at every reset")
    parser.add_argument("--fixed-rot", dest="fixed_rot", action="store_true", help="Lock the object's rotation until it is grasped")
    parser.add_argument("--delta-pos-scale", dest="delta_pos_scale", type=float, default=P.DELTA_POS_SCALE, help="Scale for hand position residuals [m]")
    parser.add_argument("--delta-rot-scale", dest="delta_rot_scale", type=float, default=P.DELTA_ROT_SCALE, help="Scale for hand rotation residuals [rad]")
    parser.add_argument("--delta-finger-scale", dest="delta_finger_scale", type=float, default=P.DELTA_FINGER_SCALE, help="Scale for finger joint residuals [rad]")
    parser.add_argument("--num-actuate-steps", type=int, default=P.NUM_ACTUATE_STEPS, help="Controller steps driving the final press")
    parser.add_argument("--press-hold-seconds", type=float, default=P.PRESS_HOLD_SECONDS,
                        help="Success requires a continuous fingertip press over this final window (0 = endpoint check only)")
    parser.add_argument("--stability-frame", choices=("world", "hand"), default=None,
                        help="Frame of the palm-to-actuation-point vector used by the stability term (default: world)")

    return parser.parse_args()

def main():
    """Entry point for training: parses args, constructs envs, and runs the RL training loop."""
    args = parse_args()

    # Resolve paths, using --scene-dir as the canonical source when individual paths are not given
    scene_dir = os.path.abspath(args.scene_dir) if args.scene_dir else None
    save_dir = args.save_dir or (os.path.join(scene_dir, "rl_runs") if scene_dir else None)
    if save_dir is None:
        raise ValueError("--save-dir is required when training with individual paths instead of --scene-dir")

    if args.mesh_path:
        mesh_path = args.mesh_path
    elif scene_dir:
        mesh_path = os.path.join(scene_dir, "mesh_coacd", "mesh.glb")
    else:
        raise ValueError("Either --mesh-path or --scene-dir must be provided.")

    if args.actuation_dir:
        actuation_dir = args.actuation_dir
    elif scene_dir:
        actuation_dir = os.path.join(scene_dir, "keypoints")
    else:
        raise ValueError("Either --actuation-dir or --scene-dir must be provided.")

    # Resolve grasp path: use the provided path, or auto-discover from <scene_dir>/grasps/
    if args.grasp_path:
        grasp_path = args.grasp_path
    elif scene_dir:
        grasps_folder = os.path.join(scene_dir, "grasps")
        candidates = sorted([
            f for f in os.listdir(grasps_folder)
            if f.endswith("_optimized.txt")
        ])
        if not candidates:
            raise ValueError(f"No *_optimized.txt grasp files found in {grasps_folder}")
        grasp_path = os.path.join(grasps_folder, candidates[0])
    else:
        raise ValueError("Either --grasp-path or --scene-dir must be provided.")

    # Derive grasp_configs and directory paths from file paths
    mesh_ids = os.path.splitext(os.path.basename(mesh_path))[0]
    grasp_configs = os.path.splitext(os.path.basename(grasp_path))[0]
    mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
    grasp_dir = os.path.dirname(os.path.abspath(grasp_path))

    # Generate a unique run ID and directory based on the run_name argument
    if args.run_name is None:
        prefix = os.path.basename(scene_dir) if scene_dir else mesh_ids
        args.run_name = f"{prefix}-{grasp_configs.removesuffix('_optimized')}"
    print(f"Using grasp: {grasp_path}")

    # Environment configuration (this remains static; you can similarly extend argparse to override these if needed)
    env_config = {
        **P.PRIMITIVE_ENV_CONFIG,
        "grasp_configs": [grasp_configs],
        "num_actuate_steps": args.num_actuate_steps,
        "init_at_standing_pose": args.standing,
        "init_at_fixed_rot": args.fixed_rot,
        "delta_pos_scale": args.delta_pos_scale,
        "delta_rot_scale": args.delta_rot_scale,
        "delta_finger_scale": args.delta_finger_scale,
        "mesh_dir": mesh_dir,
        "grasp_dir": grasp_dir,
        "actuation_dir": actuation_dir,
    }
    if args.env_id == "PrimitiveGraspForceEnv-v1":
        env_config["press_hold_seconds"] = args.press_hold_seconds
        if args.stability_frame is not None:
            env_config["stability_frame"] = args.stability_frame
    elif args.press_hold_seconds:
        raise ValueError("--press-hold-seconds requires PrimitiveGraspForceEnv-v1")
    elif args.stability_frame is not None:
        raise ValueError("--stability-frame requires PrimitiveGraspForceEnv-v1")
    goal = None
    if args.goal == "scene" and scene_dir:
        from codex.codex_train.eval import goal_pose_from_scene
        goal_pose, goal = goal_pose_from_scene(scene_dir, mesh_dir, args.goal_method)
        if goal_pose is not None:
            env_config["goal_pose"] = goal_pose.tolist()
            # The carry turns the hand: judge slip in the hand's frame.
            env_config.setdefault("stability_frame", "hand")
            print(f"Goal: {goal['source']}")
        else:
            print(f"Goal: lift 30 cm ({goal})")
            goal = None

    resume = _resume_metadata(args, env_config) if args.resume_checkpoint else None
    run_id = datetime.datetime.now().strftime("%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"{run_id}-{args.run_name}")
    if resume and Path(run_dir).resolve() == Path(resume["source_run_dir"]):
        raise ValueError("Resume output directory must differ from the source run")
    os.makedirs(run_dir, exist_ok=not bool(resume))

    # Initialize Weights & Biases with parameters overridden by command line
    # Save the full run configuration next to the run so evaluation can
    # reconstruct the environment without relying on W&B internals.
    run_config = {
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "algorithm": args.algorithm,
        "policy": args.policy,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "target_kl": args.target_kl or None,
        "goal": goal["source"] if goal else "lift",
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "press_hold_seconds": args.press_hold_seconds,
        "checkpoint_freq": args.checkpoint_freq,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "learning_rate": args.learning_rate,
        "log_std_init": resume["log_std_init"] if resume else args.log_std_init,
        "num_envs": args.num_envs,
        "resume": resume,
        "seed": args.seed,
        "eval_seed": args.seed + 10000 if args.seed is not None else None,
        "env_config": env_config,
    }
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(run_config, f)

    # W&B entity/mode come from the user's own configuration
    # (WANDB_ENTITY / WANDB_MODE env vars; use WANDB_MODE=offline to train
    # without a W&B account).
    run = wandb.init(
        project="codex",
        name=args.run_name,
        group=args.run_group,
        id=run_id,
        dir=run_dir,
        config=run_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=False,
    )

    info_keywords = [
        "lift_reward",
        "lift_success",
        "stable_reward",
        "dist_change",
        "upright_reward",
        "stable_success",
        "act_reward",
        "act_success",
        "act_dist",
        "act_dir_dist",
        "act_force",
        "est_collision"
    ]
    if args.press_hold_seconds:
        info_keywords += [
            "held_press_success", "press_contact_fraction", "press_duration_s",
            "press_selected_force", "press_min_selected_force", "legacy_act_success",
        ]

    # Friction is sampled during scene construction, before an env's reset.
    _seed_global_rngs(args.seed)

    # --- Create the parallelized training environment ---
    train_env = gym.make(
        args.env_id,
        **env_config,
        num_envs=args.num_envs,
    )
    # Wrap it with ManiSkillVecMonitor for SB3 compatibility
    train_env = ManiSkillVecMonitor(train_env, info_keywords=info_keywords)
    train_env = VecCheckNan(train_env, raise_exception=True)
    if args.seed is not None:
        train_env.seed(args.seed)
    
    # --- Create the evaluation environment ---
    eval_seed = args.seed + 10000 if args.seed is not None else None
    _seed_global_rngs(eval_seed)
    eval_env = gym.make(
        args.env_id,
        **env_config,
        num_envs=32,
        render_mode="rgb_array"
    )
    # Add recording wrapper (omit when training env holds all Vulkan contexts)
    if not args.no_eval_video:
        eval_env = CustomRecordEpisode(
            eval_env,
            output_dir=f'{run_dir}/eval_videos',
            save_trajectory=False,
            save_video=True,
            trajectory_name="eval_trajectory",
            max_steps_per_video=1,
            video_fps=120,
            render_substeps=True,
            max_render_envs=9,
            parallel_videos=False,
            info_keys=info_keywords,
        )
    eval_env = ManiSkillVecMonitor(eval_env, info_keywords=info_keywords)
    eval_env = VecCheckNan(eval_env, raise_exception=True)
    if eval_seed is not None:
        eval_env.seed(eval_seed)
    
    # --- Define callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{run_dir}/best_model',
        log_path=f'{run_dir}/results',
        eval_freq=run.config.eval_freq,
        n_eval_episodes=run.config.eval_episodes,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=run.config.checkpoint_freq,
        save_path=f'{run_dir}/checkpoints',
        name_prefix='grasp_lift_model'
    )
    progress_callback = TqdmCallback(total_timesteps=resume["target_timesteps"] if resume else run.config.total_timesteps)
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f'{run_dir}/models/',
        verbose=2,
    )
    info_metrics_callback = InfoMetricsLogger(
        keywords=info_keywords,
        verbose=1
    )
    callback = CallbackList([info_metrics_callback, wandb_callback, eval_callback, checkpoint_callback, progress_callback])

    # --- Create the model based on the chosen algorithm ---
    if resume:
        model = PPO.load(resume["checkpoint"], env=train_env, device="cuda", force_reset=True)
        if model.num_timesteps != resume["initial_timesteps"]:
            raise RuntimeError("Loaded PPO timestep counter differs from checkpoint provenance")
        if model.n_envs != args.num_envs or model.n_steps != args.n_steps or model.batch_size != args.batch_size:
            raise RuntimeError("Loaded PPO rollout settings differ from requested continuation")
        if model.target_kl != resume["target_kl"]:
            raise RuntimeError("Loaded PPO target_kl differs from requested continuation")
        # SB3 saves the first rollout checkpoint before its first optimizer
        # update; later checkpoints must carry populated optimizer moments.
        if (model.num_timesteps > resume["old_num_envs"]
                and not model.policy.optimizer.state):
            raise RuntimeError("Loaded PPO optimizer state is empty")
        # SB3 saves old episode statistics and TensorBoard path. Keep weights,
        # optimizer, and PPO settings, but start logging this continuation anew.
        model.ep_info_buffer = deque(maxlen=model._stats_window_size)
        model.ep_success_buffer = deque(maxlen=model._stats_window_size)
        model.tensorboard_log = f'{run_dir}/ppo_tensorboard/'
        _seed_global_rngs(args.seed)  # PPO.load initializes a new policy and advances RNGs.
        train_env.seed(args.seed)
        eval_env.seed(eval_seed)
    elif run.config.algorithm.upper() == "PPO":
        model = PPO(
            run.config.policy,
            train_env,
            gamma=run.config.gamma,
            gae_lambda=run.config.gae_lambda,
            learning_rate=run.config.learning_rate,
            ent_coef=run.config.ent_coef,
            target_kl=args.target_kl or None,
            n_steps=run.config.n_steps,
            batch_size=run.config.batch_size,
            n_epochs=run.config.n_epochs,
            seed=args.seed,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs={"log_std_init": run.config.log_std_init},
            verbose=1,
            tensorboard_log=f'{run_dir}/ppo_tensorboard/',
        )
    elif run.config.algorithm.upper() == "SAC":
        model = SAC(
            run.config.policy,
            train_env,
            gamma=run.config.gamma,
            learning_rate=run.config.learning_rate,
            buffer_size=run.config.buffer_size,
            learning_starts=run.config.learning_starts,
            train_freq=run.config.train_freq,
            gradient_steps=run.config.gradient_steps,
            seed=args.seed,
            verbose=1,
            tensorboard_log=f'{run_dir}/sac_tensorboard/'
        )
    else:
        raise ValueError(f"Unsupported algorithm: {run.config.algorithm}")

    # Train the model
    if resume:
        model.learn(total_timesteps=run.config.total_timesteps, callback=callback,
                    reset_num_timesteps=False)
    else:
        model.learn(total_timesteps=run.config.total_timesteps, callback=callback)

    # Save the final model
    model.save(os.path.join(run_dir, "refinement_model"))
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
