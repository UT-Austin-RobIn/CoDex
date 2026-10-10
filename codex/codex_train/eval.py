"""Evaluate a trained policy (or a grasp alone) on the grasp-move-actuate task.

All episodes run in parallel GPU environments under the training physics
(randomized friction, mass and size, hand pose noise). The policy acts deterministically
by default. Saved policies keep their training goal and stability frame; --goal scene
retargets to the current stage-2 goal, and --goal lift selects a plain lift. Results are
written to <output-dir>/metrics.json (summary plus
one row per episode); with --video-episodes N > 0, the first N episodes are
also rendered into <output-dir>/rollout.mp4, each labelled with its outcome.

    # latest run of a scene, the grasp it was trained on
    python scripts/5_evaluate_policy.py --scene-dir out/scenes/demo
    # a specific checkpoint
    python scripts/5_evaluate_policy.py --model-path out/scenes/demo/rl_runs/<run>/best_model/best_model.zip
    # the generated grasp with zero policy action (no training needed)
    python scripts/5_evaluate_policy.py --scene-dir out/scenes/demo --zero-action \\
        --grasp-path out/scenes/demo/grasps/grasp_1_optimized.txt
"""

import argparse
import json
import os
import time
import trimesh
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO, SAC

import codex.codex_train.envs  # noqa: F401  (registers the environments)
from codex.codex_train.configs import primitive as P

ENV_ID = "PrimitiveGraspForceEnv-v1"
EPISODE_KEYS = ("success", "lift_success", "goal_success", "goal_pos_err", "goal_rot_err",
                "stable_success", "act_success", "upright_success",
                "held_press_success", "est_collision", "act_dist", "act_force")
SUMMARY_KEYS = ("success", "lift_success", "goal_success", "stable_success", "act_success", "upright_success")
GOAL_METHODS = ("vlm_cem_dir", "vlm_cem")


# ---------------------------------------------------------------- discovery

def infer_model_path_from_scene(scene_dir):
    """Model of the most recent run under <scene>/rl_runs that saved one
    (best_model, then the newest checkpoint, then the final model)."""
    runs_root = Path(scene_dir).expanduser().resolve() / "rl_runs"
    runs = [p for p in runs_root.iterdir() if p.is_dir()] if runs_root.is_dir() else []
    for run_dir in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True):
        model = find_model_in_run(run_dir)
        if model is not None:
            return str(model)
    raise FileNotFoundError(f"No trained model found under {runs_root}")


def find_model_in_run(run_dir):
    run_dir = Path(run_dir)
    best = sorted((run_dir / "best_model").glob("*.zip"), key=lambda p: p.stat().st_mtime)
    ckpts = sorted((run_dir / "checkpoints").glob("*.zip"), key=lambda p: p.stat().st_mtime)
    final = run_dir / "refinement_model.zip"
    if best:
        return best[-1]
    if ckpts:
        return ckpts[-1]
    return final if final.is_file() else None


def run_dir_of(model_path):
    model_dir = Path(model_path).resolve().parent
    return model_dir.parent if model_dir.name in {"best_model", "checkpoints", "models"} else model_dir


def load_run_config(run_dir):
    path = Path(run_dir) / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; evaluation needs the config written by training")
    with open(path) as f:
        return yaml.safe_load(f)


def load_trained_model(model_path, algorithm, device):
    classes = {"PPO": PPO, "SAC": SAC}
    if algorithm not in classes:
        raise ValueError(f"Unsupported saved algorithm: {algorithm}")
    return classes[algorithm].load(model_path, device=device)


def build_env_config(run_config=None, scene_dir=None, grasp_path=None):
    """Environment config: the run's saved config when evaluating a policy,
    otherwise the shared primitive config for the scene. An explicit grasp
    replaces the one the run was trained with."""
    if run_config is not None:
        ec = dict(run_config["env_config"])
    else:
        if scene_dir is None:
            raise ValueError("--zero-action without a run needs --scene-dir")
        grasp_path = grasp_path or Path(scene_dir) / "grasps" / "grasp_1_optimized.txt"
        ec = dict(P.PRIMITIVE_ENV_CONFIG, num_actuate_steps=P.NUM_ACTUATE_STEPS,
                  press_hold_seconds=P.PRESS_HOLD_SECONDS, delta_pos_scale=P.DELTA_POS_SCALE,
                  delta_rot_scale=P.DELTA_ROT_SCALE, delta_finger_scale=P.DELTA_FINGER_SCALE)
    if scene_dir is not None:
        scene_dir = Path(scene_dir).resolve()
        # Scene paths win when the run's recorded paths no longer exist
        # (e.g. the scene directory was moved or copied).
        for key, sub in (("mesh_dir", "mesh_coacd"), ("actuation_dir", "keypoints"), ("grasp_dir", "grasps")):
            if not ec.get(key) or not os.path.isdir(ec[key]):
                ec[key] = str(scene_dir / sub)
    if grasp_path is not None:
        grasp_path = Path(grasp_path).resolve()
        ec["grasp_dir"] = str(grasp_path.parent)
        ec["grasp_configs"] = [grasp_path.stem]
    ec.pop("num_envs", None)
    grasp_file = Path(ec["grasp_dir"]) / f"{ec['grasp_configs'][0]}.txt"
    if not grasp_file.is_file():
        raise FileNotFoundError(f"Grasp {grasp_file} not found; pass --grasp-path")
    return ec


# ---------------------------------------------------------------- rollout

class _FrameGrabber:
    """Renders the first `k` environments every few simulation steps."""

    def __init__(self, base_env, k, every=8):
        self.base, self.k, self.every, self.frames, self.tick = base_env, k, every, [], 0
        self._orig = base_env._after_simulation_step
        base_env._after_simulation_step = self

    def __call__(self):
        self._orig()
        self.tick += 1
        if self.tick % self.every == 0:
            img = self.base.render()
            img = img.cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
            self.frames.append(img.reshape(-1, *img.shape[-3:])[:self.k].astype(np.uint8))


OUTCOMES = {"success": ("Success", (46, 139, 87)), "press_missed": ("Held, press missed", (217, 162, 27)),
            "not_held": ("Not held", (200, 64, 64))}


def outcome(row):
    """success; press_missed (the functional object reached the goal, or was lifted, but the press
    failed); not_held (dropped, or never carried there)."""
    if row["success"]:
        return "success"
    held = row["goal_success"] if "goal_success" in row else row["object_height_change_m"] > 0.1
    return "press_missed" if held else "not_held"


# ---------------------------------------------------------------- goal

def goal_pose_from_scene(scene_dir, mesh_dir, method=None):
    """The functional object's goal from stage 2, for carrying it there in evaluation.

    Uses the result of `method`, by default the first of GOAL_METHODS the
    scene has. Returns (goal_pose, info), or (None, reason) when the scene has
    no goal. goal_pose is the goal relative to the functional object standing upright (as
    the simulator spawns it) where it was captured; heights and orientations
    stay as searched.
    """
    import trimesh
    found = [Path(scene_dir) / "goal_gen" / m / "best_candidate.json" for m in ([method] if method else GOAL_METHODS)]
    found = [f for f in found if f.is_file()]
    if not found:
        return None, "no goal pose"
    path = found[0]
    with open(path) as f:
        g = json.load(f)
    if "transform_world" not in g or "relative_transform" not in g:
        raise ValueError(f"{path} has no goal pose; re-run stage 2 for this scene")
    T_goal = np.asarray(g["transform_world"], dtype=float)
    T_captured = T_goal @ np.linalg.inv(np.asarray(g["relative_transform"], dtype=float))
    mesh = trimesh.load(Path(mesh_dir) / "mesh.glb", force="mesh")
    spawn = np.eye(4)
    spawn[2, 3] = -float(mesh.vertices[:, 2].min())  # standing on the table at the origin
    start = spawn.copy()
    start[:2, 3] = T_captured[:2, 3]                 # ... where the functional object was captured
    G = np.linalg.inv(start) @ T_goal
    at_goal = spawn @ G                              # the goal as the simulator places it
    lowest = float(((at_goal[:3, :3] @ mesh.vertices.T)[2] + at_goal[2, 3]).min())
    if lowest < -0.005:
        raise ValueError(f"The goal in {path} puts the functional object {-lowest * 100:.1f} cm into the table; "
                         "search the goal again (stage 2).")
    return G, {"source": str(path), "position_m": at_goal[:3, 3].tolist(), "suitable": g.get("suitable")}


def write_video(frames, rows, path):
    """Episodes in a grid (two per row), each above a thin band naming its outcome."""
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    k, H, W = frames[0].shape[:3]
    band, gap = 30, 4
    cols = min(k, 2)
    rows_n = -(-k // cols)
    font = ImageFont.load_default(size=16)
    size = (cols * W + (cols - 1) * gap, rows_n * (H + band) + (rows_n - 1) * gap)
    base = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(base)
    cells = [((i % cols) * (W + gap), (i // cols) * (H + band + gap)) for i in range(k)]
    for i, (x, y) in enumerate(cells):
        text, color = OUTCOMES[outcome(rows[i])]
        draw.rectangle([x, y + H + band - 4, x + W - 1, y + H + band], fill=color)
        draw.ellipse([x + 10, y + H + 8, x + 20, y + H + 18], fill=color)
        draw.text((x + 28, y + H + 4), text, fill=(29, 39, 51), font=font)
    with imageio.get_writer(path, fps=30, macro_block_size=1) as writer:
        for fr in frames:
            canvas = base.copy()
            for i, (x, y) in enumerate(cells):
                canvas.paste(Image.fromarray(fr[i]), (x, y))
            writer.append_data(np.asarray(canvas))


def side_camera(grasp_file, goal=None, size=480, fov=0.8):
    """Render camera looking down on the approach from the side, so neither
    hand nor object hides the other. It is fitted to where the motion goes:
    the functional object, the grasping hand, the top of the 30 cm lift or, with `goal` (a
    position), the whole carry to the goal."""
    from mani_skill.utils import sapien_utils
    from scipy.spatial.transform import Rotation
    g = np.loadtxt(grasp_file)
    # The middle of the hand: its root (the grasp position) is at the wrist,
    # the palm about 10 cm along the root's z, the fingers beyond.
    hand = g[4:7] + Rotation.from_quat(np.r_[g[1:4], g[0]]).apply([0.04, 0.02, 0.1])
    points = [np.zeros(3), hand, hand + [0.0, 0.0, 0.3]]
    if goal is not None:
        goal = np.asarray(goal, dtype=float)
        points += [goal, goal + hand]
    points = np.array(points)
    az, elevation = np.arctan2(g[5], g[4]) + np.deg2rad(75), np.deg2rad(45)
    back = np.array([np.cos(az) * np.cos(elevation), np.sin(az) * np.cos(elevation), np.sin(elevation)])
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    center = (points.min(0) + points.max(0)) / 2
    # Back off until every point, with room for the hand around it, is in
    # view (nearer points need more room). The approach starts well outside.
    o = points - center
    across = np.maximum(np.abs(o @ right), np.abs(o @ up)) + 0.12
    eye = center + back * float((across / np.tan(fov / 2) + o @ back).max())
    return dict(pose=sapien_utils.look_at(eye=eye.tolist(), target=center.tolist()), width=size, height=size, fov=fov)


def evaluate(env_config, model=None, episodes=32, seed=0, deterministic=True, video_episodes=4,
             video_path=None, env_id=ENV_ID, camera=None):
    """Run `episodes` parallel episodes; returns (summary, rows)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    kwargs = dict(env_config, num_envs=episodes, sim_backend="physx_cuda")
    if video_episodes:
        kwargs["render_mode"] = "rgb_array"
        if camera is not None:
            kwargs["human_render_camera_configs"] = camera
    env = gym.make(env_id, **kwargs)
    base = env.unwrapped
    try:
        grab = _FrameGrabber(base, min(video_episodes, episodes)) if video_episodes else None
        obs, _ = env.reset(seed=seed)
        if model is None:
            action = torch.zeros((episodes, *env.single_action_space.shape), device=base.device)
        else:
            o = obs.cpu().numpy() if torch.is_tensor(obs) else obs
            action, _ = model.predict(o, deterministic=deterministic)
            action = torch.as_tensor(action, device=base.device, dtype=torch.float32)
        _, reward, _, _, info = env.step(action)
        height = (info["final_obj_pose"].p[:, 2] - info["init_obj_pose"].p[:, 2]).cpu().numpy()
        rows = []
        for i in range(episodes):
            row = {"episode": i, "reward": float(reward[i]), "object_height_change_m": float(height[i]),
                   **{k: float(v[i]) for k, v in base.physics.items()}}
            for key in EPISODE_KEYS:
                if key in info:
                    v = info[key].reshape(-1)[i]
                    row[key] = bool(v) if v.dtype == torch.bool else float(v)
            row["outcome"] = outcome(row)
            rows.append(row)
        if grab is not None and video_path is not None and grab.frames:
            write_video(grab.frames, rows, video_path)
    finally:
        env.close()
    keys = [k for k in SUMMARY_KEYS if k in rows[0] and not (k == "lift_success" and "goal_success" in rows[0])]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["episodes"] = episodes
    return summary, rows


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a trained policy or a grasp (pipeline stage 5).")
    p.add_argument("--scene-dir", help="Scene directory (latest run is used unless --run-dir/--model-path).")
    p.add_argument("--run-dir", help="Training run directory (rl_runs/<run>).")
    p.add_argument("--model-path", help="Policy checkpoint (.zip).")
    p.add_argument("--grasp-path", help="Grasp file; default: the grasp the run was trained with.")
    p.add_argument("--zero-action", action="store_true", help="Evaluate the grasp alone (no policy).")
    p.add_argument("--episodes", type=int, default=32, help="Parallel episodes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true", help="Sample policy actions instead of the mean.")
    p.add_argument("--video-episodes", type=int, default=4, help="Episodes rendered to rollout.mp4 (0: none).")
    p.add_argument("--goal-method", choices=GOAL_METHODS, default=None,
                   help="Which stage-2 result to use (default: vlm_cem_dir, else vlm_cem).")
    p.add_argument("--goal", choices=("scene", "lift"), default=None,
                   help="Default: saved training task for a policy, current scene goal for a grasp alone. scene/lift explicitly change the task.")
    p.add_argument("--output-dir", help="New directory; default: a uniquely named directory under <scene>/evals/.")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    scene = Path(a.scene_dir).resolve() if a.scene_dir else None
    model_path = run_config = None
    if not a.zero_action or a.run_dir or a.model_path:
        if a.model_path:
            model_path = Path(a.model_path).resolve()
        elif a.run_dir:
            model_path = find_model_in_run(a.run_dir)
            if model_path is None:
                raise FileNotFoundError(f"No model in {a.run_dir}")
        elif scene is not None:
            model_path = Path(infer_model_path_from_scene(scene))
        else:
            raise SystemExit("Give --scene-dir, --run-dir or --model-path (or --zero-action with a grasp).")
        run_config = load_run_config(run_dir_of(model_path))
    env_config = build_env_config(run_config, scene, a.grasp_path)
    goal = None
    goal_mode = a.goal or ("scene" if run_config is None or a.goal_method is not None else "saved")
    if goal_mode == "scene" and scene is None:
        raise ValueError("--goal scene requires --scene-dir")
    if goal_mode == "scene" and scene is not None:
        env_config.pop("goal_pose", None)
        goal_pose, goal = goal_pose_from_scene(scene, env_config["mesh_dir"], a.goal_method)
        if goal_pose is not None:
            env_config["goal_pose"] = goal_pose.tolist()
            # The carry turns the hand: judge slip in the hand's frame.
            env_config["stability_frame"] = "hand"
            print(f"Goal: {goal['source']}")
        else:
            print(f"Goal: lift 30 cm ({goal})")
            goal = None
    if goal_mode == "lift":
        env_config.pop("goal_pose", None)
        print("Goal: lift 30 cm")
    elif goal_mode == "saved" and env_config.get("goal_pose") is not None:
        mesh = trimesh.load(Path(env_config["mesh_dir"]) / "mesh.glb", force="mesh")
        position = np.asarray(env_config["goal_pose"])[:3, 3].copy()
        position[2] -= float(mesh.vertices[:, 2].min())
        goal = {"source": run_config.get("goal", "saved training task"), "position_m": position.tolist()}

    model = None
    if not a.zero_action:
        # Loading a checkpoint advances global RNG state: load first, seed after.
        model = load_trained_model(str(model_path), run_config.get("algorithm", "PPO"), "cuda")
        model.policy.eval()

    grasp = env_config["grasp_configs"][0]
    if a.output_dir:
        out = Path(a.output_dir)
    else:
        root = scene or run_dir_of(model_path)
        name = f"{grasp}-zero_action" if a.zero_action else \
            f"{run_dir_of(model_path).name}-{Path(model_path).stem}"
        out = Path(root) / "evals" / f"{name}-seed{a.seed}-{time.time_ns()}"
    out.mkdir(parents=True, exist_ok=False)
    title = (f"{grasp}, zero action" if a.zero_action else
             f"{run_dir_of(model_path).name}/{Path(model_path).stem}")
    print(f"Evaluating {title}: {a.episodes} episodes, seed {a.seed}")

    grasp_file = Path(env_config["grasp_dir"]) / f"{grasp}.txt"
    summary, rows = evaluate(
        env_config, model=model, episodes=a.episodes, seed=a.seed, deterministic=not a.stochastic,
        video_episodes=a.video_episodes, video_path=str(out / "rollout.mp4"),
        env_id=run_config["env_id"] if run_config else ENV_ID,
        camera=side_camera(grasp_file, goal and goal["position_m"]) if a.video_episodes else None)
    result = {"model": str(model_path) if model_path else None, "zero_action": a.zero_action,
              "grasp": str(grasp_file), "seed": a.seed, "deterministic": not a.stochastic, "goal": goal,
              "summary": summary, "episodes": rows, "env_config": env_config, "goal_mode": goal_mode}
    with open(out / "metrics.json", "w") as f:
        json.dump(result, f, indent=1)

    print("\n  " + "  ".join(f"{k.replace('_success', '')}: {summary[k]:.0%}" for k in SUMMARY_KEYS if k in summary))
    print(f"Wrote {out / 'metrics.json'}" + (f" and {out / 'rollout.mp4'}" if a.video_episodes else ""))
    return summary


if __name__ == "__main__":
    main()
