"""Simulation screening of candidate grasps with the RL motion primitive.

Every candidate grasp is executed with ZERO residual action through the exact
grasp-move-actuate primitive the policy is trained on (PrimitiveGraspForceEnv:
approach from 10 cm, half-curl pregrasp, +0.2 rad support squeeze, 30 cm lift
with the actuation finger parked, then the press with a 1 s held-press
window). Many candidates run in parallel GPU environments; each candidate can
occupy several environments so it sees several draws of the physical
parameters (friction, mass, size; configs/primitive.py DOMAIN_RANDOMIZATION)
and pose-noise samples.

This is the selection criterion of the grasp generator: the static
force-closure certificate does not predict whether a grasp survives the
primitive, the simulator does.
"""

import json
import os
import shutil
import tempfile

import numpy as np
import torch

from codex.codex_train.configs import primitive as P

# The training primitive (stage 4 defaults); nothing here is learned.
PRIMITIVE_CONFIG = dict(
    P.PRIMITIVE_ENV_CONFIG,
    num_actuate_steps=P.NUM_ACTUATE_STEPS,
    press_hold_seconds=P.PRESS_HOLD_SECONDS,
    delta_pos_scale=P.DELTA_POS_SCALE,
    delta_rot_scale=P.DELTA_ROT_SCALE,
    delta_finger_scale=P.DELTA_FINGER_SCALE,
    init_at_fixed_rot=False,
    init_at_standing_pose=False,
)

METRIC_KEYS = ("success", "lift_success", "stable_success", "act_success",
               "upright_success", "held_press_success", "est_collision")


def _write_placeholder_grasp(grasp_dir, vec, act_finger):
    np.savetxt(os.path.join(grasp_dir, "grasp_0_optimized.txt"), vec)
    with open(os.path.join(grasp_dir, "grasp_0_metadata.json"), "w") as f:
        json.dump({"actuation_finger_indices": [int(act_finger)]}, f)


class SimScreener:
    """Parallel zero-action primitive rollouts for many candidate grasps.

    grasps are 23-vectors in the grasp-file convention (quat wxyz, position in
    the bottom-at-z=0 object frame, 16 joints in names order).
    """

    def __init__(self, mesh_dir, keypoints_dir, num_envs=512, seed=0,
                 **env_overrides):
        """mesh_dir: the scene's CoACD directory (mesh.glb + metadata.json);
        keypoints_dir: directory holding fn_key_pts.json."""
        import gymnasium as gym
        import codex.codex_train.envs  # noqa: F401  (registers the envs)
        from codex.grasp_gen.load_grasp import quat_pos_to_pose_mat

        self._to_mat = quat_pos_to_pose_mat
        self.num_envs = num_envs
        mesh_dir = os.path.abspath(mesh_dir)
        with open(os.path.join(mesh_dir, "metadata.json")) as f:
            bbox = json.load(f)["bbox"]
        # Grasp files use the bottom-at-z=0 frame; env poses the centered mesh.
        self.offset = np.array([0.0, 0.0, bbox[2] / 2.0])

        self._tmp = tempfile.mkdtemp(prefix="sim_screen_")
        _write_placeholder_grasp(self._tmp, np.r_[1, 0, 0, 0, 0, 0, 0.5, np.zeros(16)], 0)
        cfg = dict(PRIMITIVE_CONFIG)
        cfg.update(env_overrides)
        cfg.update(
            mesh_dir=mesh_dir,
            grasp_dir=self._tmp,
            grasp_configs=["grasp_0_optimized"],
            actuation_dir=os.path.abspath(keypoints_dir),
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env = gym.make("PrimitiveGraspForceEnv-v1", num_envs=num_envs, **cfg)
        self.base = self.env.unwrapped
        self.seed = seed

    def close(self):
        self.env.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run_slots(self, vecs, act_finger, seed):
        """One primitive rollout with per-environment grasps (all sharing one
        actuation finger, which the primitive requires)."""
        n = self.num_envs
        assert len(vecs) == n
        dev = self.base.device
        joints = torch.tensor(np.stack([v[7:] for v in vecs]), dtype=torch.float32, device=dev)
        poses = []
        for v in vecs:
            pose = np.array(v[:7], dtype=np.float64)
            pose[4:] -= self.offset
            poses.append(self._to_mat(pose[:4], pose[4:]))
        poses = torch.tensor(np.stack(poses), dtype=torch.float32, device=dev)
        act = torch.full((n, 1), int(act_finger), dtype=torch.long, device=dev)

        torch.manual_seed(seed)
        self.env.reset(seed=seed)
        self.base.loaded_grasp = (joints, poses, act)
        action = torch.zeros((n, *self.env.single_action_space.shape), device=dev)
        _, _, _, _, info = self.env.step(action)
        out = {k: info[k].detach().reshape(n).float().cpu().numpy()
               for k in METRIC_KEYS if k in info}
        out["height_change"] = (info["final_obj_pose"].p[:, 2]
                                - info["init_obj_pose"].p[:, 2]).cpu().numpy()
        out["act_dist"] = info["act_dist"].detach().reshape(n).cpu().numpy()
        palm_change = torch.norm(info["final_palm_dist"] - info["pre_lifted_palm_dist"], dim=-1)
        out["palm_drift"] = palm_change.cpu().numpy()
        return out

    def screen(self, grasps, act_fingers, reps=4, seed=None, rollouts=None):
        """Screen candidates; returns a list of per-candidate metric dicts
        (rates over `reps` rollouts with different physics/noise). Pass a
        dict `rollouts` to accumulate raw per-rollout rows across calls
        (keyed by position in `grasps`); rates then pool all of them."""
        seed = self.seed if seed is None else seed
        grasps = [np.asarray(g, dtype=np.float64) for g in grasps]
        act_fingers = np.asarray(act_fingers, dtype=int)
        results = [dict() for _ in grasps]
        per_cand = {i: [] for i in range(len(grasps))}
        n = self.num_envs
        batch = 0
        for fi in np.unique(act_fingers):
            idx = np.nonzero(act_fingers == fi)[0]
            slots = np.repeat(idx, reps)  # candidate id per slot
            for start in range(0, len(slots), n):
                chunk = slots[start:start + n]
                pad = np.r_[chunk, np.resize(chunk, n - len(chunk))] if len(chunk) < n else chunk
                out = self._run_slots([grasps[i] for i in pad], fi, seed + batch)
                batch += 1
                for s, cid in enumerate(chunk):
                    per_cand[cid].append({k: float(v[s]) for k, v in out.items()})
        if rollouts is not None:
            for cid, rows in per_cand.items():
                rollouts.setdefault(cid, []).extend(rows)
                per_cand[cid] = rollouts[cid]
        for cid, rows in per_cand.items():
            r = results[cid]
            for k in rows[0]:
                vals = np.array([row[k] for row in rows])
                r[k] = float(vals.mean()) if k in METRIC_KEYS else float(np.median(vals))
            r["raw_lift"] = float(np.mean([row["height_change"] > 0.1 for row in rows]))
            r["n"] = len(rows)
        return results


def screen_score(r):
    """Scalar ranking of a screened candidate: strict task success first, then
    the components a residual policy has the hardest time creating (a hold
    that survives the lift, a finger already near the actuation point)."""
    return (4.0 * r["success"] + 2.0 * r["lift_success"] + 1.0 * r["raw_lift"]
            + 1.0 * r["stable_success"] * r["raw_lift"] + 0.5 * r["upright_success"]
            + 1.0 * np.exp(-r["act_dist"] / 0.03) - 1.0 * r["est_collision"])


def successive_halving(screener, grasps, act_fingers, schedule=((None, 2), (192, 8), (32, 32)), seed=0,
                       log=print):
    """Allocate rollouts by successive halving: every candidate gets a few
    rollouts, the best get more, the finalists many. Statistics pool across
    stages. Returns (order of finalists, pooled per-candidate results)."""
    alive = list(range(len(grasps)))
    rollouts, results = {}, [None] * len(grasps)
    for stage, (keep, reps) in enumerate(schedule):
        if keep is not None:
            alive = sorted(alive, key=lambda i: -screen_score(results[i]))[:keep]
        pooled = {k: rollouts.setdefault(i, []) for k, i in enumerate(alive)}
        res = screener.screen([grasps[i] for i in alive], [act_fingers[i] for i in alive], reps=reps,
                              seed=seed + 7919 * stage, rollouts=pooled)
        for k, i in enumerate(alive):
            results[i] = res[k]
        top = max(alive, key=lambda i: screen_score(results[i]))
        log(f"  halving stage {stage}: {len(alive)} candidates x {reps} rollouts; best "
            f"success={results[top]['success']:.2f} lift={results[top]['lift_success']:.2f} (n={results[top]['n']})")
    alive = sorted(alive, key=lambda i: -screen_score(results[i]))
    return alive, results
