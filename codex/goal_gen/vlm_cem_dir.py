#!/usr/bin/env python3
"""Generate a goal pose by scoring rotations and offsets along a mesh axis.

Stage 2 uses random exploration plus VLM proposals, then saves the highest
scoring candidate. The full translation variant lives in ``vlm_cem.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import open3d as o3d
from codex.vlm import VLMInterface, VLMSession
from codex.scene.image_scene import ImageScene
from codex.mesh.load_mesh import load_mesh, trimesh_to_o3d
from codex.mesh.estimate_mesh_pose import MeshPoseEstimator
from codex.goal_gen.synthesize_3d_goal import (
    lift_effect_point,
    estimate_fn_point_3d_from_pixel,
    render_candidate,
    VIEW_DESCRIPTION,
    ONE_VIEW_DESCRIPTION,
)
from codex.utils.utils_path import load_scene_meta


# ───────────────────────── Prompt template ─────────────────────────
PROPOSAL_TEMPLATE = """
For the task {TASK_DESCRIPTION}, the latest round's candidate placements of {OBJECT_NAME} are below. All scores in this round were assigned together:

{CANDIDATE_HISTORY}

These descriptions are the evaluator's critiques of the rendered candidates, including the incumbent from the prior round. Use them to correct specific failures.

This is the goal pose of a functional object held by a robot gripper. The gripper is omitted from these renderings. The functional object does not need to rest on the table or support itself; judge whether the held functional object can act on the target with appropriate orientation, clearance, or contact. Do not invent gripper geometry or assume reachability has been verified.

Please propose **exactly {NUM_CANDIDATES}** new candidates, each with:
- dist: <float>   (signed meters along the outward functional direction)
- "yaw": <float>
- "pitch": <float>
- "roll": <float>

The functional direction points outward from the functional surface. The functional point is placed at target + dist * rotated_functional_direction. Negative dist therefore points the outward direction toward the target; positive dist places the target behind the functional surface.

Use the evaluator's descriptions to improve placement. Consider physical clearance and contact for the task. The original functional object was removed from the background before each candidate was rendered: its blank silhouette is not another physical object, and missing points do not prove that space is empty. Table penetration measurements do not measure overlap with other objects.

You have max {MAX_NUM_TRY} tries to propose new candidates. The current try is {TRY_INDEX}.

Return a single JSON object with keys "1"–"{NUM_CANDIDATES}". Each value must contain:
- "dist": <float>
- "yaw": <float>
- "pitch": <float>
- "roll": <float>
- "reasoning": <short explanation>

Return **only** the JSON object.
"""

EVALUATION_TEMPLATE = """
We want to evaluate how well each candidate image positions the {OBJECT_NAME} for the task '{TASK_DESCRIPTION}'. 
{VIEW_DESCRIPTION}
For each candidate image (numbered at the lower right corner), assess the {OBJECT_NAME}'s placement for how well it meets the task requirements. The same incumbent may appear again in a later round; score it against this round's candidates.

This is the goal pose of a functional object held by a robot gripper. The gripper is omitted from these renderings. The functional object does not need to rest on the table or support itself; judge whether the held functional object can act on the target with appropriate orientation, clearance, or contact. Do not invent gripper geometry or assume reachability has been verified.

Scoring:
• Provide a numeric score between 0 and 1, where 1 means well positioned to accomplish the task with plausible clearance/contact, and 0 means completely unsuitable.
• You can use intermediate values (e.g., 0.22, 0.5, 0.85) if the placement is partially correct.
• Be strict with scoring: think about the physical feasibility of each candidate (avoid penetrating the environment) and avoid being overoptimistic.
• If you do not see the whole/partial object, it is probably due to either occlusion or penetration (for example, object placed beneath the table surface).

The original functional object was removed from the background before placing each candidate. Its blank silhouette is not a second physical object; missing scene points do not prove free space. Assess penetration and contact from the visible geometry and the table measurements, using your judgment rather than a hard cutoff.

The supplied outward functional axis constrains the target to lie on its line. target_forward_distance is the signed distance from the functional point to the target along that outward axis: positive means IN FRONT, negative means BEHIND, zero means contact. This is alignment evidence from the supplied semantic axis, not proof of an unobstructed path or task success. Report any visible contradiction with the supplied axis.

Below we provide table penetration (0 means no measured table penetration) and target_forward_distance in meters. Table penetration does not measure overlap with other objects; judge those from the images.
Geometry Info (meters):
{PENETRATION_TABLE}

Format Requirements:
1. Return your answer as a single, valid JSON object and nothing else. Do not include extra text, commentary, or markdown formatting.
2. The JSON object should have candidate IDs (as strings) as keys. For each key, the value should be another JSON object containing exactly two fields: "score" and "description".
3. For example, if there are 3 candidates, a valid response should look like:

{
  "1": {"score": 0.73, "description": "The object is mostly above the target and slightly tilted."},
  "2": {"score": 0.02, "description": "Thermos is upright and nowhere near the mug."},
  "3": {"score": 1.0, "description": "Perfect angle and spout is centered over the mug."}
}

Return only the JSON object, with no extra formatting or commentary.
"""

# ───────────────────────── Data class ─────────────────────────
_candidate_id = 0


def _next_id() -> int:
    global _candidate_id
    _candidate_id += 1
    return _candidate_id


@dataclass
class Candidate3D:
    dist: float
    yaw: float
    pitch: float
    roll: float
    generation: int = 0
    id: int = field(default_factory=_next_id)
    score: Optional[float] = None
    description: Optional[str] = None
    table_penetration: float = 0.0

    @property
    def params(self) -> Tuple[float, float, float, float]:
        return (
            round(self.dist, 3),
            round(self.yaw, 1),
            round(self.pitch, 1),
            round(self.roll, 1),
        )


# ───────────────────────── Math helpers ─────────────────────────
def euler_to_matrix(y: float, p: float, r: float) -> np.ndarray:
    """Convert Euler angles (yaw, pitch, roll in degrees) to a 3x3 rotation matrix."""
    y, p, r = np.deg2rad([y, p, r])
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def make_T(R: np.ndarray | None = None, t: np.ndarray | None = None) -> np.ndarray:
    """Build a 4x4 homogeneous transform from optional rotation R and translation t."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


# ───────────────────────── Pose placement ─────────────────────────
def place_pose(
    base_pose: np.ndarray,
    point_m: np.ndarray,
    dist: float,
    yaw: float,
    pitch: float,
    roll: float,
    axis: np.ndarray,
    target_w: np.ndarray,
    keep_z: bool = True,
) -> np.ndarray:
    """
    Compute the 4x4 camera-frame pose using a distance-axis approach.

    Rotates the mesh by yaw/pitch/roll, then translates so the functional point
    (shifted by dist along the object's functional axis) aligns with target_w.

    Args:
        base_pose: Initial 4x4 camera-frame mesh pose.
        point_m: Functional keypoint in mesh frame.
        dist: Distance (meters) to offset along the functional axis.
        yaw, pitch, roll: Euler angles (degrees) for local rotation.
        axis: Functional axis direction as a 3-element array-like.
        target_w: Target 3D point in camera frame.
        keep_z: Unused flag (reserved for future z-clamping logic).

    Returns:
        4x4 camera-frame transform for the posed mesh.
    """
    Rm = euler_to_matrix(yaw, pitch, roll)
    pose_rot = base_pose @ make_T(R=Rm)
    local_offset = -np.array(axis) * dist
    offset_w = pose_rot[:3, :3] @ local_offset
    fn_w = (pose_rot @ np.append(point_m, 1.0))[:3]
    delta = target_w - (fn_w + offset_w)
    return make_T(t=delta) @ pose_rot


# ───────────────────────── VLM helpers ─────────────────────────
def _clean(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    t = txt.strip()
    if t.startswith("```"):
        t = "\n".join(t.splitlines()[1:-1])
    diff = t.count("{") - t.count("}")
    if diff > 0:
        t += "}" * diff
    return t


def _load_keypoints(scene, mesh, base_pose, keypoints_dir, target_name):
    """Load the mesh fn point, lift the masked target, and choose the axis.

    Old scenes without a mesh fn point retain the pixel-to-mesh fallback.
    The direction ablation requires an explicit functional direction in the
    canonical mesh frame; actuation directions are never used here.
    """
    os.makedirs(keypoints_dir, exist_ok=True)
    keypoints_path = Path(keypoints_dir)
    kp_data = json.loads((keypoints_path / "fn_key_pts.json").read_text())
    fn_path = keypoints_path / "fn_pt_mesh.json"
    if os.path.exists(fn_path):
        fn_pt_m = np.array(json.loads(fn_path.read_text()))
    else:
        fn_px = kp_data["fn_pts"][0]
        w, h = scene.image.size
        fn_pixel = (int(fn_px[0] * w / 100.0), int(fn_px[1] * h / 100.0))
        posed_mesh_cam = mesh.copy()
        posed_mesh_cam.apply_transform(base_pose)
        scene_pcd = scene.get_pcd(transform=False)
        fn_cam = estimate_fn_point_3d_from_pixel(
            fn_pixel, posed_mesh_cam, scene.cam_intrinsics, scene_pcd, visualize=False
        )
        fn_pt_m = (np.linalg.inv(base_pose) @ np.append(fn_cam, 1.0))[:3]
        fn_path.write_text(json.dumps(fn_pt_m.tolist()))

    tgt_px = kp_data["target_pts"][0]
    w, h = scene.image.size
    t_uv = (int(tgt_px[0] * w / 100.0), int(tgt_px[1] * h / 100.0))
    target_w = lift_effect_point(scene, t_uv, target_name)
    # This ablation constrains offsets to an explicitly supplied functional
    # direction. A button/trigger press direction is a different quantity.
    raw_axis = kp_data.get("fn_direction")
    if raw_axis is None:
        raise ValueError(
            "The direction ablation requires fn_direction in fn_key_pts.json "
            "(a 3D vector in canonical mesh coordinates). "
            "Use the main vlm_cem method when no functional direction is available."
        )
    axis = np.asarray(raw_axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError("fn_direction must be a finite, nonzero 3D mesh-frame vector")
    axis = axis / np.linalg.norm(axis)
    print(f"Functional axis (from fn_direction): {axis.tolist()}")

    return fn_pt_m, target_w, axis


def _random_candidate(generation):
    return Candidate3D(
        random.uniform(-0.2, 0.2),
        random.uniform(-180, 180),
        random.uniform(-90, 90),
        random.uniform(-20, 20),
        generation=generation,
    )


def _view_description(views):
    return VIEW_DESCRIPTION if views == 3 else ONE_VIEW_DESCRIPTION


def _propose_candidates(vlm, history, task, object_name, k, iterations, iteration):
    """Propose the guided part of a batch from the latest scored critiques."""
    if k <= 0:
        return []
    lines = [
        f"ID {c.id}: dist={c.dist:.3f}, y={c.yaw:.1f}, p={c.pitch:.1f}, r={c.roll:.1f}, "
        f"score={c.score if c.score is not None else 'unknown'}, "
        f"table_penetration_m={c.table_penetration:.3f}, description={c.description or 'none'}"
        for c in history
    ]
    prompt = PROPOSAL_TEMPLATE.replace("{TASK_DESCRIPTION}", task)
    prompt = prompt.replace("{CANDIDATE_HISTORY}", "\n".join(lines))
    prompt = prompt.replace("{OBJECT_NAME}", object_name)
    prompt = prompt.replace("{NUM_CANDIDATES}", str(k))
    prompt = prompt.replace("{MAX_NUM_TRY}", str(iterations))
    prompt = prompt.replace("{TRY_INDEX}", str(iteration))
    resp = vlm.query([], prompt, include_history=False, client="gpt")
    gen = max(c.generation for c in history) + 1
    props = []
    try:
        js = json.loads(_clean(resp))
        if not isinstance(js, dict):
            raise ValueError("proposal must be a JSON object")
        for v in list(js.values())[:k]:
            try:
                dist, yaw, pitch, roll = (float(v[key]) for key in ("dist", "yaw", "pitch", "roll"))
                if not all(np.isfinite(x) for x in (dist, yaw, pitch, roll)):
                    continue
                props.append(Candidate3D(dist, yaw, pitch, roll, generation=gen))
            except (KeyError, TypeError, ValueError):
                continue
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        warnings.warn(f"Could not parse VLM proposals ({e}); filling with random candidates.")
    while len(props) < k:
        props.append(_random_candidate(gen))
    return props


def _evaluate_candidates(
    vlm, cand, pose_for, mesh, scene, hom, cache_dir, iteration, task, object_name, views, render=None
):
    """Render one batch, measure table penetration, and attach VLM scores."""
    imgs_all = []
    for c in cand:
        pose = pose_for(c)
        img = (render(pose, c.id) if render is not None
               else render_candidate(mesh, pose, scene, c.id, multi_view=(views == 3)))
        img.save(f"{cache_dir}/iter{iteration}_cand{c.id}.jpg")
        imgs_all.append(img)
        world_pose = scene.cam_extrinsics @ pose
        c.table_penetration = max(0.0, -(world_pose @ hom.T).T[:, 2].min())

    pen = "\n".join(
        f"ID {c.id}: table_penetration={c.table_penetration:.3f}, "
        f"target_forward_distance={-c.dist:.4f}" for c in cand)
    eval_p = EVALUATION_TEMPLATE.replace("{TASK_DESCRIPTION}", task)
    eval_p = eval_p.replace("{OBJECT_NAME}", object_name)
    eval_p = eval_p.replace("{PENETRATION_TABLE}", pen)
    eval_p = eval_p.replace("{VIEW_DESCRIPTION}", _view_description(views))
    print("Eval prompt:\n", eval_p)
    try:
        scores = json.loads(_clean(vlm.query(imgs_all, eval_p, include_history=False, client="gpt")))
        if not isinstance(scores, dict):
            raise ValueError("scores must be a JSON object")
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        warnings.warn(f"Could not parse VLM scores ({exc}); assigning zero scores to this batch.")
        scores = {}
    for c in cand:
        e = scores.get(str(c.id), {})
        if not isinstance(e, dict):
            e = {}
        try:
            score = float(e.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        c.score = float(np.clip(score, 0, 1)) if np.isfinite(score) else 0.0
        c.description = str(e.get("description") or "")
        print(
            f"ID{c.id}: params={c.params}, score={c.score:.3f}, pen={c.table_penetration:.3f}, desc={c.description}"
        )


def _search(mesh, scene, pose_for, task, object_name, iters, cands, cache_dir, out_dir, views, render=None):
    """Compare fresh random/proposed candidates with the incumbent each round."""
    if iters < 1 or cands < 1:
        raise ValueError("iters and cands must be positive")
    if views not in (1, 3):
        raise ValueError("views must be 1 or 3")
    vlm = VLMSession(VLMInterface(), log_enabled=True,
                     log_dir=os.path.join(out_dir, "vlm_logs"))
    latest, incumbent = [], None
    trace = []
    vertices = np.asarray(mesh.vertices)
    hom = np.column_stack((vertices, np.ones(len(vertices))))
    for it in range(iters):
        print(f"Iter {it + 1}/{iters}")
        rand_n = cands if it == 0 else max(2, cands // 2)
        rand_n = min(rand_n, cands)
        candidates = [_random_candidate(it) for _ in range(rand_n)]
        if it > 0 and latest:
            candidates.extend(
                _propose_candidates(
                    vlm,
                    latest,
                    task,
                    object_name,
                    cands - rand_n,
                    iters,
                    it + 1,
                )
            )
        if incumbent is not None:
            candidates.append(incumbent)
        _evaluate_candidates(
            vlm,
            candidates,
            pose_for,
            mesh,
            scene,
            hom,
            cache_dir,
            it + 1,
            task,
            object_name,
            views,
            render,
        )
        incumbent = max(candidates, key=lambda candidate: (
            candidate.score if candidate.score is not None else -np.inf,
            candidate is incumbent,
        ))
        trace.append({
            "round": it + 1,
            "incumbent_id": incumbent.id,
            "random_count": rand_n,
            "proposed_count": cands - rand_n,
            "candidates": [{"id": c.id, "generation": c.generation,
                            "score": c.score, "description": c.description,
                            "table_penetration": c.table_penetration,
                            "params": {"dist": c.dist, "yaw": c.yaw,
                                       "pitch": c.pitch, "roll": c.roll}}
                           for c in candidates],
        })
        latest = candidates
    (Path(out_dir) / "search_trace.json").write_text(json.dumps(trace, indent=2))
    return incumbent, trace


# ───────────────────────── Main function ─────────────────────────
def generate_3d_goal_dist(
    task: str = None,
    scene_dir: str = None,
    object_name: str = None,
    iters: int = 5,
    cands: int = 8,
    load_cached: bool = False,
    keypoints_dir: str = None,
    output_dir: str = None,
    visualize: bool = False,
    views: int = 3,
):
    """
    Run the distance-axis VLM-CEM 3D goal generation pipeline.

    Similar to generate_3d_goal but parameterizes each candidate by a distance
    along the mesh's functional axis rather than a full tx/ty/tz offset.

    Args:
        task: Natural language description of the manipulation task.
            Defaults to loading from scene_meta.json inside scene_dir.
        scene_dir: Path to scene directory.
        object_name: Human-readable name of the object.
            Defaults to loading from scene_meta.json inside scene_dir.
        iters: Number of CEM iterations.
        cands: Number of candidates per iteration.
        load_cached: If True, skip generation and load existing best_candidate.json.
        keypoints_dir: Path to keypoints directory. Defaults to <scene_dir>/keypoints.
        output_dir: Path to output directory. Defaults to <scene_dir>/goal_gen/vlm_cem_dir.
        views: Views per candidate: 3 (camera, turned 90 degrees, top; default) or 1 (camera).
    """
    if task is None or object_name is None:
        meta = load_scene_meta(scene_dir)
        if task is None:
            task = meta.get("task")
        if object_name is None:
            object_name = meta.get("functional_object")
    if task is None:
        raise ValueError("task not provided and not found in scene_meta.json")
    if object_name is None:
        raise ValueError("object_name not provided and not found in scene_meta.json")
    if views not in (1, 3):
        raise ValueError("views must be 1 or 3")
    # Default keypoints and output dirs to live inside the scene directory
    if keypoints_dir is None:
        keypoints_dir = os.path.join(scene_dir, "keypoints")
    if output_dir is None:
        output_dir = os.path.join(scene_dir, "goal_gen", "vlm_cem_dir")
    out_dir = output_dir

    cache_dir = f"{out_dir}/cache"
    os.makedirs(cache_dir, exist_ok=True)
    if not load_cached:
        for f in os.listdir(cache_dir):
            os.remove(f"{cache_dir}/{f}")

    scene = ImageScene.load(f"{scene_dir}/scene.pkl")
    mesh, _ = load_mesh(os.path.join(scene_dir, "mesh", "mesh.glb"))
    pose_pth = f"{scene_dir}/pose.txt"
    if os.path.exists(pose_pth):
        base_pose = np.loadtxt(pose_pth)
    else:
        base_pose, _ = MeshPoseEstimator(debug=1, debug_dir=f"{out_dir}/pose_dbg").estimate_pose(
            scene, mesh, 0
        )
        np.savetxt(pose_pth, base_pose)

    fn_pt_m, target_w, axis = _load_keypoints(
        scene,
        mesh,
        base_pose,
        keypoints_dir,
        load_scene_meta(scene_dir).get("target_object"),
    )

    def pose_for(candidate):
        return place_pose(
            base_pose,
            fn_pt_m,
            candidate.dist,
            candidate.yaw,
            candidate.pitch,
            candidate.roll,
            axis,
            target_w,
        )

    def render(pose, cid):
        """The functional object at `pose` with its local axes and the line along its
        functional direction from the functional point to the effect point."""
        return render_candidate(mesh, pose, scene, cid, multi_view=(views == 3), axes_origin=fn_pt_m,
                                effect_point=target_w, functional_axis=axis)

    if load_cached:
        data = json.loads((Path(out_dir) / "best_candidate.json").read_text())
        search_trace = data.get("search_trace", [])
        best = Candidate3D(
            dist=data["dist"],
            yaw=data["yaw"],
            pitch=data["pitch"],
            roll=data["roll"],
            generation=data["generation"],
            id=data["id"],
            score=data["score"],
            description=data["description"],
            table_penetration=data.get("table_penetration", 0.0),
        )
        print("Loaded cached best:", best.params, best.score)
    else:
        best, search_trace = _search(mesh, scene, pose_for, task, object_name,
                                     iters, cands, cache_dir, out_dir, views, render)
    best_pose = pose_for(best)

    best_w = scene.cam_extrinsics @ best_pose
    if visualize:
        mesh_w = mesh.copy()
        mesh_w.apply_transform(best_w)
        scene_pcd = scene.get_pcd(transform=True)
        o3d.visualization.draw_geometries(
            [trimesh_to_o3d(mesh_w), scene_pcd], window_name="Best Candidate Mesh"
        )

    final_img = render(best_pose, None)
    os.makedirs(out_dir, exist_ok=True)
    final_img.save(f"{out_dir}/best_candidate.jpg")
    transform_world = best_w.tolist()
    # get target object center in world frame
    target_obj = scene.objects[1]
    pcd = target_obj.get_pcd()
    points = np.asarray(pcd.points)
    target_obj_center = np.median(points, axis=0).tolist()
    # compute relative transform (object->best)
    rel_T = (np.linalg.inv(scene.cam_extrinsics @ base_pose) @ np.array(transform_world)).tolist()
    summary = {
        **best.__dict__,
        "transform": best_pose.tolist(),
        "transform_world": transform_world,
        "fn_pt_mesh": fn_pt_m.tolist(),
        "target_world": (scene.cam_extrinsics @ np.append(target_w, 1.0))[:3].tolist(),
        "target_camera": target_w.tolist(),
        "relative_transform": rel_T,
        "target_obj_center": target_obj_center,
        "views": views,
        "search_trace": search_trace,
    }
    with open(f"{out_dir}/best_candidate.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved best candidate:", best.params, best.score)


def main() -> None:
    """Entry point: parse arguments and run distance-axis VLM-CEM with user-provided task inputs."""
    ap = argparse.ArgumentParser(description="Goal pose search (VLM-CEM dir): the functional point on the line through the effect point along the functional direction; distance, yaw, pitch and roll are searched")
    ap.add_argument(
        "--scene-dir",
        dest="scene_dir",
        type=str,
        required=True,
        help="Path to scene directory (contains scene.pkl)",
    )
    ap.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task description for goal generation (default: load from scene_meta.json)",
    )
    ap.add_argument(
        "--object-name",
        dest="object_name",
        type=str,
        default=None,
        help="Human-readable object name for prompts (default: load from scene_meta.json)",
    )
    ap.add_argument(
        "--keypoints-dir",
        dest="keypoints_dir",
        type=str,
        default=None,
        help="Path to keypoints directory for this scene (default: <scene_dir>/keypoints)",
    )
    ap.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Optional output directory for goal outputs (default: <scene_dir>/goal_gen/vlm_cem_dir)",
    )
    ap.add_argument("--iters", type=int, default=5, help="Number of optimization iterations")
    ap.add_argument("--cands", type=int, default=8, help="Number of new candidates per iteration")
    ap.add_argument("--views", type=int, choices=(1, 3), default=3,
                    help="Views per candidate: 3 (camera, turned 90 degrees, top; default) or 1 (camera)")
    ap.add_argument(
        "--load-cached",
        action="store_true",
        help="Reuse best_candidate.json from output_dir instead of generating new candidates",
    )
    a = ap.parse_args()

    scene_dir = os.path.abspath(a.scene_dir)

    generate_3d_goal_dist(
        task=a.task,
        scene_dir=scene_dir,
        object_name=a.object_name,
        iters=a.iters,
        cands=a.cands,
        load_cached=a.load_cached,
        keypoints_dir=os.path.abspath(a.keypoints_dir) if a.keypoints_dir else None,
        output_dir=os.path.abspath(a.output_dir) if a.output_dir else None,
        views=a.views,
    )


if __name__ == "__main__":
    main()
