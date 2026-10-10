#!/usr/bin/env python3
"""Search full 6DOF goal poses with VLM-guided candidate evaluation.

Candidates rotate around the functional point and use three independent local
translation offsets. The VLM evaluates placement and penetration from renders.
"""
from __future__ import annotations

import argparse, json, math, os, random, warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import open3d as o3d
from PIL import Image
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
For the task {TASK_DESCRIPTION}, we have a history of candidate placements of {OBJECT_NAME} below. The scores and critiques are from the previous critic rounds. Use the critiques to correct specific failures.

{CANDIDATE_HISTORY}

The accompanying images correspond to the latest candidates.

Rendering context: The original {OBJECT_NAME} was removed from the background point cloud before placing each candidate. Its blank silhouette is missing scene data, not a second physical object. Do not infer a collision with an imagined obstacle from that blank region. Missing data also does not prove the region empty; judge clearance against visible geometry.

Latest critic direction: {CRITIC_EXPLANATION}

Please propose **exactly {NUM_CANDIDATES}** new candidates, each with 3-D rotation and a local offset (tx, ty, tz in meters) so that the {OBJECT_NAME} is appropriately placed for the task.

Placement convention: R = base_R @ Rz(yaw) @ Ry(pitch) @ Rx(roll), with angles in degrees. The functional point in camera coordinates is target_camera + R @ [tx, ty, tz]. The offsets are in rotated object-local axes. The displayed axes mark those local axes at the functional point; local Z is not necessarily world up.
Keeping the angles fixed, increasing tx by 0.08 moves the functional point and object 8 cm along the displayed red +X axis. Changing the angles also rotates the offset. Try both close contact and sensible standoff where the task calls for it.
Each history row gives functional_point_height_m (world Z) and local_delta_for_world_up_5cm. To raise the whole object vertically by 5 cm while keeping that row's angles fixed, ADD all three components of that vector to its current tx, ty, tz. For another desired rise in meters, multiply the vector by desired_rise_m / 0.05. Changing angles invalidates the listed vector; local +Z alone is not world up.

You should try to improve task alignment and visual feasibility. Penetration numbers measure only the table, not other objects. Explore alternatives suggested by the critiques; do not assume any axis is the functional object's functional direction.

You have max {MAX_NUM_TRY} tries to propose new candidates. The current try is {TRY_INDEX}.

Return a single JSON object with keys "1"–"{NUM_CANDIDATES}". Each value must contain:
- "yaw": <float>
- "pitch": <float>
- "roll": <float>
- "tx": <float>   (metres along local +X)
- "ty": <float>   (metres along local +Y)
- "tz": <float>   (metres along local +Z)
- "reasoning": <short explanation>

Return **only** the JSON object.
"""

EVALUATION_TEMPLATE = """
Evaluate the {OBJECT_NAME} placements for '{TASK_DESCRIPTION}'. Each image has an ID. The axes originate at the functional point and show rotated object-local X, Y, Z; local Z is not necessarily world up. The placement is defined by R = base_R @ Rz(yaw) @ Ry(pitch) @ Rx(roll), then functional_point_camera = target_camera + R @ [tx, ty, tz].

Rendering context: The original {OBJECT_NAME} was removed from the background point cloud before placing each candidate. Its blank silhouette is missing scene data, not a second physical object. Do not infer a collision with an imagined obstacle from that blank region. Missing data also does not prove the region empty; judge clearance against visible geometry.

This is round {ROUND} of at most {MAX_ROUNDS}. The next round has a fixed budget of {NUM_CANDIDATES} new candidates. If continuing, choose next_random_count between 1 and {MAX_RANDOM_COUNT}; the remaining candidates will be proposed from your critiques. Mark a pose suitable only if it can accomplish the task with plausible clearance or contact. Set stop=true once it is suitable and no named material issue remains. Continue to fix a specific material issue, not for a speculative score gain; keep judging penetration strictly.

{INCUMBENT_NOTE}

Scoring:
• Provide a numeric score between 0 and 1, where 1 means “well positioned to accomplish the task with plausible clearance or contact,” and 0 means “completely unsuitable.”
• You can use intermediate values (e.g., 0.22, 0.5, 0.85) if the placement is partially correct.
• Be strict with scoring: think about the physical feasibility of each candidate (avoid penetrating the environment) and avoid being overoptimistic.
• If you do not see the whole/partial object, it is probably due to either occlusion or penetration (for example, object placed beneath the table surface). Or the object is completely out of the camera view (which is undesired).

*Having no penetration* (with tables/other objects) is extremely important for the candidate feasibility.

Below we provide the table penetration of each candidate (0 means no measured table penetration). This is only penetration with the *table*, not other objects. Judge object overlap from the images. Do not treat zero table penetration as proof of collision freedom.
Penetration Info (meters):
{PENETRATION_TABLE}

Return one JSON object only:
{
  "candidates": {"<image ID>": {"score": <0..1>, "description": "specific critique"}},
  "best_id": <image ID integer>,
  "suitable": <boolean: selected pose is physically plausible and fit for the task>,
  "stop": <boolean: search can stop now>,
  "next_random_count": <integer: how many random candidates to allocate next round>,
  "exploration_radius_m": <float: suggested offset search radius next round>,
  "explanation": "why this pose wins and what the next round should change"
}
Score every image, including the incumbent when shown. Choose best_id among the shown IDs. If unsuitable, name the geometry or clearance issue to change. Balance random exploration with targeted proposals in the next round.
"""

# ───────────────────────── Data class ─────────────────────────
_candidate_id = 0

def _next_id() -> int:
    global _candidate_id; _candidate_id += 1; return _candidate_id

@dataclass
class Candidate3D:
    yaw: float
    pitch: float
    roll: float
    tx: float
    ty: float
    tz: float
    generation: int = 0
    id: int = field(default_factory=_next_id)
    score: Optional[float] = None
    description: Optional[str] = None
    table_penetration: float = 0.0

    @property
    def params(self) -> Tuple[float,float,float,float,float,float]:
        return (
            round(self.yaw,1),
            round(self.pitch,1),
            round(self.roll,1),
            round(self.tx,3),
            round(self.ty,3),
            round(self.tz,3),
        )

# ───────────────────────── Math helpers ─────────────────────────

def euler_to_matrix(y: float, p: float, r: float) -> np.ndarray:
    """Convert Euler angles (yaw, pitch, roll in degrees) to a 3x3 rotation matrix."""
    y, p, r = np.deg2rad([y, p, r])
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    return Rz @ Ry @ Rx

def make_T(R: np.ndarray|None=None, t: np.ndarray|None=None) -> np.ndarray:
    """Build a 4x4 homogeneous transform from optional rotation R and translation t."""
    T = np.eye(4)
    if R is not None: T[:3,:3] = R
    if t is not None: T[:3,3] = t
    return T

# pose placement

def place_pose(base_pose: np.ndarray,
               point_m: np.ndarray,
               yaw: float, pitch: float, roll: float,
               tx: float, ty: float, tz: float,
               target_w: np.ndarray,
               keep_z: bool=True) -> np.ndarray:
    """
    Compute the 4x4 camera-frame pose of the mesh given a candidate rotation and offset.

    Rotates the mesh about its functional point, then translates so that the functional
    point (plus local offset) coincides with target_w.

    Args:
        base_pose: Initial 4x4 camera-frame mesh pose.
        point_m: Functional keypoint in mesh frame coordinates.
        yaw, pitch, roll: Euler angles (degrees) for local rotation.
        tx, ty, tz: Local offset in meters (applied as negative along mesh axes).
        target_w: Target 3D point in camera frame.
        keep_z: Unused flag (reserved for future z-clamping logic).

    Returns:
        4x4 camera-frame transform for the posed mesh.
    """
    Rm = euler_to_matrix(yaw, pitch, roll)
    T_rot = make_T(R=Rm)
    pose_rot = base_pose @ make_T(t=-point_m) @ T_rot @ make_T(t=point_m)

    # world‐frame position of the functional point after rotation
    fn_w = (pose_rot @ np.append(point_m, 1.0))[:3]

    # apply full 3D offset in the object’s local frame
    local_offset = np.array([-tx, -ty, -tz])  # negative because we sit 'behind' by offset
    offset_w = pose_rot[:3, :3] @ local_offset

    delta = target_w - (fn_w + offset_w)

    return make_T(t=delta) @ pose_rot

# rendering

# VLM helpers

def _clean(txt:str)->str:
    t=txt.strip();
    if t.startswith('```'): t='\n'.join(t.splitlines()[1:-1])
    diff=t.count('{')-t.count('}');
    if diff>0: t+='}'*diff
    return t


def _world_height_and_lift(candidate: Candidate3D, base_pose: np.ndarray,
                           point_m: np.ndarray, target_w: np.ndarray,
                           cam_extrinsics: np.ndarray) -> tuple[float, np.ndarray]:
    """World height of the functional point and local offset for 5 cm world rise."""
    pose = place_pose(base_pose, point_m, candidate.yaw, candidate.pitch,
                      candidate.roll, candidate.tx, candidate.ty, candidate.tz,
                      target_w)
    world_pose = cam_extrinsics @ pose
    fn_world = (world_pose @ np.append(point_m, 1.0))[:3]
    local_lift = world_pose[:3, :3].T @ np.array([0.0, 0.0, 0.05])
    return float(fn_world[2]), local_lift


def propose_new_candidates(vlm: VLMSession,
                           history: List[Candidate3D],
                           imgs: List[Image.Image],
                           task: str,
                           obj: str,
                           k: int,
                           max_num_tries: int,
                           current_try_idx: int,
                           view_description: str = "",
                           critic_explanation: str = "Explore promising and untested placements.",
                           geometry_context: tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray] | None = None) -> List[Candidate3D]:
    """
    Ask the VLM to propose k new candidate poses based on the history of scored candidates.

    Args:
        vlm: Active VLM session for querying.
        history: Previously evaluated candidates with scores.
        imgs: Images of the latest batch of candidates.
        task: Natural language task description.
        obj: Human-readable object name.
        k: Number of new candidates to request.
        max_num_tries: Total number of iterations allowed.
        current_try_idx: Current iteration index (1-based).

    Returns:
        List of new Candidate3D instances proposed by the VLM.
    """
    lines = []
    for c in history:
        geometry = ""
        if geometry_context is not None:
            height, lift = _world_height_and_lift(c, *geometry_context)
            geometry = (f", functional_point_height_m={height:.4f}, "
                        f"local_delta_for_world_up_5cm=[{lift[0]:.4f}, {lift[1]:.4f}, {lift[2]:.4f}]")
        lines.append(
            f"ID {c.id}: y={c.params[0]}, p={c.params[1]}, r={c.params[2]}, "
            f"tx={c.params[3]}, ty={c.params[4]}, tz={c.params[5]}, "
            f"score={f'{c.score:.3f}' if c.score is not None else 'unscored'}, "
            f"table_penetration_m={c.table_penetration:.3f}{geometry}, "
            f"critique={c.description or 'none'}"
        )
    prompt = (PROPOSAL_TEMPLATE
        .replace('{TASK_DESCRIPTION}', task)
        .replace('{CANDIDATE_HISTORY}', '\n'.join(lines))
        .replace('{OBJECT_NAME}', obj)
        .replace('{NUM_CANDIDATES}', str(k))
        .replace('{MAX_NUM_TRY}', str(max_num_tries))
        .replace('{TRY_INDEX}', str(current_try_idx))
        .replace('{CRITIC_EXPLANATION}', critic_explanation)
    )
    prompt = view_description + '\n' + prompt if view_description else prompt
    resp = vlm.query(imgs, prompt, include_history=False, client='gpt')
    try:
        js = json.loads(_clean(resp))
    except json.JSONDecodeError as e:
        warnings.warn(f"Could not parse VLM proposal JSON ({e}); falling back to random candidates.")
        return []
    gen = max((c.generation for c in history), default=0) + 1
    out = []
    if not isinstance(js, dict):
        return []
    for v in js.values():
        try:
            values = [float(v[name]) for name in ('yaw', 'pitch', 'roll', 'tx', 'ty', 'tz')]
            if not all(math.isfinite(x) for x in values) or any(abs(x) > 0.5 for x in values[3:]):
                raise ValueError("nonfinite or out-of-range pose")
            out.append(Candidate3D(*values, generation=gen))
        except (KeyError, TypeError, ValueError):
            warnings.warn(f"Skipping malformed VLM proposal entry: {v!r}")
            continue
    return out


def _random_candidates(count: int, generation: int, radius_m: float,
                       incumbent: Optional[Candidate3D] = None,
                       initial: bool = False) -> List[Candidate3D]:
    """Seed the registered pose, nearby poses, and global poses; later mix searches."""
    out = []
    initial_near_count = max(0, count // 2 - 1)
    for index in range(count):
        local = incumbent is not None and index % 2 == 1 and not initial
        if initial and index == 0:
            angles = (0.0, 0.0, 0.0)
        elif initial and index <= initial_near_count:
            angles = tuple(random.uniform(-20, 20) for _ in range(3))
        elif local:
            angles = (incumbent.yaw + random.uniform(-60, 60),
                      incumbent.pitch + random.uniform(-45, 45),
                      incumbent.roll + random.uniform(-60, 60))
        else:
            angles = (random.uniform(-180, 180), random.uniform(-90, 90),
                      random.uniform(-180, 180))
        if local:
            offsets = [center + random.uniform(-radius_m, radius_m)
                       for center in (incumbent.tx, incumbent.ty, incumbent.tz)]
        else:
            scale = (0.025, 0.08, 0.2)[index % 3] if initial else radius_m
            offsets = [random.uniform(-scale, scale) for _ in range(3)]
        out.append(Candidate3D(*angles, *offsets, generation=generation))
    return out


def _critic_result(response: str, shown: List[Candidate3D], cands: int,
                   previous_radius: float) -> dict:
    """Parse a critic reply; only scores and IDs actually shown can steer search."""
    try:
        data = json.loads(_clean(response))
    except (json.JSONDecodeError, TypeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    entries = data.get('candidates', {})
    if not isinstance(entries, dict):
        entries = {}
    scored_ids = set()
    for candidate in shown:
        entry = entries.get(str(candidate.id), {})
        if not isinstance(entry, dict):
            continue
        try:
            score = float(entry['score'])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(score) or not 0 <= score <= 1:
            continue
        candidate.score = score
        candidate.description = str(entry.get('description', ''))
        scored_ids.add(candidate.id)
    try:
        best_id = int(data.get('best_id'))
    except (TypeError, ValueError):
        best_id = None
    chosen = next((c for c in shown if c.id == best_id), None) if best_id in scored_ids else None
    if chosen is None:
        valid = [c for c in shown if c.id in scored_ids]
        chosen = max(valid, key=lambda c: c.score) if valid else None
    try:
        random_count = int(data.get('next_random_count', cands // 2))
    except (TypeError, ValueError):
        random_count = cands // 2
    # Keep both exploration and critique-driven proposals when a next round exists.
    random_count = min(max(random_count, 1), cands - 1) if cands > 1 else 1
    try:
        radius_m = float(data.get('exploration_radius_m', previous_radius))
    except (TypeError, ValueError):
        radius_m = previous_radius
    if not math.isfinite(radius_m):
        radius_m = previous_radius
    radius_m = min(max(radius_m, 0.02), 0.5)
    suitable = data.get('suitable') is True and best_id in scored_ids
    stop = data.get('stop') is True and suitable and best_id in scored_ids
    return {
        'chosen': chosen, 'suitable': suitable, 'stop': stop,
        'next_random_count': random_count, 'exploration_radius_m': radius_m,
        'explanation': str(data.get('explanation', '')),
        'valid_best_id': best_id in scored_ids,
        'scored_ids': sorted(scored_ids),
    }

# ───────────────────────── Main function ─────────────────────────

def generate_3d_goal(task: str = None, scene_dir: str = None, object_name: str = None, iters: int = 5, cands: int = 8, load_cached: bool = False,
                     keypoints_dir: str = None, output_dir: str = None, visualize: bool = False,
                     views: int = 3, scene_representation: str = "points"):
    """
    Run the VLM-CEM 3D goal generation pipeline.

    Iteratively proposes and evaluates 6-DOF candidate poses for placing a mesh
    relative to a target scene point, using a VLM to score candidates.

    Args:
        task: Natural language description of the manipulation task.
            Defaults to loading from scene_meta.json inside scene_dir.
        scene_dir: Path to scene directory.
        object_name: Human-readable name of the object.
            Defaults to loading from scene_meta.json inside scene_dir.
        iters: Number of CEM iterations.
        cands: Number of candidates to evaluate per iteration.
        load_cached: If True, skip generation and load existing best_candidate.json.
        keypoints_dir: Path to keypoints directory. Defaults to <scene_dir>/keypoints.
        output_dir: Path to output directory. Defaults to <scene_dir>/goal_gen/vlm_cem.
        views: Rendered views per candidate: 3 (camera, turned 90 degrees, top) or 1 (camera).
        scene_representation: Scene renderer input ("points" or "surface").
    """
    if views not in (1, 3):
        raise ValueError("views must be 1 or 3")
    if scene_representation not in ("points", "surface"):
        raise ValueError("scene_representation must be points or surface")
    if iters < 1 or cands < 2:
        raise ValueError("iters must be positive and cands must be at least 2")
    view_description = VIEW_DESCRIPTION if views == 3 else ONE_VIEW_DESCRIPTION
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
    # Default keypoints and output dirs to live inside the scene directory
    if keypoints_dir is None:
        keypoints_dir = os.path.join(scene_dir, "keypoints")
    method_dir = "vlm_cem"
    if output_dir is None:
        out_dir = os.path.join(scene_dir, "goal_gen", method_dir)
    else:
        out_dir = output_dir
    print(f"Output directory: {out_dir}")
    cache_dir=f'{out_dir}/cache'
    os.makedirs(cache_dir,exist_ok=True)
    if not load_cached:
        for f in os.listdir(cache_dir):
            os.remove(f'{cache_dir}/{f}')

    # load scene & mesh
    scene=ImageScene.load(f'{scene_dir}/scene.pkl')
    mesh, mesh_info = load_mesh(os.path.join(scene_dir, "mesh", "mesh.glb"))

    # base pose (cached)
    pose_pth=f'{scene_dir}/pose.txt'
    if os.path.exists(pose_pth): 
        base_pose = np.loadtxt(pose_pth)
    else:
        base_pose, _ = MeshPoseEstimator(debug=1, debug_dir=f'{out_dir}/pose_dbg').estimate_pose(scene,mesh,0)
        np.savetxt(pose_pth, base_pose)

    # functional & target points
    os.makedirs(keypoints_dir, exist_ok=True)
    fn_path = os.path.join(keypoints_dir, 'fn_pt_mesh.json')
    if os.path.exists(fn_path):
        with open(fn_path) as f:
            fn_pt_m = np.array(json.load(f))
    else:
        with open(os.path.join(keypoints_dir, 'fn_key_pts.json')) as f:
            fn_px = json.load(f)['fn_pts'][0]
        w, h = scene.image.size
        fn_pixel = (int(fn_px[0]*w/100.), int(fn_px[1]*h/100.))
        # pose mesh into camera frame
        posed_mesh_cam = mesh.copy(); posed_mesh_cam.apply_transform(base_pose)
        scene_pcd = scene.get_pcd(transform=False)
        fn_cam = estimate_fn_point_3d_from_pixel(fn_pixel, posed_mesh_cam, scene.cam_intrinsics, scene_pcd, visualize=False)
        # convert back to mesh frame
        fn_pt_m = (np.linalg.inv(base_pose) @ np.append(fn_cam,1.0))[:3]
        with open(fn_path, 'w') as f:
            json.dump(fn_pt_m.tolist(), f)

    with open(os.path.join(keypoints_dir, 'fn_key_pts.json')) as f:
        kp_data = json.load(f)
    tgt_px = kp_data['target_pts'][0]
    # Drawn on the renders only (this method searches a free 6-DoF offset).
    fn_axis = (kp_data.get('fn_contact') or {}).get('dir') or kp_data.get('fn_direction')
    w,h=scene.image.size; t_uv=(int(tgt_px[0]*w/100.),int(tgt_px[1]*h/100.))
    target_w = lift_effect_point(scene, t_uv, load_scene_meta(scene_dir).get('target_object'))
    if target_w is None: raise RuntimeError('cannot infer 3‑D target point')

    history:List[Candidate3D]=[]
    search_trace = []
    suitable = False
    stop_reason = 'no_suitable_candidate'
    rounds = 0

    if not load_cached:
        vlm=VLMSession(VLMInterface(),log_enabled=True,log_dir=os.path.join(out_dir, 'vlm_logs'))
        incumbent = None
        images_by_id = {}
        last_images = []
        random_count = cands
        radius_m = 0.2
        critic_explanation = 'Explore both close contact and standoff placements.'
        for it in range(iters):
            print(f'Iter {it+1}/{iters}')
            rand_n = cands if it == 0 else random_count
            cand = _random_candidates(rand_n, it, radius_m, incumbent, initial=(it == 0))
            requested_proposals = cands - rand_n
            accepted_proposals = 0
            if requested_proposals:
                proposals = propose_new_candidates(
                    vlm, history, last_images, task, object_name, requested_proposals,
                    iters, it+1, view_description, critic_explanation,
                    (base_pose, fn_pt_m, target_w, scene.cam_extrinsics))
                accepted_proposals = min(len(proposals), requested_proposals)
                cand.extend(proposals[:requested_proposals])
                if len(cand) < cands:
                    warnings.warn(f"Only {len(proposals)} valid proposals; filling with random candidates.")
                    cand.extend(_random_candidates(cands-len(cand), it, radius_m, incumbent))

            batch_images=[]
            for c in cand:
                pose = place_pose(
                    base_pose, fn_pt_m,
                    c.yaw, c.pitch, c.roll,
                    c.tx, c.ty, c.tz,
                    target_w
                )
                img=render_candidate(mesh,pose,scene,c.id,multi_view=(views == 3),
                                     scene_representation=scene_representation,
                                     axes_origin=fn_pt_m, effect_point=target_w,
                                     functional_axis=fn_axis)
                batch_images.append(img)
                images_by_id[c.id] = img
                img.save(f'{cache_dir}/iter{it+1}_cand{c.id}.jpg')

                # — compute world‐frame penetration —
                world_pose = scene.cam_extrinsics @ pose
                verts = np.asarray(mesh.vertices)
                hom_verts = np.hstack([verts, np.ones((verts.shape[0], 1))])  # (N,4)
                world_pts = (world_pose @ hom_verts.T).T[:, :3]
                min_z = world_pts[:, 2].min()
                c.table_penetration = max(0.0, -min_z)

            shown = cand + ([incumbent] if incumbent is not None else [])
            shown_images = batch_images + ([images_by_id[incumbent.id]] if incumbent is not None else [])
            pen_lines = [
                f"ID {c.id}: {c.table_penetration:.3f}"
                for c in shown
            ]
            pen_table = "\n".join(pen_lines)

            eval_prompt = (EVALUATION_TEMPLATE
                           .replace('{OBJECT_NAME}', object_name)
                           .replace('{TASK_DESCRIPTION}', task)
                           .replace('{PENETRATION_TABLE}', pen_table)
                           .replace('{ROUND}', str(it+1))
                           .replace('{MAX_ROUNDS}', str(iters))
                           .replace('{NUM_CANDIDATES}', str(cands))
                           .replace('{MAX_RANDOM_COUNT}', str(cands-1))
                           .replace('{INCUMBENT_NOTE}',
                                    f'ID {incumbent.id} is the previous incumbent; compare it directly with the new batch.'
                                    if incumbent is not None else 'This is the first batch; there is no incumbent yet.')
                           )
            eval_prompt = view_description + "\n" + eval_prompt
            response = vlm.query(shown_images, eval_prompt, include_history=False, client='gpt')
            critique = _critic_result(response, shown, cands, radius_m)
            if critique['chosen'] is not None:
                incumbent = critique['chosen']
                suitable = critique['suitable']
            else:
                if incumbent is None:
                    incumbent = cand[0]
                suitable = False
            for c in shown:
                print(f"ID{c.id}: params={c.params}, score={c.score}, "
                      f"table_pen={c.table_penetration:.3f}, desc={c.description}")
            history.extend(cand)
            rounds = it + 1
            random_count = critique['next_random_count']
            radius_m = critique['exploration_radius_m']
            critic_explanation = critique['explanation']
            last_images = shown_images
            shown_geometry = {
                c.id: _world_height_and_lift(c, base_pose, fn_pt_m, target_w,
                                             scene.cam_extrinsics)
                for c in shown
            }
            search_trace.append({
                'round': rounds, 'new_ids': [c.id for c in cand],
                'incumbent_id_shown': shown[-1].id if len(shown) > len(cand) else None,
                'random_count': cands - accepted_proposals,
                'proposed_count': accepted_proposals,
                'requested_proposed_count': requested_proposals,
                'scored_ids': critique['scored_ids'],
                'candidates': {str(c.id): {
                    'params': c.params, 'score': c.score,
                    'description': c.description,
                    'table_penetration_m': c.table_penetration,
                    'functional_point_height_m': shown_geometry[c.id][0],
                    'local_delta_for_world_up_5cm': shown_geometry[c.id][1].tolist(),
                } for c in shown},
                'chosen_id': incumbent.id, 'critic_best_id_valid': critique['valid_best_id'],
                'suitable': suitable, 'stop_requested': critique['stop'],
                'next_random_count': random_count,
                'exploration_radius_m': radius_m,
                'explanation': critic_explanation,
            })
            if critique['stop']:
                stop_reason = 'suitable'
                break

        best = incumbent
        if stop_reason != 'suitable':
            stop_reason = 'max_rounds' if suitable else 'no_suitable_candidate'
        best_pose = place_pose(
            base_pose, fn_pt_m,
            best.yaw, best.pitch, best.roll,
            best.tx, best.ty, best.tz,
            target_w, keep_z=False,
        )

    else:
        # load cached best candidate
        with open(f'{out_dir}/best_candidate.json','r') as f:
            data = json.load(f)
            best = Candidate3D(
                data['yaw'],
                data['pitch'],
                data['roll'],
                data['tx'],
                data['ty'],
                data['tz'],
                data['generation'],
                data['id'],
                data['score'],
                data['description']
            )
        best.table_penetration = data.get('table_penetration', 0.0)
        suitable = data.get('suitable', False)
        stop_reason = data.get('stop_reason', 'cached')
        rounds = data.get('rounds', 0)
        search_trace = data.get('search_trace', [])
        best_pose = place_pose(
            base_pose, fn_pt_m,
            best.yaw, best.pitch, best.roll,
            best.tx, best.ty, best.tz,
            target_w, keep_z=False,
        )
        print('Loaded cached best candidate:', best.params, 'score', best.score)

    # Express the goal in the world frame for the saved result.
    best_pose_world = scene.cam_extrinsics @ best_pose
    print("pose world: ", best_pose_world)
    if visualize:
        mesh_world = mesh.copy()
        mesh_world.apply_transform(best_pose_world)
        scene_pcd = scene.get_pcd(transform=True)
        o3d_mesh_world = trimesh_to_o3d(mesh_world, keep_color=False)
        o3d_mesh_world.compute_vertex_normals()
        o3d.visualization.draw_geometries([
            o3d_mesh_world, scene_pcd], window_name="Best Candidate Mesh")
    
    # Get target object's center in world frame
    target_obj = scene.objects[1]

    pcd = target_obj.get_pcd()
    points = np.asarray(pcd.points)
    center = np.median(points, axis=0)
    print("Target object center in world frame:", center)
    
    # compute relative transform (object->best)
    rel_T = np.linalg.inv(scene.cam_extrinsics @ base_pose) @ best_pose_world

    # save final image & JSON
    final_img = render_candidate(mesh, best_pose, scene, None, multi_view=(views == 3),
                                 scene_representation=scene_representation, effect_point=target_w)
    os.makedirs(out_dir, exist_ok=True)
    final_img.save(f'{out_dir}/best_candidate.jpg')
    if not load_cached:
        images_by_id[best.id].save(f'{out_dir}/best_candidate_axes.jpg')
    summary = {
        **best.__dict__,
        'suitable': suitable,
        'stop_reason': stop_reason,
        'rounds': rounds,
        'search_trace': search_trace,
        'transform': best_pose.tolist(),
        'transform_world': best_pose_world.tolist(),
        'fn_pt_mesh': fn_pt_m.tolist(),
        'target_world': (scene.cam_extrinsics @ np.append(target_w, 1.0))[:3].tolist(),
        'target_camera': target_w.tolist(),
        'relative_transform': rel_T.tolist(),
        'target_obj_center': center.tolist(),
    }
    with open(f'{out_dir}/best_candidate.json','w') as f:
        json.dump(summary, f, indent=2)

    print('Saved best candidate:', best.params, 'score', best.score)

# ───────────────────────── CLI ─────────────────────────

def main():
    """Entry point: parse arguments and run VLM-CEM with user-provided task inputs."""
    ap = argparse.ArgumentParser(description='Goal pose search (VLM-CEM): full 6-DoF pose around the effect point')
    ap.add_argument('--scene-dir', dest='scene_dir', type=str, required=True, help='Path to scene directory (contains scene.pkl)')
    ap.add_argument('--task', type=str, default=None, help='Task description for goal generation (default: load from scene_meta.json)')
    ap.add_argument('--object-name', dest='object_name', type=str, default=None, help='Human-readable object name for prompts (default: load from scene_meta.json)')
    ap.add_argument('--keypoints-dir', dest='keypoints_dir', type=str, default=None, help='Path to keypoints directory for this scene (default: <scene_dir>/keypoints)')
    ap.add_argument('--output-dir', dest='output_dir', type=str, default=None, help='Optional output directory for goal outputs (default: <scene_dir>/goal_gen/vlm_cem)')
    ap.add_argument('--iters', type=int, default=5, help='Maximum search rounds (critic may stop earlier)')
    ap.add_argument('--cands', type=int, default=8, help='New candidates per round (default: 8)')
    ap.add_argument('--views', type=int, choices=(1, 3), default=3,
                    help='Views per candidate: 3 (camera, turned 90 degrees, top; default) or 1 (camera)')
    ap.add_argument('--scene-representation', choices=('points', 'surface'), default='points',
                    help='Scene renderer input (default: points)')
    ap.add_argument('--load-cached', action='store_true', help='Reuse best_candidate.json from output_dir instead of generating new candidates')
    ap.add_argument('--visualize', action='store_true', help='Show the final pose in an Open3D window')
    a = ap.parse_args()

    scene_dir = os.path.abspath(a.scene_dir)

    generate_3d_goal(
        task=a.task,
        scene_dir=scene_dir,
        object_name=a.object_name,
        iters=a.iters,
        cands=a.cands,
        load_cached=a.load_cached,
        views=a.views,
        scene_representation=a.scene_representation,
        visualize=a.visualize,
        keypoints_dir=os.path.abspath(a.keypoints_dir) if a.keypoints_dir else None,
        output_dir=os.path.abspath(a.output_dir) if a.output_dir else None,
    )

if __name__=='__main__':
    main()
