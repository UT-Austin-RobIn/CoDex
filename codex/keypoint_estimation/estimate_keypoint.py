"""Stage 1: keypoints on the object mesh, pointed out by GPT (default, "sol") or MolmoPoint."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from codex.scene.image_scene import ImageScene
from codex.keypoint_estimation.mesh_keypoints import estimate_mesh_keypoints
from codex.keypoint_estimation.pointing import create_point_backend
from codex.keypoint_estimation.utils import annotate_img
from codex.utils.utils_path import (
    get_scene_pkl_path, get_mesh_dir_from_scene, load_scene_meta, write_scene_meta,
)


def project_fn_to_scene(point, scene, scene_dir):
    """Project a canonical mesh point for the 2D overlay, when registration exists.

    The authoritative fn output is fn_pt_mesh.json; this projection does not
    claim that the point is visible in the original scene photograph.
    """
    pose_path = Path(scene_dir) / 'pose.txt'
    if scene.cam_intrinsics is None or not pose_path.exists():
        return []
    pose = np.loadtxt(pose_path)
    camera_point = (pose @ np.append(point, 1.0))[:3]
    if not np.all(np.isfinite(camera_point)) or camera_point[2] <= 0:
        return []
    pixel = np.asarray(scene.cam_intrinsics) @ camera_point
    pixel = pixel[:2] / pixel[2]
    coords = pixel / np.array(scene.image.size) * 100
    return [coords.tolist()] if np.all((coords >= 0) & (coords <= 100)) else []


def run_query(session, scene, task, visualize=False, save_dir=None,
              point_query_model="sol", scene_dir=None, functional_object=None,
              target_object=None, workers=16, n_azimuth=6, n_elevation=5,
              min_elevation=-60.0, max_elevation=60.0, backend=None):
    """Estimate fn/act in canonical mesh coordinates and target in scene pixels.

    ``session`` is retained for callers of the previous entry point; keypoint
    requests use an independent backend so parallel calls never share history.
    Returns (projected_fn_percentages, target_percentages, act_positions_3d).
    """
    if scene_dir is None:
        raise ValueError("scene_dir is required for shared fn/act mesh-view estimation")
    if not task:
        raise ValueError("A task description is required")
    if len(scene.objects) < 2:
        raise ValueError("The scene must contain the functional and target objects")
    scene_dir = os.path.abspath(scene_dir)
    save_dir = os.path.abspath(save_dir or os.path.join(scene_dir, 'keypoints'))
    meta = load_scene_meta(scene_dir)
    object_name = functional_object or meta.get('functional_object') or scene.objects[0].name
    target_name = target_object or meta.get('target_object') or scene.objects[1].name
    if not object_name or not target_name:
        raise ValueError("Functional and target object names are required")
    target_obj = next((obj for obj in scene.objects if obj.name == target_name), scene.objects[1])
    backend = backend if backend is not None else create_point_backend(point_query_model)
    target = backend.predict(
        target_obj.get_cropped(), task=task, object_name=target_name,
        descriptions={'target': f'the destination on the {target_name} where the {object_name} functional point should move to complete the task'},
    )['target']
    if not target['visible'] or target['point'] is None:
        raise RuntimeError("The task target is not visible in its scene crop; no target point was guessed")
    target_pts = target_obj.map_coords([target['point']])
    contacts = estimate_mesh_keypoints(
        os.path.join(get_mesh_dir_from_scene(scene_dir), 'mesh.glb'), task,
        object_name, save_dir, point_query_model=point_query_model, workers=workers,
        backend=backend, n_azimuth=n_azimuth, n_elevation=n_elevation,
        min_elevation=min_elevation, max_elevation=max_elevation,
    )
    fn_contact = contacts['fn']
    fn_position = fn_contact['pos']
    fn_direction = np.asarray(fn_contact['dir'], dtype=float)
    if fn_direction.shape != (3,) or not np.all(np.isfinite(fn_direction)) or not np.isclose(np.linalg.norm(fn_direction), 1.0, atol=1e-5):
        raise ValueError('Functional contact must include a finite unit outward direction')
    act_contact = contacts['act']
    fn_pts = project_fn_to_scene(fn_position, scene, scene_dir)
    data = {
        'act_pts_3d': [act_contact['pos']],
        'act_contact': act_contact,
        'fn_contact': fn_contact,
        'fn_direction': fn_direction.tolist(),
        'fn_pts': fn_pts,
        'fn_pts_3d': [fn_position],
        'fn_pts_source': 'mesh_projection',
        'target_pts': target_pts,
        'point_query_model': point_query_model,
    }
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Always replace an earlier cached fn point: goal generation reads this file
    # directly and otherwise would silently reuse a previous method's result.
    (output / 'fn_pt_mesh.json').write_text(json.dumps(fn_position, indent=4))
    (output / 'fn_key_pts.json').write_text(json.dumps(data, indent=4))
    (output / 'target_prediction.json').write_text(json.dumps(target, indent=2))
    annotate_img(scene.image, fn_pts + target_pts,
                 save_path=str(output / 'functional_pts_annotations.png'))
    return fn_pts, target_pts, data['act_pts_3d']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', required=True)
    parser.add_argument('--task', help='Default: task from scene_meta.json')
    parser.add_argument('--output-dir', help='Default: <scene-dir>/keypoints')
    parser.add_argument('--functional-object')
    parser.add_argument('--target-object')
    parser.add_argument('--point-query-model', choices=['sol', 'molmopoint'], default='sol',
                        help='sol: GPT (OPENAI_API_KEY; model from CODEX_GPT_MODEL); molmopoint: local MolmoPoint')
    parser.add_argument('--workers', type=int, default=16, help='Concurrent GPT view requests; local inference stays serial')
    parser.add_argument('--n-azimuth', type=int, default=6)
    parser.add_argument('--n-elevation', type=int, default=5)
    parser.add_argument('--min-elevation', type=float, default=-60.0)
    parser.add_argument('--max-elevation', type=float, default=60.0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    scene_dir = os.path.abspath(args.scene_dir)
    meta = load_scene_meta(scene_dir)
    task = args.task or meta.get('task')
    if not task:
        parser.error('Pass --task or set the task in scene_meta.json')
    scene = ImageScene.load(get_scene_pkl_path(scene_dir))
    run_query(None, scene, task, save_dir=args.output_dir, scene_dir=scene_dir,
              point_query_model=args.point_query_model, workers=args.workers,
              functional_object=args.functional_object, target_object=args.target_object,
              n_azimuth=args.n_azimuth, n_elevation=args.n_elevation,
              min_elevation=args.min_elevation, max_elevation=args.max_elevation)
    write_scene_meta(scene_dir,
                     functional_object=args.functional_object or meta.get('functional_object', scene.objects[0].name),
                     target_object=args.target_object or meta.get('target_object', scene.objects[1].name),
                     task=task)


if __name__ == '__main__':
    main()
