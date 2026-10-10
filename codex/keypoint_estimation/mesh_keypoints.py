"""Joint functional/actuation prediction on a shared set of mesh views."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from codex.keypoint_estimation.pointing import create_point_backend


TASK_POINTS = {
    "fn": "the functional outlet or working surface of the functional object that acts on the task target",
    "act": "the surface pressed by a finger to actuate the mechanism",
}


def predict_views(views, backend, task, object_name, descriptions=None, workers=16, save_dir=None):
    """Make one joint Sol request per view, or serial local-model predictions.

    API failures propagate instead of being mislabeled as invisible views.
    Histories and mutable VLMSessions are deliberately absent from the workers.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    descriptions = descriptions if descriptions is not None else TASK_POINTS
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def query(idx):
        result = backend.predict(Image.fromarray(views[idx]), task=task,
                                 object_name=object_name, descriptions=descriptions)
        if set(result) != set(descriptions):
            raise ValueError(f"View {idx}: prediction omitted a requested keypoint")
        for kind, record in result.items():
            point = record["point"]
            if record["visible"]:
                if point is None or len(point) != 2 or not np.all(np.isfinite(point)) or not all(0 <= x <= 100 for x in point):
                    raise ValueError(f"View {idx}: invalid {kind} coordinates")
            elif point is not None:
                raise ValueError(f"View {idx}: invisible {kind} must have no coordinates")
        if save_dir:
            (Path(save_dir) / f"view_{idx:02d}.json").write_text(json.dumps(result, indent=2))
        return idx, result

    indices = sorted(views)
    if getattr(backend, "concurrent_safe", False):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = dict(pool.map(query, indices))
    else:
        results = dict(query(idx) for idx in indices)
    return results


def estimate_mesh_keypoints(mesh_path, task, object_name, output_dir,
                            point_query_model="sol", workers=16, backend=None,
                            n_azimuth=6, n_elevation=5,
                            min_elevation=-60.0, max_elevation=60.0):
    """Return canonical fn/act contacts; both use the same cameras and recovery."""
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    from codex.mesh.load_mesh import load_mesh, trimesh_to_o3d
    from codex.keypoint_estimation.triangulation.point3d_estimator import Point3DEstimator
    from codex.keypoint_estimation.estimate_actuation_point import reconstruct_contact
    from codex.keypoint_estimation.functional_direction import select_functional_direction

    if n_azimuth < 1 or n_elevation < 1 or not -90 < min_elevation <= max_elevation < 90:
        raise ValueError("Use positive view counts and elevations strictly between -90 and 90 degrees")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh, _ = load_mesh(str(mesh_path))
    o3d_mesh = trimesh_to_o3d(mesh, keep_color=True)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    config = dict(mesh_sha256=hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest(),
                  n_azimuth=n_azimuth, n_elevation=n_elevation,
                  min_elevation=min_elevation, max_elevation=max_elevation)
    cache_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    estimator = Point3DEstimator(o3d_mesh, "mesh", output_dir=str(output_dir),
        subdir_name=f"mesh_view_cache/{cache_id}", n_azimuth=n_azimuth,
        n_elevation=n_elevation, min_elevation=min_elevation, max_elevation=max_elevation)
    views = estimator.render_mesh_views()
    backend = backend if backend is not None else create_point_backend(point_query_model)
    run_dir = output_dir / "view_predictions" / point_query_model
    results = predict_views(views, backend, task, object_name, workers=workers, save_dir=run_dir)
    (run_dir / "manifest.json").write_text(json.dumps(dict(
        **config, task=task, object_name=object_name, backend=point_query_model,
        workers=workers if getattr(backend, "concurrent_safe", False) else 1,
        view_count=len(views), cache_dir=estimator.cache_dir,
    ), indent=2))

    contacts = {}
    for kind in TASK_POINTS:
        annotations = {str(i): [r[kind]["point"]] if r[kind]["visible"] else [] for i, r in results.items()}
        point_dir = output_dir / f"{kind}_point_cache"
        point_dir.mkdir(parents=True, exist_ok=True)
        (point_dir / "auto_annotations.json").write_text(json.dumps(annotations, indent=2))
        # Geometry is shared; outputs and selected points belong to this kind.
        estimator.output_dir = str(point_dir)
        estimator.points_2d = {int(i): pts for i, pts in annotations.items() if pts}
        estimator.points_3d = []
        estimator.points_3d_free = []
        contact = reconstruct_contact(estimator, mesh, kind, visualize=False,
                                      contact_name=kind, include_direction=(kind == "act"))
        if contact is None:
            raise RuntimeError(f"No visible {kind} predictions hit the mesh. Inspect {point_dir}; hidden views are not guessed.")
        contacts[kind] = contact
    selection = select_functional_direction(
        mesh, contacts['fn']['pos'], task, object_name, views,
        estimator.camera_data, output_dir / 'functional_direction',
        visible_view_ids=[i for i, r in results.items() if r['fn']['visible']],
    )
    contacts['fn']['dir'] = selection['direction']
    contacts['fn']['direction_axis'] = selection['axis']
    contacts['fn']['direction_source'] = 'sol_canonical_axis'
    for suffix in ('', '_single', '_all'):
        path = output_dir / 'fn_point_cache' / f'{estimator.mesh_name}_fn{suffix}.json'
        if path.exists():
            data = json.loads(path.read_text())
            for record in data.get('actuation_contacts', []):
                record.update(dir=selection['direction'], direction_axis=selection['axis'],
                              direction_source='sol_canonical_axis')
            path.write_text(json.dumps(data, indent=2))
    return contacts
