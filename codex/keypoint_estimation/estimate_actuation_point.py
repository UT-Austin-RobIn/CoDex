import os
import json
import numpy as np
from PIL import Image
import argparse
from typing import Optional, Dict
from codex.mesh.load_mesh import load_mesh, trimesh_to_o3d
from codex.keypoint_estimation.triangulation.point3d_estimator import Point3DEstimator
from codex.keypoint_estimation.triangulation.vis_utils import annotate_img
from codex.utils.utils_path import OUT_DIR
import matplotlib.pyplot as plt
from codex.vlm import VLMSession
import trimesh as _trimesh

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

DEFAULT_CACHE_OUTPUT_DIR = os.path.join(OUT_DIR, "actuation_3d")


def render_views_for_annotation(
    mesh_path: str,
    output_dir: str = DEFAULT_CACHE_OUTPUT_DIR,
    n_azimuth: int = 6,
    n_elevation: int = 3,
    min_elevation: float = 0.0,
    radius: float = None,
    image_width: int = 640,
    image_height: int = 480,
) -> str:
    """Render mesh views and save them for manual 2D annotation."""
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    trimesh_mesh, _ = load_mesh(mesh_path)
    mesh_o3d = trimesh_to_o3d(trimesh_mesh, keep_color=True)
    mesh_o3d.compute_vertex_normals()

    estimator = Point3DEstimator(
        mesh=mesh_o3d, mesh_name=mesh_name, output_dir=output_dir,
        n_azimuth=n_azimuth, n_elevation=n_elevation, min_elevation=min_elevation,
        radius=radius, image_width=image_width, image_height=image_height,
    )

    print(f"Rendering {n_azimuth * n_elevation} views of {mesh_name}...")
    views = estimator.render_mesh_views()

    # Save a grid overview
    views_dir = os.path.join(estimator.output_dir, "rendered_views")
    n_views = len(views)
    cols = min(4, n_views)
    rows = (n_views + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    for i in range(n_views):
        r, c = i // cols, i % cols
        axes[r, c].imshow(views[i])
        axes[r, c].set_title(f"View {i}")
        axes[r, c].axis('off')
    for i in range(n_views, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis('off')
    plt.tight_layout()
    grid_path = os.path.join(estimator.output_dir, "views_grid.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()

    template = {str(i): [] for i in range(n_views)}
    template_path = os.path.join(estimator.output_dir, "annotations_template.json")
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Views rendered to: {views_dir}")
    print(f"Views grid saved to: {grid_path}")
    print(f"Annotation template saved to: {template_path}")
    print(f"\nTo annotate: fill in [X%, Y%] coords (0-100) in {template_path}")
    print(f"Then run with --annotations {template_path}")
    return views_dir


def auto_annotate_views(
    views: Dict[int, np.ndarray],
    description: str,
    point_query_model: str = "sol",
    save_dir: Optional[str] = None,
    session: Optional[VLMSession] = None,
    workers: int = 16,
) -> Dict[str, list]:
    """Locate actuation contacts directly, retaining hidden views as empty."""
    from codex.keypoint_estimation.pointing import create_point_backend
    from codex.keypoint_estimation.mesh_keypoints import predict_views
    backend = create_point_backend(point_query_model)
    results = predict_views(views, backend, task="Identify the actuation contact",
                           object_name="object", descriptions={"act": description},
                           workers=workers, save_dir=save_dir)
    annotations = {str(i): [result["act"]["point"]] if result["act"]["visible"] else []
                   for i, result in results.items()}
    if save_dir:
        with open(os.path.join(save_dir, "auto_annotations.json"), "w") as f:
            json.dump(annotations, f, indent=2)
    return annotations


def run_actuation_point_estimation(
    mesh_path: str,
    description: str = "",
    annotations_path: Optional[str] = None,
    auto_annotate: Optional[str] = None,
    output_dir: str = DEFAULT_CACHE_OUTPUT_DIR,
    n_azimuth: int = 6,
    n_elevation: int = 3,
    min_elevation: float = 0.0,
    radius: float = None,
    visualize: bool = True,
    cache_subdir: Optional[str] = None,
    session: Optional[VLMSession] = None,
) -> Optional[np.ndarray]:
    """Estimate 3D actuation points by projecting 2D annotations onto the mesh.

    Args:
        mesh_path: Full path to the mesh file. metadata.json must be in the same directory.
        description: Short description of the point to find (e.g. "trigger").
            Used for auto-annotation and naming the output folder.
        annotations_path: Path to JSON file with 2D annotations per view.
            Format: {"0": [[x_pct, y_pct], ...], "1": [...], ...}
        auto_annotate: Point backend ("sol" or "molmopoint").
            Requires description.
        output_dir: Single output root directory for all artifacts,
            including rendered views, annotations, and final
            <mesh_name>_actuation.json.
        n_azimuth: Number of azimuth views per ring
        n_elevation: Number of elevation rings
        radius: Camera orbit radius. None = auto-scale to mesh size.
        visualize: Whether to open interactive 3D visualization
    """
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    assert mesh_name, "mesh_path must point to a named file"

    # If no annotations and no auto-annotate, render views for manual annotation
    if annotations_path is None and auto_annotate is None:
        render_views_for_annotation(
            mesh_path, output_dir, n_azimuth, n_elevation, radius,
        )
        return None

    print(f"Estimating 3D actuation points on {mesh_name}...")
    trimesh_mesh, _ = load_mesh(mesh_path)
    mesh_o3d = trimesh_to_o3d(trimesh_mesh, keep_color=True)
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.compute_triangle_normals()

    estimator = Point3DEstimator(
        mesh=mesh_o3d, mesh_name=mesh_name, output_dir=output_dir,
        n_azimuth=n_azimuth, n_elevation=n_elevation, min_elevation=min_elevation,
        radius=radius, subdir_name=cache_subdir,
    )

    print("Rendering mesh views...")
    views = estimator.render_mesh_views()
    print(f"Rendered {len(views)} views")

    # Load or generate 2D annotations
    if auto_annotate is not None:
        assert description, "--description is required for auto-annotation"
        print(f"Auto-annotating with {auto_annotate}...")
        auto_annotate_views(
            views, description,
            point_query_model=auto_annotate,
            save_dir=estimator.output_dir,
            session=session,
        )
        annotations_path = os.path.join(estimator.output_dir, "auto_annotations.json")
    estimator.load_annotations(annotations_path)

    return reconstruct_contact(estimator, trimesh_mesh, description, visualize=visualize)


def estimate_contact_direction(mesh, point, outward=False, radius=0.010):
    """Estimate the press direction at a mesh point, optionally reversing outward.

    The inward direction uses the established area-clustered nearby face normals
    and finger-clearance arbitration. Functional directions use the same
    calculation at the functional point, with its sign reversed.
    """
    point = np.asarray(point, dtype=float).reshape(3)
    if not np.all(np.isfinite(point)):
        raise ValueError("contact point must contain three finite coordinates")
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces
    normal_lengths = np.linalg.norm(face_normals, axis=1)
    valid_faces = (np.isfinite(face_areas) & (face_areas > 0) &
                   np.isfinite(normal_lengths) & (normal_lengths > 0))
    if not np.any(valid_faces):
        raise ValueError("mesh has no valid face normals for contact direction")
    near = np.where(np.linalg.norm(mesh.triangles_center - point, axis=1) < radius)[0]
    near = near[valid_faces[near]]
    if len(near) == 0:
        _, _, tri = _trimesh.proximity.closest_point(mesh, point[None])
        face_index = int(tri[0])
        if not valid_faces[face_index]:
            valid_indices = np.flatnonzero(valid_faces)
            closest = np.argmin(np.linalg.norm(
                mesh.triangles_center[valid_indices] - point, axis=1))
            face_index = int(valid_indices[closest])
        press = -face_normals[face_index]
    else:
        normals = face_normals[near]
        areas = face_areas[near]
        clusters = []
        for i in np.argsort(-areas):
            n, a = normals[i], areas[i]
            for c in clusters:
                if np.dot(c[0] / np.linalg.norm(c[0]), n) > np.cos(np.radians(35)):
                    c[0] += n * a
                    c[1] += a
                    break
            else:
                clusters.append([n * a, a])
        clusters.sort(key=lambda c: -c[1])

        def clearance(press_direction):
            ts = np.linspace(0.010, 0.045, 5)
            centers = point[None] - press_direction[None] * ts[:, None]
            _, dist, _ = _trimesh.proximity.closest_point(mesh, centers)
            return float(np.min(dist))

        scored = []
        for c in clusters[:3]:
            inward = -(c[0] / np.linalg.norm(c[0]))
            scored.append((inward, clearance(inward)))
        clear = [s for s in scored if s[1] > 0.007]
        press = (clear[0] if clear else scored[0])[0]

    direction = -press if outward else press
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("could not estimate a finite contact direction")
    return direction / length


def reconstruct_contact(estimator, trimesh_mesh, description, visualize=False,
                        contact_name="actuation", include_direction=True,
                        direction_outward=False):
    """Recover one surface contact using the established consensus and normals."""
    if not any(estimator.points_2d.values()):
        return None
    mesh_name = estimator.mesh_name
    views = estimator.rendered_views
    # Save annotated views for verification
    annotated_dir = os.path.join(estimator.output_dir, "annotated_views")
    os.makedirs(annotated_dir, exist_ok=True)
    for idx, pts in estimator.points_2d.items():
        if pts and idx in views:
            annotate_img(
                Image.fromarray(views[idx]), pts,
                save_path=os.path.join(annotated_dir, f"view_{idx:02d}.png")
            )

    # Project 2D to 3D via ray-mesh intersection
    print("Projecting 2D points to 3D mesh surface...")
    pts3d = estimator.project_2d_to_3d(visualize=visualize)
    print(f"Done: found {len(pts3d)} 3D points.")

    # 3D consensus: points from views that truly show the part cluster tightly
    # on it; guesses from occluded or ambiguous views scatter across the mesh.
    # Keep the points near the geometric median (two refinement rounds).
    hit_tris = list(getattr(estimator, "hit_tris", []))
    if len(pts3d) >= 3:
        arr = np.array(pts3d)
        center = np.median(arr, axis=0)
        for _ in range(2):
            keep = np.linalg.norm(arr - center, axis=1) < 0.025
            if keep.sum() < 3:
                break
            center = np.median(arr[keep], axis=0)
        keep = np.linalg.norm(arr - center, axis=1) < 0.025
        if 3 <= keep.sum() < len(arr):
            print(f"  3D consensus: kept {int(keep.sum())}/{len(arr)} points "
                  f"within 2.5 cm of the median")
            pts3d = [p for p, k in zip(pts3d, keep) if k]
            if len(hit_tris) == len(keep):
                hit_tris = [t for t, k in zip(hit_tris, keep) if k]

    final_single_pt = None

    # Save in the format expected by grasp generation under the same output dir.
    if pts3d:
        vertices = np.asarray(trimesh_mesh.vertices)
        triangles = np.asarray(trimesh_mesh.faces)
        tm = _trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        pts_arr = np.array(pts3d)

        # Actuation uses the established inward press normal. Functional
        # contacts use that same local calculation with the sign reversed.
        def contact_direction(point):
            direction = estimate_contact_direction(tm, point, outward=direction_outward)
            return direction.tolist()

        actuation_contacts_all = [
            {"pos": pt.tolist(), "dir": contact_direction(pt) if include_direction else None}
            for pt in pts_arr
        ]

        # 2) Compute one final actuation point: centroid projected to mesh.
        centroid = pts_arr.mean(axis=0)
        single_pt, _, _ = _trimesh.proximity.closest_point(tm, np.array([centroid]))
        final_single_pt = single_pt[0]
        actuation_contacts_single = [
            {"pos": single_pt[0].tolist(), "dir": contact_direction(single_pt[0]) if include_direction else None}
        ]

        os.makedirs(estimator.output_dir, exist_ok=True)
        actuation_path_default = os.path.join(estimator.output_dir, f"{mesh_name}_{contact_name}.json")
        actuation_path_all = os.path.join(estimator.output_dir, f"{mesh_name}_{contact_name}_all.json")
        actuation_path_single = os.path.join(estimator.output_dir, f"{mesh_name}_{contact_name}_single.json")

        # Default file used by downstream code is single-point.
        with open(actuation_path_default, "w") as f:
            json.dump({"actuation_contacts": actuation_contacts_single}, f, indent=2)
        with open(actuation_path_all, "w") as f:
            json.dump({"actuation_contacts": actuation_contacts_all}, f, indent=2)
        with open(actuation_path_single, "w") as f:
            json.dump({"actuation_contacts": actuation_contacts_single}, f, indent=2)

        print(f"Actuation contacts (default single-point) saved to {actuation_path_default}")
        print(f"Actuation contacts (all points) saved to {actuation_path_all}")
        print(f"Actuation contacts (single point) saved to {actuation_path_single}")

    results_dir = estimator.save_results(task_name=contact_name, highlight_point=final_single_pt)
    if final_single_pt is not None:
        vis_path = os.path.join(results_dir, "visualization", "final_actuation_point.png")
        print(f"Final actuation visualization saved to {vis_path}")

    # Return the full actuation contact (pos + dir) for the single consolidated point,
    # or just the point if direction is unavailable.
    if final_single_pt is not None and pts3d:
        return {"pos": actuation_contacts_single[0]["pos"], "dir": actuation_contacts_single[0]["dir"]}
    return None


def parse_args():
    parser = argparse.ArgumentParser("3D actuation point estimation")
    parser.add_argument("--mesh-path", required=True, help="Path to the mesh file (e.g. /path/to/mesh/mesh.glb)")
    parser.add_argument("--description", default="trigger",
                        help="Short description of the point to find (e.g. 'trigger').")
    parser.add_argument("--annotations", default=None,
                        help="Path to 2D annotations JSON. If omitted, renders views for annotation.")
    parser.add_argument("--auto-annotate", default="sol", choices=["sol", "molmopoint"],
                        help="Auto-annotate views using a VLM (requires --description).")
    parser.add_argument("--output", default=DEFAULT_CACHE_OUTPUT_DIR)
    parser.add_argument("--azimuth", type=int, default=6)
    parser.add_argument("--elevation", type=int, default=2)
    parser.add_argument("--radius", type=float, default=None,
                        help="Camera orbit radius. Default: auto-scale to mesh size.")
    parser.add_argument("--no-visualize", action="store_true")
    parser.set_defaults(no_visualize=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_actuation_point_estimation(
        mesh_path=args.mesh_path,
        description=args.description,
        annotations_path=args.annotations,
        auto_annotate=None if args.annotations else args.auto_annotate,
        output_dir=args.output,
        n_azimuth=args.azimuth,
        n_elevation=args.elevation,
        radius=args.radius,
        visualize=not args.no_visualize,
    )


if __name__ == "__main__":
    main()
