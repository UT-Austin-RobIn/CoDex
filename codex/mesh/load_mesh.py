"""Mesh loading and conversion utilities.

Scene meshes are stored canonical: gravity-up along +Z and metric scale,
baked into the file at scene creation (see
MeshPoseEstimator.register_and_canonicalize). Loading therefore applies no
transforms; this module only merges multi-part files (e.g. CoACD output)
and returns the optional metadata (bbox, com, ...) stored alongside.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
import argparse


def load_mesh(mesh_path: str) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Load a canonical scene mesh and its metadata.

    Args:
        mesh_path: Full path to the mesh file (e.g. /path/to/mesh/mesh.glb or .obj).
            The mesh format is inferred from the file extension. ``metadata.json``
            is read from the same directory when present.

    Returns:
        (mesh, metadata):
            mesh: Loaded `trimesh.Trimesh` (multi-part files are merged).
            metadata: Parsed JSON metadata dictionary ({} when absent).
    """
    mesh_path = os.path.abspath(mesh_path)
    metadata_path = os.path.join(os.path.dirname(mesh_path), "metadata.json")

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    loaded = trimesh.load(mesh_path)
    if isinstance(loaded, trimesh.Scene):
        # Multi-part files (e.g. CoACD output) are merged; single-mesh scenes
        # are unaffected.
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded

    metadata: dict[str, Any] = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Meshes used to be stored raw, with the canonical transform described by
    # metadata and re-applied on every load. Refuse such data instead of
    # silently loading it un-transformed.
    if metadata.get("orientation") is not None or any(
        d is not None for d in metadata.get("dimensions", [])
    ):
        raise ValueError(
            f"{metadata_path} carries legacy transform metadata "
            "(orientation/dimensions); this mesh predates canonical storage. "
            "Re-register it: python -m codex.mesh.estimate_mesh_pose "
            "--scene-dir <scene_dir>, then codex/scene/post_mesh_metadata.py."
        )

    return mesh, metadata


def trimesh_to_o3d(
    trimesh_mesh: trimesh.Trimesh,
    keep_color: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Convert a `trimesh` mesh into an Open3D mesh.

    Args:
        trimesh_mesh: Input mesh.
        keep_color: If True, attempt to preserve per-vertex colors/texture-derived colors.

    Returns:
        mesh_o3d: Open3D triangle mesh.
    """
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(trimesh_mesh.vertices))

    if hasattr(trimesh_mesh, "faces") and trimesh_mesh.faces is not None:
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(trimesh_mesh.faces, dtype=np.int32))

    mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])
    if keep_color:
        if hasattr(trimesh_mesh.visual, "vertex_colors") and trimesh_mesh.visual.vertex_colors is not None:
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(trimesh_mesh.visual.vertex_colors[:, :3] / 255.0)
        elif isinstance(trimesh_mesh.visual, trimesh.visual.texture.TextureVisuals):
            try:
                colors = trimesh_mesh.visual.to_color().vertex_colors
                if colors is not None and len(colors) == len(trimesh_mesh.vertices):
                    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0)
            except Exception:
                mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])

    mesh_o3d.compute_vertex_normals()
    mesh_o3d.compute_triangle_normals()
    return mesh_o3d


def main() -> None:
    """CLI entrypoint to load and visualize a mesh."""
    parser = argparse.ArgumentParser(description="Load and visualize a mesh")
    parser.add_argument(
        "--mesh-path",
        required=True,
        help="Path to the mesh file (e.g. /path/to/mesh/mesh.glb).",
    )
    parser.add_argument("--keep-color", action="store_true", help="Preserve vertex colors when possible")
    args = parser.parse_args()

    mesh, metadata = load_mesh(args.mesh_path)
    mesh_name = os.path.splitext(os.path.basename(args.mesh_path))[0]
    print("Loaded mesh:", mesh_name)
    print("Metadata keys:", sorted(metadata.keys()))

    mesh_o3d = trimesh_to_o3d(mesh, keep_color=args.keep_color)
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    print(f"Bounding-box extent (meters): {bbox.get_extent()}")

    o3d.visualization.draw_geometries(
        [mesh_o3d],
        window_name=f"Mesh: {mesh_name}",
    )


if __name__ == "__main__":
    main()
