from __future__ import annotations

from typing import Any
import argparse
import json
import os
from codex.mesh.load_mesh import load_mesh, trimesh_to_o3d
from codex.mesh.mesh_to_coacd import run_coacd




def _normalize_mesh_filename(mesh_path: str) -> str:
    """Rename mesh to mesh.<ext> in-place when needed and return canonical path."""
    mesh_dir = os.path.dirname(mesh_path)
    _, ext = os.path.splitext(mesh_path)
    if not ext:
        raise ValueError(f"Mesh file must have an extension: {mesh_path}")

    canonical_path = os.path.join(mesh_dir, f"mesh{ext.lower()}")
    if os.path.abspath(mesh_path) == os.path.abspath(canonical_path):
        return canonical_path

    # Replace any pre-existing canonical file so the requested mesh becomes canonical.
    os.replace(mesh_path, canonical_path)
    print(f"Renamed mesh to canonical filename: {canonical_path}")
    return canonical_path


def post_mesh_metadata(mesh_path: str) -> None:
    """Compute bbox, write metadata.json, and run CoACD decomposition for a mesh."""
    mesh_path = os.path.abspath(mesh_path)
    mesh_path = _normalize_mesh_filename(mesh_path)
    mesh_dir = os.path.dirname(mesh_path)
    metadata_path = os.path.join(mesh_dir, "metadata.json")

    mesh, metadata = load_mesh(mesh_path)
    mesh_o3d = trimesh_to_o3d(mesh)
    extent: list[float] = [float(v) for v in mesh_o3d.get_axis_aligned_bounding_box().get_extent()]

    metadata_dict: dict[str, Any] = dict(metadata)
    metadata_dict["bbox"] = extent
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"Saved metadata to {metadata_path}")

    try:
        coacd_dir = run_coacd(mesh_path, output_dir=os.path.join(mesh_dir, "..", "mesh_coacd"))
        print(f"CoACD mesh saved to {coacd_dir}")
    except ImportError as exc:
        raise RuntimeError(
            "CoACD is required (training uses the decomposed collision "
            "mesh). Install it with: pip install coacd"
        ) from exc


def _resolve_mesh_path(scene_dir: str, mesh_file: str) -> str:
    return os.path.join(os.path.abspath(scene_dir), "mesh", mesh_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a mesh after manual replacement by recomputing metadata bbox "
            "and CoACD decomposition."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--scene-dir",
        type=str,
        default=None,
        help="Scene directory containing mesh/<mesh-file> (e.g. out/scenes/test_spray).",
    )
    src.add_argument(
        "--mesh-path",
        type=str,
        default=None,
        help="Direct path to a mesh .glb file.",
    )
    parser.add_argument(
        "--mesh-file",
        type=str,
        default="mesh.glb",
        help="Mesh filename inside <scene-dir>/mesh/ when --scene-dir is used.",
    )
    args = parser.parse_args()

    mesh_path = args.mesh_path or _resolve_mesh_path(args.scene_dir, args.mesh_file)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    post_mesh_metadata(mesh_path)
