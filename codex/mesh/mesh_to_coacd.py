"""Generate a CoACD (Convex Approximate Convex Decomposition) mesh from an existing mesh.

Usage:
    python -m codex.mesh.mesh_to_coacd \
        --mesh_path /path/to/mesh/mesh.glb \
        --output_dir /path/to/output \
        [--threshold 0.05]

For each mesh, the script:
  1. Loads the canonical mesh using load_mesh.
  2. Runs CoACD convex decomposition.
  3. Exports the decomposed mesh as a .glb scene into --output_dir.
  4. Copies the metadata JSON alongside the CoACD mesh.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
import trimesh
import coacd
from codex.mesh.load_mesh import load_mesh

def run_coacd(
    mesh_path: str,
    output_dir: str,
    threshold: float = 0.05,
) -> str:
    """Decompose a mesh with CoACD and save the result to output_dir.

    Args:
        mesh_path: Full path to the source mesh file. metadata.json must be in the same directory.
        output_dir: Output directory for the CoACD mesh.
        threshold: CoACD concavity threshold (lower = finer decomposition).

    Returns:
        Path to the output directory.
    """


    # --- paths -----------------------------------------------------------
    mesh_path = os.path.abspath(mesh_path)
    src_dir = Path(os.path.dirname(mesh_path))
    stem = Path(mesh_path).stem
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_json = src_dir / "metadata.json"
    dst_json = dst_dir / "metadata.json"
    dst_glb = dst_dir / f"{stem}.glb"

    if not src_json.exists():
        raise FileNotFoundError(f"Metadata not found: {src_json}")

    # --- load mesh --------------------------------------------------------
    mesh, metadata = load_mesh(mesh_path)

    # --- run CoACD -------------------------------------------------------
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
    print(f"CoACD decomposed '{stem}' into {len(parts)} convex parts")

    # --- build a trimesh Scene from the convex hulls --------------------
    scene = trimesh.Scene()
    for i, (vertices, faces) in enumerate(parts):
        convex = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        scene.add_geometry(convex, node_name=f"{stem}_part{i}")

    scene.export(str(dst_glb), file_type="glb")
    print(f"Saved CoACD mesh to {dst_glb}")

    # --- copy / update metadata -----------------------------------------
    shutil.copy2(str(src_json), str(dst_json))

    # Recompute bbox from the decomposed mesh
    all_verts = np.concatenate([p[0] for p in parts], axis=0)
    bbox_extent = all_verts.max(axis=0) - all_verts.min(axis=0)
    metadata["bbox"] = bbox_extent.tolist()

    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {dst_json}")

    return str(dst_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CoACD convex decomposition of a mesh."
    )
    parser.add_argument(
        "--mesh-path",
        dest="mesh_path",
        required=True,
        help="Path to the source mesh file (e.g. /path/to/mesh/mesh.glb). metadata.json must be in the same directory.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Output directory for CoACD mesh.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="CoACD concavity threshold (default: 0.05). Lower = finer.",
    )
    args = parser.parse_args()

    run_coacd(args.mesh_path, output_dir=args.output_dir, threshold=args.threshold)


if __name__ == "__main__":
    main()
