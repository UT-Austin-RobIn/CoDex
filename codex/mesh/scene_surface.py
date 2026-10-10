"""Build a colored, organized surface mesh from a single RGB-D scene."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def build_scene_surface(
    scene,
    exclude_objs=None,
    reverse_selection: bool = False,
    max_edge_m: float = 0.02,
    max_depth_jump_m: float = 0.02,
) -> o3d.geometry.TriangleMesh:
    """Triangulate neighboring RGB-D pixels, rejecting invalid or discontinuous cells.

    ``scene.points`` is expected to be an organized ``(H, W, 3)`` array in the
    camera frame. Object masks follow ``ImageScene.get_pcd`` selection rules.
    The returned vertex colors are the corresponding scene RGB values in [0, 1].
    """
    if max_edge_m <= 0 or max_depth_jump_m <= 0:
        raise ValueError("max_edge_m and max_depth_jump_m must be positive")
    if scene.points is None or np.asarray(scene.points).ndim != 3:
        raise ValueError("scene.points must be an organized (H, W, 3) array")

    points = np.asarray(scene.points, dtype=np.float64)
    if points.shape[2] != 3:
        raise ValueError("scene.points must have three coordinates per pixel")
    h, w, _ = points.shape
    rgb = np.asarray(scene.image)
    if rgb.shape[:2] != (h, w) or rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("scene.image must be an RGB image matching scene.points")
    colors = rgb[..., :3].astype(np.float64) / 255.0

    selected = np.ones((h, w), dtype=bool)
    requested = [] if exclude_objs is None else (
        exclude_objs if isinstance(exclude_objs, (list, tuple)) else [exclude_objs]
    )
    excluded = np.zeros((h, w), dtype=bool)
    for obj in requested:
        if isinstance(obj, (int, np.integer)):
            if obj < 0 or obj >= len(scene.objects):
                raise IndexError(f"object index {obj} is out of range")
            obj = scene.objects[obj]
        mask = np.asarray(obj.mask)
        if mask.shape != (h, w):
            raise ValueError("object masks must match scene.points resolution")
        excluded |= mask > 0
    if reverse_selection:
        if not requested:
            raise ValueError("reverse_selection=True requires at least one excluded object")
        selected &= excluded
    else:
        selected &= ~excluded

    valid = selected & np.isfinite(points).all(axis=2) & np.isfinite(points[..., 2]) & (points[..., 2] > 0)
    vertex_map = np.full((h, w), -1, dtype=np.int32)
    vertex_map[valid] = np.arange(np.count_nonzero(valid), dtype=np.int32)
    vertices = points[valid]
    vertex_colors = colors[valid]

    triangles = np.empty((0, 3), dtype=np.int32)
    if h > 1 and w > 1:
        # Arrange per-cell corners in row-major order. Requiring all four
        # corners keeps faces from leaking across invalid/masked boundaries.
        tl = vertex_map[:-1, :-1]
        tr = vertex_map[:-1, 1:]
        bl = vertex_map[1:, :-1]
        br = vertex_map[1:, 1:]
        valid_cells = (tl >= 0) & (tr >= 0) & (bl >= 0) & (br >= 0)
        cell_ids = np.stack((tl, tr, bl, br), axis=-1)[valid_cells]
        cell_points = np.stack(
            (points[:-1, :-1], points[:-1, 1:], points[1:, :-1], points[1:, 1:]),
            axis=-2,
        )[valid_cells]

        # Two consistently wound triangles per cell; each is filtered on its
        # own so a bad diagonal can leave the neighboring triangle intact.
        tri_corner_indices = np.array(((0, 1, 2), (1, 3, 2)), dtype=np.intp)
        tri_ids = cell_ids[:, tri_corner_indices]
        tri_points = cell_points[:, tri_corner_indices]
        edge01 = tri_points[:, :, 0] - tri_points[:, :, 1]
        edge12 = tri_points[:, :, 1] - tri_points[:, :, 2]
        edge20 = tri_points[:, :, 2] - tri_points[:, :, 0]
        max_edge_sq = np.maximum.reduce((np.sum(edge01 * edge01, axis=-1),
                                         np.sum(edge12 * edge12, axis=-1),
                                         np.sum(edge20 * edge20, axis=-1)))
        depths = tri_points[..., 2]
        depth_jumps = np.maximum.reduce((np.abs(depths[..., 0] - depths[..., 1]),
                                         np.abs(depths[..., 1] - depths[..., 2]),
                                         np.abs(depths[..., 2] - depths[..., 0])))
        keep = (max_edge_sq <= max_edge_m * max_edge_m) & (depth_jumps <= max_depth_jump_m)
        triangles = tri_ids[keep].astype(np.int32, copy=False)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles.reshape(-1, 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    extrinsics = getattr(scene, "cam_extrinsics", None)
    if extrinsics is not None:
        mesh.transform(np.asarray(extrinsics, dtype=np.float64))
    return mesh
