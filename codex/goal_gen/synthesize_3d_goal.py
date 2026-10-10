import argparse
from typing import Tuple, Optional
import os
import json
import numpy as np
import trimesh
import open3d as o3d
import cv2
from copy import deepcopy
from PIL import Image, ImageDraw
from codex.mesh.load_mesh import trimesh_to_o3d
from codex.mesh.render_mesh import render_mesh_pcd
from codex.scene.image_scene import ImageScene

def candidate_views(scene: ImageScene, effect_world: np.ndarray) -> list:
    """Camera poses (camera to world, OpenCV axes) of the three candidate views:
    the capture camera; the same camera turned 90 degrees about the vertical line
    through the effect point (same height and distance); and a camera straight
    above the effect point, looking down, oriented like the capture camera."""
    C = np.asarray(scene.cam_extrinsics, dtype=np.float64)
    e = np.asarray(effect_world, dtype=np.float64).reshape(3)
    turn = np.eye(4)
    turn[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turn[:3, 3] = e - turn[:3, :3] @ e
    x = C[:3, 0].copy()
    x[2] = 0.0
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    top = np.eye(4)
    top[:3, 0], top[:3, 1], top[:3, 2] = x, np.cross(z, x), z
    top[:3, 3] = e + [0.0, 0.0, np.linalg.norm(C[:3, 3] - e)]
    return [C, turn @ C, top]


def render_candidate(mesh: trimesh.Trimesh, pose: np.ndarray, scene: ImageScene, cid: int,
                     multi_view: bool = False, scene_representation: str = "points",
                     axes_origin: np.ndarray | None = None, effect_point: np.ndarray | None = None,
                     functional_axis: np.ndarray | None = None) -> Image.Image:
    """Render the posed functional object in the scene.

    pose: mesh to camera. axes_origin (mesh frame) draws the functional object's local axes
    there; effect_point (camera frame) marks where the functional object should act, and with
    functional_axis (mesh frame) a line runs from axes_origin along it.
    multi_view: the capture view on top and, below at half size, the view turned
    90 degrees about the vertical through the effect point and the view from
    straight above it (see candidate_views); needs effect_point.
    """
    from codex.mesh.scene_axes import draw_goal_markers
    # Converting the textured mesh to a coloured Open3D mesh is the same for
    # every candidate of a search: do it once per mesh and pose a copy.
    base = getattr(mesh, "_render_o3d", None)
    if base is None:
        base = mesh._render_o3d = trimesh_to_o3d(mesh)
    posed = o3d.geometry.TriangleMesh(base)
    posed.transform(np.asarray(pose, dtype=np.float64))

    def m():  # a fresh copy per view: render_mesh_pcd moves it into the world in place
        return o3d.geometry.TriangleMesh(posed)

    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    axes_frame = None
    if axes_origin is not None:
        fn_local = np.asarray(axes_origin, dtype=np.float64).reshape(3)
        axes_frame = (pose[:3, :3] @ fn_local + pose[:3, 3], pose[:3, :3])
    annotate = effect_world = None
    if effect_point is not None:
        to_world = np.asarray(scene.cam_extrinsics, dtype=np.float64)
        effect_world = (to_world @ np.append(np.asarray(effect_point, dtype=np.float64).reshape(3), 1.0))[:3]
        fn_world = dir_world = None
        if axes_frame is not None and functional_axis is not None:
            fn_world = (to_world @ np.append(axes_frame[0], 1.0))[:3]
            dir_world = to_world[:3, :3] @ pose[:3, :3] @ np.asarray(functional_axis, dtype=np.float64).reshape(3)

        def markers(image, world_to_camera, K):
            return draw_goal_markers(image, world_to_camera, K, effect_world, fn_world, dir_world)
        annotate = markers

    def view(v):
        return render_mesh_pcd(m(), scene, object_index=[0], display_mode="mesh_on_pcd_without_obj",
                               view=v, scene_representation=scene_representation,
                               axes_origin=axes_frame, annotate=annotate)

    if multi_view:
        if effect_world is None:
            raise ValueError("multi_view needs effect_point")
        camera, side, top = candidate_views(scene, effect_world)
        main = view("scene")
        w, h = main.size
        small = [view(side).resize((w // 2, h // 2)), view(top).resize((w - w // 2, h // 2))]
        img = Image.new("RGB", (w, h + h // 2), "white")
        img.paste(main, (0, 0))
        img.paste(small[0], (0, h))
        img.paste(small[1], (w // 2, h))
        ImageDraw.Draw(img).line((w // 2, h, w // 2, h + h // 2), fill=(255, 255, 255), width=4)
        ImageDraw.Draw(img).line((0, h, w, h), fill=(255, 255, 255), width=4)
    else:
        img = view("scene")

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if cid is not None:
        txt = str(cid); (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 3, 8)
        h, w = bgr.shape[:2]
        cv2.putText(bgr, txt, (w-tw-20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 8)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


VIEW_DESCRIPTION = (
    "Each candidate image shows the same pose in three views. Top: the original camera. "
    "Bottom left: the same camera turned 90 degrees around the vertical line through the effect "
    "point, at the same height. Bottom right: looking straight down onto the effect point. "
    "In every view the green ring marks the effect point, where the functional object should act; the blue line "
    "runs from the functional object's functional point along its functional direction; the red, green and blue "
    "arrows are the functional object's local X, Y and Z axes at the functional point.")
ONE_VIEW_DESCRIPTION = (
    "Each candidate image shows the original camera view. The green ring marks the effect point, "
    "where the functional object should act; the blue line runs from the functional object's functional point along its "
    "functional direction; the red, green and blue arrows are the functional object's local X, Y and Z axes.")


def pixel_to_ray(px: Tuple[int,int], K: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Compute camera‑frame ray origin and direction for pixel px."""
    u, v = px
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    # Image coords: x right, y down. Camera frame: x right, y up, -Z forward
    x_cam = (u - cx) / fx
    y_cam = (cy - v) / fy  # invert image‑y axis
    dir_cam = np.array([x_cam, -y_cam, 1.0], dtype=float)  # forward along +Z
    dir_cam /= np.linalg.norm(dir_cam)
    origin_cam = np.zeros(3, dtype=float)
    return origin_cam, dir_cam


def ray_intersect_mesh(
    origin: np.ndarray,
    direction: np.ndarray,
    mesh_cam: trimesh.Trimesh,
    fallback: bool = True
) -> Optional[np.ndarray]:
    """Intersect ray (cam frame) with mesh_cam; fallback to nearest vertex."""
    locations, _, _ = mesh_cam.ray.intersects_location([origin], [direction])
    if len(locations) > 0:
        d = np.linalg.norm(locations - origin, axis=1)
        return locations[np.argmin(d)]
    if not fallback:
        return None
    # fallback: nearest vertex to ray
    v = mesh_cam.vertices
    o2v = v - origin
    t = o2v @ direction
    proj = origin + np.outer(t, direction)
    perp = v - proj
    idx = np.argmin(np.linalg.norm(perp, axis=1))
    return v[idx]


def estimate_fn_point_3d_from_pixel(
    pixel_xy: Tuple[int,int],
    posed_mesh_cam: trimesh.Trimesh,
    K: np.ndarray,
    scene_pcd,
    visualize: bool = False
) -> Tuple[np.ndarray,np.ndarray]:
    """Cast ray in camera frame; return fn_pt in mesh frame and camera frame."""
    origin_cam, dir_cam = pixel_to_ray(pixel_xy, K)
    hit_cam = ray_intersect_mesh(origin_cam, dir_cam, posed_mesh_cam, fallback=True)
    if hit_cam is None:
        raise RuntimeError("Ray miss and no fallback")
    if visualize:
        mesh_o3d = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(posed_mesh_cam.vertices),
            triangles=o3d.utility.Vector3iVector(posed_mesh_cam.faces)
        )
        mesh_o3d.compute_vertex_normals(); mesh_o3d.paint_uniform_color([0.8,0.8,0.8])
        length = np.linalg.norm(hit_cam - origin_cam)
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([origin_cam, origin_cam + dir_cam*length]),
            lines=o3d.utility.Vector2iVector([[0,1]])
        )
        line.colors = o3d.utility.Vector3dVector([[1,0,0]])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(hit_cam); sphere.paint_uniform_color([0,1,0])
        o3d.visualization.draw_geometries([mesh_o3d, line, sphere, scene_pcd], window_name="FN Debug")
    return hit_cam

def lift_effect_point(scene: ImageScene, pixel, target_name: Optional[str]) -> Optional[np.ndarray]:
    """3D effect point (camera frame) of the pixel the VLM chose on the target.

    The lift is restricted to the target's mask, since a raw depth lookup can
    fall through sparse geometry (e.g. foliage) onto the background. Stage 0
    rejects implausible masks (codex.scene.create_scene.check_masks).
    """
    objects = list(scene.objects)
    target = next((o for o in objects if o.name == target_name), None)
    if target is None and len(objects) > 1:
        target = objects[1]
    return infer_3d_point_from_mask(pixel, scene.points, mask=None if target is None else target.mask)


def infer_3d_point_from_mask(
    pixel,
    scene_points: np.ndarray,
    mask: Optional[np.ndarray],
    radius: int = 20,
    color: Optional[np.ndarray] = None,
    visualize: bool = False,
) -> Optional[np.ndarray]:
    """
    Given a 2D pixel coordinate, infer its 3D position using the scene's depth points
    and a binary mask (to restrict the search to valid object pixels). If the pixel
    is outside the mask, find the closest masked pixel first. If depth at that pixel
    is NaN, search in a small window around it.
    """
    if mask is None:
        mask = np.ones_like(scene_points[..., 0], dtype=bool)
    h, w = mask.shape
    x, y = int(pixel[0]), int(pixel[1])
    mask = mask.copy() > 0

    # If the query pixel isn't on the mask, snap it to the nearest mask pixel.
    if not mask[y, x]:
        ys, xs = np.nonzero(mask)
        # compute squared distances
        d2 = (xs - x)**2 + (ys - y)**2
        idx = np.argmin(d2)
        x, y = int(xs[idx]), int(ys[idx])

    # Now search depth around (x,y) within 'radius'
    x_min = max(0, x - radius)
    x_max = min(w, x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(h, y + radius + 1)

    # Restrict to object mask
    obj_pts = scene_points.copy()
    obj_pts[~mask] = np.nan
    # Extract local window points
    window = obj_pts[y_min:y_max, x_min:x_max].reshape(-1, 3)
    valid = window[~np.isnan(window).any(axis=1)]
    if valid.shape[0] == 0:
        print(f"No valid depth inside radius {radius} of pixel ({x},{y}).")
        return None

    inferred = np.median(valid, axis=0)

    if visualize:
        # visualize full object
        all_pts = obj_pts.reshape(-1,3)
        all_pts = all_pts[~np.isnan(all_pts).any(axis=1)]
        pcd_obj = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_pts))
        pcd_obj.paint_uniform_color([0.7,0.7,0.7])
        # visualize local window
        win_mask = np.zeros_like(mask, bool)
        win_mask[y_min:y_max, x_min:x_max] = True
        local_pts = scene_points.copy()
        local_pts[~(mask & win_mask)] = np.nan
        loc = local_pts.reshape(-1,3)
        loc = loc[~np.isnan(loc).any(axis=1)]
        pcd_loc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(loc))
        pcd_loc.paint_uniform_color([1,0,0])
        # sphere at inferred
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sph.translate(inferred)
        sph.paint_uniform_color([0,1,0])
        o3d.visualization.draw_geometries([pcd_obj, pcd_loc, sph],
                                          window_name="3D Estimation")
    return inferred


def synthesize_goal_pcd(
    best_candidate_json: dict,
    scene: ImageScene,
    color: Optional[np.ndarray] = None,
    visualize: bool = False,
) -> Optional[o3d.geometry.PointCloud]:
    """
    Synthesize the goal (transformed) point cloud in 3D using a 2D transformation
    (from the best candidate's JSON) and the scene's depth information.
    
    Process:
      a. Infer the 3D positions of the keypoint (on object1) and the targetpoint (on object2)
         from their 2D pixel locations.
      b. Compute the transformation: a rotation (about the image plane normal) and a translation,
         so that the keypoint moves to the targetpoint.
      c. Apply this transformation to all 3D points corresponding to object1.
    
    Args:
        best_candidate_json (dict): JSON data from the best candidate containing at least:
            - "key_px": [x, y] of the keypoint on object1.
            - "target_px": [x, y] of the targetpoint on object2.
            - "rotation": 2D rotation (in degrees) to be applied.
        scene (ImageScene): Your scene instance that contains:
            - scene.points: a (H, W, 3) array of 3D points.
            - scene.objects: a list where object1 (index 0) and object2 (index 1) have a 'mask' attribute.
        color (np.ndarray, optional): Array of shape (H, W, 3) with the RGB image data.
                                      If provided, colors are extracted for the synthesized point cloud.
        visualize (bool): If True, visualize the resulting goal point cloud along with the inferred key and target points.
    
    Returns:
        The transformed (goal) point cloud for object1, or None if key/target 3D points
        could not be inferred.
    """
    if "relative_transform" in best_candidate_json:
        T = np.asarray(best_candidate_json["relative_transform"], dtype=float)
        obj_pcd = deepcopy(scene.objects[0].get_pcd())
        obj_pcd.transform(T)
        if visualize:
            scene_pcd = scene.get_pcd()
            o3d.visualization.draw_geometries([scene_pcd, obj_pcd], window_name="Transformed Object")
        return obj_pcd

    # Extract 2D pixel coordinates and rotation from the JSON.
    key_px = best_candidate_json["key_px"]      # e.g., [x, y]
    target_px = best_candidate_json["target_px"]  # e.g., [x, y]
    rotation_angle = best_candidate_json["rotation"]
    
    # Infer the 3D positions for keypoint and targetpoint.
    key3d = infer_3d_point_from_mask(key_px, scene.points, scene.objects[0].mask,
                                     radius=10, color=color, visualize=visualize)
    target3d = infer_3d_point_from_mask(target_px, scene.points, scene.objects[1].mask,
                                        radius=10, color=color, visualize=visualize)
    
    if key3d is None or target3d is None:
        print("Failed to infer valid 3D key or target points. Aborting 3D synthesis.")
        return None
    
    # Compute the rotation matrix about the image plane normal (assumed to be the z-axis).
    theta = np.deg2rad(-rotation_angle)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0,              0,             1]])
    
    # Compute the translation vector to move the keypoint to the targetpoint.
    t = target3d - np.dot(R, key3d)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # visualize
    scene_pcd = scene.get_pcd()
    obj_pcd = deepcopy(scene.objects[0].get_pcd())
    obj_pcd.transform(T)
    if visualize:
        o3d.visualization.draw_geometries([scene_pcd, obj_pcd], window_name="Transformed Object")
    return obj_pcd
    


def _find_best_candidate(scene_dir: str, output_dir: Optional[str]) -> str:
    """Resolve the goal candidate JSON from an explicit or standard output directory."""
    if output_dir:
        candidate_dirs = [os.path.abspath(output_dir)]
    else:
        candidate_dirs = [os.path.join(os.path.abspath(scene_dir), "goal_gen", m) for m in ("vlm_cem_dir", "vlm_cem")]

    for candidate_dir in candidate_dirs:
        best_candidate_path = os.path.join(candidate_dir, "best_candidate.json")
        if os.path.exists(best_candidate_path):
            return best_candidate_path

    raise FileNotFoundError(
        "Could not find best_candidate.json. Checked: "
        + ", ".join(os.path.join(d, "best_candidate.json") for d in candidate_dirs)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize a 3D goal point cloud from a saved goal candidate.")
    parser.add_argument(
        "--scene-dir",
        dest="scene_dir",
        required=True,
        help="Path to the scene directory (contains scene.pkl).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Goal-generation output directory containing best_candidate.json. Defaults to scene-local goal_gen and legacy out/goal_gen/<scene_name> locations.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the synthesized goal point cloud.",
    )
    args = parser.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)
    best_candidate_path = _find_best_candidate(scene_dir, args.output_dir)
    scene = ImageScene.load(os.path.join(scene_dir, "scene.pkl"))

    with open(best_candidate_path, "r") as f:
        best_candidate_data = json.load(f)

    synthesize_goal_pcd(best_candidate_data, scene, visualize=args.visualize)


if __name__ == '__main__':
    main()
