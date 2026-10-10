import argparse
import os
import numpy as np
import open3d as o3d
from PIL import Image
from typing import List

from codex.mesh.load_mesh import load_mesh, trimesh_to_o3d
from codex.scene.image_scene import ImageScene


def visualize_mesh_pcd(
    mesh = None,
    scene: ImageScene = None,
    object_index=0,
    display_mode="mesh_on_pcd",
    view="mesh",
    background_color=(0.8, 0.8, 0.8),
    point_size=2.0,
    window_name="Mesh and Point Cloud Visualization"
):
    """
    Visualize mesh and point cloud together with togglable display modes.
    
    Args:
        mesh: The mesh to visualize (Open3D or trimesh)
        scene: ImageScene object containing the point cloud
        object_index: Index of the object to exclude when removing from point cloud
        display_mode: One of "mesh_only", "pcd_only", "mesh_on_pcd", "mesh_on_pcd_without_obj"
        view: Initial view to display. One of "mesh", "scene"
        background_color: RGB background color (0-1 for each channel)
        point_size: Size of points in the point cloud
        window_name: Name of the visualization window
    """
    # Validate display mode
    valid_modes = ["mesh_only", "pcd_only", "mesh_on_pcd", "mesh_on_pcd_without_obj"]
    assert display_mode in valid_modes, f"Invalid mode. Choose from: {valid_modes}"
    
    # Convert trimesh to Open3D mesh if needed
    mesh_o3d = trimesh_to_o3d(mesh)
            
    # Apply camera extrinsics to mesh if provided by scene
    if scene is not None and scene.cam_extrinsics is not None:
        mesh_o3d.transform(scene.cam_extrinsics)

    # Extract point clouds from scene
    scene_pcd = None
    scene_without_obj_pcd = None
    if scene is not None:
        scene_pcd = scene.get_pcd(return_o3d=True, transform=True)
        scene_without_obj_pcd = scene.get_pcd(
            return_o3d=True, 
            exclude_objs=[object_index], 
            transform=True
        )

    # Interactive visualization with key toggles
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name)
    
    # Setup render options
    opt = vis.get_render_option()
    opt.background_color = np.array(background_color)
    opt.point_size = point_size
    
    # Add geometries
    geometries = []
    
    if mesh_o3d is not None:
        vis.add_geometry(mesh_o3d)
        mesh_triangles = np.asarray(mesh_o3d.triangles).copy()
        geometries.append(mesh_o3d)
    
    if scene_pcd is not None:
        vis.add_geometry(scene_pcd)
        scene_points = np.asarray(scene_pcd.points).copy()
        geometries.append(scene_pcd)
    
    if scene_without_obj_pcd is not None:
        vis.add_geometry(scene_without_obj_pcd)
        without_obj_points = np.asarray(scene_without_obj_pcd.points).copy()
        # Initially hide
        scene_without_obj_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        geometries.append(scene_without_obj_pcd)
    
    # Display modes configuration
    display_configs = {
        1: {
            "name": "Mesh overlaid on original point cloud",
            "setup": lambda: configure_display(
                mesh_o3d, scene_pcd, scene_without_obj_pcd,
                show_mesh=True, show_full_pcd=True, show_without_obj=False,
                mesh_triangles=mesh_triangles, scene_points=scene_points, 
                without_obj_points=without_obj_points
            )
        },
        2: {
            "name": "Original point cloud only",
            "setup": lambda: configure_display(
                mesh_o3d, scene_pcd, scene_without_obj_pcd,
                show_mesh=False, show_full_pcd=True, show_without_obj=False,
                mesh_triangles=mesh_triangles, scene_points=scene_points
            )
        },
        3: {
            "name": "Mesh overlaid on point cloud without object",
            "setup": lambda: configure_display(
                mesh_o3d, scene_pcd, scene_without_obj_pcd,
                show_mesh=True, show_full_pcd=False, show_without_obj=True,
                mesh_triangles=mesh_triangles, without_obj_points=without_obj_points
            )
        },
        4: {
            "name": "Mesh only",
            "setup": lambda: configure_display(
                mesh_o3d, scene_pcd, scene_without_obj_pcd,
                show_mesh=True, show_full_pcd=False, show_without_obj=False,
                mesh_triangles=mesh_triangles
            )
        }
    }
    
    # Set initial display based on mode
    display_num = 1  # Default
    if display_mode == "mesh_only":
        display_num = 4
    elif display_mode == "pcd_only":
        display_num = 2
    elif display_mode == "mesh_on_pcd_without_obj":
        display_num = 3
        
    current_display = display_num
    display_configs[current_display]["setup"]()
    for geo in geometries:
        vis.update_geometry(geo)
    
    # Function to switch display modes
    def switch_display(vis_param, display_num):
        nonlocal current_display
        if current_display == display_num:
            return False
        
        # Setup the new display mode
        display_configs[display_num]["setup"]()
        
        # Update all geometries
        for geo in geometries:
            vis_param.update_geometry(geo)
        
        print(f"Display mode {display_num}: {display_configs[display_num]['name']}")
        current_display = display_num
        return False
    
    # Register callbacks
    for num in display_configs.keys():
        vis.register_key_callback(ord(str(num)), lambda vis_param, n=num: switch_display(vis_param, n))
    
    # Set camera view
    view_control = vis.get_view_control()
    if view == "mesh":
        center_camera_on_mesh(view_control, mesh_o3d)
    elif view == "scene":
        assert scene is not None, "Scene object is required for scene view"
        set_camera_from_scene(view_control, scene)
    
    # Print instructions
    print("\nVisualization Controls:")
    for num, config in display_configs.items():
        print(f"  Press '{num}': {config['name']}")
    print("  Press 'Q': Close the visualization\n")
    
    # Run visualizer
    vis.run()
    vis.destroy_window()


def configure_display(
    mesh_o3d, scene_pcd, scene_without_obj_pcd,
    show_mesh=True, show_full_pcd=True, show_without_obj=False,
    mesh_triangles=None, scene_points=None, without_obj_points=None
):
    """Helper function to configure geometries for different display modes."""
    # Configure mesh visibility
    if mesh_o3d is not None and mesh_triangles is not None:
        if show_mesh:
            mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_triangles)
        else:
            mesh_o3d.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
    
    # Configure full point cloud visibility
    if scene_pcd is not None and scene_points is not None:
        if show_full_pcd:
            scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
        else:
            scene_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    
    # Configure point cloud without object visibility
    if scene_without_obj_pcd is not None and without_obj_points is not None:
        if show_without_obj:
            scene_without_obj_pcd.points = o3d.utility.Vector3dVector(without_obj_points)
        else:
            scene_without_obj_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))


def set_camera_from_scene(view_control, scene):
    """Set the view using camera parameters from the scene."""
    if scene.cam_intrinsics is not None:
        intrinsic = scene.cam_intrinsics.copy()
        width, height = scene.image.size
        
        # Create a PinholeCameraParameters object
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic.set_intrinsics(
            width=width,
            height=height,
            fx=intrinsic[0, 0],
            fy=intrinsic[1, 1],
            cx=intrinsic[0, 2],
            cy=intrinsic[1, 2]
        )
        
        params.extrinsic = np.linalg.inv(scene.cam_extrinsics.copy())
        view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def center_camera_on_mesh(view_control, mesh):
    """Center the view on the mesh."""
    if mesh is not None:
        center = mesh.get_center()
        view_control.set_lookat(center)
        view_control.set_zoom(0.7)
        view_control.camera_local_rotate(0, 0)  # Reset rotation


def render_mesh_pcd(
    mesh=None,
    scene: ImageScene | None = None,
    size: tuple[int, int] | None = (640, 480),
    object_index: int | List[int] = 0,
    reverse_object_selection: bool = False,
    display_mode: str = "mesh_on_pcd",
    view: str | np.ndarray = "mesh",
    background_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
    point_size: float = 2.0,
    save_path: str | None = None,
    camera_intrinsics: np.ndarray | None = None,
    scene_representation: str = "points",
    axes_origin: tuple[np.ndarray, np.ndarray] | None = None,
    annotate=None,
) -> Image.Image:
    """
    Headless render of a mesh / point cloud with several overlay modes.

    Added behaviour
    ---------------
    • For any mode that uses the *scene point cloud* (pcd_only,
      mesh_on_pcd, mesh_on_pcd_without_obj) the camera intrinsics are
      always set to the original `scene.cam_intrinsics`, ensuring the
      rendering lines up with the captured photograph.
    • Supports a new view: "topdown" for an overhead orthographic view.
    """
    MODES = {
        "mesh_only",
        "pcd_only",
        "mesh_on_pcd",
        "mesh_on_pcd_without_obj",
        "mesh_on_bg",
    }
    if display_mode not in MODES:
        raise ValueError(f"Invalid display_mode {display_mode!r}")

    object_indices = object_index if isinstance(object_index, list) else [object_index]

    # --------------------------------------------------------------- #
    # 1.  Geometry preparation
    # --------------------------------------------------------------- #
    mesh_o3d = None
    if mesh is not None:
        mesh_o3d = mesh if isinstance(mesh, o3d.geometry.TriangleMesh) else trimesh_to_o3d(mesh)
        if scene is not None and scene.cam_extrinsics is not None:
            mesh_o3d.transform(scene.cam_extrinsics)

    need_pcd = display_mode in {"pcd_only", "mesh_on_pcd", "mesh_on_pcd_without_obj"}
    if scene_representation not in {"points", "surface"}:
        raise ValueError("scene_representation must be 'points' or 'surface'")
    scene_pcd = scene_without_pcd = None
    if need_pcd:
        if scene is None:
            raise ValueError("display_mode needs a valid `scene`")
        if scene_representation == "surface":
            from codex.mesh.scene_surface import build_scene_surface
            # A scene is fixed throughout a candidate search. Cache its visible
            # surface rather than rebuilding up to two million triangles per view.
            cache = getattr(scene, "_render_surface_cache", None)
            if cache is None:
                cache = scene._render_surface_cache = {}
            excluded = object_indices if display_mode == "mesh_on_pcd_without_obj" else []
            key = (tuple(excluded), reverse_object_selection)
            if key not in cache:
                cache[key] = build_scene_surface(
                    scene, exclude_objs=excluded, reverse_selection=reverse_object_selection,
                )
            scene_pcd = scene_without_pcd = cache[key]
        else:
            # The scene is fixed throughout a candidate search: build each
            # point cloud once (about a million points) and only the one this
            # mode draws; Open3D does not modify geometry it renders.
            cache = getattr(scene, "_render_pcd_cache", None)
            if cache is None:
                cache = scene._render_pcd_cache = {}
            if display_mode == "mesh_on_pcd_without_obj":
                key = (tuple(object_indices), reverse_object_selection)
                if key not in cache:
                    cache[key] = scene.get_pcd(return_o3d=True, exclude_objs=object_indices, transform=True,
                                               reverse_selection=reverse_object_selection)
                scene_without_pcd = cache[key]
            else:
                if "all" not in cache:
                    cache["all"] = scene.get_pcd(return_o3d=True, transform=True)
                scene_pcd = cache["all"]

    # --------------------------------------------------------------- #
    # 2.  Size and intrinsics
    # --------------------------------------------------------------- #
    uses_scene_intr = need_pcd
    if display_mode == "mesh_on_bg":
        uses_scene_intr = True

    if uses_scene_intr:
        if scene is None or scene.cam_intrinsics is None:
            raise ValueError("scene.cam_intrinsics required for this mode")
        if size is None or size == (640, 480):
            size = scene.image.size
        elif size != scene.image.size:
            raise ValueError(
                f"Requested size {size} must match scene image {scene.image.size}"
            )

    width, height = size
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.point_size = point_size
    if scene_representation == "surface":
        opt.mesh_show_back_face = True

    # --------------------------------------------------------------- #
    # 3.  Add geometries
    # --------------------------------------------------------------- #
    if display_mode != "pcd_only":
        if mesh_o3d is None:
            raise ValueError("Mesh must be supplied for this display_mode")
        vis.add_geometry(mesh_o3d)

    if display_mode in {"pcd_only", "mesh_on_pcd"}:
        vis.add_geometry(scene_pcd)
    elif display_mode == "mesh_on_pcd_without_obj":
        vis.add_geometry(scene_without_pcd)

    # --------------------------------------------------------------- #
    # 4.  Camera placement
    # --------------------------------------------------------------- #
    vc = vis.get_view_control()
    if isinstance(view, str):
        if view == "scene":
            set_camera_from_scene(vc, scene)
        elif view == "mesh":
            center_camera_on_mesh(vc, mesh_o3d)
        elif view == "topdown":
            eye = np.array([0.0, 0.0, 2])
            center = np.array([0.6, 0.0, 0.0])
            up_dir = np.array([0.0, 1.0, 0.0])
            forward = center - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, up_dir)
            right /= np.linalg.norm(right)
            true_up = np.cross(right, forward)
            R_wc = np.vstack([true_up, -right, forward])
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R_wc
            extrinsic[:3, 3] = -R_wc @ eye
            params = vc.convert_to_pinhole_camera_parameters()
            params.extrinsic = extrinsic
            vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            vc.set_zoom(0.3)
        else:
            raise ValueError(f"Unknown view {view!r}")
    else:
        params = vc.convert_to_pinhole_camera_parameters()
        params.extrinsic = np.linalg.inv(view)
        if camera_intrinsics is not None:
            params.intrinsic.set_intrinsics(
                width=width, height=height,
                fx=camera_intrinsics[0, 0], fy=camera_intrinsics[1, 1],
                cx=camera_intrinsics[0, 2], cy=camera_intrinsics[1, 2],
            )
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    # 6. Enforce intrinsics
    # skip intrinsics override for topdown to preserve our extrinsic
    if uses_scene_intr and not (isinstance(view, str) and view == "topdown"):
        K = scene.cam_intrinsics
        p = vc.convert_to_pinhole_camera_parameters()
        p.intrinsic.set_intrinsics(
            width=scene.image.size[0],
            height=scene.image.size[1],
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
        )
        vc.convert_from_pinhole_camera_parameters(p, allow_arbitrary=True)

    # 7. Render
    vis.poll_events(); vis.update_renderer()
    frame = (np.asarray(vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
    camera_params = vc.convert_to_pinhole_camera_parameters() if (axes_origin is not None or annotate) else None
    vis.destroy_window()

    # 8. Composite for mesh_on_bg
    if display_mode == "mesh_on_bg":
        bg = scene.isolate_objects(objs=object_indices, inverse=True)
        bg = np.asarray(bg).copy()
        bg_col = (np.array(background_color) * 255).astype(np.uint8)
        mask = (np.abs(frame - bg_col) > 3).any(axis=2)
        bg[mask] = frame[mask]
        frame = bg

    if axes_origin is not None:
        from codex.mesh.scene_axes import draw_axes_overlay, project_axes

        origin_cam, rotation_cam = axes_origin
        origin_cam = np.asarray(origin_cam, dtype=np.float64).reshape(3)
        rotation_cam = np.asarray(rotation_cam, dtype=np.float64).reshape(3, 3)
        camera_to_world = np.asarray(scene.cam_extrinsics) if scene is not None and scene.cam_extrinsics is not None else np.eye(4)
        origin_world = (camera_to_world @ np.r_[origin_cam, 1.0])[:3]
        rotation_world = camera_to_world[:3, :3] @ rotation_cam
        K_render = np.asarray(camera_params.intrinsic.intrinsic_matrix)
        segments = project_axes(
            origin_world, rotation_world, camera_params.extrinsic, K_render,
        )
        frame = np.asarray(draw_axes_overlay(Image.fromarray(frame), segments))
    if annotate is not None:
        # annotate(image, world_to_camera, K) draws more world-frame markers.
        frame = np.asarray(annotate(Image.fromarray(frame), camera_params.extrinsic,
                                    np.asarray(camera_params.intrinsic.intrinsic_matrix)))

    out = Image.fromarray(frame)
    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out.save(save_path)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively visualize a mesh (and optional scene point cloud) with optional bounding box overlay.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mesh-path", dest="mesh_path", type=str, required=True,
                        help="Path to the mesh file (.glb, .obj, etc.).")
    parser.add_argument("--scene-dir", dest="scene_dir", type=str, default=None,
                        help="Path to the scene directory (contains scene.pkl). When provided, the point cloud is shown alongside the mesh.")
    parser.add_argument("--pose-path", dest="pose_path", type=str, default=None,
                        help="Path to a pose.txt (4x4 T_cam_obj). Auto-detected from the scene directory if not set.")
    parser.add_argument("--display-mode", dest="display_mode", type=str, default="mesh_only",
                        choices=["mesh_only", "pcd_only", "mesh_on_pcd", "mesh_on_pcd_without_obj"],
                        help="What geometry to show (default: mesh_only).")
    parser.add_argument("--view", type=str, default="mesh", choices=["mesh", "scene"],
                        help="Initial camera view (default: mesh).")
    parser.add_argument("--bbox", action="store_true",
                        help="Overlay the axis-aligned bounding box of the mesh in red.")
    parser.add_argument("--object-index", dest="object_index", type=int, default=0,
                        help="Scene object index for *_without_obj modes (default: 0).")
    args = parser.parse_args()

    mesh_path = os.path.abspath(args.mesh_path)

    scene_dir = None
    if args.scene_dir is not None:
        scene_dir = os.path.abspath(args.scene_dir)

    pose_path = args.pose_path

    mesh, _ = load_mesh(mesh_path)
    mesh_o3d = trimesh_to_o3d(mesh)

    scene = None
    if scene_dir is not None:
        scene = ImageScene.load(os.path.join(scene_dir, "scene.pkl"))

    if args.bbox:
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]

        # Resolve pose: try explicit arg, then auto-detect from scene dir.
        if pose_path is None and scene_dir is not None:
            candidate = os.path.join(scene_dir, "pose.txt")
            if os.path.exists(candidate):
                pose_path = candidate
                print(f"Auto-detected pose: {pose_path}")

        if pose_path is not None:
            # pose.txt is T_cam_obj: transforms mesh from object frame to camera frame.
            pose = np.loadtxt(pose_path)
            mesh_o3d.transform(pose)
            print(f"Applied pose from {pose_path}")
        else:
            print("No pose found — mesh shown in its canonical (object) frame.")

        # Everything stays in camera space: pcd with transform=False, mesh already in cam frame.
        geoms = [mesh_o3d]
        bbox = mesh_o3d.get_axis_aligned_bounding_box()
        bbox.color = (1.0, 0.0, 0.0)
        geoms.append(bbox)
        if scene is not None:
            pcd = scene.get_pcd(return_o3d=True, transform=False)
            geoms.append(pcd)

        vis = o3d.visualization.Visualizer()
        w, h = (scene.image.size if scene is not None else (1280, 720))
        vis.create_window(window_name=f"Mesh + BBox: {mesh_name}", width=w, height=h)
        for g in geoms:
            vis.add_geometry(g)
        # Set camera to view from origin looking into the scene (camera frame convention).
        if scene is not None and scene.cam_intrinsics is not None:
            K = scene.cam_intrinsics
            vc = vis.get_view_control()
            p = vc.convert_to_pinhole_camera_parameters()
            p.intrinsic.set_intrinsics(
                width=w, height=h,
                fx=K[0, 0], fy=K[1, 1],
                cx=K[0, 2], cy=K[1, 2],
            )
            p.extrinsic = np.eye(4)  # camera at origin, looking into the scene
            vc.convert_from_pinhole_camera_parameters(p, allow_arbitrary=True)
        vis.run()
        vis.destroy_window()
    else:
        visualize_mesh_pcd(
            mesh=mesh,
            scene=scene,
            object_index=args.object_index,
            display_mode=args.display_mode,
            view=args.view,
        )
