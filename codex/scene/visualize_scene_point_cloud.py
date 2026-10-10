import argparse
import os

import numpy as np
import open3d as o3d

from codex.scene.image_scene import ImageScene
from codex.scene.estimation.estimate_extrinsic import estimate_scene_transform
from codex.utils.utils_path import get_scene_pkl_path


def _set_camera_pov(vis: o3d.visualization.Visualizer, scene: "ImageScene", T_cam_to_world: np.ndarray) -> None:
	"""Set the Open3D visualizer camera to match the real camera's POV."""
	if scene.cam_intrinsics is None:
		return
	width, height = scene.image.size
	K = scene.cam_intrinsics
	params = o3d.camera.PinholeCameraParameters()
	params.intrinsic.set_intrinsics(
		width=width, height=height,
		fx=K[0, 0], fy=K[1, 1],
		cx=K[0, 2], cy=K[1, 2],
	)
	params.extrinsic = np.linalg.inv(T_cam_to_world)
	vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def visualize_scene_point_cloud(
	scene_dir: str,
	transform: bool,
	point_size: float,
	use_estimated_transform: bool,
) -> None:
	scene_path = get_scene_pkl_path(scene_dir)
	if not os.path.isfile(scene_path):
		raise FileNotFoundError(f"Scene file not found: {scene_path}")

	scene = ImageScene.load(scene_path)

	if use_estimated_transform:
		# Use the same estimated scene transform path as estimate_object_size.py.
		T = estimate_scene_transform(scene)
		pcd = scene.get_pcd(return_o3d=True, transform=False)
		pcd.transform(T)
		frame_b = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
		frame_b.transform(T)
		window_name = "Scene Point Cloud (Estimated Transform)"
		T_cam_to_world = T
	else:
		# If no extrinsics are available, gracefully fall back to raw camera-frame points.
		use_transform = transform and scene.cam_extrinsics is not None
		if transform and scene.cam_extrinsics is None:
			print("Requested transform, but scene has no cam_extrinsics. Showing untransformed point cloud.")
		pcd = scene.get_pcd(return_o3d=True, transform=use_transform)
		frame_b = None
		window_name = "Scene Point Cloud"
		T_cam_to_world = scene.cam_extrinsics if use_transform else np.eye(4)

	frame_a = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name=window_name)
	vis.add_geometry(pcd)
	vis.add_geometry(frame_a)
	if frame_b is not None:
		vis.add_geometry(frame_b)

	render_opt = vis.get_render_option()
	render_opt.point_size = point_size

	_set_camera_pov(vis, scene, T_cam_to_world)
	vis.run()
	vis.destroy_window()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Load and visualize a scene point cloud from a scene.pkl file.")
	parser.add_argument(
		"--scene-dir",
		type=str,
		required=True,
		help="Path to the scene directory (contains scene.pkl).",
	)
	parser.add_argument(
		"--transform",
		action="store_true",
		help="Apply scene camera extrinsics before visualization (if available).",
	)
	parser.add_argument(
		"--estimated-transform",
		action="store_true",
		default=True,
		help="Estimate scene transform (T) and visualize the point cloud under T.",
	)
	parser.add_argument(
		"--point-size",
		type=float,
		default=2.0,
		help="Open3D point size.",
	)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	scene_dir = os.path.abspath(args.scene_dir)
	visualize_scene_point_cloud(
		scene_dir=scene_dir,
		transform=args.transform,
		point_size=args.point_size,
		use_estimated_transform=args.estimated_transform,
	)
