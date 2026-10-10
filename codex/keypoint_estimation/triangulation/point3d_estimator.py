import os
import json
import numpy as np
from PIL import Image
import open3d as o3d
import open3d.visualization.rendering as rendering
import trimesh
from typing import List, Dict, Optional

from codex.keypoint_estimation.triangulation.vis_utils import visualize_mesh_with_keypoints
from codex.utils.utils_path import PROJECT_ROOT_DIR


class Point3DEstimator:
    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        mesh_name: str,
        output_dir: str = os.path.join(PROJECT_ROOT_DIR, "out", "actuation_3d"),
        n_views: Optional[int] = None,
        n_azimuth: Optional[int] = None,
        n_elevation: Optional[int] = None,
        min_elevation: float = 30.0,
        max_elevation: float = 60.0,
        radius: Optional[float] = None,
        image_width: int = 640,
        image_height: int = 480,
        fov: float = 60.0,
        up_vector: Optional[List[float]] = None,
        surface_constrained: bool = False,
        surface_method: str = 'closest_point',
        subdir_name: Optional[str] = None,
    ):
        assert (n_views is not None) ^ (n_azimuth is not None and n_elevation is not None), \
            "Specify exactly one of: n_views or (n_azimuth and n_elevation)"
        self.mesh = mesh
        self.mesh_name = mesh_name
        self.output_dir = os.path.join(output_dir, subdir_name if subdir_name is not None else mesh_name)
        self.cache_dir = os.path.join(self.output_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.surface_constrained = surface_constrained
        self.surface_method = surface_method
        if radius is None:
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox_size = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
            radius = bbox_size * 1.2
        self.radius = radius
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov
        self.up_vector = np.array(up_vector if up_vector is not None else [0, 0, 1], dtype=float)
        if n_views is not None:
            self.views_per_ring = n_views
            mid = (min_elevation + max_elevation) / 2.0
            self.elevations = [mid]
        else:
            self.views_per_ring = n_azimuth
            self.elevations = list(np.linspace(min_elevation, max_elevation, n_elevation))

        # Compute intrinsics from FOV
        W, H = image_width, image_height
        fx = fy = (W / 2) / np.tan(np.radians(fov / 2))
        self.intrinsic_matrix = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])

        self.view_fronts: List[np.ndarray] = []
        self.camera_data: Dict[int, dict] = {}
        self.rendered_views: Dict[int, np.ndarray] = {}
        self.points_2d: Dict[int, List[List[float]]] = {}
        self.points_3d: List[np.ndarray] = []
        self.points_3d_free: List[np.ndarray] = []

    def _look_at_extrinsic(self, cam_pos, target, up):
        """Compute a world-to-camera extrinsic from look-at parameters (OpenCV convention)."""
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        # OpenCV convention: x=right, y=down, z=forward
        R = np.array([right, -true_up, forward])
        t = -R @ cam_pos
        extr = np.eye(4)
        extr[:3, :3] = R
        extr[:3, 3] = t
        return extr

    def render_mesh_views(self, force_rerender: bool = False) -> Dict[int, np.ndarray]:
        """Render the mesh from multiple viewpoints using OffscreenRenderer."""
        assert self.mesh is not None, "Mesh not loaded"
        cache = os.path.join(self.cache_dir, "rendered_views")
        os.makedirs(cache, exist_ok=True)

        center = self.mesh.get_center()

        # Generate view directions: orbit in XY, height in Z, interleaved rings
        if not self.view_fronts or force_rerender:
            self.view_fronts = []
            azimuth_step = 2 * np.pi / self.views_per_ring
            for ring_idx, elev in enumerate(self.elevations):
                elev_rad = np.radians(elev)
                offset = ring_idx * azimuth_step / len(self.elevations)
                for i in range(self.views_per_ring):
                    angle = azimuth_step * i + offset
                    front = np.array([
                        np.cos(angle) * np.cos(elev_rad),
                        np.sin(angle) * np.cos(elev_rad),
                        np.sin(elev_rad),
                    ])
                    self.view_fronts.append(front / np.linalg.norm(front))
            if force_rerender:
                self.rendered_views.clear()
                self.camera_data.clear()

        cam_cache = os.path.join(self.cache_dir, "camera_data.json")

        # Load cached camera data
        if not force_rerender and os.path.exists(cam_cache):
            with open(cam_cache, 'r') as f:
                raw = json.load(f)
            self.camera_data = {
                int(k): {
                    'extrinsic': np.array(v['extrinsic']),
                    'intrinsic': np.array(v['intrinsic']),
                } for k, v in raw.items()
            }

        # Check which views need rendering
        needs_render = []
        for idx in range(len(self.view_fronts)):
            cache_file = os.path.join(cache, f"view_{idx:02d}.png")
            if not force_rerender and os.path.exists(cache_file) and idx in self.camera_data:
                self.rendered_views[idx] = np.array(Image.open(cache_file))
            else:
                needs_render.append(idx)

        if not needs_render:
            return self.rendered_views

        # Set up OffscreenRenderer (exact camera control, no readback issues)
        W, H = self.image_width, self.image_height
        renderer = rendering.OffscreenRenderer(W, H)
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit'
        renderer.scene.add_geometry('mesh', self.mesh, mat)
        renderer.scene.set_background([0.8, 0.8, 0.8, 1.0])

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            W, H,
            self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1],
            self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2],
        )

        for idx in needs_render:
            front = self.view_fronts[idx]
            cam_pos = center + front * self.radius

            # Compute and store exact camera parameters
            extr = self._look_at_extrinsic(cam_pos, center, self.up_vector)
            self.camera_data[idx] = {
                'extrinsic': extr,
                'intrinsic': self.intrinsic_matrix.copy(),
            }

            # Render with exact intrinsics + extrinsics (OpenCV convention)
            renderer.setup_camera(intrinsic_o3d, extr)
            img = renderer.render_to_image()
            frame = np.asarray(img)

            cache_file = os.path.join(cache, f"view_{idx:02d}.png")
            Image.fromarray(frame).save(cache_file)
            self.rendered_views[idx] = frame
    
        del renderer

        # Cache camera data
        cam_serial = {}
        for idx, cd in self.camera_data.items():
            cam_serial[str(idx)] = {
                'extrinsic': cd['extrinsic'].tolist(),
                'intrinsic': cd['intrinsic'].tolist(),
            }
        with open(cam_cache, 'w') as f:
            json.dump(cam_serial, f, indent=2)

        return self.rendered_views

    def add_points_for_view(self, idx: int, points: List[List[float]]):
        """Add 2D point annotations for a view. Points in [X%, Y%] format (0-100)."""
        assert idx in self.rendered_views, f"View {idx} not rendered"
        self.points_2d[idx] = points

    def load_annotations(self, annotations_path: str):
        """Load 2D annotations from a JSON file.

        Expected format: {"0": [[x_pct, y_pct], ...], ...}
        """
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        for key, pts in data.items():
            idx = int(key)
            if pts:
                self.add_points_for_view(idx, pts)
        print(f"Loaded annotations for {sum(1 for v in data.values() if v)} views from {annotations_path}")

    def _pixel_ray_from_extrinsic(self, pixel_pct, extrinsic, intrinsic):
        """Cast a ray from camera through a pixel coordinate."""
        u = pixel_pct[0] * self.image_width / 100.0
        v = pixel_pct[1] * self.image_height / 100.0

        K_inv = np.linalg.inv(intrinsic)
        ray_cam = K_inv @ np.array([u, v, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        ray_world = R.T @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        cam_pos = -R.T @ t

        return cam_pos, ray_world

    def project_2d_to_3d(self, visualize: bool = False,
                         outlier_rejection: bool = True) -> List[np.ndarray]:
        """Project 2D points to 3D via ray-mesh intersection."""
        assert self.points_2d, "No 2D points to project"
        assert self.camera_data, "No camera data — run render_mesh_views() first"

        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        tm = trimesh.Trimesh(vertices=vertices, faces=triangles)

        all_3d_points = []
        # Triangle each annotation ray first hits: the face the annotator was
        # actually looking at. Consumers use these normals to derive the press
        # direction (self.hit_tris stays parallel to the returned point list).
        self.hit_tris = []

        for idx, pts in self.points_2d.items():
            assert idx in self.camera_data, f"No camera data for view {idx}"
            extr = self.camera_data[idx]['extrinsic']
            intr = self.camera_data[idx]['intrinsic']

            for pt_2d in pts:
                origin, direction = self._pixel_ray_from_extrinsic(pt_2d, extr, intr)
                locations, _, index_tri = tm.ray.intersects_location(
                    ray_origins=[origin],
                    ray_directions=[direction],
                    multiple_hits=True,
                )
                if len(locations) > 0:
                    dists = np.linalg.norm(locations - origin, axis=1)
                    hit = int(np.argmin(dists))
                    all_3d_points.append(locations[hit])
                    self.hit_tris.append(int(index_tri[hit]))
                    print(f"  View {idx}, pt {pt_2d} -> 3D {locations[hit]}")
                else:
                    print(f"  View {idx}, pt {pt_2d} -> no mesh intersection")

        if outlier_rejection and len(all_3d_points) >= 3:
            pts_arr = np.array(all_3d_points)
            median = np.median(pts_arr, axis=0)
            dists_to_median = np.linalg.norm(pts_arr - median, axis=1)
            med_dist = np.median(dists_to_median)
            threshold = max(med_dist * 2.0, 1e-6)
            keep = dists_to_median <= threshold
            n_rejected = np.sum(~keep)
            if n_rejected > 0:
                print(f"  Outlier rejection: dropped {n_rejected}/{len(all_3d_points)} points "
                      f"(threshold={threshold:.6f})")
            all_3d_points = [p for p, k in zip(all_3d_points, keep) if k]
            self.hit_tris = [t for t, k in zip(self.hit_tris, keep) if k]

        self.points_3d = all_3d_points
        self.points_3d_free = all_3d_points

        if visualize and all_3d_points:
            visualize_mesh_with_keypoints(self.mesh, all_3d_points)

        return all_3d_points

    def render_result_visualization(self, save_dir: str, highlight_point: Optional[np.ndarray] = None):
        """Render the mesh with projected 3D points from multiple angles."""
        if not self.points_3d:
            return

        bbox = self.mesh.get_axis_aligned_bounding_box()
        bbox_size = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        center = self.mesh.get_center()
        mean_pt = np.mean(self.points_3d, axis=0)

        W, H = 800, 600
        renderer = rendering.OffscreenRenderer(W, H)

        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit'
        renderer.scene.add_geometry('mesh', self.mesh, mat)

        # Individual points in red
        mat_red = rendering.MaterialRecord()
        mat_red.shader = 'defaultUnlit'
        mat_red.base_color = [1, 0, 0, 1]
        for i, pt in enumerate(self.points_3d):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=bbox_size * 0.015)
            s.translate(pt)
            s.paint_uniform_color([1, 0, 0])
            renderer.scene.add_geometry(f'pt_{i}', s, mat_red)

        # Mean point in green
        mat_green = rendering.MaterialRecord()
        mat_green.shader = 'defaultUnlit'
        mat_green.base_color = [0, 1, 0, 1]
        s = o3d.geometry.TriangleMesh.create_sphere(radius=bbox_size * 0.02)
        s.translate(mean_pt)
        s.paint_uniform_color([0, 1, 0])
        renderer.scene.add_geometry('mean', s, mat_green)

        if highlight_point is not None:
            # Draw the final selected actuation point as a larger green marker.
            mat_highlight = rendering.MaterialRecord()
            mat_highlight.shader = 'defaultUnlit'
            mat_highlight.base_color = [0, 0.8, 0.2, 1]
            highlight = o3d.geometry.TriangleMesh.create_sphere(radius=bbox_size * 0.025)
            highlight.translate(np.asarray(highlight_point))
            highlight.paint_uniform_color([0, 0.8, 0.2])
            renderer.scene.add_geometry('highlight', highlight, mat_highlight)

        renderer.scene.set_background([0.9, 0.9, 0.9, 1.0])

        # Camera distance and focal length scaled to mesh size
        dist = bbox_size * 1.5
        fx = fy = W * 1.2  # moderate telephoto

        angles = [
            ('side', [1, 0.3, 0.5]),
            ('front', [0, 0.3, 1]),
            ('3_4', [0.7, 0.7, 0.6]),
        ]
        up = [0, 0, 1]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, W / 2, H / 2)

        os.makedirs(save_dir, exist_ok=True)
        for label, direction in angles:
            d = np.array(direction, dtype=float)
            d = d / np.linalg.norm(d)
            cam_pos = center + d * dist
            extr = self._look_at_extrinsic(cam_pos, center, np.array(up, dtype=float))
            renderer.setup_camera(intrinsic, extr)
            img = np.asarray(renderer.render_to_image())
            Image.fromarray(img).save(os.path.join(save_dir, f"result_{label}.png"))

        if highlight_point is not None:
            d = np.array([0.9, 0.35, 0.55], dtype=float)
            d = d / np.linalg.norm(d)
            cam_pos = center + d * dist
            extr = self._look_at_extrinsic(cam_pos, center, np.array(up, dtype=float))
            renderer.setup_camera(intrinsic, extr)
            img = np.asarray(renderer.render_to_image())
            Image.fromarray(img).save(os.path.join(save_dir, "final_actuation_point.png"))

        del renderer
        print(f"Result visualizations saved to {save_dir}")

    def save_results(self, task_name: str = "", highlight_point: Optional[np.ndarray] = None) -> str:
        res_dir = os.path.join(self.output_dir, "results", task_name)
        os.makedirs(res_dir, exist_ok=True)
        np.savez(
            os.path.join(res_dir, f"{self.mesh_name}_points.npz"),
            points_3d=self.points_3d,
            points_3d_free=self.points_3d_free,
            surface_constrained=self.surface_constrained,
            surface_method=self.surface_method,
        )
        annotations = {str(k): v for k, v in self.points_2d.items()}
        with open(os.path.join(res_dir, "annotations_2d.json"), 'w') as f:
            json.dump(annotations, f, indent=2)
        cam_serial = {}
        for idx, cd in self.camera_data.items():
            cam_serial[str(idx)] = {
                'extrinsic': cd['extrinsic'].tolist(),
                'intrinsic': cd['intrinsic'].tolist(),
            }
        with open(os.path.join(res_dir, "camera_data.json"), 'w') as f:
            json.dump(cam_serial, f, indent=2)
        for idx, img in self.rendered_views.items():
            Image.fromarray(img).save(os.path.join(res_dir, f"view_{idx:02d}.png"))

        # Render result visualization
        self.render_result_visualization(
            os.path.join(res_dir, "visualization"),
            highlight_point=highlight_point,
        )

        print(f"Results saved to {res_dir}")
        return res_dir
