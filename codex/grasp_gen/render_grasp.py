"""Visualize hand grasps loaded from SDF/URDF and txt descriptions."""

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Any, Optional
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
from codex.mesh.load_mesh import load_mesh, trimesh_to_o3d
from codex.utils.utils_path import PROJECT_ROOT_DIR
from codex.grasp_gen.load_grasp import load_grasp_config

HAND_COLOR = (0.86, 0.74, 0.52)  # tan LEAP hand in grasp renders

ALLEGRO_RH_JOINT_NAMES = [
    "algr_rh_if_axl",
    "algr_rh_if_mcp",
    "algr_rh_if_pip",
    "algr_rh_if_dip",
    "algr_rh_mf_axl",
    "algr_rh_mf_mcp",
    "algr_rh_mf_pip",
    "algr_rh_mf_dip",
    "algr_rh_rf_axl",
    "algr_rh_rf_mcp",
    "algr_rh_rf_pip",
    "algr_rh_rf_dip",
    "algr_rh_th_cmc",
    "algr_rh_th_axl",
    "algr_rh_th_mcp",
    "algr_rh_th_ipl",
]


# LEAP joint names (right hand), ordered to match the joint-vector convention used
# throughout this repo (index, middle, ring, thumb; each finger as axl/mcp/pip/dip
# except thumb cmc/axl/mcp/ipl).
LEAP_RH_JOINT_NAMES = [
    "leap_rh_if_axl",
    "leap_rh_if_mcp",
    "leap_rh_if_pip",
    "leap_rh_if_dip",
    "leap_rh_mf_axl",
    "leap_rh_mf_mcp",
    "leap_rh_mf_pip",
    "leap_rh_mf_dip",
    "leap_rh_rf_axl",
    "leap_rh_rf_mcp",
    "leap_rh_rf_pip",
    "leap_rh_rf_dip",
    "leap_rh_th_cmc",
    "leap_rh_th_axl",
    "leap_rh_th_mcp",
    "leap_rh_th_ipl",
]


class GraspVisualizer:
    def __init__(
        self,
        sdf_file_path: Optional[str] = None,
        joint_names: Optional[list[str]] = None,
    ) -> None:
        """Initialize the visualizer by loading an SDF/URDF model.

        For SDF, link ``<pose>`` elements are assumed to be in the model frame.
        For URDF, the kinematic tree is constructed from joint origins.

        Args:
            sdf_file_path: Path to an SDF or URDF file. If omitted, uses the
                LEAP right-hand solver URDF.
            joint_names: Optional joint-name ordering to map a 16-dof grasp
                vector into joint angles by name.
        """
        if sdf_file_path is None:

            sdf_file_path = os.path.join(PROJECT_ROOT_DIR, "assets/models/leap_rh/leap_solver.urdf")
        self.sdf_file_path = sdf_file_path
        self.sdf_dir = os.path.dirname(os.path.abspath(sdf_file_path))
        self.model_format = "urdf" if sdf_file_path.lower().endswith(".urdf") else "sdf"

        self.links: dict[str, dict[str, Any]] = {}
        self.joints: dict[str, dict[str, Any]] = {}
        self.meshes: dict[str, list[o3d.geometry.TriangleMesh]] = {}

        self.joint_names: list[str] = joint_names or []

        if self.model_format == "urdf":
            self.parse_urdf()
            # Link poses for URDF models come from pytorch_kinematics (the
            # same FK the grasp solver uses); the hand-rolled traversal below
            # is kept only for the legacy SDF (Allegro) models.
            import pytorch_kinematics as pk
            with open(sdf_file_path) as fh:
                self._pk_chain = pk.build_chain_from_urdf(fh.read())
            self._pk_joint_order = self._pk_chain.get_joint_parameter_names()
        else:
            self.parse_sdf()
            self.compute_joint_params()

        self.root_link = self._infer_root_link() or (
            "base_link" if self.model_format == "urdf" else "algr_rh_palm"
        )

        if not self.joint_names:
            self.joint_names = self._infer_joint_names()
        
        # Object mesh
        self.object_mesh = None

    def _infer_root_link(self) -> Optional[str]:
        if not self.links:
            return None
        children = {j["child"] for j in self.joints.values() if "child" in j}
        candidates = sorted(set(self.links.keys()) - children)
        if not candidates:
            return None
        # Prefer common roots when present.
        for preferred in ("base_link", "algr_rh_palm"):
            if preferred in candidates:
                return preferred
        return candidates[0]

    def _infer_joint_names(self) -> list[str]:
        joint_set = set(self.joints.keys())
        if all(name in joint_set for name in LEAP_RH_JOINT_NAMES):
            return LEAP_RH_JOINT_NAMES
        return ALLEGRO_RH_JOINT_NAMES

    def parse_sdf(self) -> None:
        """Parse the SDF and populate link/joint metadata."""
        with open(self.sdf_file_path, "r", encoding="utf-8") as f:
            xml_content = f.read()
        xml_content = re.sub(r"drake:", "", xml_content)
        root = ET.fromstring(xml_content)
        model = root.find("model") if root.tag == "sdf" else root

        for link_elem in model.findall('link'):
            link_name = link_elem.get('name')
            pose_elem = link_elem.find('pose')
            if pose_elem is not None and pose_elem.text.strip():
                pose_vals = [float(x) for x in pose_elem.text.strip().split()]
            else:
                pose_vals = [0, 0, 0, 0, 0, 0]
            self.links[link_name] = {'pose': pose_vals, 'visuals': []}
            for visual_elem in link_elem.findall('visual'):
                mesh_elem = visual_elem.find('.//mesh')
                if mesh_elem is not None:
                    uri_elem = mesh_elem.find('uri')
                    if uri_elem is None or uri_elem.text is None:
                        continue
                    mesh_uri = uri_elem.text.strip()
                    vpose_elem = visual_elem.find('pose')
                    if vpose_elem is not None and vpose_elem.text.strip():
                        vpose_vals = [float(x) for x in vpose_elem.text.strip().split()]
                    else:
                        vpose_vals = [0, 0, 0, 0, 0, 0]
                    self.links[link_name]['visuals'].append({
                        'mesh_uri': mesh_uri,
                        'pose': vpose_vals
                    })

        for joint_elem in model.findall('joint'):
            joint_name = joint_elem.get('name')
            joint_type = joint_elem.get('type')
            parent = joint_elem.find('parent').text.strip()
            child = joint_elem.find('child').text.strip()
            
            # Extract joint pose if available
            pose_elem = joint_elem.find('pose')
            if pose_elem is not None and pose_elem.text.strip():
                pose_vals = [float(x) for x in pose_elem.text.strip().split()]
            else:
                pose_vals = [0, 0, 0, 0, 0, 0]
                
            # Extract joint axis
            axis = np.array([0, 0, 1], dtype=float)
            axis_elem = joint_elem.find('axis/xyz')
            if axis_elem is not None and axis_elem.text is not None:
                parts = axis_elem.text.strip().split()
                if len(parts) >= 3:
                    axis = np.array([float(x) for x in parts[:3]])
                    
            self.joints[joint_name] = {
                'type': joint_type,
                'parent': parent,
                'child': child,
                'pose': pose_vals,
                'axis': axis  # axis expressed in the joint frame
            }

    def parse_urdf(self) -> None:
        """Parse a URDF and populate link/joint metadata."""
        with open(self.sdf_file_path, "r", encoding="utf-8") as f:
            xml_content = f.read()
        root = ET.fromstring(xml_content)

        def _origin_to_pose(elem: Optional[ET.Element]) -> list[float]:
            if elem is None:
                return [0, 0, 0, 0, 0, 0]
            xyz = elem.get("xyz", "0 0 0").split()
            rpy = elem.get("rpy", "0 0 0").split()
            xyz_f = [float(v) for v in xyz[:3]]
            rpy_f = [float(v) for v in rpy[:3]]
            return xyz_f + rpy_f

        for link_elem in root.findall("link"):
            link_name = link_elem.get("name")
            if not link_name:
                continue
            self.links[link_name] = {"visuals": []}

            for visual_elem in link_elem.findall("visual"):
                mesh_elem = visual_elem.find("geometry/mesh")
                if mesh_elem is None:
                    continue
                mesh_filename = mesh_elem.get("filename")
                if not mesh_filename:
                    continue
                origin_elem = visual_elem.find("origin")
                self.links[link_name]["visuals"].append(
                    {
                        "mesh_uri": mesh_filename,
                        "pose": _origin_to_pose(origin_elem),
                    }
                )

        for joint_elem in root.findall("joint"):
            joint_name = joint_elem.get("name")
            joint_type = joint_elem.get("type")
            if not joint_name or not joint_type:
                continue
            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            if parent_elem is None or child_elem is None:
                continue
            parent = parent_elem.get("link")
            child = child_elem.get("link")
            if not parent or not child:
                continue

            origin_elem = joint_elem.find("origin")
            origin_pose = _origin_to_pose(origin_elem)
            T_origin = self._pose_to_matrix(origin_pose)

            axis = np.array([0.0, 0.0, 1.0], dtype=float)
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_xyz = axis_elem.get("xyz")
                if axis_xyz:
                    parts = axis_xyz.split()
                    if len(parts) >= 3:
                        axis = np.array([float(x) for x in parts[:3]], dtype=float)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0:
                axis = axis / axis_norm

            self.joints[joint_name] = {
                "type": joint_type,
                "parent": parent,
                "child": child,
                "T_origin": T_origin,
                "axis": axis,
            }

    def _pose_to_matrix(self, pose: list[float]) -> np.ndarray:
        """Convert a 6D pose [x, y, z, roll, pitch, yaw] to a 4x4 homogeneous transform matrix."""
        pos = np.array(pose[:3])
        rot = pose[3:]
        R_mat = R.from_euler('xyz', rot).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = pos
        return T

    def compute_joint_params(self) -> None:
        """Compute joint parameters including relative transforms and rotation axes."""
        for jname, jdata in self.joints.items():
            parent = jdata['parent']
            child = jdata['child']
            
            # Get the poses in model frame
            T_parent_model = self._pose_to_matrix(self.links[parent]['pose'])
            T_child_model = self._pose_to_matrix(self.links[child]['pose'])
            
            # Compute relative transform from parent to child (in zero configuration)
            T_rel = np.linalg.inv(T_parent_model) @ T_child_model
            jdata['T_rel'] = T_rel
            
            # Transform joint axis to parent frame for later use
            axis_model = jdata['axis'] 
            R_model_to_parent = T_parent_model[:3, :3].T
            axis_in_parent = R_model_to_parent @ axis_model
            jdata['axis_in_parent'] = axis_in_parent / np.linalg.norm(axis_in_parent)

    def compute_link_transforms(self, joint_configs: dict[str, float]) -> dict[str, np.ndarray]:
        """
        Compute transforms for all links given the joint configurations.
        Uses a breadth-first approach to ensure parent links are processed first.
        """
        if self.model_format == "urdf":
            import torch
            q = torch.tensor(
                [[float(joint_configs.get(name, 0.0)) for name in self._pk_joint_order]],
                dtype=torch.float32,
            )
            fk = self._pk_chain.forward_kinematics(q)
            return {link: fk[link].get_matrix()[0].numpy() for link in fk}

        # Start with the transform for the root link
        if self.model_format == "urdf":
            link_transforms = {self.root_link: np.eye(4)}
        else:
            link_transforms = {
                self.root_link: self._pose_to_matrix(self.links[self.root_link]['pose'])
            }
        
        # Use a breadth-first approach to process joints
        processed = {self.root_link}
        queue = [self.root_link]
        
        while queue:
            parent = queue.pop(0)
            T_parent = link_transforms[parent]
            
            # Find all joints that have this link as a parent
            for jname, jdata in self.joints.items():
                if jdata['parent'] == parent:
                    child = jdata['child']
                    
                    # Skip if already processed
                    if child in processed:
                        continue
                    
                    # Get joint angle
                    angle = joint_configs.get(jname, 0.0)

                    if self.model_format == "urdf":
                        T_origin = jdata["T_origin"].copy()
                        if jdata["type"] == "fixed":
                            T_motion = np.eye(4)
                        else:
                            axis = jdata["axis"]
                            R_joint = R.from_rotvec(angle * axis).as_matrix()
                            T_motion = np.eye(4)
                            T_motion[:3, :3] = R_joint
                        T_child = T_parent @ T_origin @ T_motion
                    else:
                        # Get the axis in parent frame (pre-computed)
                        axis_in_parent = jdata['axis_in_parent']

                        # Get relative transform (pre-computed)
                        T_rel = jdata['T_rel'].copy()

                        # Extract pivot point from relative transform
                        pivot = T_rel[:3, 3]

                        # Create rotation matrix for joint angle
                        R_joint = R.from_rotvec(angle * axis_in_parent).as_matrix()

                        # Create transformation matrices
                        T_to_pivot = np.eye(4)
                        T_to_pivot[:3, 3] = pivot

                        T_rot = np.eye(4)
                        T_rot[:3, :3] = R_joint

                        T_from_pivot = np.eye(4)
                        T_from_pivot[:3, 3] = -pivot

                        # Apply transformations in sequence to get the joint motion
                        T_joint = T_to_pivot @ T_rot @ T_from_pivot

                        # Apply joint motion to the relative transform
                        T_rel_with_rotation = T_joint @ T_rel

                        # Calculate child transform
                        T_child = T_parent @ T_rel_with_rotation
                    
                    # Store transform
                    link_transforms[child] = T_child
                    processed.add(child)
                    queue.append(child)
        
        return link_transforms

    def load_meshes(self) -> None:
        """Load meshes for all links in the model."""
        for link, data in self.links.items():
            self.meshes[link] = []
            for visual in data['visuals']:
                mesh_path = os.path.join(self.sdf_dir, visual['mesh_uri'])
                if os.path.exists(mesh_path):
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    mesh.compute_vertex_normals()
                    T_visual = self._pose_to_matrix(visual['pose'])
                    vertices = np.asarray(mesh.vertices)
                    ones = np.ones((vertices.shape[0], 1))
                    vertices_h = np.hstack([vertices, ones])
                    transformed_vertices = (T_visual @ vertices_h.T).T[:, :3]
                    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
                    self.meshes[link].append(mesh)
    
    def load_object_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """Load an object mesh for grasping visualization."""
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        self.object_mesh = mesh

    def _leap_solver_meshes(self, joint_configs: dict) -> tuple[dict, dict]:
        """Link transforms + visual meshes from the grasp solver's validated
        LEAP model (assets/models/leap_rh): pytorch_kinematics FK on the
        solver URDF and the solver's visual-mesh offset table. This is the
        same geometry the solver optimized, so renders match the grasp
        exactly."""
        import torch
        import pytorch_kinematics as pk
        import trimesh as _trimesh
        from codex.grasp_gen.batched_grasp_solver import _MODELS_DIR, _leap_visual_meshes

        if not hasattr(self, "_solver_chain"):
            with open(os.path.join(_MODELS_DIR, "leap_rh/leap_solver.urdf")) as fh:
                self._solver_chain = pk.build_chain_from_urdf(fh.read())
            self._solver_joint_order = self._solver_chain.get_joint_parameter_names()
            self._solver_visuals = {}
            mesh_dir = os.path.join(_MODELS_DIR, "leap_rh")
            for link, entries in _leap_visual_meshes("rh").items():
                metas = []
                for fname, pose in entries:
                    tm = _trimesh.load(os.path.join(mesh_dir, fname), force="mesh")
                    if pose is not None:
                        T_vis = np.eye(4)
                        T_vis[:3, 3] = pose[:3]
                        T_vis[:3, :3] = R.from_euler("xyz", pose[3:]).as_matrix()
                        tm.apply_transform(T_vis)
                    om = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(tm.vertices),
                        o3d.utility.Vector3iVector(tm.faces))
                    om.compute_vertex_normals()
                    metas.append(om)
                self._solver_visuals[link] = metas
        q = torch.tensor(
            [[float(joint_configs.get(n, 0.0)) for n in self._solver_joint_order]],
            dtype=torch.float32)
        fk = self._solver_chain.forward_kinematics(q)
        link_T = {link: fk[link].get_matrix()[0].numpy() for link in self._solver_visuals}
        return link_T, self._solver_visuals

    def get_transformed_meshes(
        self,
        joint_config_array: np.ndarray,
        hand_pose: Optional[np.ndarray] = None,
        object_transform: Optional[np.ndarray] = None,
        include_object: bool = True,
    ) -> list[o3d.geometry.TriangleMesh]:
        """
        Compute the current transforms for every link and apply them to the corresponding visual meshes.
        If a hand_pose is provided, it is applied as a post-transformation to the entire hand.
        """
        joint_configs = {
            name: (joint_config_array[i] if i < len(joint_config_array) else 0.0)
            for i, name in enumerate(self.joint_names)
        }

        if self.joint_names == LEAP_RH_JOINT_NAMES:
            link_transforms, mesh_table = self._leap_solver_meshes(joint_configs)
            self.meshes = mesh_table
        else:
            link_transforms = self.compute_link_transforms(joint_configs)
        # If hand_pose is provided, post-multiply every link's transform.
        if hand_pose is not None:
            T_hand = hand_pose.copy()
            for link in link_transforms:
                link_transforms[link] = T_hand @ link_transforms[link]
                
        if not self.meshes:
            self.load_meshes()
            
        all_meshes: list[o3d.geometry.TriangleMesh] = []
        for link, T in link_transforms.items():
            if link in self.meshes:
                for mesh in self.meshes[link]:
                    mesh_copy = deepcopy(mesh)
                    vertices = np.asarray(mesh_copy.vertices)
                    ones = np.ones((vertices.shape[0], 1))
                    vertices_h = np.hstack([vertices, ones])
                    transformed = (T @ vertices_h.T).T[:, :3]
                    mesh_copy.vertices = o3d.utility.Vector3dVector(transformed)
                    all_meshes.append(mesh_copy)
        if self.object_mesh is not None:
            object_mesh_copy = deepcopy(self.object_mesh)
            if object_transform is not None:
                vertices = np.asarray(object_mesh_copy.vertices)
                ones = np.ones((vertices.shape[0], 1))
                vertices_h = np.hstack([vertices, ones])
                transformed = (object_transform @ vertices_h.T).T[:, :3]
                object_mesh_copy.vertices = o3d.utility.Vector3dVector(transformed)
            if include_object:
                all_meshes.append(object_mesh_copy)
        return all_meshes

    def visualize(
        self,
        joint_config_array: np.ndarray,
        hand_pose: Optional[np.ndarray] = None,
        object_transform: Optional[np.ndarray] = None,
        extra_geoms: list[Any] | None = None,
        width: int = 960,
        height: int = 720,
    ) -> Image.Image:
        """Render the hand (and object) offscreen and return the image."""
        if extra_geoms is None:
            extra_geoms = []

        meshes = self.get_transformed_meshes(joint_config_array, hand_pose, object_transform)
        n_hand = len(meshes) - (self.object_mesh is not None)
        for mesh in meshes[:n_hand]:
            mesh.paint_uniform_color(HAND_COLOR)

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        renderer.scene.scene.set_sun_light([-0.4, -0.4, -1.0], [1.0, 1.0, 1.0], 90000)
        renderer.scene.scene.enable_sun_light(True)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        bounds = o3d.geometry.AxisAlignedBoundingBox()
        for i, mesh in enumerate([*meshes, *extra_geoms]):
            mesh.compute_vertex_normals()
            renderer.scene.add_geometry(f"geom_{i}", mesh, material)
            bounds += mesh.get_axis_aligned_bounding_box()

        center = bounds.get_center()
        radius = float(np.linalg.norm(bounds.get_extent())) * 0.9
        # View from the object/finger side (opposite the palm) when the hand
        # pose is known; otherwise use a fixed three-quarter view.
        if hand_pose is not None:
            to_obj = center - hand_pose[:3, 3]
            az = float(np.arctan2(to_obj[1], to_obj[0]))
            eye = center + radius * np.array([np.cos(az) * 0.9, np.sin(az) * 0.9, 0.45])
        else:
            eye = center + radius * np.array([0.7, -0.7, 0.45])
        renderer.setup_camera(50.0, center.astype(np.float32),
                              eye.astype(np.float32), np.array([0, 0, 1], np.float32))
        img = renderer.render_to_image()
        return Image.fromarray(np.asarray(img))

def render_grasp_files(mesh_path: str, grasp_files: list[str], model_path: Optional[str] = None) -> list[str]:
    """Render each grasp file next to itself as <grasp>.png; returns the PNG paths."""
    mesh, _ = load_mesh(mesh_path)
    offset = np.array([0.0, 0.0, -mesh.bounds[0, 2]])
    visualizer = GraspVisualizer(model_path, joint_names=LEAP_RH_JOINT_NAMES)
    visualizer.load_object_mesh(trimesh_to_o3d(mesh))
    outputs = []
    for grasp_file in grasp_files:
        joint_config, hand_pose, _ = load_grasp_config(grasp_file, offset=offset)
        png = os.path.splitext(grasp_file)[0] + ".png"
        visualizer.visualize(joint_config, hand_pose).save(png)
        outputs.append(png)
        print(f"Rendered {png}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LEAP grasps on their object mesh.")
    parser.add_argument("--scene-dir", help="Render every <scene>/grasps/grasp_*_optimized.txt "
                                            "on <scene>/mesh/mesh.glb.")
    parser.add_argument("--grasp-file", help="A single grasp file to render.")
    parser.add_argument("--mesh-path", help="Object mesh (default with --scene-dir: <scene>/mesh/mesh.glb).")
    parser.add_argument("--model-path", help="Hand URDF/SDF (default: LEAP solver URDF).")
    args = parser.parse_args()

    if args.scene_dir:
        grasp_files = sorted(glob.glob(os.path.join(args.scene_dir, "grasps", "grasp_*_optimized.txt")))
        mesh_path = args.mesh_path or os.path.join(args.scene_dir, "mesh", "mesh.glb")
    elif args.grasp_file and args.mesh_path:
        grasp_files, mesh_path = [args.grasp_file], args.mesh_path
    else:
        parser.error("Give --scene-dir, or --grasp-file with --mesh-path")
    if not grasp_files:
        parser.error("No grasp files found")
    render_grasp_files(mesh_path, grasp_files, args.model_path)


if __name__ == "__main__":
    main()
