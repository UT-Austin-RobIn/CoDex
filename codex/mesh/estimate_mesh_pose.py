import os
import argparse
import ctypes
import sys
import numpy as np
import torch
import trimesh
import cv2
from scipy.spatial import cKDTree
from PIL import Image
from typing import Union
from codex.utils.utils_path import PROJECT_ROOT_DIR, OUT_DIR
from codex.scene.image_scene import ImageScene
from codex.scene.estimation.estimate_functional_object_meta_data import estimate_object_height
from codex.mesh.load_mesh import load_mesh
from codex.mesh.render_mesh import visualize_mesh_pcd, render_mesh_pcd

_FOUNDATIONPOSE_RUNTIME = None


def _load_foundationpose_runtime() -> dict[str, object]:
    """Import FoundationPose lazily so the module stays importable without CUDA setup."""
    global _FOUNDATIONPOSE_RUNTIME
    if _FOUNDATIONPOSE_RUNTIME is not None:
        return _FOUNDATIONPOSE_RUNTIME

    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_lib = os.path.join(conda_prefix, "lib") if conda_prefix else None
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    if conda_lib and conda_lib not in ld_library_path.split(":"):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_library_path}" if ld_library_path else conda_lib
        if __name__ == "__main__" and env.get("_CODEX_MESH_POSE_REEXEC") != "1":
            env["_CODEX_MESH_POSE_REEXEC"] = "1"
            os.execvpe(
                sys.executable,
                [sys.executable, "-m", "codex.mesh.estimate_mesh_pose", *sys.argv[1:]],
                env,
            )

        libstdcpp_path = os.path.join(conda_lib, "libstdc++.so.6")
        if os.path.exists(libstdcpp_path):
            ctypes.CDLL(libstdcpp_path, mode=ctypes.RTLD_GLOBAL)

    # Imported here (not at module level) so that image/mesh-based pipeline
    # modes work without the FoundationPose + nvdiffrast CUDA stack installed.
    import nvdiffrast.torch as dr
    from foundationpose.Utils import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed
    from foundationpose.estimater import FoundationPose
    from foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor
    from foundationpose.learning.training.predict_score import ScorePredictor

    _FOUNDATIONPOSE_RUNTIME = {
        "FoundationPose": FoundationPose,
        "PoseRefinePredictor": PoseRefinePredictor,
        "ScorePredictor": ScorePredictor,
        "draw_posed_3d_box": draw_posed_3d_box,
        "draw_xyz_axis": draw_xyz_axis,
        "dr": dr,
        "set_logging_format": set_logging_format,
        "set_seed": set_seed,
    }
    return _FOUNDATIONPOSE_RUNTIME


class MeshPoseEstimator:
    def __init__(self, debug: int = 0, debug_dir: str = None) -> None:
        """Initialize the pose estimator using FoundationPose.
        
        Args:
            debug (int): Debug level (0=minimal, 1=basic, 2=detailed, 3=verbose). Defaults to 0.
            debug_dir (str): Directory for debug outputs. Defaults to out/mesh_pose/default.
            
        Returns:
            None
        """
        runtime = _load_foundationpose_runtime()

        # Setup logging and directories
        runtime["set_logging_format"]()
        runtime["set_seed"](0)
        if debug_dir is None:
            debug_dir = os.path.join(PROJECT_ROOT_DIR, "out", "mesh_pose", "default")
        os.makedirs(debug_dir, exist_ok=True)
        if debug > 0:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        
        self.debug = debug
        self.debug_dir = debug_dir
        self.runtime = runtime
        
        # Initialize predictors
        self.scorer = self.runtime["ScorePredictor"]()
        self.refiner = self.runtime["PoseRefinePredictor"]()
        self.glctx = self.runtime["dr"].RasterizeCudaContext()

    @staticmethod
    def _scene_inputs(scene: ImageScene, object_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract the (rgb, depth, mask, K) registration inputs from a scene."""
        if scene.points is None:
            raise ValueError("Scene does not have 3D points. Load them with scene.load_points()")
        if len(scene.objects) <= object_index:
            raise ValueError(f"Scene only has {len(scene.objects)} objects, but object_index is {object_index}")
        assert scene.cam_intrinsics is not None, "Camera intrinsics are required for pose estimation"

        from codex.scene.input_validation import validate_rgbd
        valid = validate_rgbd(scene.image, scene.points, scene.cam_intrinsics)

        rgb = np.array(scene.image)
        depth = np.where(valid, scene.points[:, :, 2], 0.0)
        mask = scene.objects[object_index].mask > 0
        if mask.shape != valid.shape or not (mask & valid).any():
            raise ValueError("Object mask must match the RGB-D image and contain valid depth pixels")
        mask = mask & valid
        return rgb, depth, mask, scene.cam_intrinsics

    @staticmethod
    def _prepare_mesh_for_foundationpose(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Normalize textured mesh materials to FoundationPose expectations."""
        mesh_prepared = mesh.copy()
        visual = getattr(mesh_prepared, "visual", None)

        if isinstance(visual, trimesh.visual.texture.TextureVisuals):
            material = getattr(visual, "material", None)
            if material is not None:
                has_image = hasattr(material, "image") and (getattr(material, "image", None) is not None)
                if not has_image:
                    pbr_texture = getattr(material, "baseColorTexture", None)
                    if pbr_texture is not None:
                        visual.material = trimesh.visual.material.SimpleMaterial(image=pbr_texture)
                    else:
                        visual.material = trimesh.visual.material.SimpleMaterial(
                            image=Image.new("RGB", (2, 2), (128, 128, 128))
                        )

        return mesh_prepared
        
    def estimate_pose(self, scene: ImageScene, mesh: trimesh.Trimesh, object_index: int = 0, est_refine_iter: int = 5) -> tuple[np.ndarray, dict]:
        """Estimate the pose of a mesh in the scene.
        
        Args:
            scene (ImageScene): The scene containing the object.
            mesh (trimesh.Trimesh): The input mesh.
            object_index (int): Index of the object in the scene. Defaults to 0.
            est_refine_iter (int): Number of refinement iterations. Defaults to 5.
            
        Returns:
            (pose, mesh_info):
                - pose (np.ndarray): Estimated pose as a 4x4 transformation matrix.
                - mesh_info (dict): Additional mesh information for visualization.
        """
        rgb, depth, mask, K = self._scene_inputs(scene, object_index)


        # Get oriented bounds for visualization
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        
        # FoundationPose expects a texture material with `image` on textured meshes.
        mesh_for_pose = self._prepare_mesh_for_foundationpose(mesh)

        # Initialize foundation pose estimator
        estimator = self.runtime["FoundationPose"](
            model_pts=mesh_for_pose.vertices,
            model_normals=mesh_for_pose.vertex_normals,
            mesh=mesh_for_pose,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=self.glctx
        )
        
        # Estimate the pose. Two local guards for one-shot registration:
        # autocast off — some imports (lang_sam) leave a global bfloat16
        # autocast active, which breaks FoundationPose's full-precision
        # linear algebra; cudnn off — the cudnn-9/cu11 wheel aborts
        # (free(): invalid pointer) in its precompiled engines when the
        # scene-creation stack is loaded in the same process. The cuBLAS
        # fallback costs a few seconds per registration.
        with torch.backends.cudnn.flags(enabled=False), torch.autocast("cuda", enabled=False):
            pose = estimator.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=est_refine_iter
            )
        
        # If debugging, save additional visualization
        if self.debug >= 2:
            # Save the estimated pose for reference
            np.savetxt(os.path.join(self.debug_dir, "estimated_pose.txt"), pose)
            
            # Save visualization of inputs
            cv2.imwrite(os.path.join(self.debug_dir, "input_rgb.png"), rgb[..., ::-1])
            cv2.imwrite(os.path.join(self.debug_dir, "input_depth.png"), (depth*1000).astype(np.uint16))
            cv2.imwrite(os.path.join(self.debug_dir, "input_mask.png"), mask.astype(np.uint8)*255)
        
        mesh_info = {
            'to_origin': to_origin,
            'bbox': bbox,
            'K': K
        }
        np.savez(os.path.join(self.debug_dir, "mesh_info.npz"), **mesh_info)

        return pose, mesh_info

    def register_and_canonicalize(self, scene: ImageScene, mesh_path: str, object_index: int = 0, est_refine_iter: int = 5) -> tuple[np.ndarray, str]:
        """Register a generated mesh against the scene and bake its canonical frame.

        Generated meshes arrive at arbitrary scale and orientation; the only
        metric anchor is the object height measured from the scene depth.
        Each choice of mesh up-axis implies a different metric scale
        (height / extent along that axis), and FoundationPose searches over
        rotations on its own, so a hypothesis only has to fix the scale.
        The mesh is registered once per axis hypothesis. Hypotheses are
        compared by how well the posed mesh explains the observed object
        points (a trimmed two-sided surface residual) — FoundationPose's
        internal score is not comparable across differently scaled models.
        The up-axis is read off the winning rotation (the mesh axis that
        maps closest to the world up direction).

        The canonical frame — up-axis rotated to +Z, metric scale — is baked
        into the mesh file IN PLACE, so downstream stages load the mesh
        as-is with no transform metadata to carry around. The canonical mesh
        is then registered once more for the returned pose.

        Args:
            scene (ImageScene): The scene containing the object.
            mesh_path (str): Path to the mesh file; overwritten with the
                canonicalized mesh.
            object_index (int): Index of the object in the scene. Defaults to 0.
            est_refine_iter (int): Refinement iterations per hypothesis. Defaults to 5.

        Returns:
            (pose, up_axis):
                pose (np.ndarray): Canonical-mesh-to-camera transform (4x4).
                up_axis (str): The selected raw-mesh up-axis (e.g. "+y").
        """
        rgb, depth, mask, K = self._scene_inputs(scene, object_index)

        # World-up expressed in the camera frame, from the scene extrinsics
        # (T_world_cam, estimated against the table plane at scene creation).
        if scene.cam_extrinsics is None:
            from codex.scene.estimation.estimate_extrinsic import estimate_scene_transform
            scene.cam_extrinsics = estimate_scene_transform(scene)
        up_cam = scene.cam_extrinsics[:3, :3].T @ np.array([0.0, 0.0, 1.0])

        height = float(estimate_object_height(scene, scene.cam_extrinsics, obj_idx=object_index))
        print(f"Measured object height: {height*1000:.0f} mm")

        loaded = trimesh.load(mesh_path)
        if isinstance(loaded, trimesh.Scene):
            raw = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            raw = loaded

        # Observed object surface (camera frame), for hypothesis arbitration.
        obj_pts = np.asarray(
            scene.objects[object_index].get_pcd(use_cutout=True, return_o3d=True, transform=False).points
        )

        hypotheses = []
        for axis in range(3):
            scale = float(height) / float(raw.extents[axis])
            mesh_h = self._prepare_mesh_for_foundationpose(raw)
            mesh_h.apply_scale(scale)
            estimator = self.runtime["FoundationPose"](
                model_pts=mesh_h.vertices,
                model_normals=mesh_h.vertex_normals,
                mesh=mesh_h,
                scorer=self.scorer,
                refiner=self.refiner,
                debug_dir=self.debug_dir,
                debug=self.debug,
                glctx=self.glctx,
            )
            # cudnn + autocast disabled: see the note in estimate_pose.
            with torch.backends.cudnn.flags(enabled=False), torch.autocast("cuda", enabled=False):
                pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=est_refine_iter)

            # Two-sided surface residual between the posed hypothesis and the
            # observed points: observed->model (trimmed mean, robust to mask
            # bleed) + model->observed (40th percentile — only the visible
            # side of the model can be near a single-view cloud). A wrongly
            # scaled model cannot make both small.
            samples = (pose[:3, :3] @ mesh_h.sample(4096).T).T + pose[:3, 3]
            d_obs = np.sort(cKDTree(samples).query(obj_pts)[0])
            d_model = cKDTree(obj_pts).query(samples)[0]
            residual = float(np.mean(d_obs[: max(1, int(0.8 * len(d_obs)))]) + np.percentile(d_model, 40))

            hypotheses.append({"axis": axis, "scale": scale, "pose": pose, "residual": residual})
            print(f"up-axis hypothesis {'xyz'[axis]}: scale {scale:.4f}, surface residual {residual*1000:.1f} mm")
            del estimator
            torch.cuda.empty_cache()

        best = min(hypotheses, key=lambda h: h["residual"])

        # A clean registration sits well under 30 mm (typical: ~10 mm).
        # Above that, every hypothesis registered poorly (weak depth — dark
        # or reflective objects); retry the two best scales with more
        # refinement before giving up.
        if best["residual"] > 0.03:
            print(f"best residual {best['residual']*1000:.1f} mm is poor; retrying with more refinement")
            for h in sorted(hypotheses, key=lambda h: h["residual"])[:2]:
                mesh_h = self._prepare_mesh_for_foundationpose(raw)
                mesh_h.apply_scale(h["scale"])
                estimator = self.runtime["FoundationPose"](
                    model_pts=mesh_h.vertices,
                    model_normals=mesh_h.vertex_normals,
                    mesh=mesh_h,
                    scorer=self.scorer,
                    refiner=self.refiner,
                    debug_dir=self.debug_dir,
                    debug=self.debug,
                    glctx=self.glctx,
                )
                with torch.backends.cudnn.flags(enabled=False), torch.autocast("cuda", enabled=False):
                    pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=3 * est_refine_iter)
                samples = (pose[:3, :3] @ mesh_h.sample(4096).T).T + pose[:3, 3]
                d_obs = np.sort(cKDTree(samples).query(obj_pts)[0])
                d_model = cKDTree(obj_pts).query(samples)[0]
                residual = float(np.mean(d_obs[: max(1, int(0.8 * len(d_obs)))]) + np.percentile(d_model, 40))
                print(f"retry {'xyz'[h['axis']]}: surface residual {residual*1000:.1f} mm")
                if residual < best["residual"]:
                    best = {"axis": h["axis"], "scale": h["scale"], "pose": pose, "residual": residual}
                del estimator
                torch.cuda.empty_cache()
            if best["residual"] > 0.03:
                import warnings
                warnings.warn(
                    f"Registration residual is still {best['residual']*1000:.1f} mm; "
                    "the selected up-axis and pose may be unreliable. Inspect the scene "
                    "(dark/reflective objects give weak depth)."
                )

        # The winning rotation identifies the up-axis: the mesh-frame
        # direction of world-up is R_cam_mesh^T @ up_cam.
        up_mesh = best["pose"][:3, :3].T @ up_cam
        axis = int(np.argmax(np.abs(up_mesh)))
        tilt = np.degrees(np.arccos(min(1.0, abs(up_mesh[axis]) / np.linalg.norm(up_mesh))))
        if tilt > 10:
            import warnings
            warnings.warn(
                f"The object's up direction is {tilt:.0f} deg from the mesh's nearest axis; only "
                "axis-aligned meshes are supported (one of x/y/z pointing up), so the canonical "
                "mesh will be tilted and mis-scaled. Re-export the mesh with an axis up."
            )
        sign = "+" if up_mesh[axis] >= 0 else "-"
        up_axis = sign + "xyz"[axis]
        tilt_deg = float(np.degrees(np.arccos(np.clip(np.abs(up_mesh[axis]), 0.0, 1.0))))
        print(f"Selected up-axis {up_axis} (tilt from world-up: {tilt_deg:.1f} deg)")
        if tilt_deg > 30.0:
            import warnings
            warnings.warn(
                f"Registered mesh is tilted {tilt_deg:.1f} deg from world-up; "
                "the up-axis selection may be unreliable. Inspect the scene registration."
            )

        # Bake the canonical frame into the mesh file: rotate the up-axis to
        # +Z, then scale that (now vertical) extent to the measured height.
        up_vec = np.zeros(3)
        up_vec[axis] = 1.0 if sign == "+" else -1.0
        rot_canon = trimesh.geometry.align_vectors(up_vec, [0.0, 0.0, 1.0])
        baked = raw.copy()
        baked.apply_transform(rot_canon)
        baked.apply_scale(height / float(baked.extents[2]))
        baked.export(mesh_path)
        print(f"Baked canonical mesh (extents {np.round(baked.extents, 4)}) to {mesh_path}")

        # Register the canonical mesh once more for the returned pose (the
        # hypothesis pose transformed by rot_canon would also work, but a
        # fresh registration of the final model is what downstream consumers
        # historically got, and it re-checks the bake end to end).
        pose, _ = self.estimate_pose(scene, baked, object_index, est_refine_iter=est_refine_iter)
        return pose, up_axis

    def visualize_pose(self, scene: ImageScene, mesh: trimesh.Trimesh, pose: np.ndarray, mesh_info: dict, object_index: int = 0, display: bool = True, save_path: str = None) -> Union[np.ndarray, None]:
        """Visualize the estimated pose by overlaying the mesh on the scene.
        
        Args:
            scene (ImageScene): The scene containing the object.
            mesh (trimesh.Trimesh): The input mesh.
            pose (np.ndarray): The estimated pose.
            mesh_info (dict): Additional mesh information from estimate_pose.
            object_index (int): Index of the object in the scene to exclude. Defaults to 0.
            display (bool): Whether to display the visualization interactively. Defaults to True.
            save_path (str, optional): Path to save the visualization image. Defaults to None.
            
        Returns:
            vis_2d: 2D visualization with the mesh overlaid or None
        """
        # Use the interactive visualization function
        return visualize_pose_interactive(
            scene=scene,
            mesh=mesh,
            pose=pose,
            mesh_info=mesh_info,
            object_index=object_index,
            display=display,
            save_path=save_path
        )

def visualize_pose_interactive(
    scene: ImageScene, 
    mesh: trimesh.Trimesh, 
    pose: np.ndarray, 
    mesh_info: dict, 
    object_index: int = 0, 
    display: bool = True, 
    save_path: str = None
) -> Union[np.ndarray, None]:
    """Visualize the pose with interactive view toggling by modifying geometry.
    
    Args:
        scene (ImageScene): Scene object.
        mesh (trimesh.Trimesh): Mesh object.
        pose (np.ndarray): Pose matrix for the mesh.
        mesh_info (dict): Dictionary containing mesh info (to_origin, bbox, K).
        object_index (int): Index of the object in the scene. Defaults to 0.
        display (bool): Whether to display the visualization. Defaults to True.
        save_path (str, optional): Path to save the 2D visualization image. Defaults to None.
        
    Returns:
        vis_2d: 2D visualization with the mesh overlaid or None
    """
    # Extract mesh info
    to_origin = mesh_info['to_origin']
    bbox = mesh_info['bbox']
    K = mesh_info['K']
    
    # Apply pose to mesh
    mesh_posed = mesh.copy()
    mesh_posed.apply_transform(pose)
    
    # 3D Visualization
    # Launch interactive visualization
    visualize_mesh_pcd(
        mesh=mesh_posed,
        scene=scene,
        object_index=object_index,
        display_mode="mesh_on_pcd",
        view="mesh",
        window_name="Mesh Pose Visualization"
    )
    
    # 2D Visualization
    if display or save_path:
        runtime = _load_foundationpose_runtime()
        img = np.array(scene.image)
        vis_2d = img.copy()
        
        # Draw bounding box and axes
        center_pose = pose @ np.linalg.inv(to_origin)
        vis_2d = runtime["draw_posed_3d_box"](K, img=vis_2d, ob_in_cam=center_pose, bbox=bbox)
        vis_2d = runtime["draw_xyz_axis"](
            vis_2d,
            ob_in_cam=center_pose,
            scale=0.1,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        
        # Display and/or save
        vis_2d = Image.fromarray(vis_2d)
        if display:
            vis_2d.show()
        
        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vis_2d.save(save_path)
        
        return vis_2d
    
    return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register a scene's mesh: bake its canonical frame (up-axis to +Z, metric scale) "
                    "into the mesh file and cache pose.txt. Run this after replacing a mesh, "
                    "then post_mesh_metadata to rebuild bbox/CoACD."
    )
    parser.add_argument("--scene-dir", required=True, help="Path to the scene directory (contains scene.pkl)")
    parser.add_argument("--output-dir", default=None, help="Output directory for debug artifacts (default: OUT_DIR/mesh_pose/<scene_name>)")
    parser.add_argument("--show", action="store_true", help="Show the registered mesh over the scene point cloud")
    args = parser.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)
    scene_name = os.path.basename(scene_dir)
    mesh_path = os.path.join(scene_dir, "mesh", "mesh.glb")
    scene_path = os.path.join(scene_dir, "scene.pkl")

    scene = ImageScene.load(scene_path)

    debug_dir = args.output_dir or os.path.join(OUT_DIR, "mesh_pose", scene_name)
    estimator = MeshPoseEstimator(debug=0, debug_dir=debug_dir)
    pose, _ = estimator.register_and_canonicalize(scene, mesh_path)
    np.savetxt(os.path.join(scene_dir, "pose.txt"), pose)
    print(f"Saved pose to {os.path.join(scene_dir, 'pose.txt')}")

    if args.show:
        mesh, _ = load_mesh(mesh_path)
        mesh_posed = mesh.copy()
        mesh_posed.apply_transform(pose)
        render_mesh_pcd(mesh=mesh_posed, scene=scene, object_index=0, display_mode="mesh_on_bg", view="scene").show()
