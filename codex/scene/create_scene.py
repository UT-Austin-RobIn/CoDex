from __future__ import annotations

from typing import List, Optional
import argparse
import shutil
import gc
import os
from pathlib import Path
import tempfile
import uuid
import numpy as np
import torch
from PIL import Image
from codex.scene.image_scene import ImageScene, ImageSceneObject
from codex.utils.utils_path import OUT_DIR, load_scene_meta, write_scene_meta
from codex.vlm import VLMInterface, VLMSession
from codex.scene.estimation.estimate_extrinsic import (
    estimate_scene_transform,
    visualize_transform,
)
from codex.scene.estimation.estimate_lang_seg import predict_mask as sam_mask, LangSAM
from codex.mesh.image_to_mesh import image_to_mesh
from codex.mesh.estimate_mesh_pose import MeshPoseEstimator
from codex.scene.post_mesh_metadata import post_mesh_metadata


# Outputs of stages 1-3; copied on hot-start only with copy_outputs=True.
STAGE_OUTPUT_DIRS = {"keypoints", "goal_gen", "grasps"}
# Training/evaluation outputs; never copied.
RUN_OUTPUT_DIRS = {"rl_runs", "evals", "wandb", "__pycache__", ".history"}


def _copy_ignore(copy_outputs: bool):
    skip = RUN_OUTPUT_DIRS | (set() if copy_outputs else STAGE_OUTPUT_DIRS)

    def ignore(_src: str, names: list[str]) -> list[str]:
        return [name for name in names if name in skip]
    return ignore


def _infer_objects_from_task(img: Image.Image, task: str) -> tuple[str, str]:
    """Query GPT to infer functional and target object names from a task + image."""

    def _extract(text: str) -> str:
        if "<" in text and ">" in text:
            try:
                return text.split("<", 1)[1].split(">", 1)[0].strip()
            except Exception:
                pass
        return text.strip()

    iface = VLMInterface()
    sess = VLMSession(iface, log_enabled=False)

    fn_prompt = f"What object should I use to {task}? Express in <>."
    fn_response = sess.query([img], fn_prompt, include_history=False, client="gpt")
    functional_object = _extract(fn_response)
    print(f"Inferred functional object: {functional_object}")

    tgt_prompt = f"What is the target object for the task '{task}'? Be concise. Express in <>."
    tgt_response = sess.query([img], tgt_prompt, include_history=False, client="gpt")
    target_object = _extract(tgt_response)
    print(f"Inferred target object: {target_object}")

    return functional_object, target_object


def _validate_hotstart_scene_dir(scene_dir: str) -> None:
    """Validate that a hot-start scene directory contains required artifacts."""
    required_files = [
        "scene.pkl",
        "scene_meta.json",
        "scene_img.png",
        "pose.txt",
        os.path.join("mesh", "mesh.glb"),
        os.path.join("mesh", "metadata.json"),
        os.path.join("mesh_coacd", "mesh.glb"),
        os.path.join("mesh_coacd", "metadata.json"),
    ]
    missing_files = [
        rel_path
        for rel_path in required_files
        if not os.path.exists(os.path.join(scene_dir, rel_path))
    ]
    if missing_files:
        raise FileNotFoundError(
            "Hot-start scene is missing required files: "
            + ", ".join(sorted(missing_files))
        )

def create_scene_from_cam(
    scene_img: Image.Image,
    object_names: List[str],
    scene_points: Optional[np.ndarray] = None,
    cam_intrinsics: Optional[np.ndarray] = None,
    cam_extrinsics: Optional[np.ndarray] = None,
    *,
    estimate_extrinsics: bool = True,
    visualize: bool = False,
) -> ImageScene:
    """Build an ImageScene from one RGB capture.

    Args:
        scene_img (Image.Image): The input scene RGB image.
        object_names (List[str]): List of object names to segment in the scene.
        scene_points (Optional[np.ndarray], optional): Scene point cloud or depth map. Defaults to None.
        cam_intrinsics (Optional[np.ndarray], optional): Camera intrinsics. Defaults to None.
        cam_extrinsics (Optional[np.ndarray], optional): Camera extrinsics. Defaults to None.
        estimate_extrinsics (bool, optional): Whether to estimate camera extrinsics. Defaults to True.
        visualize (bool, optional): Whether to visualize outcomes. Defaults to False.

    Returns:
        ImageScene: The initialized ImageScene object containing segmented objects.
    """
    # ------------------------------------------------------------------ init

    from codex.scene.input_validation import validate_rgbd
    validate_rgbd(scene_img, scene_points, cam_intrinsics)

    scene = ImageScene(
        scene_img,
        points=scene_points,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
    )

    # --------------------------------------------------------- segmentation
    objs: List[ImageSceneObject] = []

    if not hasattr(create_scene_from_cam, "_sam"):
        create_scene_from_cam._sam = LangSAM(sam_type="sam2.1_hiera_large")
    sam = create_scene_from_cam._sam

    def seg_fn(label: str) -> np.ndarray:
        masks = sam_mask(sam, scene_img, label, visualize=False)
        if masks is None or len(masks) == 0:
            raise RuntimeError(f"LangSAM found no mask for '{label}' in the scene image.")
        return masks[0]

    # segment each requested object
    for name in object_names:
        mask = seg_fn(name)
        objs.append(scene.add_object(mask, margin=0.1, name=name))

    # -------------------------------------------------- override table with SAM
    table_masks = sam_mask(sam, scene_img, "table", visualize=False)
    if table_masks is None or len(table_masks) == 0:
        raise RuntimeError("SAM found no mask for 'table' in the scene image.")
    table_mask = table_masks[0]
    table = scene.add_object(table_mask, margin=0.1, name="table")
    check_masks({o.name: o.mask for o in objs}, table_mask)

    if visualize:
        scene.isolate_objects([*objs, table]).show()

    # ---------------------------------------------------- extrinsics
    if estimate_extrinsics:
        T = estimate_scene_transform(scene)
        scene.cam_extrinsics = T
        if visualize:
            visualize_transform(scene, T)

    # 1) Delete singletons
    for attr in ("_sam", "_point_sess", "_gem_sess"):
        if hasattr(create_scene_from_cam, attr):
            delattr(create_scene_from_cam, attr)

    # 2) Empty CUDA cache
    torch.cuda.empty_cache()

    # 3) Collect Python garbage
    gc.collect()

    return scene


def check_masks(masks: dict, table_mask: np.ndarray, min_fraction: float = 1e-4,
                max_table_iou: float = 0.5) -> None:
    """Reject object masks that cannot be right: empty, or essentially the table.

    Later stages trust these masks (the functional object crop for its mesh, the target for
    the effect point), so a failed segmentation must stop scene creation.
    """
    table = np.asarray(table_mask) > 0
    for name, mask in masks.items():
        m = np.asarray(mask) > 0
        if m.mean() < min_fraction:
            raise RuntimeError(f"The segmentation found nothing for '{name}'. Check the object name "
                               "(--functional-object/--target-object) or the image.")
        iou = (m & table).sum() / max((m | table).sum(), 1)
        if iou > max_table_iou:
            raise RuntimeError(f"The segmentation of '{name}' is essentially the table (IoU {iou:.2f}). "
                               "Try a more specific object name.")


def _capture_from_camera(camera_serial: Optional[str] = None):
    """Capture an RGB image, point cloud, and intrinsics from a RealSense camera.

    Args:
        camera_serial: Optional RealSense device serial number. If None, the first
            available device is used.

    Returns:
        Tuple of (img: PIL.Image, points: np.ndarray, intrinsics: np.ndarray).
    """

    # Imported lazily: pyrealsense2 is only needed for camera capture, not
    # for image- or mesh-based scene creation.
    from codex.scene.rs_camera import RealSenseCamera

    kwargs = {"serial_number": camera_serial} if camera_serial else {}
    cam = RealSenseCamera(**kwargs)

    try:
        for _ in range(20):
            cam.get_camera_obs()
        obs = cam.get_camera_obs()
        img = Image.fromarray(obs['image'].astype(np.uint8))
        points, _ = cam.get_points(remove_outlier=True, observation=obs)
        return img, points, cam.get_intrinsics()
    finally:
        cam.stop()


def _load_from_files(
    image_path: str,
    points_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
):
    """Load an RGB image and optional point cloud / intrinsics from files.

    Args:
        image_path: Path to the RGB image (any PIL-supported format).
        points_path: Optional .npy/.npz containing pixel-aligned (H, W, 3)
            camera-frame coordinates in metres, matching the RGB image.
        intrinsics_path: Optional path to a .npy file containing a (3, 3) camera
            intrinsics matrix.

    Returns:
        Tuple of (img: PIL.Image, points: np.ndarray | None, intrinsics: np.ndarray | None).
    """
    img = Image.open(image_path).convert("RGB")

    def _load_array(path: str) -> np.ndarray:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            with arr:
                key = "points" if "points" in arr.files else arr.files[0]
                return arr[key]
        return arr

    points = _load_array(points_path) if points_path else None
    intrinsics = _load_array(intrinsics_path) if intrinsics_path else None
    if points is not None and intrinsics is not None:
        from codex.scene.input_validation import validate_rgbd
        validate_rgbd(img, points, intrinsics)
    return img, points, intrinsics


MESH_TYPES = (".glb", ".gltf", ".obj", ".ply", ".stl", ".off")


def build_object_mesh(scene_dir: str, mesh_file: Optional[str] = None, api_key: Optional[str] = None,
                      scene: Optional[ImageScene] = None) -> None:
    """Object mesh of the functional object, ready for the later stages.

    The mesh is generated from the object crop (``obj_0.png``) with Tripo, or
    imported from ``mesh_file`` (any scale, one axis pointing up). FoundationPose then
    registers it against the capture, which fixes its metric scale and up-axis
    (baked into ``mesh/mesh.glb``) and its pose in the scene (``pose.txt``);
    CoACD decomposes it for simulation (``mesh_coacd/``). Keypoints, goal and
    grasps of an earlier mesh are archived after the new mesh is complete.
    """
    import trimesh
    imported = None
    if mesh_file is not None:
        if os.path.splitext(mesh_file)[1].lower() not in MESH_TYPES:
            raise ValueError(f"Unsupported mesh file {mesh_file}; use one of {', '.join(MESH_TYPES)}")
        imported = trimesh.load(mesh_file, force="mesh")  # node transforms applied, parts merged
        if len(imported.faces) == 0:
            raise ValueError(f"{mesh_file} contains no triangles")
    stale = [d for d in sorted(STAGE_OUTPUT_DIRS) if os.path.isdir(os.path.join(scene_dir, d))]
    scene = scene or ImageScene.load(os.path.join(scene_dir, "scene.pkl"))
    # No published artifact is changed until registration and CoACD finish.
    with tempfile.TemporaryDirectory(prefix=".codex-mesh-", dir=Path(scene_dir).parent) as temporary:
        staging = Path(temporary)
        (staging / "mesh").mkdir()
        mesh_path = str(staging / "mesh" / "mesh.glb")
        if imported is not None:
            imported.export(mesh_path)
            print(f"Imported {mesh_file} ({len(imported.faces)} faces)")
        else:
            shutil.copy2(Path(scene_dir) / "obj_0.png", staging / "obj_0.png")
            image_to_mesh(scene_dir=str(staging), api_key=api_key)
        pose, _ = MeshPoseEstimator(debug=0).register_and_canonicalize(scene, mesh_path, 0)
        pose = np.asarray(pose)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Mesh registration returned an invalid 4×4 pose")
        np.savetxt(staging / "pose.txt", pose)
        post_mesh_metadata(mesh_path)
        required = ("mesh/mesh.glb", "mesh/metadata.json", "mesh_coacd/mesh.glb", "mesh_coacd/metadata.json", "pose.txt")
        if not all((staging / name).is_file() for name in required):
            raise RuntimeError("Mesh rebuild did not produce all registration/CoACD artifacts")
        _publish_mesh(staging, Path(scene_dir), stale)


def _publish_mesh(staging: Path, destination: Path, stale: list[str]):
    """Archive old derived artifacts and roll back publication if a rename fails."""
    names = ["mesh", "mesh_coacd", "pose.txt"]
    archive = destination / ".history" / f"mesh-{uuid.uuid4().hex}"
    archive.mkdir(parents=True)
    old, published = [], []
    try:
        for name in names + stale:
            path = destination / name
            if path.exists() or path.is_symlink():
                os.replace(path, archive / name)
                old.append(name)
        for name in names:
            os.replace(staging / name, destination / name)
            published.append(name)
    except BaseException:
        for name in reversed(published):
            os.replace(destination / name, staging / name)
        for name in reversed(old):
            os.replace(archive / name, destination / name)
        raise
    print(f"Archived previous mesh-dependent outputs at {archive}; training runs are retained.")


def create_scene(
    scene_name: str = "default_scene",
    *,
    task: Optional[str] = None,
    functional_object: Optional[str] = None,
    target_object: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    # --- camera source (live capture) ---
    camera_serial: Optional[str] = None,
    # --- pre-captured image source ---
    image_path: Optional[str] = None,
    points_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
    # --- hot-start from an existing scene directory ---
    from_scene: Optional[str] = None,
    copy_outputs: bool = False,
    # --- object mesh ---
    mesh_file: Optional[str] = None,
    remesh: bool = False,
) -> None:
    """Build, save, and generate a mesh for a scene.

    Provide exactly one image source:
    - **Camera mode**: leave ``image_path`` unset (optionally pass ``camera_serial``
      to target a specific RealSense device).
    - **Image mode**: pass ``image_path`` to an existing RGB image file.
                ``points_path`` and ``intrinsics_path`` are required so downstream
                VLM-CEM rendering and object-size estimation have depth geometry and
                camera parameters.
    - **Hot-start mode**: pass ``from_scene`` to copy an existing scene directory as
            the starting point, skipping camera capture and segmentation entirely.
    - **Remesh mode**: ``remesh=True`` only rebuilds the object mesh of the existing
            scene (see ``build_object_mesh``).

    The object mesh is generated with Tripo unless ``mesh_file`` is given.

    The mesh name is derived automatically from ``functional_object``.

    Outputs written to ``<scene_dir>`` (default ``out/scenes/<scene_name>/``, where
    ``out/`` is ``<repo>/out/``):
    - ``scene.pkl`` — structured scene artifact
    - ``scene_img.png`` — raw RGB scene capture
    - ``obj_0.png`` — segmented crop of the functional object used for mesh generation
    - ``mesh/mesh.glb`` — 3D mesh of the functional object, stored canonical
      (metric scale, gravity-up along +Z, baked in during registration)
    - ``mesh/metadata.json`` — mesh metadata (bbox)
    - ``pose.txt`` — estimated mesh-in-scene pose (4×4 matrix, used by goal gen)

    Args:
        scene_name: Subdirectory name under ``out/scenes/``.
        task: Natural-language task description (e.g. "spray the plant"). Used to
            automatically infer ``functional_object`` and ``target_object`` via GPT
            when they are not supplied explicitly.
        functional_object: Override segmentation label for the functional object.
            Inferred from ``task`` if omitted.
        target_object: Override segmentation label for the target object.
            Inferred from ``task`` if omitted.
        api_key: Tripo API key. Falls back to the ``TRIPO_API_KEY`` env var.
        output_dir: Optional output directory override (default: out/scenes/<scene_name>/).
        camera_serial: RealSense serial number for live capture (camera mode only).
        image_path: Path to an existing RGB image (image mode).
        points_path: Path to a .npy point cloud file (image mode, required).
        intrinsics_path: Path to a .npy camera intrinsics file (image mode, required).
        from_scene: Path to an existing scene directory to use as a hot-start.
            Its contents are copied to the output location (``out/scenes/<scene_name>/``
            or ``--output_dir``) before any new assets are applied.  Camera capture
            and segmentation are skipped; ``scene.pkl`` is loaded from the copy.
        copy_outputs: With ``from_scene``, also copy the source scene's
            keypoints, goal and grasps so later stages can start from them.
        mesh_file: Object mesh to use instead of generating one with Tripo.
        remesh: Only rebuild the object mesh of the existing scene.
    """
    # Resolve the output directory early so hot-start can copy into it.
    scene_dir = output_dir or os.path.join(OUT_DIR, "scenes", scene_name)
    destination = Path(scene_dir).resolve()
    source = Path(from_scene).resolve() if from_scene is not None else None
    if source is not None and source != destination and (source in destination.parents or destination in source.parents):
        raise ValueError("Source and destination scene directories must not contain one another")
    if not remesh and source != destination and destination.exists():
        raise FileExistsError(f"Scene destination exists: {destination}; choose another --scene-name or --output-dir to preserve its runs.")

    copied_from_scene = from_scene is not None

    if remesh:
        if not os.path.exists(os.path.join(scene_dir, "scene.pkl")):
            raise FileNotFoundError(f"No scene to remesh at {scene_dir}")
        build_object_mesh(scene_dir, mesh_file=mesh_file, api_key=api_key)
        return

    # ---------------------------------------------------------------- hot-start
    if from_scene is not None:
        from_scene = os.path.abspath(from_scene)
        _validate_hotstart_scene_dir(from_scene)
        scene_dir_abs = os.path.abspath(scene_dir)
        if scene_dir_abs != from_scene:
            shutil.copytree(from_scene, scene_dir, ignore=_copy_ignore(copy_outputs))
            print(f"Hot-start: copied {from_scene} → {scene_dir}")
        else:
            print(f"Hot-start: source and destination are the same ({scene_dir}), skipping copy.")

        scene = ImageScene.load(os.path.join(scene_dir, "scene.pkl"))
    else:
        if image_path is not None:
            if points_path is None:
                raise ValueError(
                    "Image mode requires --points: point cloud/depth data is needed "
                    "for VLM-CEM rendering and object-size estimation."
                )
            if intrinsics_path is None:
                raise ValueError(
                    "Image mode requires --intrinsics: camera intrinsics are needed "
                    "for scene geometry processing."
                )
            img, points, intrinsics = _load_from_files(image_path, points_path, intrinsics_path)
        else:
            img, points, intrinsics = _capture_from_camera(camera_serial)

        if points is None:
            raise ValueError(
                "Scene creation requires point cloud/depth data to render VLM-CEM and "
                "estimate object size."
            )

        # Infer object names from task if not explicitly provided.
        if (functional_object is None or target_object is None) and task is not None:
            inferred_fn, inferred_tgt = _infer_objects_from_task(img, task)
            functional_object = functional_object or inferred_fn
            target_object = target_object or inferred_tgt

        if functional_object is None or target_object is None:
            raise ValueError(
                "Could not determine object names. Pass --task to infer them automatically, "
                "or supply --functional-object and --target-object explicitly."
            )

        # Create the scene
        scene = create_scene_from_cam(
            img,
            object_names=[functional_object, target_object],
            scene_points=points,
            cam_intrinsics=intrinsics,
            estimate_extrinsics=True,
            visualize=False,
        )

        # Save the scene
        scene_path = os.path.join(scene_dir, "scene.pkl")
        scene.save(scene_path, save_previews=True)

        # Preserve the raw scene capture separately; obj_0.png is the segmented
        # functional-object crop written by scene.save(...).
        scene_img_path = os.path.join(scene_dir, "scene_img.png")
        img.save(scene_img_path)

    # Persist scene-level metadata so downstream scripts can load object names and task
    # without requiring them as CLI arguments.
    existing_meta = load_scene_meta(scene_dir)
    write_scene_meta(
        scene_dir,
        functional_object=(
            functional_object
            or existing_meta.get("functional_object")
            or scene.objects[0].name
        ),
        target_object=(
            target_object
            or existing_meta.get("target_object")
            or scene.objects[1].name
        ),
        task=task or existing_meta.get("task"),
    )

    if not copied_from_scene or mesh_file is not None:
        build_object_mesh(scene_dir, mesh_file=mesh_file, api_key=api_key, scene=scene)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a scene and generate a mesh from either a RealSense camera or a pre-captured RGBD image.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scene-name", dest="scene_name", type=str, default="default_scene", help="Name for the saved scene (out/scenes/<scene_name>/).")
    parser.add_argument("--task", dest="task", type=str, default=None, help="Task description (e.g. 'spray the plant'). Used to auto-infer functional/target objects via GPT.")
    parser.add_argument("--functional-object", dest="functional_object", type=str, default=None, help="Override segmentation label for the functional object (inferred from --task if omitted).")
    parser.add_argument("--target-object", dest="target_object", type=str, default=None, help="Override segmentation label for the target object (inferred from --task if omitted).")
    parser.add_argument("--api-key", dest="api_key", type=str, default=None, help="Tripo API key (falls back to TRIPO_API_KEY env var).")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None, help="Optional output directory override (default: out/scenes/<scene_name>/).")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera-serial", dest="camera_serial", type=str, default=None,
                     help="RealSense serial number for live capture.\n"
                          "Omit to use the first available device.")
    src.add_argument("--image", type=str, dest="image_path", default=None,
                     help="Path to a pre-captured RGB image (skips live capture).")

    parser.add_argument("--points", type=str, dest="points_path", default=None,
                        help="Pixel-aligned H×W×3 camera-frame coordinates in metres (.npy/.npz), matching the RGB image.")
    parser.add_argument("--intrinsics", type=str, dest="intrinsics_path", default=None,
                        help="Path to a .npy camera intrinsics matrix (3,3) (required in image mode).")
    src.add_argument("--from-scene", dest="from_scene", type=str, default=None,
                     help="Existing scene directory to copy (e.g. assets/scenes/<name>);\n"
                          "capture, segmentation and mesh generation are skipped.")
    src.add_argument("--remesh", action="store_true",
                     help="Only rebuild the object mesh of the existing scene (Tripo, or --mesh);\n"
                          "its previous mesh, pose, keypoints, goal and grasps are archived on success.")
    parser.add_argument("--mesh", dest="mesh_file", type=str, default=None,
                        help="Object mesh (.glb, .obj, .ply, .stl) to use instead of generating one\n"
                             "with Tripo. Any scale, with one of its axes pointing up; it is registered\n"
                             "against the capture.")
    parser.add_argument("--copy-outputs", action="store_true",
                        help="With --from-scene, also copy the source scene's keypoints,\n"
                             "goal and grasps so you can start at a later stage.")
    args = parser.parse_args()

    create_scene(
        scene_name=args.scene_name,
        task=args.task,
        functional_object=args.functional_object,
        target_object=args.target_object,
        api_key=args.api_key,
        output_dir=args.output_dir,
        camera_serial=args.camera_serial,
        image_path=args.image_path,
        points_path=args.points_path,
        intrinsics_path=args.intrinsics_path,
        from_scene=args.from_scene,
        copy_outputs=args.copy_outputs,
        mesh_file=args.mesh_file,
        remesh=args.remesh,
    )


if __name__ == "__main__":
    main()
