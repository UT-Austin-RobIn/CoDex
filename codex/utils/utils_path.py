import os
import json

_CURR_DIR = os.path.dirname(os.path.realpath(__file__))

PROJECT_ROOT_DIR = os.path.join(_CURR_DIR, '..', '..')
OUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'out')


def get_mesh_dir_from_scene(scene_dir: str) -> str:
    """Get the mesh directory path for a scene (scene_dir/mesh/)."""
    return os.path.join(scene_dir, "mesh")


def get_scene_pkl_path(scene_dir: str) -> str:
    """Get the scene.pkl path for a scene directory."""
    return os.path.join(scene_dir, "scene.pkl")


def get_scene_meta_path(scene_dir: str) -> str:
    """Get the scene_meta.json path for a scene directory."""
    return os.path.join(scene_dir, "scene_meta.json")


def write_scene_meta(
    scene_dir: str,
    functional_object: str,
    target_object: str,
    task: str = None,
) -> None:
    """Write scene metadata to scene_meta.json inside scene_dir."""
    meta = {
        "functional_object": functional_object,
        "target_object": target_object,
        "task": task,
    }
    os.makedirs(scene_dir, exist_ok=True)
    with open(get_scene_meta_path(scene_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def load_scene_meta(scene_dir: str) -> dict:
    """Load scene metadata from scene_meta.json. Returns {} if the file is missing."""
    meta_path = get_scene_meta_path(scene_dir)
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
