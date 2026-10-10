"""
Load a scene and estimate the height of an object (z) in the scene.

The measured height is the metric anchor for mesh registration
(MeshPoseEstimator.register_and_canonicalize), which scales the generated
mesh so its up-axis extent matches this height.
"""
import os
import argparse

import numpy as np

from codex.scene.image_scene import ImageScene
from codex.scene.estimation.estimate_extrinsic import estimate_scene_transform
from codex.utils.utils_path import get_scene_pkl_path

# Assumes the point cloud has been transformed so the frame is touching the table and z is up.
def estimate_object_height(scene: ImageScene, T, obj_idx=0):
    obj = scene.objects[obj_idx]
    # Always fetch raw (untransformed) points and apply T exactly once below.
    obj_pcd = obj.get_pcd(use_cutout=True, return_o3d=True, transform=False)

    obj_pcd.transform(T)

    obj_points = np.asarray(obj_pcd.points)
    obj_z = obj_points[:, 2]

    height = np.percentile(obj_z, 98) - np.percentile(obj_z, 2)

    return height


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate object height from a scene.")
    parser.add_argument("--scene-dir", required=True, help="Path to the scene directory (contains scene.pkl)")
    args = parser.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)
    scene = ImageScene.load(get_scene_pkl_path(scene_dir))

    T = scene.cam_extrinsics if scene.cam_extrinsics is not None else estimate_scene_transform(scene)
    for idx, obj in enumerate(scene.objects):
        height = estimate_object_height(scene, T, obj_idx=idx)
        print(f"object {idx} ({obj.name}): height {height*1000:.0f} mm")
