import os
import numpy as np
import open3d as o3d
from typing import Tuple
import argparse
from codex.scene.image_scene import ImageScene
from codex.utils.utils_path import get_scene_pkl_path
from copy import deepcopy

def estimate_scene_transform(scene: ImageScene) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate transform to reorient scene such that:
    1. Table surface becomes xy plane with z=0 
    2. Object side has positive z
    3. x-axis aligns with cross product of new and old z directions
    
    Args:
        scene: ImageScene object containing point cloud and object segmentations
        
    Returns:
        4x4 camera-to-world transform aligning the table plane with z=0.
    """
    # Get table points using the table mask
    table_obj = next((o for o in scene.objects if o.name == "table"), None)
    if table_obj is None:
        raise ValueError("Scene has no object named 'table'; cannot estimate extrinsics.")
    pcd_table = table_obj.get_pcd(use_cutout=True, return_o3d=True, transform=False)
    table_points = np.asarray(pcd_table.points)
    table_center = np.mean(table_points, axis=0)

    plane_model, inliers = pcd_table.segment_plane(distance_threshold=0.01,
                                                  ransac_n=3,
                                                  num_iterations=1000)
    
    # Get plane normal (a,b,c,d) where ax+by+cz+d=0
    a, b, c, d = plane_model
    old_normal = np.array([a, b, c])
    old_normal = old_normal / np.linalg.norm(old_normal)
    
    # Determine if normal should be flipped
    # Get object center
    obj_points, _ = scene.objects[0].get_pcd(use_cutout=True, return_o3d=False, transform=False)
    obj_center = np.mean(obj_points, axis=0)
    
    # Flip normal if it points away from objects
    if np.dot(old_normal, obj_center - table_center) < 0:
        old_normal = -old_normal
        
    # New coordinate system
    new_z = old_normal
    old_z = np.array([0, 0, 1]) # Camera view direction
    
    # Get x direction as cross product
    new_x = np.cross(new_z, np.cross(new_z, old_z))
    new_x = new_x / np.linalg.norm(new_x)
    
    # Get y direction to complete right-handed system
    new_y = np.cross(new_z, new_x)
    
    # Construct rotation matrix
    R = np.vstack([new_x, new_y, new_z]).T
    
    # Construct transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = table_center  
    T = np.linalg.inv(T)  

    return T

def visualize_transform(scene, T):
    """
    Visualize original and transformed point clouds
    
    Args:
        scene: ImageScene object
        T: 4x4 transformation matrix
    """

    # Get original point cloud
    pcd_orig = scene.get_pcd(return_o3d=True, transform=False)
    
    # Create transformed point cloud
    pcd_trans = deepcopy(pcd_orig)
    pcd_trans.transform(T)
    
    # pcd_trans.points = o3d.utility.Vector3dVector(points_trans)
    # pcd_trans.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frames
    coord_orig = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    coord_trans = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    coord_trans.transform(T)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd_trans, coord_trans, coord_orig])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate scene transform from a scene directory.")
    parser.add_argument("--scene-dir", required=True, help="Path to the scene directory (contains scene.pkl)")
    args = parser.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)

    scene = ImageScene.load(get_scene_pkl_path(scene_dir))

    T = estimate_scene_transform(scene)
    visualize_transform(scene, T)
