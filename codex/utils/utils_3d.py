from typing import Tuple, Optional, Union
import numpy as np
import open3d as o3d
import cv2
from PIL import Image

def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply a transformation matrix to a point cloud.

    Args:
        pts (np.ndarray): The point cloud array of shape (N, 3) or (H, W, 3).
        T (np.ndarray): The 4x4 transformation matrix to apply.

    Returns:
        np.ndarray: The transformed point cloud of the same shape as `pts`.
    """
    pts = np.array(pts)
    pts_shape = pts.shape
    if len(pts_shape) == 1:
        pts = pts.reshape(1, -1)
        pts_shape = pts.shape
    pts_homo = np.concatenate([pts, np.ones((*pts_shape[:-1], 1))], axis=-1)
    pts_transformed = pts_homo @ T.T
    pts_transformed = pts_transformed / pts_transformed[..., 3][..., None]
    return pts_transformed[..., :-1]

def depth_to_point_cloud(
    depth: np.ndarray, 
    intrinsic: np.ndarray, 
    color: Optional[Union[np.ndarray, Image.Image]] = None, 
    keep_shape: bool = True, 
    visualize: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate a point cloud from a depth map using the provided intrinsic matrix.
    
    Args:
        depth (np.ndarray): A 2D array (H x W) with depth values.
        intrinsic (np.ndarray): A 3x3 intrinsic matrix.
        color (Optional[Union[np.ndarray, "Image.Image"]], optional): A color image corresponding to the depth map. Defaults to None.
        keep_shape (bool, optional): If True, returns points in a (H, W, 3) array; 
            if False, returns a flattened (H*W, 3) array. Defaults to True.
        visualize (bool, optional): If True, visualize the point cloud with Open3D. Defaults to False.
    
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: 
            - points (np.ndarray): Computed 3D points.
            - color (Optional[np.ndarray]): Computed colors, or None if no color image was provided.
    """
    # Get image dimensions from the depth map.
    H, W = depth.shape
    # Extract intrinsic parameters.
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

    # Create a mesh grid of pixel coordinates.
    u = np.arange(W)
    v = np.arange(H)
    u_grid, v_grid = np.meshgrid(u, v)

    # Compute 3D coordinates:
    # X = (u - cx) * depth / fx
    # Y = (v - cy) * depth / fy
    # Z = depth
    X = (u_grid - cx) * depth / fx
    Y = (v_grid - cy) * depth / fy
    Z = depth
    points = np.stack((X, Y, Z), axis=-1)  # shape (H, W, 3)

    if not keep_shape:
        points = points.reshape(-1, 3)

    # Process the color image if provided.
    output_color = None
    if color is not None:
        # If color is a PIL Image, convert to NumPy.
        if hasattr(color, "convert"):
            color = np.array(color)
        # Ensure the color image matches the depth resolution.
        if color.shape[0] != H or color.shape[1] != W:
            color = cv2.resize(color, (W, H), interpolation=cv2.INTER_LINEAR)
        # Normalize to [0, 1]
        output_color = np.float32(color) / 255.0
        if not keep_shape:
            output_color = output_color.reshape(-1, 3)

    # Visualization using Open3D.
    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        if output_color is not None:
            pcd.colors = o3d.utility.Vector3dVector(output_color.reshape(-1, 3))
        o3d.visualization.draw_geometries([pcd])

    return points, output_color

def points_to_pcd(points: np.ndarray, colors: Optional[np.ndarray] = None, visualize: bool = True) -> o3d.geometry.PointCloud:
    """
    Visualize a point cloud using Open3D and return the PointCloud object.

    Args:
        points (np.ndarray): An array of 3D points with shape (N, 3) or (H, W, 3).
        colors (Optional[np.ndarray], optional): An array of RGB colors with shape (N, 3) or (H, W, 3). Defaults to None.
        visualize (bool, optional): If True, visualizations are shown. Defaults to True.

    Returns:
        o3d.geometry.PointCloud: Open3D PointCloud object computed from the passed points and colors.
    """
    points_flat = points.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3) if colors is not None else None

    valid_mask = ~np.isnan(points_flat).any(axis=1)
    points_flat = points_flat[valid_mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_flat)
    if colors is not None:
        colors_flat = colors_flat[valid_mask]
        pcd.colors = o3d.utility.Vector3dVector(colors_flat)
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def remove_outliers_2d(
    points_2d: np.ndarray, 
    method: str = "statistical",
    nb_neighbors: int = 20, 
    std_ratio: float = 2.5,
    radius: float = 0.05, 
    min_points: int = 5,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from a point cloud in (H, W, 3) format using either 
    statistical or radius-based removal in Open3D.
    
    Args:
        points_2d (np.ndarray): Array of shape (H, W, 3), where each (x, y, z) is a point.
        method (str, optional): "statistical" or "radius" outlier removal. Defaults to "statistical".
        nb_neighbors (int, optional): Number of neighbors for statistical outlier removal. Defaults to 20.
        std_ratio (float, optional): Threshold on the standard deviation for statistical outlier removal. Defaults to 2.5.
        radius (float, optional): Search radius for radius-based outlier removal. Defaults to 0.05.
        min_points (int, optional): Minimum neighbors within 'radius' for radius-based removal. Defaults to 5.
        visualize (bool, optional): If True, shows an Open3D visualization with inliers (white) and outliers (red). Defaults to False.
                          
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - filtered_points (np.ndarray): Same shape as points_2d (H, W, 3), with outliers set to [NaN, NaN, NaN].
            - inlier_mask (np.ndarray): A boolean mask (H, W) where True = inlier, False = outlier.
    """
    # Flatten (H, W, 3) -> (N, 3)
    H, W, _ = points_2d.shape
    points_flat = points_2d.reshape(-1, 3)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_flat)

    # Outlier removal
    if method.lower() == "statistical":
        cl, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
    else:  # "radius"
        cl, inlier_indices = pcd.remove_radius_outlier(
            radius=radius, 
            min_points=min_points
        )

    # Build a mask of which points are inliers
    N = len(points_flat)
    inlier_mask_flat = np.zeros(N, dtype=bool)
    inlier_mask_flat[inlier_indices] = True
    inlier_mask = inlier_mask_flat.reshape(H, W)

    # Optionally visualize inliers vs. outliers
    if visualize:
        inlier_cloud = pcd.select_by_index(inlier_indices, invert=False)
        outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # white
        outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0]) # red
        
        print(f"Showing {len(inlier_cloud.points)} inliers (white) "
              f"and {len(outlier_cloud.points)} outliers (red).")
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # Create a filtered version of points_2d with outliers set to NaN
    filtered_points = points_2d.copy()
    filtered_points[~inlier_mask] = [np.nan, np.nan, np.nan]

    return filtered_points, inlier_mask
