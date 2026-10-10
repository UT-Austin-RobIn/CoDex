import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import open3d as o3d

def annotate_img(img, coords, arrows=None, save_path=None, scale=1) -> Image.Image:
    """Annotate an image with points and optional direction arrows.

    Args:
        img: PIL image to annotate.
        coords: List of [X%, Y%] percentile coordinates for keypoints.
        arrows: Optional list of tuples [(x_start%, y_start%, x_end%, y_end%)].
        save_path: Optional path to save the annotated image.
        scale: Scale factor for marker sizes.

    Returns:
        annotated_img: The annotated image.
    """
    # Convert PIL Image to numpy array for plotting and OpenCV
    img_array = np.array(img)

    # Get image dimensions
    height, width, _ = img_array.shape

    # Convert relative coordinates to absolute pixel coordinates
    absolute_coords = [(int(coord[0] * width / 100), int(coord[1] * height / 100)) for coord in coords]

    # Draw circles at the actuation points
    for coord in absolute_coords:
        cv2.circle(img_array, coord, radius=10*scale, color=(0, 0, 255), thickness=5*scale)  # Red circles

    # Draw arrows if directions are provided
    if arrows:
        for arrow in arrows:
            x_start = int(arrow[0] * width / 100)
            y_start = int(arrow[1] * height / 100)
            x_end = int(arrow[2] * width / 100)
            y_end = int(arrow[3] * height / 100)
            cv2.arrowedLine(img_array, (x_start, y_start), (x_end, y_end), color=(255, 0, 0), thickness=2*scale)

    # Convert back to RGB for PIL compatibility
    annotated_img = Image.fromarray(img_array)
    
    # Save the image if a save_path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        annotated_img.save(save_path)
        print(f"Annotated image saved to {save_path}")
    else:
        annotated_img.show()
    
    return annotated_img

def visualize_mesh_with_keypoints(mesh: o3d.geometry.TriangleMesh,
                                 keypoints: List[np.ndarray],
                                 target_points: Optional[List[np.ndarray]] = None,
                                 rays: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                                 ray_length: float = 1.0,
                                 show_wireframe: bool = True,
                                 save_path: Optional[str] = None) -> None:
    """
    Visualize a mesh with key points, optional target points, and optional rays.
    
    Args:
        mesh: The 3D mesh
        keypoints: List of key points (3D coordinates)
        target_points: Optional list of target points
        rays: Optional list of rays as (origin, direction) tuples
        ray_length: Length of ray visualization
        show_wireframe: Whether to show mesh wireframe
        save_path: Optional path to save the visualization
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add mesh
    mesh_to_display = o3d.geometry.TriangleMesh(mesh)  # Create a copy
    mesh_to_display.compute_vertex_normals()
    vis.add_geometry(mesh_to_display)
    
    # Add rays if provided
    if rays:
        for origin, direction in rays:
            # Normalize direction
            direction = direction / np.linalg.norm(direction)
            
            # Create ray endpoint
            end_point = origin + direction * ray_length
            
            # Create line set for ray
            points = np.vstack([origin, end_point])
            lines = np.array([[0, 1]])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(np.array([[0, 0.8, 0]]))  # Green color
            
            vis.add_geometry(line_set)
    
    # Add key points as red spheres
    for point in keypoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0, 0])  # Red color
        vis.add_geometry(sphere)
    
    # Add target points as blue spheres if provided
    if target_points:
        for point in target_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.translate(point)
            sphere.paint_uniform_color([0, 0, 1])  # Blue color
            vis.add_geometry(sphere)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.8, 0.8, 0.8])
    opt.mesh_show_wireframe = show_wireframe
    
    # Run visualization or save image
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Set a default view
        view = vis.get_view_control()
        view.set_zoom(0.8)
        
        # Update and render
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
    else:
        vis.run()
        vis.destroy_window()
