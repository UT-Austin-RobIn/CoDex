import os
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

from codex.vlm import VLMSession


def query_point(
        session: VLMSession,
        image: Image.Image,
        description: str,
        use_history: bool = False,
        point_query_model="sol") -> List[Tuple[float, float, str]]:
    """Locate a described point; coordinates are x,y image percentages.

    Sol and MolmoPoint handle visibility in the same request as localization.
    """
    from codex.keypoint_estimation.pointing import create_point_backend
    # Share the local model across calls without coupling it to chat histories.
    if not hasattr(session.interface, "_point_backends"):
        session.interface._point_backends = {}
    backends = session.interface._point_backends
    if point_query_model not in backends:
        backends[point_query_model] = create_point_backend(point_query_model)
    result = backends[point_query_model].predict(
        image, task="Locate the described part", object_name="object",
        descriptions={"target": description},
    )["target"]
    return [(*result["point"], description)] if result["visible"] else []


def annotate_img(img, coords, arrows=None, save_path=None, scale=1) -> None:
    """
    Draws circles on the specified coords on an image. Optionally draws arrows to indicate directions.

    Args:
        img: PIL.Image to annotate.
        coords: List of [X%, Y%] percentile coordinates.
        arrows: Optional list of tuples [(x_start%, y_start%, x_end%, y_end%)].
        save_path: Optional path to save the annotated image.
        scale: Scale factor applied to marker sizes.

    Returns:
        None
    """
    # Convert PIL Image to numpy array for plotting and OpenCV
    img_array = np.array(img)

    # Get image dimensions
    height, width, _ = img_array.shape

    # Convert relative coordinates to absolute pixel coordinates
    absolute_coords = [(int(coord[0] * width / 100), int(coord[1] * height / 100)) for coord in coords]

    # Draw circles at the functional points
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
