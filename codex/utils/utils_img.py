import os
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def show_anns(masks: List[np.ndarray], borders: bool = False) -> None:
    """
    Display annotations (masks) on a plot.

    Args:
        masks (List[np.ndarray]): A list of binary mask arrays.
        borders (bool, optional): If True, draw borders around the masks. Defaults to False.

    Returns:
        None
    """
    if len(masks) == 0:
        return

    # Sort masks by area (descending)
    sorted_masks = sorted(masks, key=lambda x: np.sum(x), reverse=True)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Prepare an RGBA canvas
    img_height, img_width = sorted_masks[0].shape
    img = np.ones((img_height, img_width, 4))
    img[:, :, 3] = 0  # Set alpha to 0 initially

    for mask in sorted_masks:
        color_mask = np.concatenate([np.random.random(3), [0.5]])  # Random color with 50% opacity
        img[mask > 0] = color_mask  # Apply color to mask

        if borders:
            # Find and smooth contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            smoothed_contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, smoothed_contours, -1, (0, 0, 1, 0.4), thickness=1)  # Blue borders with alpha

    ax.imshow(img)

# -----------------------------
# Utility Functions
# -----------------------------
def save_image(image: np.ndarray, path: str) -> None:
    """
    Saves a NumPy image array to disk, creating the directory if needed.

    Args:
        image (np.ndarray): The image array to save.
        path (str): The file path where the image will be saved.

    Returns:
        None
    """
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    pil_img = Image.fromarray(image)
    pil_img.save(path)
    print(f"Image saved successfully at {path}")
