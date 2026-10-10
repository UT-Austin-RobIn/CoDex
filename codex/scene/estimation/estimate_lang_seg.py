import torch
import os
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Any

from lang_sam import LangSAM

from codex.utils.utils_img import show_anns

def reset_autocast():
    """
    LangSAM sets global autocast, which can cause issues with other parts of the code.
    Reset autocast state by exiting the context that wasn't properly closed
    """
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.autocast(device).__exit__(None, None, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_autocast_enabled(device, False)


# Importing lang_sam enters a global bfloat16 autocast context and never
# exits it, silently downcasting (or breaking) every later CUDA op in the
# process — e.g. FoundationPose registration. Neutralize it at the source;
# predict_mask re-enables autocast around SAM inference only.
reset_autocast()


def set_autocast():
    """Return a bounded autocast context for LangSAM inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device(device)
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

def predict_mask(
    model: LangSAM,
    image: Image.Image,
    text: str,
    visualize: bool = False,
    verbose_return: bool = False,
) -> Any:
    """Predict segmentation masks for a text prompt.

    Args:
        model: LangSAM model.
        image: Input image.
        text: Text prompt to segment.
        visualize: If True, visualize masks overlaid on the image.
        verbose_return: If True, return the full results structure.

    Returns:
        masks_or_results: If ``verbose_return`` is False, returns masks; otherwise returns
            the full results.
    """
    # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #     results = mask_generator.generate(image)
    with set_autocast():
        results = model.predict([image], [text])
    # print(results)

    if visualize:
        # print(masks.shape)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.array(image))
        show_anns(results[0]["masks"])
        plt.axis('off')
        plt.show()
    
    if len(results) == 0:
        warnings.warn("No masks detected.")
        return None
    
    if verbose_return:
        return results
    else:
        return results[0]["masks"]

def normalize_logistic(logits: np.ndarray) -> np.ndarray:
    """Apply a logistic normalization to a logits array.

    Args:
        logits: Input logits array.

    Returns:
        normalized: Logits mapped to [0, 1].
    """
    val = 1 / (1 + np.exp(-logits))
    val = (val - np.min(val)) / (np.max(val) - np.min(val))
    return val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangSAM segmentation demo.")
    parser.add_argument("--scene-dir", required=True, help="Path to the scene directory containing scene.png")
    args = parser.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)
    filename = os.path.join(scene_dir, "scene.png")  # Assuming the image file is named "scene.png"

    model = LangSAM(
        sam_type="sam2.1_hiera_large",
    )

    # Read the image
    image = Image.open(filename)
    image = image.convert("RGB")

    # Resize the image
    width, height = image.size
    # resize_ratio = np.minimum(640 / width, 360 / height)
    # image = image.resize((int(width * resize_ratio), int(height * resize_ratio)))

    # Predict the mask
    masks = predict_mask(
        model, image, "sprayer", visualize=True)
    
    # Full results
    # results1 = predict_mask(
    #     model, image, "hot_glue_gun", visualize=False, verbose_return=True)
    # logits1 = results1[0]["logits"][0]
    
    # results2 = predict_mask(
    #     model, image, "metal", visualize=False, verbose_return=True)
    # logits2 = results2[0]["logits"][0]

    # plt.figure()
    # plt.imshow(normalize_logistic(logits1))

    # plt.figure()
    # plt.imshow(normalize_logistic(logits2))

    # plt.figure()
    # plt.imshow(normalize_logistic(logits1 + logits2))

    # plt.show()

    
