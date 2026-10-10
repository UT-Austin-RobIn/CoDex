"""Validate pixel-aligned RGB-D geometry before inference or registration."""
import numpy as np


def validate_rgbd(image, points, intrinsics):
    expected = (image.height, image.width, 3)
    points = np.asarray(points)
    if points.shape != expected or not np.issubdtype(points.dtype, np.number):
        raise ValueError(f"points must be pixel-aligned H×W×3 camera coordinates in metres, shape {expected}; got {points.shape}")
    # Invalid depth pixels may be NaN (as in the bundled RealSense captures).
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    if not valid.any():
        raise ValueError("points contain no finite positive camera depth in metres")
    K = np.asarray(intrinsics, dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("intrinsics must be a finite 3×3 pinhole matrix with positive focal lengths and last row [0, 0, 1]")
    return valid
