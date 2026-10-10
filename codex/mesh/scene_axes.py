"""Small helpers for projecting object-local axes onto rendered scene images."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


AXIS_LENGTH_M = 0.08
AXIS_COLORS = ((235, 45, 45), (35, 180, 65), (45, 95, 235))
AXIS_LABELS = ("X", "Y", "Z")


def project_axes(origin_world, rotation_world, world_to_camera, intrinsics,
                 axis_length_m: float = AXIS_LENGTH_M) -> np.ndarray:
    """Return ``(3, 2, 2)`` pixel line segments for X/Y/Z, NaN if behind camera."""
    origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_world, dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(world_to_camera, dtype=np.float64).reshape(4, 4)
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    local_endpoints = origin[None, :] + axis_length_m * rotation.T
    points = np.vstack((origin, local_endpoints))
    points_h = np.column_stack((points, np.ones(4)))
    cam = (extrinsic @ points_h.T).T[:, :3]
    z = cam[:, 2]
    pixels = np.full((4, 2), np.nan, dtype=np.float64)
    visible = z > 1e-8
    projected = (K @ cam[visible].T).T
    pixels[visible] = projected[:, :2] / projected[:, 2:3]
    return np.stack((np.repeat(pixels[:1], 3, axis=0), pixels[1:]), axis=1)


def draw_axes_overlay(image: Image.Image, segments: np.ndarray,
                      axis_length_m: float = AXIS_LENGTH_M) -> Image.Image:
    """Draw labeled, fixed-length local axes and a compact legend on an image."""
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    legend = f"local axes, {axis_length_m * 100:.0f} cm"
    box = draw.textbbox((16, 14), legend, font=font)
    draw.rectangle((10, 10, box[2] + 8, box[3] + 8), fill=(255, 255, 255), outline=(20, 20, 20), width=1)
    draw.text((16, 14), legend, fill=(20, 20, 20), font=font)

    for segment, color, label in zip(segments, AXIS_COLORS, AXIS_LABELS):
        if not np.isfinite(segment).all():
            continue
        start = tuple(np.rint(segment[0]).astype(int))
        end = tuple(np.rint(segment[1]).astype(int))
        # Outline first so the axis remains legible over either light or dark RGB.
        draw.line((start, end), fill=(255, 255, 255), width=6)
        draw.line((start, end), fill=color, width=3)
        delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        norm = np.linalg.norm(delta)
        if norm > 1:
            direction = delta / norm
            normal = np.array([-direction[1], direction[0]])
            base = np.asarray(end, dtype=float) - 9 * direction
            arrow = [end, tuple(np.rint(base + 4 * normal).astype(int)),
                     tuple(np.rint(base - 4 * normal).astype(int))]
            draw.polygon(arrow, fill=color)
        label_at = (end[0] + 3, end[1] - 8)
        draw.text(label_at, "+" + label, fill=color, font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    return result


def project_points(points_world, world_to_camera, intrinsics) -> np.ndarray:
    """Pixels ``(N, 2)`` of world points, NaN behind the camera."""
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    cam = (np.asarray(world_to_camera, dtype=np.float64) @ np.column_stack((pts, np.ones(len(pts)))).T).T[:, :3]
    out = np.full((len(pts), 2), np.nan)
    ok = cam[:, 2] > 1e-8
    proj = (np.asarray(intrinsics, dtype=np.float64) @ cam[ok].T).T
    out[ok] = proj[:, :2] / proj[:, 2:3]
    return out


EFFECT_COLOR = (46, 160, 90)
FUNCTIONAL_COLOR = (45, 95, 235)


def draw_goal_markers(image: Image.Image, world_to_camera, intrinsics, effect_world,
                      functional_world=None, direction_world=None, length_m: float = 0.08) -> Image.Image:
    """The effect point (green ring, where the functional object should act) and a line from
    the functional object's functional point along its functional direction (blue)."""
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    if functional_world is not None and direction_world is not None:
        start = np.asarray(functional_world, dtype=np.float64)
        d = np.asarray(direction_world, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        reach = float(np.dot(np.asarray(effect_world) - start, d))  # up to the effect point when it lies ahead
        end = start + d * (reach if reach > 0.02 else length_m)
        a, b = project_points([start, end], world_to_camera, intrinsics)
        if np.isfinite(a).all() and np.isfinite(b).all():
            draw.line((tuple(a), tuple(b)), fill=(255, 255, 255), width=6)
            draw.line((tuple(a), tuple(b)), fill=FUNCTIONAL_COLOR, width=3)
    (e,) = project_points([effect_world], world_to_camera, intrinsics)
    if np.isfinite(e).all():
        r = 13
        draw.ellipse((e[0] - r, e[1] - r, e[0] + r, e[1] + r), outline=(255, 255, 255), width=7)
        draw.ellipse((e[0] - r, e[1] - r, e[0] + r, e[1] + r), outline=EFFECT_COLOR, width=4)
        draw.text((e[0] + r + 5, e[1] - r - 4), "effect point", fill=EFFECT_COLOR, font=font,
                  stroke_width=2, stroke_fill=(255, 255, 255))
    return result
