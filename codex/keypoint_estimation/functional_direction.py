"""Select a functional action direction from the six canonical mesh axes."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


AXES = {
    "+X": [1.0, 0.0, 0.0], "-X": [-1.0, 0.0, 0.0],
    "+Y": [0.0, 1.0, 0.0], "-Y": [0.0, -1.0, 0.0],
    "+Z": [0.0, 0.0, 1.0], "-Z": [0.0, 0.0, -1.0],
}
COLORS = {
    "+X": (235, 45, 45), "-X": (145, 20, 20),
    "+Y": (25, 180, 50), "-Y": (10, 100, 35),
    "+Z": (45, 105, 245), "-Z": (25, 55, 145),
}


def project_mesh_point(point_m, extrinsic, intrinsic):
    """Project mesh-frame XYZ with cached OpenCV world-to-camera matrices."""
    point = np.asarray(point_m, dtype=float).reshape(3)
    extrinsic = np.asarray(extrinsic, dtype=float).reshape(4, 4)
    intrinsic = np.asarray(intrinsic, dtype=float).reshape(3, 3)
    camera = (extrinsic @ np.append(point, 1.0))[:3]
    if not np.all(np.isfinite(camera)) or camera[2] <= 1e-6:
        return None
    pixel = intrinsic @ camera
    xy = pixel[:2] / pixel[2]
    return tuple(float(x) for x in xy) if np.all(np.isfinite(xy)) else None


def _font(size=20):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except OSError:
        return ImageFont.load_default()


def annotate_axes(view, camera, point_m, axis_length_m, view_id):
    """Overlay projected signed canonical axes at the fixed functional point."""
    image = Image.fromarray(np.asarray(view)).convert('RGB')
    origin = project_mesh_point(point_m, camera['extrinsic'], camera['intrinsic'])
    if origin is None or not (0 <= origin[0] < image.width and 0 <= origin[1] < image.height):
        raise ValueError(f"View {view_id}: functional point does not project into the image")
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.rectangle((0, 0, image.width, 32), fill=(15, 20, 28))
    draw.text((8, 4), f"View {view_id}: mesh axes at functional point", fill='white', font=font)
    point = np.asarray(point_m, dtype=float)
    for label, direction in AXES.items():
        endpoint = project_mesh_point(point + axis_length_m * np.asarray(direction),
                                      camera['extrinsic'], camera['intrinsic'])
        if endpoint is None:
            continue
        color = COLORS[label]
        draw.line((origin, endpoint), fill=color, width=5)
        x, y = endpoint
        draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
        draw.text((x+5, y-11), label, fill='white', font=font,
                  stroke_width=2, stroke_fill=color)
    x, y = origin
    draw.ellipse((x-7, y-7, x+7, y+7), fill=(255, 205, 30),
                 outline=(15, 15, 15), width=2)
    return image


def _image_url(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def select_functional_direction(mesh, point_m, task, object_name, views,
                                camera_data, output_dir, visible_view_ids=None,
                                client=None, model=None, max_views=6):
    """Ask Sol once which signed mesh axis is the task's outward action axis.

    ``views`` and ``camera_data`` are the existing mesh-view cache. The chosen
    point is fixed; this request only chooses among six directions. Invalid or
    incomplete replies raise rather than falling back to geometric normals.
    """
    if max_views < 2:
        raise ValueError('max_views must be at least 2')
    point = np.asarray(point_m, dtype=float).reshape(3)
    if not np.all(np.isfinite(point)):
        raise ValueError('functional point must have three finite coordinates')
    bounds = np.asarray(mesh.bounds, dtype=float).reshape(2, 3)
    axis_length = float(np.linalg.norm(bounds[1] - bounds[0]) * 0.18)
    if not np.isfinite(axis_length) or axis_length <= 0:
        raise ValueError('mesh must have nonzero finite extent')
    available = sorted(int(i) for i in views if int(i) in camera_data or str(i) in camera_data)
    preferred = (sorted(set(int(i) for i in visible_view_ids) & set(available))
                 if visible_view_ids is not None else available)
    if len(preferred) < 2:
        preferred = available
    valid = []
    for view_id in preferred:
        camera = camera_data.get(view_id, camera_data.get(str(view_id)))
        view = views.get(view_id, views.get(str(view_id)))
        xy = project_mesh_point(point, camera['extrinsic'], camera['intrinsic'])
        if xy is not None and 0 <= xy[0] < np.asarray(view).shape[1] and 0 <= xy[1] < np.asarray(view).shape[0]:
            valid.append(view_id)
    if len(valid) < 2:
        raise ValueError('Functional point must project into at least two mesh views')
    positions = np.linspace(0, len(valid)-1, min(max_views, len(valid)), dtype=int)
    selected_ids = [valid[i] for i in positions]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for view_id in selected_ids:
        camera = camera_data.get(view_id, camera_data.get(str(view_id)))
        view = views.get(view_id, views.get(str(view_id)))
        clean = Image.fromarray(np.asarray(view)).convert('RGB')
        image = annotate_axes(view, camera, point, axis_length, view_id)
        clean.save(output_dir / f'view_{view_id:02d}_clean.png')
        image.save(output_dir / f'view_{view_id:02d}_axes.png')
        images.extend((clean, image))
    prompt = (
        f"Object: {object_name}. Task: {task}. The yellow marker is the already selected "
        "functional point on this object's mesh. Each view is shown twice, first "
        "clean and then with the six signed canonical mesh axes overlaid at the "
        "same point. First identify the physical outlet or working face in the "
        "clean views and distinguish it from the trigger and body. Then trace "
        "which signed arrow points outward through that outlet or away from that "
        "working face across the paired views. Choose the ONE outward direction "
        "along which this functional point acts when performing the task (for "
        "example emission from an outlet or light, or the working direction of a "
        "functional object). Do not pick the finger press direction, a "
        "surface normal by default, or a direction toward an imagined target. "
        "The target scene is not shown; decide the object's own action axis. "
        "Return exactly one of +X, -X, +Y, -Y, +Z, -Z and a short reason."
    )
    if client is None:
        from codex.keypoint_estimation.pointing import SolPointing
        backend = SolPointing(model=model)
        client, model = backend.client, backend.model
    else:
        model = model or os.environ.get('CODEX_GPT_MODEL') or 'gpt-6-sol'
    schema = {
        'type': 'object', 'properties': {
            'axis': {'type': 'string', 'enum': list(AXES)},
            'reason': {'type': 'string'},
        }, 'required': ['axis', 'reason'], 'additionalProperties': False,
    }
    content = [{'type': 'input_text', 'text': prompt}]
    content.extend({'type': 'input_image', 'image_url': _image_url(image),
                    'detail': 'high'} for image in images)
    response = client.responses.create(
        model=model, reasoning={'effort': 'high'},
        input=[{'role': 'user', 'content': content}],
        text={'format': {'type': 'json_schema', 'name': 'functional_direction',
                         'strict': True, 'schema': schema}},
        max_output_tokens=4096,
    )
    if getattr(response, 'status', None) != 'completed' or not getattr(response, 'output_text', None):
        raise RuntimeError('Sol functional direction request did not complete')
    try:
        choice = json.loads(response.output_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Sol functional direction response was not valid JSON') from exc
    if not isinstance(choice, dict) or set(choice) != {'axis', 'reason'} or choice['axis'] not in AXES or not isinstance(choice['reason'], str) or not choice['reason'].strip():
        raise ValueError('Sol functional direction response omitted a valid axis or reason')
    usage = getattr(response, 'usage', None)
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump()
    record = {
        'axis': choice['axis'], 'direction': AXES[choice['axis']],
        'reason': choice['reason'], 'model': getattr(response, 'model', model),
        'usage': usage, 'raw_response': response.output_text,
        'point_m': point.tolist(), 'view_ids': selected_ids,
        'axis_length_m': axis_length, 'prompt': prompt,
    }
    (output_dir / 'selection.json').write_text(json.dumps(record, indent=2))
    return record
