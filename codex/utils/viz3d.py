"""Interactive 3D views (Plotly) of a scene's object mesh, keypoints, goal pose and grasps.

Each function reads a scene folder and returns a plotly Figure; the app shows
them with gr.Plot, and they also work in a notebook (fig.show()).
"""

import functools
import json
import os

import numpy as np
import plotly.graph_objects as go

HAND = "#e2a93b"       # LEAP hand
ACT = "#d64545"        # actuation point / press direction
FN = "#2f6fde"         # functional point / functional direction
TARGET = "#2e8b57"     # effect point (on the environment)
GHOST = "#9aa7b3"      # functional object at its captured pose


def _layout(fig, height=560, eye=(1.9, 1.9, 1.25)):
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.75)", font=dict(size=12)),
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]), up=dict(x=0, y=0, z=1)),
                   bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12),
        modebar=dict(orientation="v", bgcolor="rgba(255,255,255,0)", color="#9aa7b3", activecolor="#2f6fde"),
    )
    return fig


def _decimate(mesh_o3d, max_faces):
    if len(mesh_o3d.triangles) > max_faces:
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(max_faces)
    return mesh_o3d


def _hex(colors01):
    c = (np.clip(colors01, 0, 1) * 255).astype(int)
    return ["#%02x%02x%02x" % tuple(x) for x in c]


def _merge(meshes):
    """Concatenate open3d meshes into one (fewer, smaller traces)."""
    import open3d as o3d
    out = o3d.geometry.TriangleMesh()
    for m in meshes:
        out += m
    return out


def _mesh_trace(mesh_o3d, name, color=None, opacity=1.0, max_faces=40000, transform=None, **kw):
    mesh_o3d = _decimate(mesh_o3d, max_faces)
    v = np.asarray(mesh_o3d.vertices)
    if transform is not None:
        v = v @ transform[:3, :3].T + transform[:3, 3]
    v = np.round(v, 4)
    f = np.asarray(mesh_o3d.triangles)
    args = dict(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], name=name,
                opacity=opacity, flatshading=False, hoverinfo="name", showlegend=True,
                lighting=dict(ambient=0.55, diffuse=0.75, specular=0.15, roughness=0.7, fresnel=0.1),
                lightposition=dict(x=100, y=200, z=300))
    if color is None and mesh_o3d.has_vertex_colors():
        args["vertexcolor"] = _hex(np.asarray(mesh_o3d.vertex_colors))
    else:
        args["color"] = color or "#c9cfd6"
    args.update(kw)
    return go.Mesh3d(**args)


def _object_mesh(scene_dir, max_faces=12000):
    path = os.path.join(scene_dir, "mesh", "mesh.glb")
    return _load_object(os.path.abspath(path), os.path.getmtime(path), max_faces)


@functools.lru_cache(maxsize=16)
def _load_object(path, _mtime, max_faces):
    from codex.mesh.load_mesh import load_mesh, trimesh_to_o3d
    mesh, _ = load_mesh(path)
    return mesh, _decimate(trimesh_to_o3d(mesh, keep_color=True), max_faces)


def _point(p, color, name, text, size=9):
    return go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name=name,
                        marker=dict(size=size, color=color, line=dict(color="white", width=2)),
                        hovertext=[text], hoverinfo="text")


def _arrow(p, d, color, name, length=0.05, showlegend=False):
    """A line plus a cone, starting at p and pointing along d."""
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    q = np.asarray(p) + d * length
    line = go.Scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode="lines", name=name,
                        line=dict(color=color, width=6), hoverinfo="skip", showlegend=showlegend)
    head = go.Cone(x=[q[0]], y=[q[1]], z=[q[2]], u=[d[0]], v=[d[1]], w=[d[2]], anchor="tip",
                   sizemode="absolute", sizeref=min(length * 0.35, 0.02), colorscale=[[0, color], [1, color]],
                   showscale=False, hoverinfo="skip", name=name)
    return [line, head]


def _fmt(v):
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


# ---------------------------------------------------------------------------

def mesh_figure(scene_dir, height=300):
    """The functional object's mesh (canonical frame: metric, bottom at its lowest point, +Z up)."""
    _, mesh = _object_mesh(scene_dir)
    return _layout(go.Figure([_mesh_trace(mesh, "functional object", showlegend=False)]), height=height, eye=(2.1, 2.1, 1.3))


def keypoints_image(scene_dir, margin=0.18):
    """The scene image with the functional point (on the functional object) and the effect
    point (on the environment, where the functional object should act), cropped around them."""
    from PIL import Image, ImageDraw, ImageFont
    with open(os.path.join(scene_dir, "keypoints", "fn_key_pts.json")) as f:
        kp = json.load(f)
    img = Image.open(os.path.join(scene_dir, "scene_img.png")).convert("RGB")
    W, H = img.size
    points = [(p, FN, "functional point") for p in kp.get("fn_pts", [])[:1]] + \
             [(p, TARGET, "effect point") for p in kp.get("target_pts", [])[:1]]
    if not points:
        return None
    xy = np.array([[p[0] * W / 100.0, p[1] * H / 100.0] for p, _, _ in points])
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(16, W // 55))
    r = max(9, W // 90)
    for (x, y), (_, color, name) in zip(xy, points):
        draw.ellipse([x - r, y - r, x + r, y + r], outline="white", width=max(3, r // 3))
        draw.ellipse([x - r + 3, y - r + 3, x + r - 3, y + r - 3], outline=color, width=max(3, r // 3))
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        box = draw.textbbox((x + r + 8, y - r), name, font=font)
        draw.rounded_rectangle([box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3], radius=4, fill="white")
        draw.text((x + r + 8, y - r), name, fill=color, font=font)
    # Crop to the two points with a margin, keeping at least half the image.
    lo, hi = xy.min(0) - margin * np.array([W, H]), xy.max(0) + margin * np.array([W, H])
    size = np.maximum(hi - lo, 0.5 * np.array([W, H]))
    centre = (lo + hi) / 2
    lo = np.clip(centre - size / 2, 0, None)
    hi = np.minimum(lo + size, [W, H])
    lo = np.clip(hi - size, 0, None)
    return img.crop((int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])))


def keypoints_figure(scene_dir):
    """Functional object mesh with the actuation point (press direction) and the functional
    point (the part that acts, e.g. the nozzle, with its direction). The mesh is
    drawn see-through so points on hidden faces stay visible."""
    kp_path = os.path.join(scene_dir, "keypoints", "fn_key_pts.json")
    with open(kp_path) as f:
        kp = json.load(f)
    _, mesh = _object_mesh(scene_dir)
    traces = [_mesh_trace(mesh, "functional object", showlegend=False, opacity=0.6)]
    act = kp.get("act_contact") or {}
    if act.get("pos") is not None:
        p, d = np.asarray(act["pos"]), np.asarray(act["dir"])
        # Draw the press arrow outside the surface, pointing in.
        traces += _arrow(p - d / np.linalg.norm(d) * 0.05, d, ACT, "press direction", length=0.045)
        traces.append(_point(p, ACT, "actuation point",
                             f"actuation point {_fmt(p)}<br>press direction {_fmt(d)}"))
    fn = kp.get("fn_contact") or {}
    fn_pos = fn.get("pos") or (kp.get("fn_pts_3d") or [None])[0]
    if fn_pos is not None:
        fn_dir = fn.get("dir") or kp.get("fn_direction")
        traces.append(_point(fn_pos, FN, "functional point", f"functional point {_fmt(fn_pos)}"
                             + (f"<br>functional direction {_fmt(fn_dir)}" if fn_dir is not None else "")))
        if fn_dir is not None:
            traces += _arrow(np.asarray(fn_pos), fn_dir, FN, "functional direction", length=0.05)
    return _layout(go.Figure(traces), height=440)


def goal_figure(scene_dir, max_points=45000, method="vlm_cem"):
    """Scene point cloud, the functional object at its captured pose and at the goal pose."""
    from codex.scene.image_scene import ImageScene
    with open(os.path.join(scene_dir, "goal_gen", method, "best_candidate.json")) as f:
        goal = json.load(f)
    scene = ImageScene.load(os.path.join(scene_dir, "scene.pkl"))
    captured = scene.cam_extrinsics @ np.loadtxt(os.path.join(scene_dir, "pose.txt"))
    T_goal = np.asarray(goal["transform_world"])
    # Frame the functional object (now and at the goal) and the target, not the whole room.
    anchors = np.array([captured[:3, 3], T_goal[:3, 3]]
                       + ([goal["target_world"]] if goal.get("target_world") is not None else []))
    center = (anchors.min(0) + anchors.max(0)) / 2
    half = max((anchors.max(0) - anchors.min(0)).max() / 2 + 0.15, 0.25)
    box = np.stack([center - half, center + half], 1)
    pts, cols = scene.get_pcd(return_o3d=False, exclude_objs=[0], transform=True)
    inside = np.all((pts >= box[:, 0]) & (pts <= box[:, 1]), 1)
    pts, cols = pts[inside], cols[inside]
    if len(pts) > max_points:
        keep = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[keep], cols[keep]
    pts = np.round(pts, 4)
    traces = [go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name="scene",
                           marker=dict(size=2.4, color=_hex(cols)),
                           hoverinfo="skip")]
    _, mesh = _object_mesh(scene_dir)
    traces.append(_mesh_trace(mesh, "functional object now", color=GHOST, opacity=0.35, transform=captured))
    traces.append(_mesh_trace(mesh, "functional object at goal", transform=T_goal))
    fn_w = None
    if goal.get("fn_pt_mesh") is not None:
        fn_w = T_goal[:3, :3] @ np.asarray(goal["fn_pt_mesh"]) + T_goal[:3, 3]
        traces.append(_point(fn_w, FN, "functional point", f"functional point at goal {_fmt(fn_w)}"))
    if goal.get("target_world") is not None:
        effect = np.asarray(goal["target_world"], dtype=float)
        traces.append(_point(effect, TARGET, "effect point", f"effect point {_fmt(effect)}"))
        if fn_w is not None and np.linalg.norm(effect - fn_w) > 0.005:
            # The searched placement: from the functional object's functional point to the effect point.
            gap = np.linalg.norm(effect - fn_w)
            traces += _arrow(fn_w, effect - fn_w, "#1d2733", f"functional to effect point, {gap * 100:.1f} cm",
                             length=gap, showlegend=True)
    fig = _layout(go.Figure(traces), eye=(-1.2, -1.4, 1.0))
    fig.update_layout(scene=dict(aspectmode="cube", **{f"{a}axis_range": r.tolist() for a, r in zip("xyz", box)}))
    return fig


@functools.lru_cache(maxsize=1)
def _hand_visualizer():
    """LEAP visual meshes, decimated once in their link frames."""
    from codex.grasp_gen.render_grasp import GraspVisualizer, LEAP_RH_JOINT_NAMES
    vis = GraspVisualizer(None, joint_names=LEAP_RH_JOINT_NAMES)
    vis._leap_solver_meshes({})
    for link, meshes in vis._solver_visuals.items():
        vis._solver_visuals[link] = [_decimate(m, 2500) for m in meshes]
    return vis


def grasp_figure(scene_dir, grasp_file):
    """The functional object with the LEAP hand at one grasp, and the actuation point."""
    from codex.grasp_gen.load_grasp import load_grasp_config
    mesh_tm, mesh = _object_mesh(scene_dir)
    offset = np.array([0.0, 0.0, -mesh_tm.bounds[0, 2]])
    joints, pose, _ = load_grasp_config(grasp_file, offset=offset)
    links = _hand_visualizer().get_transformed_meshes(joints, pose, include_object=False)
    traces = [_mesh_trace(mesh, "functional object", max_faces=12000, showlegend=False),
              _mesh_trace(_merge(links), "LEAP hand", color=HAND, max_faces=10 ** 7)]
    kp_path = os.path.join(scene_dir, "keypoints", "fn_key_pts.json")
    if os.path.exists(kp_path):
        with open(kp_path) as f:
            act = json.load(f).get("act_contact") or {}
        if act.get("pos") is not None:
            traces.append(_point(act["pos"], ACT, "actuation point", f"actuation point {_fmt(act['pos'])}", 7))
    return _layout(go.Figure(traces), height=420)
