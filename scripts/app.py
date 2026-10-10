"""CoDex in the browser: run each pipeline stage and inspect its result.

    python scripts/app.py            # then open http://localhost:7860

Buttons run the stage scripts (scripts/<N>_*.py) in a subprocess and stream
their logs; results are read back from the scene folder.
"""

import collections
import functools
import re
import glob
import html
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import gradio as gr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from codex.utils import api_keys  # noqa: E402

EXAMPLES = os.path.join(ROOT, "assets", "scenes")
SCENES = os.path.join(ROOT, "out", "scenes")
STAGES = ["Scene", "Keypoints", "Goal", "Grasps", "Training", "Evaluation"]
MESH_TYPES = (".glb", ".gltf", ".obj", ".ply", ".stl", ".off")   # as codex.scene.create_scene
MESH_SOURCES = [("Reconstruct from the image (Tripo)", "tripo"), ("Use my own mesh file", "file")]
GOAL_METHODS = [("VLM-CEM (dir)", "vlm_cem_dir"), ("VLM-CEM", "vlm_cem")]
GRASP_METHODS = [("Primitive-aware", "primitive"), ("Force closure", "fc")]
JOBS = {}      # stage index -> running subprocess
FIGURES = {}   # (view, scene, file mtimes) -> plotly figure

api_keys.load_saved_keys()

# ------------------------------------------------------------------ files


def rel(path):
    return os.path.relpath(path, ROOT) if path else ""


def full(scene):
    """Absolute scene folder of a picker value: a path (relative to the repository)
    or just a scene's name under out/scenes."""
    if not scene:
        return ""
    path = scene if os.path.isabs(scene) else os.path.join(ROOT, scene)
    named = os.path.join(SCENES, scene)
    return named if not os.path.isdir(path) and os.path.isdir(named) else path


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def list_scenes():
    return sorted(rel(p) for p in glob.glob(os.path.join(SCENES, "*")) if os.path.isdir(p))


def latest_scene():
    """The scene worked on most recently (its folder or a stage output changed last)."""
    def touched(scene):
        path = full(scene)
        return max([os.path.getmtime(path)] + [os.path.getmtime(e.path) for e in os.scandir(path)])
    scenes = list_scenes()
    return max(scenes, key=touched) if scenes else None


def scene_choices():
    return [(os.path.basename(p), p) for p in list_scenes()]


def examples():
    """Example scenes: (folder, capture image, task)."""
    out = []
    for d in sorted(glob.glob(os.path.join(EXAMPLES, "*"))):
        task = read_json(os.path.join(d, "scene_meta.json")).get("task") or os.path.basename(d)
        out.append((d, os.path.join(d, "scene_img.png"), task[:1].upper() + task[1:]))
    return out


def free_name(base):
    """`base`, or base_2, base_3, ... if a scene of that name exists."""
    name, k = base, 2
    while os.path.exists(os.path.join(SCENES, name)):
        name, k = f"{base}_{k}", k + 1
    return name


def landing_stage(scene):
    """The tab a scene opens on: its furthest stage with results."""
    done = [i for i, ok in enumerate(stage_done(scene)) if ok]
    return done[-1] if done else 0


def new_scene_heading(first):
    if first:
        return ('<div class="newhead"><h2>Create your first scene</h2><p>A scene is one RGB-D capture of a '
                'functional object and the target it acts on; every stage works on it. Start from an '
                'example, or from your own capture.</p></div>')
    return '<div class="newhead"><h2>New scene</h2></div>'


def runs(scene):
    return sorted(glob.glob(os.path.join(full(scene), "rl_runs", "*")), key=os.path.getmtime, reverse=True)


def grasp_files(scene):
    return sorted(glob.glob(os.path.join(full(scene), "grasps", "grasp_*_optimized.txt")),
                  key=lambda p: int(os.path.basename(p).split("_")[1]))


def policy_evals(scene):
    found = glob.glob(os.path.join(full(scene), "evals", "*", "metrics.json"))
    return sorted((m for m in found if not os.path.dirname(m).endswith("zero_action")), key=os.path.getmtime)


def stage_done(scene):
    s = full(scene)
    if not os.path.isdir(s):
        return [False] * 6
    return [
        os.path.exists(os.path.join(s, "mesh_coacd", "mesh.glb")),
        os.path.exists(os.path.join(s, "keypoints", "fn_key_pts.json")),
        bool(goal_file(s)),
        bool(grasp_files(s)),
        any(glob.glob(os.path.join(r, "best_model", "*.zip")) for r in runs(s)),
        bool(policy_evals(s)),
    ]


def goal_file(scene, method=None):
    """The goal result of `method`, or the most recent of either method ('' if none)."""
    methods = [method] if method else [m for _, m in GOAL_METHODS]
    found = [os.path.join(full(scene), "goal_gen", m, "best_candidate.json") for m in methods]
    found = [f for f in found if os.path.exists(f)]
    return max(found, key=os.path.getmtime) if found else ""


def pretty_grasp(name):
    """grasp_2_optimized -> Grasp 2"""
    return name.split("_optimized")[0].replace("_", " ").capitalize()


# ------------------------------------------------------------------ status


def rail(scene):
    items = []
    for i, (name, ok) in enumerate(zip(STAGES, stage_done(scene))):
        state = "running" if i in JOBS else ("done" if ok else "todo")
        items.append(f'<li class="step {state}" data-i="{i}" role="button" tabindex="0">'
                     f'<span class="dot">{i}</span><span class="name">{name}</span></li>')
    return f'<ol class="rail">{"".join(items)}</ol>'


KEY_USE = {"OPENAI_API_KEY": "keypoints, goal pose and new captures",
           "TRIPO_API_KEY": "meshes generated from a capture"}


def missing_keys(*names):
    return [v for v in names if not api_keys.get_key(v)]


@functools.lru_cache(maxsize=1)
def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0).replace("NVIDIA ", "") if torch.cuda.is_available() else None
    except Exception:
        return None


def chips():
    """Header status: a chip per key the pipeline uses, the GPU, and the key-panel button,
    which asks for attention while a key is missing."""
    missing = missing_keys("OPENAI_API_KEY", "TRIPO_API_KEY")
    parts = [f'<span class="chip {"missing" if var in missing else "ok"}" title="Used for {KEY_USE[var]}">'
             f'{api_keys.KEYS[var]}{" key missing" if var in missing else ""}</span>'
             for var in ("OPENAI_API_KEY", "TRIPO_API_KEY")]
    gpu = gpu_name()
    parts.append(f'<span class="chip {"ok" if gpu else "off"}">{gpu or "No GPU"}</span>')
    parts.append(f'<button class="keys-btn{" attention" if missing else ""}">'
                 f'{"Add API keys" if missing else "API keys"}</button>')
    flag = ' data-open-keys="1"' if "OPENAI_API_KEY" in missing else ""
    return f'<div class="chips"{flag}>{"".join(parts)}</div>'


def key_label(var):
    value, source = api_keys.get_key(var), api_keys.key_source(var)
    name = api_keys.KEYS[var]
    if value:
        return f"{name}: {api_keys.masked(value)} ({source})"
    return f"{name}: not set, needed for {KEY_USE[var]}"


def key_notice(*names):
    """Inline warning for an action that needs keys that are not set ('' when all are)."""
    missing = missing_keys(*names)
    if not missing:
        return ""
    which = " and ".join(api_keys.KEYS[v] for v in missing)
    return (f'<div class="notice">Needs your {which} API key{"s" if len(missing) > 1 else ""}. '
            f'<a href="#" class="open-keys">Add {"them" if len(missing) > 1 else "it"}</a></div>')


def notices(kp_model, source, cap_mesh):
    capture = key_notice("OPENAI_API_KEY", *(["TRIPO_API_KEY"] if cap_mesh == "tripo" else []))
    return (key_notice("OPENAI_API_KEY") if kp_model == "sol" else "",
            key_notice("OPENAI_API_KEY"),
            capture if source == "capture" else "")


def save_keys(*values):
    for var, value in zip(api_keys.KEYS, values):
        if value and value.strip():
            api_keys.save_key(var, value.strip())
    return (chips(),) + tuple(gr.update(value="", label=key_label(var)) for var in api_keys.KEYS)


# ------------------------------------------------------------------ jobs


def _pump(pipe, out):
    for line in pipe:
        out.put(line)
    out.put(None)


Tick = collections.namedtuple("Tick", "log new rail status")


def clock(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def run_status(state, text, seconds):
    """A stage's status bar: running (what it is doing, elapsed), done, failed or stopped."""
    head = {"running": f"Running <time>{clock(seconds)}</time>", "done": f"Done in {clock(seconds)}",
            "failed": f"Failed after {clock(seconds)}", "stopped": f"Stopped after {clock(seconds)}"}[state]
    mark = {"running": '<span class="spin"></span>', "done": '<span class="mark">&#10003;</span>',
            "failed": '<span class="mark">&#10005;</span>', "stopped": '<span class="mark">&#9632;</span>'}[state]
    what = f'<span class="what">{html.escape(text)}</span>' if text else ""
    return f'<div class="run {state}">{mark}<b>{head}</b>{what}</div>'


def stream(stage, cmd, scene, env=None, progress=None, summary=None):
    """Run a stage command. Yields Ticks: the log (new=True when it changed), the
    stage rail and the status bar, at least once a second so the clock runs even
    while the stage prints nothing (e.g. waiting for the VLM)."""
    head = "$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n"
    t0 = time.time()
    # Unbuffered: a piped Python stage would otherwise print only at the end.
    proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1", **(env or {})),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    JOBS[stage] = proc
    # Read in a thread: helpers a stage starts in their own session (wandb's
    # service) inherit the pipe and may keep it open after the stage exits.
    out = queue.Queue()
    threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True).start()
    lines, log, shown = [], head, 0.0
    status = lambda: run_status("running", (progress(log) if progress else "") or "Starting", time.time() - t0)
    try:
        yield Tick(log, True, rail(scene), status())
        while True:
            try:
                line = out.get(timeout=0.5)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                if time.time() - shown >= 1:
                    shown = time.time()
                    yield Tick(log, False, None, status())
                continue
            if line is None:
                break
            if "Unknown tag" in line or "Unknown attribute" in line:  # URDF parser noise
                continue
            lines.append(line.rstrip("\n"))
            log = head + "\n".join(lines[-400:])
            shown = time.time()
            yield Tick(log, True, rail(scene), status())
        proc.wait()
    finally:
        JOBS.pop(stage, None)
    seconds = time.time() - t0
    if proc.returncode == 0:
        end, bar = "Done.", run_status("done", summary(log) if summary else "", seconds)
    elif proc.returncode < 0:
        end, bar = "Stopped.", run_status("stopped", "", seconds)
    else:
        end = f"Failed (exit code {proc.returncode})."
        bar = run_status("failed", failure_reason(log) or "see the log", seconds)
    yield Tick(log + f"\n\n{end}", True, rail(scene), bar)


def ui(t):
    """A Tick as outputs (log, rail, status); unchanged parts are left alone."""
    return (t.log, t.rail, t.status) if t.new else (gr.update(), gr.update(), t.status)


def milestone(log, steps):
    """The text of the last of `steps` (log substring, text) the log has reached."""
    found = [(log.rfind(key), text) for key, text in steps if key in log]
    return max(found)[1] if found else ""


SCENE_STEPS = [("Hot-start", "Copying the example"), ("Removed outputs", "Clearing the old mesh's outputs"),
               ("Created task", "Reconstructing the mesh with Tripo"), ("Imported", "Importing your mesh"),
               ("Measured object height", "Registering the mesh against the capture"),
               ("up-axis hypothesis", "Registering the mesh against the capture"),
               ("CoACD", "Splitting the mesh into collision parts")]
KEYPOINT_STEPS = [("Rendering", "Rendering views of the functional object"), ("Auto-annotating", "Asking the model for points"),
                  ("Projecting 2D points", "Placing the points on the mesh"),
                  ("Functional axis", "Choosing the functional direction")]


def goal_progress_text(log):
    rounds = [line for line in log.splitlines() if line.startswith("Iter ")]
    if not rounds:
        return "Preparing the scene"
    k, n = rounds[-1].split()[1].split("/")
    after = log[log.rfind("Iter "):]
    if "Eval prompt" in after or "\nID" in after:
        doing = "the VLM is scoring the candidates"
    else:  # from round 2 on the VLM first proposes candidates from its critique
        doing = "rendering candidates" if k == "1" else "the VLM is proposing candidates"
    return f"Round {k} of {n}: {doing}"


def eval_progress(log):
    line = next((l for l in reversed(log.splitlines()) if l.startswith("Evaluating ")), None)
    return f"Simulating {line.rsplit(':', 1)[1].split(',')[0].strip()}" if line else "Setting up the simulation"


def eval_summary(log):
    line = next((l for l in reversed(log.splitlines()) if l.strip().startswith("success:")), "")
    rates = dict(zip(line.split()[::2], line.split()[1::2]))
    parts = [f"task success {rates['success:']}"] if "success:" in rates else []
    if "goal:" in rates:
        parts.append(f"at the goal {rates['goal:']}")
    elif "lift:" in rates:
        parts.append(f"lifted {rates['lift:']}")
    return ", ".join(parts).capitalize() + "." if parts else ""


def grasp_summary(log):
    found = [l.strip() for l in log.splitlines() if l.strip().startswith("grasp_") and "simulated success" in l]
    if not found:
        wrote = next((l for l in reversed(log.splitlines()) if l.startswith("Wrote ") and " grasps " in l), "")
        return f"Wrote {wrote.split()[1]} grasps." if wrote else ""
    best = found[0].split("simulated success")[1].split(",")[0].strip()
    return f"{len(found)} grasps; the best succeeds in {best} of simulated trials."
def stop(stage):
    proc = JOBS.get(stage)
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)


def script(n, *args):
    name = next(f for f in os.listdir(os.path.join(ROOT, "scripts")) if f.startswith(f"{n}_"))
    return [sys.executable, os.path.join("scripts", name), *map(str, args)]


def need_scene(scene):
    if not os.path.isdir(full(scene)):
        raise gr.Error("Pick a scene first.")
    return full(scene)


def need_key(var):
    if not api_keys.get_key(var):
        raise gr.Error(f"Set the {api_keys.KEYS[var]} API key first (API keys, top right).")


# ------------------------------------------------------------------ 3D views


def figure(view, scene, *files):
    """Build (or reuse) a 3D view; returns (figure or None, message)."""
    s = full(scene)
    paths = [os.path.join(s, f) for f in files]
    if not all(os.path.exists(p) for p in paths):
        return None, "Nothing to show yet."
    key = (view, s, tuple(os.path.getmtime(p) for p in paths))
    if key not in FIGURES:
        from codex.utils import viz3d
        build = {"mesh": lambda: viz3d.mesh_figure(s),
                 "keypoints": lambda: viz3d.keypoints_figure(s),
                 "goal": lambda: viz3d.goal_figure(s, method=os.path.basename(os.path.dirname(paths[0]))),
                 "grasp": lambda: viz3d.grasp_figure(s, paths[0])}[view]
        try:
            FIGURES[key] = build()
        except Exception as exc:  # show the problem instead of a blank panel
            return None, f"Could not draw this view: {exc}"
    return FIGURES[key], ""


def mesh_view(scene):
    fig, _ = figure("mesh", scene, "mesh/mesh.glb")
    if fig is None:
        return None, "The mesh has not been reconstructed yet (see the log, or rebuild it below)."
    bbox = read_json(os.path.join(full(scene), "mesh", "metadata.json")).get("bbox")
    return fig, "Mesh size " + " × ".join(f"{v * 100:.1f}" for v in bbox) + " cm" if bbox else ""


def keypoint_view(scene):
    """The scene image with the functional and effect points, and the functional object in 3D."""
    fig, msg = figure("keypoints", scene, "keypoints/fn_key_pts.json")
    img = None
    if fig is not None:
        from codex.utils import viz3d
        try:
            img = viz3d.keypoints_image(full(scene))
        except (OSError, ValueError, KeyError):
            img = None
    return img, fig, msg


def goal_view(scene, method):
    best = read_json(goal_file(scene, method)) if goal_file(scene, method) else {}
    if best and "transform_world" not in best:
        return None, "This goal result is from an older version without a pose; search again."
    fig, msg = figure("goal", scene, f"goal_gen/{method}/best_candidate.json", "scene.pkl")
    return fig, goal_summary(read_json(goal_file(scene, method))) if fig is not None else msg


def goal_summary(best):
    """One line on how the goal search ended."""
    trace = best.get("search_trace") or []
    if "stop_reason" in best:  # VLM-CEM: the critic may accept early
        rounds = best.get("rounds") or len(trace)
        if best["stop_reason"] == "suitable":
            return f"Accepted in round {rounds}: the critic judged the pose suitable and stopped the search."
        rounds_text = f"{rounds} round{'s' if rounds != 1 else ''}"
        if best.get("suitable"):
            return f"Best pose after {rounds_text}."
        return f"Best effort after {rounds_text}: no pose was judged suitable."
    score, n = best.get("score"), len(trace)
    return (f"Best of {n} round{'s' if n != 1 else ''}"
            + (f", score {score:.2f}." if score is not None else "."))


def failure_reason(log):
    """The error a failed job's log ends with (None if there is none)."""
    errors = [l for l in log.splitlines() if "Error" in l and ":" in l]
    if not errors:
        return None
    reason = errors[-1].split(":", 1)[1].strip()
    if "insufficient_quota" in reason or "no credits" in reason:
        return "the OpenAI account has no credits left"
    return reason[:180]


CANDIDATE = re.compile(r"^ID(\d+): params=.*?score=([0-9.]+|None).*?desc=(.*)$")


def goal_rounds(scene, method, log=None):
    """Every round of a goal search: each candidate's render, score and critique.

    While a search runs (log given) the rounds are read off its log as they come
    in; afterwards from the saved search trace. Renders are the search's cache.
    """
    s = full(scene)
    rounds = {}  # round -> {candidate id: {"score", "desc", "img"}}
    chosen = {}
    if log is not None:
        k = 0
        for line in log.splitlines():
            if line.startswith("Iter "):
                k = int(line.split()[1].split("/")[0])
                rounds.setdefault(k, {})
            m = CANDIDATE.match(line)
            if m and k:
                rounds[k][m.group(1)] = {"score": None if m.group(2) == "None" else float(m.group(2)),
                                         "desc": m.group(3).strip()}
    else:
        best = read_json(goal_file(s, method)) if goal_file(s, method) else {}
        for r in best.get("search_trace") or []:
            cands = r.get("candidates") or {}
            items = cands.items() if isinstance(cands, dict) else ((str(c.get("id")), c) for c in cands)
            rounds[r["round"]] = {str(cid): {"score": c.get("score"), "desc": c.get("description") or ""}
                                  for cid, c in items}
            pick = r.get("chosen_id", r.get("incumbent_id"))
            if pick is not None:
                chosen[r["round"]] = str(pick)
    for f in glob.glob(os.path.join(s, "goal_gen", method, "cache", "iter*_cand*.jpg")):
        name = os.path.basename(f)
        k, cid = int(name[4:].split("_")[0]), name.split("_cand")[1].split(".")[0]
        if log is not None or k in rounds:
            rounds.setdefault(k, {}).setdefault(cid, {"score": None, "desc": ""})["img"] = f
    if not rounds:
        return ""
    cards = []
    for k in sorted(rounds, reverse=True):
        cands = rounds[k]
        scored = {cid: c for cid, c in cands.items() if c.get("score") is not None}
        pick = chosen.get(k) or (max(scored, key=lambda cid: scored[cid]["score"]) if scored else None)
        seen_before = {cid for j in rounds if j < k for cid in rounds[j]}
        tiles = []
        for cid in sorted(cands, key=int):
            c = cands[cid]
            img = (f'<a href="/gradio_api/file={html.escape(c["img"])}" target="_blank">'
                   f'<img loading="lazy" src="/gradio_api/file={html.escape(c["img"])}?t={os.path.getmtime(c["img"]):.0f}"></a>'
                   if c.get("img") else '<div class="noimg">no render kept</div>')
            score = (f'<span class="score" style="--s:{c["score"]:.2f}">{c["score"]:.2f}</span>'
                     if c.get("score") is not None else '<span class="score pending">scoring</span>')
            tag = " (kept)" if cid in seen_before else ""
            tiles.append(f'<div class="cand{" best" if cid == pick else ""}">{img}'
                         f'<div class="meta"><b>#{cid}{tag}</b>{score}</div>'
                         f'<p>{html.escape(c.get("desc") or "")}</p></div>')
        head = f"Round {k}" + (f": best #{pick}, score {cands[pick]['score']:.2f}" if pick else "")
        cards.append(f'<section class="round"><h4>{head}</h4><div class="cands">{"".join(tiles)}</div></section>')
    return f'<div class="rounds">{"".join(cards)}</div>'


def grasp_progress(log):
    """What grasp generation is doing, read off its log."""
    lines = log.splitlines()
    last = lambda key: next((l for l in reversed(lines) if key in l), None)
    stage = last("halving stage")
    if stage:
        k = int(stage.split("halving stage")[1].split(":")[0]) + 1
        what = stage.split(":", 1)[1].split(";")[0].strip()
        best = stage.split("best success=")[1].split()[0] if "best success=" in stage else None
        return (f"Simulating, round {k} of 3: {what}"
                + (f"; best success so far {float(best):.0%}" if best else ""))
    sim = last("Simulating the grasp primitive")
    if sim:
        return sim.strip().rstrip(".").replace("Simulating the grasp primitive for", "Simulating")
    batch = last("=== batch")
    if batch:  # force-closure solver
        k, rest = batch.split("batch")[1].strip().split(" ", 1)
        return f"Force closure, batch {k.replace('/', ' of ')}: {rest.strip(' =()').replace('so far', 'found so far')}"
    step, total = last(" step "), last("Optimising")
    if step and total:
        n = total.split("(")[1].split()[0] if "(" in total else "?"
        return f"Optimising candidate grasps: step {step.split()[1]} of {n}"
    return "Optimising candidate grasps" if total else "Loading the scene"


def grasp_view(scene, grasp):
    if not grasp:
        return None, "Nothing to show yet."
    return figure("grasp", scene, os.path.relpath(grasp, full(scene)))


def grasp_label(path):
    sim = read_json(path.replace("_optimized.txt", "_metadata.json")).get("sim")
    name = pretty_grasp(os.path.basename(path))
    return f"{name}: {sim['success']:.0%} success, {sim['raw_lift']:.0%} lift" if sim else name


def grasp_choices(scene, keep=None):
    paths = grasp_files(scene)
    value = keep if keep in paths else (paths[0] if paths else None)
    return (gr.update(choices=[(grasp_label(f), f) for f in paths], value=value),
            gr.update(choices=[(pretty_grasp(os.path.basename(f)), f) for f in paths], value=value))


def grasp_video(scene, grasp):
    if not grasp:
        return None
    name = os.path.splitext(os.path.basename(grasp))[0]
    path = os.path.join(full(scene), "evals", f"{name}-zero_action", "rollout.mp4")
    return path if os.path.exists(path) else None


def scene_view(scene):
    """The scene step's outputs: capture, segmented functional object, heading (mesh: mesh_view)."""
    s = full(scene)
    if not os.path.isdir(s):
        return None, None, ""
    meta = read_json(os.path.join(s, "scene_meta.json"))
    task = meta.get("task") or "No task set"
    text = (f'<div class="task"><h2>{task[:1].upper() + task[1:]}</h2>'
            f'<p>{meta.get("functional_object", "?")} to {meta.get("target_object", "?")}</p></div>')
    img, obj = os.path.join(s, "scene_img.png"), os.path.join(s, "obj_0.png")
    return (img if os.path.exists(img) else None), (obj if os.path.exists(obj) else None), text


# ------------------------------------------------------------------ training and evaluation


def run_label(run):
    name = os.path.basename(run)
    stamp = name.split("-", 1)[0]
    try:
        import yaml
        with open(os.path.join(run, "config.yaml")) as f:
            grasp = pretty_grasp(yaml.safe_load(f)["env_config"]["grasp_configs"][0])
    except Exception:
        grasp = name.split("-", 1)[-1]
    try:
        when = datetime.strptime(stamp, "%m%d_%H%M%S").strftime("%b %-d, %H:%M")
    except ValueError:
        return name
    return f"{grasp}, {when}"


def run_choices(scene, keep=None):
    choices = [(run_label(r), r) for r in runs(scene)]
    paths = [c[1] for c in choices]
    return gr.update(choices=choices, value=keep if keep in paths else (paths[0] if paths else None))


CURVES = [("rollout/full_batch_success", "Task success, training", "#2f6fde", "lines"),
          ("rollout/lift_success", "Lifted, training", "#9aa7b4", "lines"),
          ("eval/success_rate", "Task success, evaluation", "#2e8b57", "lines+markers")]


def training_curve(run):
    """Success over training steps from the run's tensorboard log (None if empty)."""
    events = glob.glob(os.path.join(run or "", "*_tensorboard", "**", "events.*"), recursive=True)
    if not events:
        return None
    key = ("curve", run, tuple(os.path.getmtime(e) for e in events))
    if key in FIGURES:
        return FIGURES[key]
    import plotly.graph_objects as go
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    series = {}
    for path in events:
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag, *_ in CURVES:
            if tag in acc.Tags()["scalars"]:
                series.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    if not series:
        return None
    fig = go.Figure()
    for tag, name, color, mode in CURVES:
        pts = sorted(series.get(tag, []))
        if pts:
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], name=name, mode=mode,
                                     line=dict(color=color, width=2.5 if tag.startswith("eval") else 2),
                                     marker=dict(size=7)))
    fig.update_layout(
        height=360, margin=dict(l=56, r=20, t=16, b=44), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Public Sans, system-ui, sans-serif", size=13, color="#1d2733"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Environment steps", gridcolor="#eef1f4", zeroline=False),
        yaxis=dict(range=[0, 1.02], tickformat=".0%", gridcolor="#eef1f4", zeroline=False),
        hovermode="x unified",
        modebar_remove=["lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "toImage"])
    FIGURES[key] = fig
    return fig


def training_status(run, log="", live=False):
    if not run:
        return '<p class="empty">No training runs yet.</p>'
    steps = [line for line in log.splitlines() if "total_timesteps" in line]
    parts = [f"{'Training' if live else 'Latest run'} <b>{run_label(run)}</b>"]
    if steps:
        parts.append(f"{int(steps[-1].split('|')[2]):,} steps")
    evals = [line for line in log.splitlines() if line.startswith("Success rate:")]
    if evals:
        parts.append(f"last evaluation {evals[-1].split(':', 1)[1].strip()}")
    return f'<p class="status">{", ".join(parts)}</p>'


def training_view(scene):
    found = runs(scene)
    run = found[0] if found else None
    return training_status(run), training_curve(run)


def evaluation_view(scene, run):
    if not run:
        return '<p class="empty">Train a policy to evaluate it here.</p>', None
    prefix = os.path.basename(run) + "-"
    found = [m for m in policy_evals(scene) if os.path.basename(os.path.dirname(m)).startswith(prefix)]
    if not found:
        return '<p class="empty">This run has not been evaluated yet.</p>', None
    data = read_json(found[-1])
    sm = data.get("summary", {})
    bars = "".join(
        f'<div class="row"><span>{label}</span><div class="track"><i style="width:{sm[k] * 100:.1f}%"></i></div>'
        f'<b>{sm[k]:.0%}</b></div>'
        for label, k in (("At the goal", "goal_success"), ("Lifted", "lift_success"), ("Held steady", "stable_success"),
                         ("Pressed", "act_success"), ("Upright", "upright_success")) if k in sm)
    goal = data.get("goal")
    motion = (f"carried to the goal pose ({os.path.basename(os.path.dirname(goal['source'])).replace('_', '-')})"
              if goal else "lifted 30 cm")
    ckpt = os.path.basename(os.path.dirname(found[-1]))[len(prefix):].replace("_", " ")
    when = datetime.fromtimestamp(os.path.getmtime(found[-1])).strftime("%b %-d, %H:%M")
    html = f"""<div class="ev">
      <div class="lead"><div class="num">{sm.get("success", 0):.0%}</div>
        <div class="cap">task success<small>{sm.get("episodes", "?")} episodes, {motion}, {ckpt}, {when}</small></div></div>
      <div class="bars">{bars}</div>
    </div>"""
    video = os.path.join(os.path.dirname(found[-1]), "rollout.mp4")
    return html, (video if os.path.exists(video) else None)


# ------------------------------------------------------------------ stages


def scene_job(cmd, scene_dir, done):
    for t in stream(0, cmd, scene_dir, progress=lambda log: milestone(log, SCENE_STEPS) or "Working",
                    summary=lambda log: done):
        yield ui(t)


def create_example(example, name, copy_outputs):
    if not example or not name:
        raise gr.Error("Choose an example and a name.")
    yield from scene_job(script(0, "--scene-name", name, "--from-scene", example,
                                *(["--copy-outputs"] if copy_outputs else [])), os.path.join(SCENES, name),
                         f"Scene {name} is ready.")


def create_capture(image, points, intrinsics, name, task, mesh_mode, mesh_file):
    if not (image and points and intrinsics and name and task):
        raise gr.Error("Add the image, point cloud, intrinsics, a name and the task.")
    if mesh_mode == "file" and not mesh_file:
        raise gr.Error("Choose your mesh file, or generate the mesh with Tripo.")
    need_key("OPENAI_API_KEY")
    if mesh_mode == "tripo":
        need_key("TRIPO_API_KEY")
    mesh = ["--mesh", mesh_file] if mesh_mode == "file" else []
    yield from scene_job(script(0, "--scene-name", name, "--image", image, "--points", points,
                                "--intrinsics", intrinsics, "--task", task, *mesh), os.path.join(SCENES, name),
                         f"Scene {name} is ready.")


def create_new(source, example, copy_outputs, image, points, intrinsics, task, mesh_mode, mesh_file, name):
    if source == "example":
        yield from create_example(example, name, copy_outputs)
    else:
        yield from create_capture(image, points, intrinsics, name, task, mesh_mode, mesh_file)


def remesh(scene, mesh_mode, mesh_file):
    """Rebuild the object mesh of the current scene: with Tripo, or from mesh_file."""
    s = need_scene(scene)
    if not os.path.exists(os.path.join(s, "scene.pkl")):
        raise gr.Error("This scene has no capture to register a mesh against.")
    if mesh_mode == "file" and not mesh_file:
        raise gr.Error("Choose your mesh file.")
    mesh_file = mesh_file if mesh_mode == "file" else None
    if mesh_file is None:
        need_key("TRIPO_API_KEY")
    yield from scene_job(script(0, "--output-dir", s, "--remesh", *(["--mesh", mesh_file] if mesh_file else [])),
                         s, "The mesh is rebuilt; find the keypoints again.")


def run_keypoints(scene, model):
    s = need_scene(scene)
    if model == "sol":
        need_key("OPENAI_API_KEY")
    for t in stream(1, script(1, "--scene-dir", s, "--point-query-model", model), s,
                    progress=lambda log: milestone(log, KEYPOINT_STEPS) or "Working",
                    summary=lambda log: "Keypoints found."):
        yield ui(t)


def run_goal(scene, method, iters, cands):
    s = need_scene(scene)
    need_key("OPENAI_API_KEY")
    if not stage_done(s)[1]:
        raise gr.Error("Find keypoints first.")
    last = 0.0
    cmd = script(2, "--method", method, "--scene-dir", s, "--iters", int(iters), "--cands", int(cands))
    for t in stream(2, cmd, s, progress=goal_progress_text,
                    summary=lambda log: goal_summary(read_json(goal_file(s, method)))):
        if t.new and time.time() - last > 2:  # re-read the rounds at most every 2 s
            last = time.time()
            yield (*ui(t), goal_rounds(s, method, t.log))
        else:
            yield (*ui(t), gr.update())
    yield (gr.update(), gr.update(), gr.update(), goal_rounds(s, method, t.log))


def run_grasps(scene, method, n, candidates, seed):
    s = need_scene(scene)
    if not stage_done(s)[1]:
        raise gr.Error("Find keypoints first.")
    extra = (["--num-candidates", int(candidates), "--seed", int(seed)] if method == "primitive"
             else ["--num-envs", int(candidates), "--seed-offset", int(seed)])
    for t in stream(3, script(3, "--scene-dir", s, "--method", method, "--num-grasps", int(n), *extra), s,
                    progress=grasp_progress, summary=grasp_summary):
        yield ui(t)


def simulate_grasp(scene, grasp):
    s = need_scene(scene)
    if not grasp:
        raise gr.Error("Generate grasps first.")
    # The grasp alone, under the training primitive (the lift it was selected for).
    for t in stream(5, script(5, "--scene-dir", s, "--zero-action", "--grasp-path", grasp, "--goal", "lift",
                              "--video-episodes", 2), s,
                    progress=eval_progress, summary=eval_summary):
        yield ui(t)


def training_progress(log, total):
    steps = [line for line in log.splitlines() if "total_timesteps" in line]
    evals = [line for line in log.splitlines() if line.startswith("Success rate:")]
    if not steps:
        return "Building the simulation"
    done = int(steps[-1].split("|")[2])
    text = f"{done:,} of {total:,} steps ({done / total:.0%})"
    return text + (f"; last evaluation {evals[-1].split(':', 1)[1].strip()}" if evals else "")


def training_summary(log):
    steps = [line for line in log.splitlines() if "total_timesteps" in line]
    evals = [line for line in log.splitlines() if line.startswith("Success rate:")]
    if not steps:
        return ""
    return (f"Trained {int(steps[-1].split('|')[2]):,} steps"
            + (f"; last evaluation {evals[-1].split(':', 1)[1].strip()}." if evals else "."))


def run_training(scene, grasp, steps, envs, seed):
    s = need_scene(scene)
    if not grasp:
        raise gr.Error("Generate grasps first.")
    before = set(runs(s))
    cmd = script(4, "--scene-dir", s, "--grasp-path", grasp, "--total-timesteps", int(steps),
                 "--num-envs", int(envs), "--seed", int(seed), "--no-eval-video")
    last, run, fig = 0.0, None, None
    # Without a configured wandb account, log offline instead of prompting.
    for t in stream(4, cmd, s, env={"WANDB_MODE": os.environ.get("WANDB_MODE", "offline")},
                    progress=lambda log: training_progress(log, int(steps)), summary=training_summary):
        if time.time() - last > 10:  # the curve is re-read at most every 10 s
            last = time.time()
            run = run or next((r for r in runs(s) if r not in before), None)
            fig = training_curve(run) or fig
        yield (*ui(t), training_status(run, t.log, live=True) if run else "", fig)
    run = run or next((r for r in runs(s) if r not in before), None)
    yield (gr.update(), gr.update(), gr.update(), training_status(run, t.log), training_curve(run) or fig)


def run_evaluation(scene, run, episodes):
    s = need_scene(scene)
    if not run:
        raise gr.Error("Train a policy first.")
    # Use the goal and stability frame saved with the policy.
    for t in stream(5, script(5, "--scene-dir", s, "--run-dir", run, "--episodes", int(episodes)), s,
                    progress=eval_progress, summary=eval_summary):
        yield ui(t)


# ------------------------------------------------------------------ UI

CSS = """
:root { --ink:#1d2733; --line:#dde3e9; --run:#d9a21b; --ok:#2e8b57; --blue:#2f6fde; --muted:#6b7886; --paper:#f4f6f8; }
footer { display: none !important; }
.gradio-container { max-width: 1240px !important; background: var(--paper) !important; }
#top { align-items: center; gap: 12px; }
#top h1 { font-size: 1.45rem; font-weight: 700; margin: 0; color: var(--ink); letter-spacing: -.01em; }
#top .chips { display: flex; gap: 6px; justify-content: flex-end; align-items: center; flex-wrap: wrap; }
.chip { font-size: .8rem; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--line); background: #fff; color: var(--ink); white-space: nowrap; }
.chip::before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; background: var(--ok); }
.chip.off { color: var(--muted); } .chip.off::before { background: none; border: 1px solid var(--muted); }
.chip.missing { color: #7a4f00; background: #fff6e0; border-color: #ebc77a; }
.chip.missing::before { background: var(--run); }
.keys-btn { font: inherit; font-size: .8rem; padding: 3px 12px; border-radius: 6px; border: 1px solid var(--line);
            background: #fff; color: var(--ink); cursor: pointer; white-space: nowrap; }
.keys-btn:hover { border-color: #b9c3cd; }
.keys-btn.attention { background: var(--run); border-color: var(--run); color: #fff; font-weight: 650; }
#keys { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; }
#keys.closed { display: none !important; }
.notice { background: #fff6e0; border: 1px solid #ebc77a; color: #6d4b00; border-radius: 6px; padding: 7px 12px; font-size: .88rem; }
.notice a { color: var(--blue); font-weight: 650; text-decoration: none; margin-left: 4px; }
.card { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px !important; }
/* one choice of scene source: a segmented control */
.seg .wrap { display: inline-flex !important; flex-wrap: nowrap !important; gap: 2px !important; padding: 4px;
             background: #e3e8ee; border-radius: 10px; }
.seg label { background: transparent !important; border: none !important; box-shadow: none !important;
             border-radius: 7px !important; padding: 8px 18px !important; cursor: pointer; }
.seg label span { font-weight: 600; color: var(--muted); }
.seg label.selected { background: #fff !important; box-shadow: 0 1px 3px rgba(29,39,51,.18) !important; }
.seg label.selected span { color: var(--ink); }
.seg input[type=radio] { display: none !important; }
#keys .note p { margin: 0; color: var(--muted); font-size: .85rem; }

/* stage rail: the navigation (the tab bar underneath is hidden) */
#stages > .tab-wrapper { height: 0; min-height: 0; overflow: hidden; visibility: hidden; margin: 0; padding: 0; border: 0; }
#stages { border: none; gap: 0 !important; }
#stages > .tabitem { padding: 14px 0 0 !important; border: none !important; }
.rail { list-style: none; display: grid; grid-template-columns: repeat(6, 1fr); margin: 0; padding: 12px 16px;
        background: #fff; border: 1px solid var(--line); border-radius: 8px; }
.rail li { margin: 0 !important; }
.step { display: flex; align-items: center; gap: 8px; cursor: pointer; border-radius: 6px; outline: none; }
.step::after { content: ""; flex: 1; height: 2px; background: var(--line); margin: 0 10px 0 2px; }
.step:last-child::after { display: none; }
.step.done::after { background: var(--ok); }
.dot { width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; font-size: .82rem; font-weight: 700;
       border: 2px solid var(--line); background: #fff; color: var(--muted); flex: none; transition: box-shadow .15s; }
.step.done .dot { background: var(--ok); border-color: var(--ok); color: #fff; }
.step.running .dot { border-color: var(--run); background: #fff7df; color: var(--ink); }
@media (prefers-reduced-motion: no-preference) { .step.running .dot { animation: pulse 1.6s ease-in-out infinite; } }
@keyframes pulse { 50% { box-shadow: 0 0 0 6px rgba(217,162,27,.2); } }
.name { font-weight: 550; color: var(--muted); font-size: .93rem; }
.step:hover .name { color: var(--ink); }
.step.active .name { color: var(--ink); font-weight: 700; }
.step.active .dot { box-shadow: 0 0 0 4px #dbe6fb; }
.step:focus-visible .dot { box-shadow: 0 0 0 3px var(--blue); }

/* rows of inputs with their action: one baseline, buttons as tall as the inputs */
.bar, .stagebar { align-items: flex-end !important; gap: 14px !important; }
/* the right-hand column of a stage panel: its start (and stop) buttons, stacked */
.actions { gap: 8px !important; justify-content: flex-end; align-self: stretch; }
.actions > button { height: 42px !important; min-height: 42px !important; }
.stagebar .row { gap: 14px !important; }
.bar > button, .stagebar > button { height: 42px !important; min-height: 42px !important; max-height: 42px !important;
                                 margin-bottom: 6px; }  /* level with the input boxes */
/* status lines and key notices take no room while empty */
.slot:not(:has(p)):not(:has(.notice)):not(:has(.ev)):not(:has(.run)) { display: none !important; }
/* goal search rounds: each candidate's render, score and critique */
.rounds { display: grid; gap: 12px; }
.round { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.round h4 { margin: 0 0 8px; font-size: .95rem; font-weight: 700; color: var(--ink); }
.cands { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; }
.cand { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: #fbfcfd; }
.cand.best { border-color: var(--ok); box-shadow: 0 0 0 2px rgba(46,139,87,.25); }
.cand img { width: 100%; display: block; }
.cand .noimg { padding: 20px 8px; text-align: center; color: var(--muted); font-size: .8rem; }
.cand .meta { display: flex; justify-content: space-between; align-items: center; padding: 6px 8px 0; font-size: .85rem; }
.cand .score { font-variant-numeric: tabular-nums; font-weight: 700; padding: 1px 7px; border-radius: 999px;
               background: color-mix(in srgb, var(--ok) calc(var(--s) * 100%), #c84040); color: #fff; }
.cand .score.pending { background: #e7ebef; color: var(--muted); font-weight: 600; }
.cand p { margin: 4px 8px 8px; font-size: .78rem; line-height: 1.35; color: #3b4652; }
/* a stage's job: running / done / failed / stopped */
.run { display: flex; align-items: center; gap: 10px; padding: 9px 14px; border-radius: 8px; border: 1px solid;
       font-size: .93rem; }
.run b { font-weight: 700; white-space: nowrap; } .run b time { font-weight: 600; font-variant-numeric: tabular-nums; margin-left: 4px; }
.run .what { color: var(--ink); }
.run.running { background: #fff7df; border-color: #ecd28b; color: #6d4b00; }
.run.done { background: #eaf5ee; border-color: #b9dcc6; color: #1e6b41; }
.run.failed { background: #fbecec; border-color: #e8b8b8; color: #8f2f2f; }
.run.stopped { background: #f1f3f5; border-color: var(--line); color: var(--muted); }
.run .mark { font-weight: 800; width: 16px; text-align: center; }
.spin { width: 14px; height: 14px; flex: none; border-radius: 50%; border: 2px solid rgba(217,162,27,.3);
        border-top-color: var(--run); animation: spin .9s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .spin { animation: none; } }
/* a stage's settings and start button: one plain panel, no boxes inside */
.stagebar { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px !important; }
.stagebar .form, .card .form { border: none !important; background: transparent !important; box-shadow: none !important; gap: 14px !important; }
.stagebar .block, .card > .form > .block, .card .bar .block, .card .column > .form > .block { border: none !important; background: transparent !important; box-shadow: none !important; }
.rescan { height: 40px !important; background: #fff !important; border: 1px solid var(--line) !important;
          font-weight: 550 !important; color: var(--ink) !important; }
.rescan:hover { border-color: #b9c3cd !important; }
.log textarea { font-family: 'JetBrains Mono', ui-monospace, monospace !important; font-size: .74rem !important;
                 line-height: 1.45 !important; background: #fbfcfd !important; }
.work { margin-top: 8px; align-items: flex-start; }
.note, .note p { color: var(--muted); }
.empty { color: var(--muted); margin: 18px 2px; }
.status { margin: 6px 2px; color: var(--muted); } .status b { color: var(--ink); font-weight: 650; }
.status.fail { color: #a23a3a; }
.task h2 { font-size: 1.35rem; font-weight: 700; margin: 4px 0 2px; color: var(--ink); }
.task p { margin: 0 0 8px; color: var(--muted); }
/* creating a scene */
.newhead h2 { font-size: 1.35rem; font-weight: 700; margin: 6px 0 4px; color: var(--ink); }
.newhead p { margin: 0 0 12px; color: var(--muted); max-width: 62em; }
.examples .caption-label { font-size: .85rem; }
/* simulation videos: as wide as the result column */
.rollout, .rollout > div, .rollout .wrap { height: auto !important; max-height: none !important; }
.rollout video { width: 100%; height: auto !important; max-height: none !important; }
.js-plotly-plot .modebar-group { background: transparent !important; }

/* evaluation summary */
.ev { display: grid; gap: 16px;
      align-items: center; background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 18px 22px; }
.ev .lead { display: flex; align-items: center; gap: 16px; }
.ev .num { font-size: 3rem; font-weight: 700; line-height: 1; color: var(--ink); font-variant-numeric: tabular-nums; }
.ev .cap { font-weight: 650; color: var(--ink); }
.ev .cap small { display: block; font-weight: 400; color: var(--muted); font-size: .82rem; margin-top: 2px; }
.ev .bars { display: grid; gap: 7px; }
.ev .row { display: grid; grid-template-columns: 7.5em 1fr 3em; align-items: center; gap: 10px; font-size: .9rem; color: var(--ink); }
.ev .row b { text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }
.ev .track { height: 8px; background: #edf1f5; border-radius: 4px; overflow: hidden; }
.ev .track i { display: block; height: 100%; background: var(--blue); border-radius: 4px; }
@media (max-width: 760px) {
  .rail { grid-template-columns: repeat(3, 1fr); row-gap: 10px; } .step::after { display: none; }
  .ev { grid-template-columns: 1fr; gap: 18px; }
}
"""

# The rail drives the (hidden) tabs and shows which one is open.
HEAD = """<script>
(() => {
  const tabs = () => [...document.querySelectorAll('#stages .tab-container button[role=tab]')];
  const sync = () => {
    const i = tabs().findIndex(b => b.getAttribute('aria-selected') === 'true');
    document.querySelectorAll('.rail .step').forEach((li, j) => li.classList.toggle('active', j === i));
  };
  const go = e => {
    const li = e.target.closest('.rail .step');
    if (!li || (e.type === 'keydown' && e.key !== 'Enter' && e.key !== ' ')) return;
    e.preventDefault();
    tabs()[+li.dataset.i]?.click();
    requestAnimationFrame(sync);
  };
  document.addEventListener('click', go);
  document.addEventListener('keydown', go);
  // API key panel: header button toggles it, "Add it" links open it.
  const keys = () => document.querySelector('#keys');
  document.addEventListener('click', e => {
    if (e.target.closest('.keys-btn')) keys()?.classList.toggle('closed');
    if (e.target.closest('.open-keys')) {
      e.preventDefault();
      keys()?.classList.remove('closed');
      window.scrollTo({top: 0, behavior: 'smooth'});
    }
  });
  let checked = false;  // open the panel once at start when the OpenAI key is missing
  setInterval(() => {
    sync();
    const c = document.querySelector('.chips');
    if (!checked && c && keys()) { checked = true; if (c.dataset.openKeys) keys().classList.remove('closed'); }
  }, 400);
})();
</script>"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(*["#eef3fd", "#dbe6fb", "#b7cdf6", "#8eb0ef", "#5f8ee7", "#2f6fde",
                                  "#2458b5", "#1c448c", "#153266", "#0e2244", "#08152b"]),
    neutral_hue="slate", radius_size="sm",
    font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(body_background_fill="#f4f6f8", block_border_color="#dde3e9", block_shadow="none")


def log_box():
    """A stage's log: always visible, left of its result."""
    return gr.Textbox(label="Log", lines=24, max_lines=24, autoscroll=True, elem_classes="log", interactive=False)


def build():
    with gr.Blocks(title="CoDex", theme=THEME, css=CSS, head=HEAD) as demo:
        with gr.Row(elem_id="top", equal_height=True):
            with gr.Column(scale=0, min_width=110):
                gr.HTML("<h1>CoDex</h1>")
            scene = gr.Dropdown(scene_choices(), value=latest_scene(), show_label=False,
                                container=False, allow_custom_value=True, scale=2)
            rescan = gr.Button("Rescan", scale=0, min_width=90, elem_classes="rescan")
            new_btn = gr.Button("New scene", scale=0, min_width=120, elem_classes="rescan")
            with gr.Column(scale=3, min_width=560):
                chip_bar = gr.HTML(chips())
        with gr.Column(elem_id="keys", elem_classes="closed"):
            with gr.Row(elem_classes="bar"):
                key_boxes = [gr.Textbox(label=key_label(var), type="password", placeholder="Paste a new key")
                             for var in api_keys.KEYS]
                save_btn = gr.Button("Save", variant="primary", scale=0, min_width=100)
            where = str(api_keys.keys_file())
            gr.Markdown(f"Saved to `{rel(where) if where.startswith(ROOT) else where}`, which git ignores. "
                        "Environment variables take precedence.", elem_classes="note")
        first = not list_scenes()
        # Creating a scene is its own view; the stage workspace shows the
        # scene picked at the top.
        with gr.Column(visible=first) as new_view:
            new_head = gr.HTML(new_scene_heading(first), padding=False)
            source = gr.Radio([("From an example", "example"), ("From your RGB-D capture", "capture")],
                              value="example", show_label=False, container=False, elem_classes="seg")
            with gr.Column(elem_classes="card"):
                with gr.Column(visible=True) as ex_form:
                    ex_gallery = gr.Gallery([(img, task) for _, img, task in examples()], columns=6,
                                            height=220, allow_preview=False, show_label=False,
                                            object_fit="cover", elem_classes="examples")
                    example = gr.State(None)
                    copy_out = gr.Checkbox(label="Keep its keypoints, goal and grasps")
                with gr.Column(visible=False) as cap_form:
                    with gr.Row():
                        cap_img = gr.File(label="RGB image", type="filepath", height=130)
                        cap_pts = gr.File(label="Points (.npy/.npz, H×W×3, metres)", type="filepath", height=130)
                        cap_k = gr.File(label="Intrinsics (.npy, 3×3)", type="filepath", height=130)
                    cap_task = gr.Textbox(label="Task", placeholder="spray the plant")
                    cap_mesh_mode = gr.Radio(MESH_SOURCES, value="tripo", label="Functional object mesh")
                    cap_mesh = gr.File(label="Mesh file: any scale, one axis up", type="filepath",
                                       file_types=list(MESH_TYPES), height=130, visible=False)
                with gr.Row(elem_classes="bar"):
                    new_name = gr.Textbox(label="Scene name", placeholder="Pick an example first", scale=3)
                    create_btn = gr.Button("Create scene", variant="primary", scale=1)
                    cancel_btn = gr.Button("Cancel", scale=0, min_width=100, visible=not first)
                cap_notice = gr.HTML(padding=False, elem_classes="slot")
            run_new = gr.HTML(padding=False, elem_classes="slot")
            log_new = gr.Textbox(label="Log", lines=12, max_lines=12, autoscroll=True, elem_classes="log",
                                 interactive=False)

        with gr.Column(visible=not first) as work_view:
            stage_rail = gr.HTML(rail(scene.value), padding=False)

            # Every stage: settings and its start button on top; below, the log
            # (left) and the result (right).
            with gr.Tabs(elem_id="stages", selected=landing_stage(scene.value) if scene.value else 0) as stages:
                with gr.Tab("Scene", id=0) as tab_scene:
                    run0 = gr.HTML(padding=False, elem_classes="slot")
                    with gr.Row(elem_classes="work"):
                        with gr.Column(scale=2, min_width=300):
                            log0 = log_box()
                        with gr.Column(scale=3):
                            scene_md = gr.HTML(padding=False)
                            with gr.Row(equal_height=True):
                                scene_img = gr.Image(label="Capture", interactive=False, height=240, scale=3,
                                                     show_download_button=False, show_fullscreen_button=False)
                                obj_img = gr.Image(label="Segmented functional object", interactive=False, height=240,
                                                   scale=2, show_download_button=False, show_fullscreen_button=False)
                            mesh_plot = gr.Plot(label="Reconstructed mesh")
                            mesh_msg = gr.Markdown(elem_classes="note")
                            with gr.Accordion("Rebuild the mesh", open=False):
                                rb_mode = gr.Radio(MESH_SOURCES, value="tripo", label="Functional object mesh")
                                rb_file = gr.File(label="Mesh file: any scale, one axis up", type="filepath",
                                                  file_types=list(MESH_TYPES), height=130, visible=False)
                                rb_btn = gr.Button("Rebuild mesh")

                with gr.Tab("Keypoints", id=1) as tab_kp:
                    with gr.Row(elem_classes="stagebar"):
                        with gr.Column(scale=4):
                            kp_model = gr.Radio([("GPT", "sol"), ("MolmoPoint (local)", "molmopoint")], value="sol",
                                                label="Model")
                        with gr.Column(scale=0, min_width=200, elem_classes="actions"):
                            kp_btn = gr.Button("Find keypoints", variant="primary")
                    kp_notice = gr.HTML(padding=False, elem_classes="slot")
                    run1 = gr.HTML(padding=False, elem_classes="slot")
                    with gr.Row(elem_classes="work"):
                        with gr.Column(scale=2, min_width=300):
                            log1 = log_box()
                        with gr.Column(scale=3):
                            kp_img = gr.Image(label="Functional and effect points", interactive=False, height=300,
                                              show_download_button=False, show_fullscreen_button=False)
                            kp_plot = gr.Plot(label="Functional object: actuation and functional points")
                            kp_msg = gr.Markdown(elem_classes="note")

                with gr.Tab("Goal", id=2) as tab_goal:
                    with gr.Row(elem_classes="stagebar"):
                        with gr.Column(scale=4):
                            g_method = gr.Radio(GOAL_METHODS, value="vlm_cem_dir", label="Method",
                                                info="VLM-CEM (dir) produces more sensible goals when the functional object has "
                                                     "a clear effect direction.")
                            with gr.Row():
                                g_iters = gr.Slider(1, 10, value=5, step=1, label="Rounds", min_width=200)
                                g_cands = gr.Slider(2, 16, value=8, step=1, label="Poses per round", min_width=200)
                        with gr.Column(scale=0, min_width=200, elem_classes="actions"):
                            g_btn = gr.Button("Search goal pose", variant="primary")
                            g_stop = gr.Button("Stop")
                    goal_notice = gr.HTML(padding=False, elem_classes="slot")
                    run2 = gr.HTML(padding=False, elem_classes="slot")
                    with gr.Row(elem_classes="work"):
                        with gr.Column(scale=2, min_width=300):
                            log2 = log_box()
                        with gr.Column(scale=3):
                            goal_plot = gr.Plot(label="Goal pose")
                            goal_msg = gr.Markdown(elem_classes="note")
                            goal_rounds_html = gr.HTML(padding=False, elem_classes="slot")

                with gr.Tab("Grasps", id=3) as tab_grasp:
                    with gr.Row(elem_classes="stagebar"):
                        with gr.Column(scale=4):
                            gr_method = gr.Radio(GRASP_METHODS, value="primitive", label="Method",
                                                 info="Primitive-aware grasps are chosen by simulating the task motion; "
                                                      "force closure is the static solver it replaced.")
                            with gr.Row():
                                gr_n = gr.Slider(1, 10, value=5, step=1, label="Grasps", min_width=180)
                                gr_cands = gr.Slider(256, 4096, value=2048, step=256, label="Candidates", min_width=200)
                                gr_seed = gr.Number(value=0, precision=0, label="Seed", min_width=90, scale=0)
                        with gr.Column(scale=0, min_width=200, elem_classes="actions"):
                            gr_btn = gr.Button("Generate grasps", variant="primary")
                            gr_stop = gr.Button("Stop")
                    run3 = gr.HTML(padding=False, elem_classes="slot")
                    with gr.Row(elem_classes="work"):
                        with gr.Column(scale=2, min_width=300):
                            log3 = log_box()
                        with gr.Column(scale=3):
                            with gr.Row(elem_classes="stagebar"):
                                gr_pick = gr.Dropdown(label="Grasp", scale=3)
                                sim_btn = gr.Button("Simulate", scale=1)
                            grasp_plot = gr.Plot(show_label=False)
                            grasp_msg = gr.Markdown(elem_classes="note")
                            sim_video = gr.Video(show_label=False, interactive=False, show_download_button=False,
                                                 elem_classes="rollout")

                with gr.Tab("Training", id=4) as tab_train:
                    with gr.Row(elem_classes="stagebar"):
                        with gr.Column(scale=4):
                            with gr.Row():
                                tr_grasp = gr.Dropdown(label="Grasp", scale=2, min_width=160)
                                tr_steps = gr.Number(value=400_000, precision=0, label="Steps", min_width=110)
                                tr_envs = gr.Number(value=2048, precision=0, label="Environments", min_width=110)
                                tr_seed = gr.Number(value=0, precision=0, label="Seed", min_width=90)
                        with gr.Column(scale=0, min_width=200, elem_classes="actions"):
                            tr_btn = gr.Button("Start training", variant="primary")
                            tr_stop = gr.Button("Stop")
                    run4 = gr.HTML(padding=False, elem_classes="slot")
                    with gr.Row(elem_classes="work"):
                        with gr.Column(scale=2, min_width=300):
                            log4 = log_box()
                        with gr.Column(scale=3):
                            tr_status = gr.HTML(padding=False, elem_classes="slot")
                            tr_plot = gr.Plot(show_label=False)

                with gr.Tab("Evaluation", id=5) as tab_eval:
                    with gr.Row(elem_classes="stagebar"):
                        with gr.Column(scale=4):
                            with gr.Row():
                                ev_run = gr.Dropdown(label="Run", scale=3, min_width=200)
                                ev_eps = gr.Slider(8, 128, value=32, step=8, label="Episodes", scale=2, min_width=200)
                        with gr.Column(scale=0, min_width=200, elem_classes="actions"):
                            ev_btn = gr.Button("Evaluate", variant="primary")
                            ev_stop = gr.Button("Stop")
                    run5 = gr.HTML(padding=False, elem_classes="slot")
                    with gr.Row(elem_classes="work"):
                        with gr.Column(scale=2, min_width=300):
                            log5 = log_box()
                        with gr.Column(scale=3):
                            ev_summary = gr.HTML(padding=False)
                            ev_video = gr.Video(show_label=False, interactive=False, show_download_button=False,
                                               elem_classes="rollout")

        # ---------------------------------------------------------- events
        key_notices = [kp_notice, goal_notice, cap_notice]
        notice_in = [kp_model, source, cap_mesh_mode]
        save_btn.click(save_keys, key_boxes, [chip_bar, *key_boxes]).then(notices, notice_in, key_notices)
        demo.load(notices, notice_in, key_notices)
        kp_model.change(notices, notice_in, key_notices)
        source.change(lambda v, ex: (gr.update(visible=v == "example"), gr.update(visible=v == "capture"),
                                     free_name(os.path.basename(ex)) if v == "example" and ex else
                                     ("" if v == "example" else free_name("my_scene"))),
                      [source, example], [ex_form, cap_form, new_name]).then(notices, notice_in, key_notices)

        def pick_example(evt: gr.SelectData):
            folder = examples()[evt.index][0]
            return folder, free_name(os.path.basename(folder))

        ex_gallery.select(pick_example, None, [example, new_name])
        cap_mesh_mode.change(lambda m: gr.update(visible=m == "file"), cap_mesh_mode, cap_mesh).then(
            notices, notice_in, key_notices)
        rb_mode.change(lambda m: gr.update(visible=m == "file"), rb_mode, rb_file)

        panel = [scene_img, obj_img, scene_md, mesh_plot, mesh_msg]
        light = [stage_rail, *panel, gr_pick, tr_grasp, ev_run]

        def on_scene(s, pick=None, _train_pick=None, run=None):
            p, t = grasp_choices(s, pick)
            img, obj, md = scene_view(s)
            fig, mmsg = mesh_view(s) if os.path.isdir(full(s)) else (None, "")
            return (rail(s), img, obj, md, fig, mmsg, p, t, run_choices(s, run))

        views = [new_view, work_view, new_head, cancel_btn]

        def show_new():
            return gr.update(visible=True), gr.update(visible=False), new_scene_heading(False), gr.update(visible=True)

        def show_work(s):
            """The workspace on scene `s`, opened on its furthest stage."""
            if not os.path.isdir(full(s)):
                first = not list_scenes()
                return (gr.update(visible=True), gr.update(visible=False), new_scene_heading(first),
                        gr.update(visible=not first), gr.update())
            return (gr.update(visible=False), gr.update(visible=True), gr.update(), gr.update(),
                    gr.Tabs(selected=landing_stage(s)))

        # Picking a scene at the top opens it.
        scene.change(on_scene, scene, light).then(show_work, scene, [*views, stages])
        demo.load(on_scene, scene, light)
        new_btn.click(show_new, None, views)
        cancel_btn.click(lambda s: show_work(s)[:4], scene, views)
        rescan.click(lambda s: gr.update(choices=scene_choices(), value=s), scene, scene)

        def goal_tab(s, method):
            fig, msg = goal_view(s, method)
            # while a search runs its rounds come from the job; otherwise from the saved trace
            return fig, msg, gr.update() if 2 in JOBS else goal_rounds(s, method)

        goal_views = [goal_plot, goal_msg, goal_rounds_html]

        # Views are drawn when their tab opens and when their inputs change.
        tab_scene.select(on_scene, [scene, gr_pick, tr_grasp, ev_run], light)
        tab_kp.select(keypoint_view, scene, [kp_img, kp_plot, kp_msg])
        tab_goal.select(goal_tab, [scene, g_method], goal_views)
        g_method.change(goal_tab, [scene, g_method], goal_views)
        tab_grasp.select(grasp_view, [scene, gr_pick], [grasp_plot, grasp_msg]).then(
            grasp_video, [scene, gr_pick], sim_video)
        gr_pick.change(grasp_view, [scene, gr_pick], [grasp_plot, grasp_msg]).then(
            grasp_video, [scene, gr_pick], sim_video)
        tab_train.select(training_view, scene, [tr_status, tr_plot])
        tab_eval.select(evaluation_view, [scene, ev_run], [ev_summary, ev_video])
        ev_run.change(evaluation_view, [scene, ev_run], [ev_summary, ev_video])

        def select_new(name):
            target = rel(os.path.join(SCENES, name))
            return gr.update(choices=scene_choices(), value=target if target in list_scenes() else None)

        refresh = [scene, gr_pick, tr_grasp, ev_run]

        def start(btn, label, job, inputs, outputs):
            """Run `job` from `btn`; the button reads Running and is disabled until the
            job ends (also when it fails). Returns the event to chain views after."""
            return btn.click(lambda: gr.update(value="Running…", interactive=False), None, btn).then(
                job, inputs, outputs).then(lambda: gr.update(value=label, interactive=True), None, btn)

        start(create_btn, "Create scene", create_new,
              [source, example, copy_out, cap_img, cap_pts, cap_k, cap_task, cap_mesh_mode, cap_mesh, new_name],
              [log_new, stage_rail, run_new]).then(select_new, new_name, scene).then(
            on_scene, refresh, light).then(
            lambda name: (*show_work(rel(os.path.join(SCENES, name)))[:4], gr.Tabs(selected=0))
            if os.path.isdir(os.path.join(SCENES, name)) else (gr.update(),) * 5, new_name, [*views, stages])
        start(rb_btn, "Rebuild mesh", remesh, [scene, rb_mode, rb_file], [log0, stage_rail, run0]).then(
            on_scene, refresh, light)
        start(kp_btn, "Find keypoints", run_keypoints, [scene, kp_model], [log1, stage_rail, run1]).then(
            keypoint_view, scene, [kp_img, kp_plot, kp_msg])
        start(g_btn, "Search goal pose", run_goal, [scene, g_method, g_iters, g_cands],
              [log2, stage_rail, run2, goal_rounds_html]).then(
            lambda s, m: goal_view(s, m), [scene, g_method], [goal_plot, goal_msg])
        start(gr_btn, "Generate grasps", run_grasps, [scene, gr_method, gr_n, gr_cands, gr_seed],
              [log3, stage_rail, run3]).then(on_scene, refresh, light)
        start(sim_btn, "Simulate", simulate_grasp, [scene, gr_pick], [log3, stage_rail, run3]).then(
            on_scene, refresh, light).then(grasp_video, [scene, gr_pick], sim_video)
        start(tr_btn, "Start training", run_training, [scene, tr_grasp, tr_steps, tr_envs, tr_seed],
              [log4, stage_rail, run4, tr_status, tr_plot]).then(on_scene, refresh, light)
        start(ev_btn, "Evaluate", run_evaluation, [scene, ev_run, ev_eps],
              [log5, stage_rail, run5]).then(evaluation_view, [scene, ev_run], [ev_summary, ev_video])
        for button, stage in ((g_stop, 2), (gr_stop, 3), (tr_stop, 4), (ev_stop, 5)):
            button.click(lambda stage=stage: stop(stage))
    return demo


if __name__ == "__main__":
    # Goal-search renders are shown from the scene folders.
    build().queue(default_concurrency_limit=None).launch(show_api=False, allowed_paths=[SCENES, EXAMPLES])
