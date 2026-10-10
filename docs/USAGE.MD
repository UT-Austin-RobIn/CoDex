# Usage guide

Each stage is a script in `scripts/` that reads from and writes to one scene folder. Stages only
communicate through these files, so you can re-run a stage, inspect its output, or replace it with
your own (for example, hand-written keypoints).

```
<scene>/
├── scene.pkl, scene_img.png, scene_meta.json   capture, segmentation, task      (stage 0)
├── mesh/mesh.glb, mesh_coacd/mesh.glb, pose.txt object mesh, collision parts, pose (stage 0)
├── keypoints/fn_key_pts.json                    actuation / functional / target  (stage 1)
├── goal_gen/<method>/best_candidate.json        goal pose                        (stage 2)
├── grasps/grasp_N_optimized.txt                 grasps, best first               (stage 3)
├── rl_runs/<run>/                               config, checkpoints, best model  (stage 4)
└── evals/<run>/                                 metrics.json, rollout.mp4        (stage 5)
```

Run any script with `--help` for all options. `python scripts/app.py` runs the same scripts from
the browser.

## Stage 0: create a scene

```bash
python scripts/0_create_scene.py --scene-name demo --from-scene assets/scenes/grey_spray_bottle_plant
```

Choose one input:

- `--from-scene <dir>` copies an existing scene. Add `--copy-outputs` to also copy its keypoints,
  goal pose and grasps.
- `--image rgb.png --points pts.npy --intrinsics K.npy` builds a scene from an RGB-D frame
  (pixel-aligned points H×W×3 in camera coordinates, in metres, matching the RGB image; intrinsics 3×3).
- `--camera-serial <id>` captures from a RealSense camera (needs `pyrealsense2`).

A new capture must show the functional object, the target and the table. `--task "spray the
plant"` lets GPT name the two objects, or pass `--functional-object` and `--target-object`. LangSAM
segments them; scene creation stops if a mask is empty or covers the table. Tripo
(`TRIPO_API_KEY`) generates the functional object's mesh from its crop (or `--mesh object.obj`
supplies your own), and FoundationPose registers the mesh in the scene. The mesh is stored upright
(+Z) in metres, and CoACD splits it into the convex parts used for simulation. Simulation and grasp
generation stand the functional object upright on the table. `--remesh` rebuilds the mesh of an
existing scene; see [MESH_CREATION.MD](MESH_CREATION.MD).
`python -m codex.scene.visualize_scene_point_cloud --scene-dir <scene>` shows the segmented scene.

Scene creation rejects an occupied destination; choose a new scene name or output directory.
Load scene pickles only from trusted sources.

## Stage 1: keypoints

```bash
python scripts/1_estimate_keypoints.py --scene-dir out/scenes/demo
```

The model finds three points: the **actuation point** a finger presses (trigger, button) and the
**functional point** that acts (nozzle, lens), with its direction, both on the functional object; and the
**effect point** on the environment where the functional object should act (`target_pts`, in the scene image;
the anchor of the goal search). The points on the functional object are found on 30 rendered
views of the mesh and triangulated onto it. GPT (`OPENAI_API_KEY`) is the default;
`--point-query-model molmopoint` uses the local
[MolmoPoint](https://huggingface.co/allenai/MolmoPoint-8B) model instead, which needs its own
environment (Transformers 4.57):

```bash
python -m venv --system-site-packages out/molmopoint-env
out/molmopoint-env/bin/pip install --ignore-installed "numpy<2" torch==2.6.0+cu118 torchvision==0.21.0+cu118 \
    --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118
out/molmopoint-env/bin/pip install transformers==4.57.1 decord2
PYTHONPATH="$PWD" out/molmopoint-env/bin/python scripts/1_estimate_keypoints.py \
    --scene-dir out/scenes/demo --point-query-model molmopoint
```

To supply keypoints yourself, write `keypoints/fn_key_pts.json`. Grasp generation and training only
read `act_contact`, in the mesh frame (metres); `dir` is the direction the finger presses, into the
surface:

```json
{"act_contact": {"pos": [x, y, z], "dir": [dx, dy, dz]},
 "fn_contact": {"pos": [x, y, z], "dir": [dx, dy, dz]},
 "target_pts": [[u, v]]}
```

## Stage 2: goal pose

```bash
python scripts/2_generate_goal.py --scene-dir out/scenes/demo --iters 5 --cands 8
```

Searches the pose of the functional object relative to the effect point. Each round, GPT scores
rendered candidates and proposes new ones from its critique. Each candidate is shown in three
views: the capture camera, the same camera turned 90° around the vertical through the effect point,
and from straight above it. Every view marks the effect point, the functional direction and the
object's local axes. The effect point is placed in 3D on the target's segmentation mask. Two
methods:

- `--method vlm_cem_dir` (default) keeps the functional point on the line through the effect
  point along the functional direction, and searches the distance from the effect point and the
  rotation. It gives more sensible goals when the functional object has a clear effect direction
  (spray, glue, air).
- `--method vlm_cem` searches the full 6-DoF pose (rotation and a free offset). Its critic may
  accept a pose and stop before the last round; `stop_reason` and `rounds` in
  `best_candidate.json` record it.

The goal is saved in `goal_gen/<method>/best_candidate.json`, in world coordinates and relative to the captured functional object. Training reads the goal and saves it in the run configuration; evaluation keeps that saved task unless explicitly retargeted.

## Stage 3: grasps

```bash
python scripts/3_generate_grasps.py --scene-dir out/scenes/demo
```

A grasp is useful only if the fixed motion the policy executes can carry it out: the hand slides in
along its palm normal with the fingers half closed, closes the supporting fingers slightly past the
grasp, lifts the functional object 30 cm, then closes the pressing finger onto the actuation
point. The hand's position has about 2 cm of error the policy cannot observe.

Grasp generation therefore works in two steps:

1. **Optimise.** With the functional object standing on the table, 2,048 candidates start with a
   fingertip on the actuation point and are optimised jointly over hand pose and 16 joints. The
   objective checks every state of the motion: the pressing fingertip sits a few millimetres past
   the actuation point; the supporting fingers are commanded slightly into the object, so closing
   them grips it despite the position error; the half-closed hand clears the object and the table
   on the way in; the pressing finger has room to press; the fingers neither collide nor cross over
   one another.
2. **Select by simulation.** Every candidate that passes the geometric checks runs the plain lift
   and press in ManiSkill3 with no policy, under the training physics. This does not screen the
   goal carry. Promising candidates get more episodes
   (2, then 8, then 32 per candidate), and the best are written in order of task success, then lift.

Output: `grasps/grasp_N_optimized.txt` (23 numbers: hand orientation as a wxyz quaternion, position,
16 LEAP joint angles; the frame is the mesh frame shifted so the object's bottom is at z = 0),
`grasp_N_metadata.json` (pressing finger, simulated success and lift rates). About ten minutes on a 24 GB GPU. Options: `--num-grasps`, `--num-candidates`,
`--screen-envs` (parallel simulations), `--seed`, and `--no-sim` to skip selection.

`--method fc` runs a force-closure solver instead, which optimises a static force-closure
certificate for the supporting fingers; its grasps rarely survive the motion.

Render the grasps:

```bash
python codex/grasp_gen/render_grasp.py --scene-dir out/scenes/demo
```

## Stage 4: training

```bash
WANDB_MODE=offline python scripts/4_train_policy.py --scene-dir out/scenes/demo --total-timesteps 400000
```

PPO learns one correction per episode (hand pose, pre-grasp and grasp finger joints) on top of a
grasp. Each episode is a single step that runs the whole task: grasp, lift clear, carry the
functional object to the scene's stage-2 goal pose, and press the actuation point for the final
second (`--goal lift` trains on a plain 30 cm lift instead; scenes without a goal use the lift).
Success requires the object at the goal (within 5 cm and 30°), held steadily, with the press held.

Training runs 2,048 parallel simulations by default (`--num-envs`). The policy does not observe
what varies between them: the hand pose error (1 cm and 3° per axis), the hand/object friction
U(0.4, 1.0), the table/object friction U(0.2, 0.8), and the object's mass (×U(0.9, 1.1)) and size
(×U(0.98, 1.02)); the ranges are `DOMAIN_RANDOMIZATION` in
`codex/codex_train/configs/primitive.py`. Exploration is kept small (`--log-std-init -3`, no
entropy bonus, updates capped by `--target-kl 0.02`) so that sampled episodes still succeed often
enough to learn from. The first grasp file is used unless `--grasp-path` is given.

A run is written to `rl_runs/<time>-<name>/` with `config.yaml`, checkpoints every
`--checkpoint-freq` rollouts and `best_model/`, the checkpoint with the best evaluation so far.
Allow about two to six hours for 400k steps on a 48 GB L40S GPU at 4,096 environments.
Logging uses Weights & Biases; `WANDB_MODE=offline` needs no account (the
app trains offline unless `WANDB_MODE` is set). The app plots the success curves live from the
run's tensorboard log.

Resume requires matching saved PPO settings, including `target_kl`. With individual input
paths instead of `--scene-dir`, provide `--save-dir`.

## Stage 5: evaluation

```bash
python scripts/5_evaluate_policy.py --scene-dir out/scenes/demo
```

Runs the latest run's best model, with the grasp it was trained on, on 32 parallel episodes, prints
the success rates and writes a uniquely named directory under `evals/` with `metrics.json` (summary and one row per
episode) and `rollout.mp4` (the first four episodes, labelled with their outcome).

Evaluation preserves the saved training goal and stability frame. After the grasp the functional object is lifted clear,
carried to the stage-2 goal (height and orientation as searched, the position on the table relative
to where the object was captured) and held there while the finger presses. While the hand holds
it, the hand is corrected by the object's measured pose error at the goal, as a system tracking
the object would. Success means the object is within 5 cm and 30° of the goal, held steadily, with
the press held. A goal that would put the object into the table is an error (search the goal
again). `--goal lift` selects a plain 30 cm lift; `--goal scene` explicitly retargets to the current
stage-2 result, using hand-frame stability. `--goal-method` also requests the current scene goal
(default `vlm_cem_dir`, else `vlm_cem`). A grasp alone uses the current scene goal by default.
`--output-dir` must be new to preserve earlier metrics and videos.

Other options: `--run-dir` or `--model-path` for a specific policy, `--episodes`, `--seed`,
`--video-episodes`, and `--zero-action --grasp-path <grasp>` to evaluate a grasp without a policy.
