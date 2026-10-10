# LEAP right-hand models

All LEAP hand assets live here; the two URDF variants share the same
kinematics (identical joints, origins, axes, limits) and the same visual
meshes (`meshes_obj/`), but carry different collision geometry because they
serve different consumers:

- `leap_solver.urdf` — used by the grasp solver
  (`codex/grasp_gen/batched_grasp_solver.py`) and grasp rendering.
  Collision geometry is 69 calibrated boxes (ported from the MuJoCo
  Menagerie model) that the solver's collision and self-collision checks
  were validated against. Do not swap these for mesh collisions.
- `leap_sim.urdf` — the hand with per-link convex mesh collisions used for
  contact dynamics in ManiSkill.
- `floating_leap.urdf` — `leap_sim` plus a 6-DOF floating base; this is the
  robot the RL agent (`codex/codex_train/agents/leap_rh/`) loads.

Grasp files produced by the solver transfer to the simulation directly
because the joint frames of both variants are identical.
