"""Functional grasp generation (pipeline stage 3).

Two methods write the same grasp files:

  primitive (default)  Optimise grasps against the states of the RL
                       grasp-move-actuate primitive (`primitive_grasp.py`),
                       then select them by simulating that primitive with zero
                       policy action (`sim_screen.py`). Grasps are ranked by
                       simulated task success.
  fc                   The batched force-closure solver
                       (`batched_grasp_solver.py`), kept for comparison with
                       the static formulation.

Inputs (from --scene-dir unless overridden)
    mesh_coacd/          collision mesh used by simulation (primitive)
    mesh/mesh.glb        visual mesh (fc)
    keypoints/fn_key_pts.json   "act_contact": {"pos": [x,y,z], "dir": [dx,dy,dz]}
                                in the mesh frame; dir is the press direction

Outputs (in --output-dir, default <scene_dir>/grasps)
    grasp_<i>_optimized.txt   23 numbers: base quaternion (wxyz), base position
                              (xyz), 16 LEAP joints in grasp-file order. Frame:
                              object mesh frame translated so its bottom sits at z=0.
    grasp_<i>_metadata.json   actuation finger index + method-specific scores
    selection.json (primitive) / raw_results.pt (fc)

Usage
    python scripts/3_generate_grasps.py --scene-dir out/scenes/<scene>
    python scripts/3_generate_grasps.py --scene-dir out/scenes/<scene> --method fc
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from codex.mesh.load_mesh import load_mesh

# Joint order of the grasp .txt files (the order codex_train loads):
# [axl, mcp, pip, dip] per finger for index/middle/ring, then thumb
# [cmc, axl, mcp, ipl]. pytorch_kinematics traverses the LEAP URDF in exactly
# this order, so solver joint vectors are written unchanged.
_TXT_JOINT_ORDER = [
    "leap_rh_if_axl", "leap_rh_if_mcp", "leap_rh_if_pip", "leap_rh_if_dip",
    "leap_rh_mf_axl", "leap_rh_mf_mcp", "leap_rh_mf_pip", "leap_rh_mf_dip",
    "leap_rh_rf_axl", "leap_rh_rf_mcp", "leap_rh_rf_pip", "leap_rh_rf_dip",
    "leap_rh_th_cmc", "leap_rh_th_axl", "leap_rh_th_mcp", "leap_rh_th_ipl",
]
SIM_KEYS = ("success", "lift_success", "raw_lift", "stable_success", "upright_success",
            "act_success", "act_dist", "n")


def _load_actuation(keypoints_path, offset):
    """Actuation contact (position + press direction) from fn_key_pts.json."""
    with open(keypoints_path) as f:
        c = json.load(f).get("act_contact")
    if c is None:
        raise KeyError(f"{keypoints_path} has no 'act_contact' entry; run stage 1 for this scene.")
    return [(np.asarray(c["pos"], dtype=float) + offset, np.asarray(c["dir"], dtype=float))]


def _clear_previous(output_dir):
    """Remove grasp files from an earlier run so indices never mix runs."""
    stale = [p for pattern in ("grasp_*_optimized.txt", "grasp_*_optimized.png", "grasp_*_metadata.json")
             for p in glob.glob(os.path.join(output_dir, pattern))]
    for p in stale:
        os.remove(p)
    if stale:
        print(f"Removed {len(stale)} grasp files from a previous run in {output_dir}")


def _write_grasp(output_dir, i, vec, metadata):
    np.savetxt(os.path.join(output_dir, f"grasp_{i}_optimized.txt"), vec)
    with open(os.path.join(output_dir, f"grasp_{i}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def _require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("Grasp generation requires a CUDA GPU.")


# ---------------------------------------------------------------------------
# Primitive-aware generation + simulated selection (default)
# ---------------------------------------------------------------------------

def generate_primitive(coacd_dir, keypoints_path, output_dir, num_grasps=5, num_candidates=2048,
                       steps=300, seed=0, screen_envs=1024, max_screen=1536, simulate=True):
    """Returns the number of grasp files written."""
    from codex.grasp_gen.primitive_grasp import (PrimitiveGraspSynthesizer, load_scene,
                                                 to_grasp_vectors, JOINT_ORDER)
    _require_cuda()
    assert JOINT_ORDER == _TXT_JOINT_ORDER
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    parts, act_pos, act_dir = load_scene(coacd_dir, keypoints_path)
    syn = PrimitiveGraspSynthesizer(parts, act_pos, act_dir, seed=seed)
    print(f"Optimising {num_candidates} candidates ({steps} steps) ...")
    res = syn.optimize(B=num_candidates, steps=steps)
    ok, gates = syn.gates(res)
    print(f"Proposals passing geometric gates: {int(ok.sum())}/{len(ok)} "
          f"({time.time() - t0:.0f}s; " + ", ".join(f"{k} {int(v.sum())}" for k, v in gates.items()) + ")")
    if not ok.any():
        print("ERROR: no proposal passed the geometric gates. Check that the actuation point lies on "
              "the object surface and its direction points into the object; try another --seed.")
        return 0
    order = torch.where(ok, res["energy"], torch.full_like(res["energy"], float("inf"))).argsort()
    sel = order[:min(max_screen, int(ok.sum()))].tolist()
    vecs = to_grasp_vectors(res, sel)
    fingers = [int(res["fa"][i]) for i in sel]
    energies = [float(res["energy"][i]) for i in sel]
    del syn, res
    torch.cuda.empty_cache()

    if simulate:
        from codex.grasp_gen.sim_screen import SimScreener, screen_score, successive_halving
        t0 = time.time()
        print(f"Simulating the grasp primitive for {len(sel)} proposals ({screen_envs} parallel envs) ...")
        screener = SimScreener(coacd_dir, os.path.dirname(keypoints_path), num_envs=screen_envs,
                               seed=1000 + seed)
        ranked, stats = successive_halving(screener, vecs, fingers, seed=5000 + seed)
        screener.close()
        print(f"Simulation selection took {time.time() - t0:.0f}s")
        ranked = [i for i in ranked if stats[i]["raw_lift"] > 0] or ranked
    else:
        print("WARNING: --no-sim: writing unverified proposals ranked by energy.")
        ranked, stats = list(range(len(sel))), [None] * len(sel)

    _clear_previous(output_dir)
    written = ranked[:num_grasps]
    for k, i in enumerate(written):
        meta = {"method": "primitive", "actuation_finger_indices": [fingers[i]], "energy": energies[i]}
        if stats[i] is not None:
            meta["sim"] = {key: stats[i][key] for key in SIM_KEYS}
        _write_grasp(output_dir, k + 1, vecs[i], meta)
    with open(os.path.join(output_dir, "selection.json"), "w") as f:
        json.dump([{"grasp": k + 1 if k < len(written) else None, "actuation_finger": fingers[i],
                    "energy": energies[i], "score": float(screen_score(stats[i])) if stats[i] else None,
                    "sim": {key: stats[i][key] for key in SIM_KEYS} if stats[i] else None}
                   for k, i in enumerate(ranked[:32])], f, indent=1)

    print(f"\nWrote {len(written)} grasps to {output_dir}")
    for k, i in enumerate(written):
        s = stats[i]
        extra = (f"simulated success {s['success']:.0%}, lift {s['raw_lift']:.0%} over {s['n']} rollouts"
                 if s else "not simulated")
        print(f"  grasp_{k + 1}: actuation finger {fingers[i]}, {extra}")
    return len(written)


# ---------------------------------------------------------------------------
# Force-closure solver (--method fc)
# ---------------------------------------------------------------------------

def generate_fc(mesh_path, keypoints_path, output_dir, num_grasps=5, num_envs=4000,
                max_batches=3, steps=300, seed_offset=0, variant="P"):
    """Returns the number of grasp files written."""
    from codex.grasp_gen.batched_grasp_solver import (BatchedSDF, BatchedGraspOptimizer,
                                                      FC_LSTAR_MIN_MARGIN)
    _require_cuda()
    os.makedirs(output_dir, exist_ok=True)
    mesh, _ = load_mesh(mesh_path)
    offset = np.array([0.0, 0.0, -mesh.bounds[0, 2]])
    X_WO = np.eye(4)
    X_WO[:3, 3] = offset
    obj_center = mesh.centroid + offset
    targets = _load_actuation(keypoints_path, offset)

    pooled = []
    for batch_i in range(max_batches):
        n_feas = sum(1 for g in pooled if g.get("feasible", False))
        if n_feas >= num_grasps:
            break
        print(f"\n=== batch {batch_i + 1}/{max_batches} ({n_feas}/{num_grasps} feasible so far) ===")
        sdf = BatchedSDF(mesh, X_WO, bounds_padding=0.15, resolution=128, device="cuda")
        sdf.add_clearance_volume(targets[0][0], targets[0][1], radius=0.020, height=0.03)
        sdf.add_floor(0.0)
        seed = 1000 * batch_i + 7 + seed_offset
        torch.manual_seed(seed)
        np.random.seed(seed)
        opt = BatchedGraspOptimizer(sdf, num_envs=num_envs, device="cuda", hand="rh",
                                    hand_type="leap", palm_contact=True)
        if list(opt.chain.get_joint_parameter_names()) != _TXT_JOINT_ORDER:
            raise RuntimeError("Solver joint order differs from the grasp-file convention")
        results = opt.optimize(actuation_targets=targets, object_center=obj_center,
                               steps=steps, lr=0.005, opt_variant=variant)
        for g in results:
            g["batch_idx"] = batch_i
        pooled.extend(results)

    torch.save(pooled, os.path.join(output_dir, "raw_results.pt"))
    feasible = sorted((g for g in pooled if g.get("feasible", False)), key=lambda g: -g["score"])
    if not feasible:
        print(f"ERROR: no feasible grasp after {max_batches} batches x {num_envs} candidates. "
              f"Try more --max-batches, another --seed-offset, or the default primitive method.")
        return 0
    _clear_previous(output_dir)
    for i, g in enumerate(feasible[:num_grasps]):
        quat_wxyz = Rotation.from_matrix(g["base_rot"]).as_quat()[[3, 0, 1, 2]]
        vec = np.concatenate([quat_wxyz, g["base_pos"], np.asarray(g["q_joints"])])
        _write_grasp(output_dir, i + 1, vec, {
            "method": "fc",
            "actuation_finger_indices": [int(g["act_finger"])],
            "l_star": g["l_star"],
            "l_star_acceptance_margin": g.get("l_star_acceptance_margin", FC_LSTAR_MIN_MARGIN),
            "palm_used_for_fc": g.get("palm_used_for_fc", False),
            "sigma_min": g["sigma_min"],
            "act_dist": g.get("act_dist", 0.0),
            "solver": {"num_envs": num_envs, "steps": steps, "variant": variant,
                       "seed_offset": seed_offset, "batch_idx": g.get("batch_idx", 0)},
        })
    n = min(len(feasible), num_grasps)
    print(f"Wrote {n} grasps to {output_dir} ({len(feasible)} feasible of {len(pooled)} kept)")
    return n


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate functional grasps (pipeline stage 3).")
    p.add_argument("--scene-dir", help="Scene directory; inputs/outputs are discovered from it.")
    p.add_argument("--method", choices=["primitive", "fc"], default="primitive",
                   help="primitive (default): primitive-aware optimisation + simulated selection; "
                        "fc: force-closure solver.")
    p.add_argument("--keypoints-path", help="fn_key_pts.json (default: <scene>/keypoints/fn_key_pts.json).")
    p.add_argument("--output-dir", help="Output directory (default: <scene>/grasps).")
    p.add_argument("--num-grasps", type=int, default=5, help="Number of grasps to write.")
    g = p.add_argument_group("primitive method")
    g.add_argument("--coacd-dir", help="CoACD mesh directory (default: <scene>/mesh_coacd).")
    g.add_argument("--num-candidates", type=int, default=2048, help="Grasp proposals optimised in parallel.")
    g.add_argument("--screen-envs", type=int, default=1024, help="Parallel simulation environments.")
    g.add_argument("--max-screen", type=int, default=1536, help="Maximum proposals simulated.")
    g.add_argument("--seed", type=int, default=0, help="Random seed.")
    g.add_argument("--no-sim", action="store_true", help="Skip simulated selection (unverified grasps).")
    g = p.add_argument_group("fc method")
    g.add_argument("--mesh-path", help="Visual mesh (default: <scene>/mesh/mesh.glb).")
    g.add_argument("--num-envs", type=int, default=4000, help="Parallel candidates per batch.")
    g.add_argument("--max-batches", type=int, default=3, help="Maximum sampling batches.")
    g.add_argument("--seed-offset", type=int, default=0, help="Offset added to per-batch seeds.")
    p.add_argument("--steps", type=int, default=300, help="Optimisation steps.")
    a = p.parse_args()

    scene = os.path.abspath(a.scene_dir) if a.scene_dir else None

    def resolve(value, *default):
        if value:
            return os.path.abspath(value)
        if scene is None:
            p.error("--scene-dir is required unless every input/output path is given")
        return os.path.join(scene, *default)

    keypoints = resolve(a.keypoints_path, "keypoints", "fn_key_pts.json")
    output_dir = resolve(a.output_dir, "grasps")
    if a.method == "primitive":
        n = generate_primitive(resolve(a.coacd_dir, "mesh_coacd"), keypoints, output_dir,
                               num_grasps=a.num_grasps, num_candidates=a.num_candidates,
                               steps=a.steps, seed=a.seed, screen_envs=a.screen_envs,
                               max_screen=a.max_screen, simulate=not a.no_sim)
    else:
        n = generate_fc(resolve(a.mesh_path, "mesh", "mesh.glb"), keypoints, output_dir,
                        num_grasps=a.num_grasps, num_envs=a.num_envs, max_batches=a.max_batches,
                        steps=a.steps, seed_offset=a.seed_offset)
    sys.exit(0 if n > 0 else 1)


if __name__ == "__main__":
    main()
