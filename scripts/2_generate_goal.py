"""Stage 2: generate the goal pose of the functional object with a VLM-guided search.

    --method vlm_cem_dir   (default) search along the functional object's effect direction:
                           distance plus rotation; more sensible goals when the
                           functional object has a clear effect direction (spray, glue, air)
    --method vlm_cem       full 6-DoF search (rotation plus a free offset)

Other options are those of codex.goal_gen.vlm_cem_dir / vlm_cem (run with
--method <m> --help). Results: <scene>/goal_gen/<method>/best_candidate.json.
"""
import argparse
import sys

METHODS = ("vlm_cem_dir", "vlm_cem")

if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--method", choices=METHODS, default="vlm_cem_dir")
    args, rest = pre.parse_known_args()
    if args.method == "vlm_cem_dir":
        from codex.goal_gen.vlm_cem_dir import main
    else:
        from codex.goal_gen.vlm_cem import main
    if {"-h", "--help"} & set(rest):
        print(f"--method {{{','.join(METHODS)}}} (default vlm_cem_dir); options of {args.method}:\n")
    sys.argv = [sys.argv[0], *rest]
    main()
