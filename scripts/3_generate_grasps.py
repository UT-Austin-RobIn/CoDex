"""Stage 3: Synthesize functional grasps with the batched GPU solver.

Thin wrapper around codex.grasp_gen.generate_grasp — run with --help for all options.
"""
from codex.grasp_gen.generate_grasp import main

if __name__ == "__main__":
    main()
