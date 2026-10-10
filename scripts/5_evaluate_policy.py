"""Stage 5: Evaluate a trained policy (or a grasp alone) and record a rollout video.

Thin wrapper around codex.codex_train.eval — run with --help for all options.
"""
from codex.codex_train.eval import main

if __name__ == "__main__":
    main()
