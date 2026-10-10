"""Stage 4: Train the grasp-actuate RL policy (ManiSkill3 + PPO).

Thin wrapper around codex.codex_train.train — run with --help for all options.
"""
from codex.codex_train.train import main

if __name__ == "__main__":
    main()
