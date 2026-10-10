"""Stage 1: Estimate functional/target/actuation keypoints with VLMs.

Thin wrapper around codex.keypoint_estimation.estimate_keypoint — run with --help for all options.
"""
from codex.keypoint_estimation.estimate_keypoint import main

if __name__ == "__main__":
    main()
