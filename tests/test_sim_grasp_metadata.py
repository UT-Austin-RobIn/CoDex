"""Simulation grasps require the metadata that generic grasp rendering may omit."""

import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from codex.codex_train.envs.pick_mesh import load_sim_grasp_config
from codex.grasp_gen.load_grasp import load_grasp_config


SCRATCH = Path(tempfile.gettempdir())


class SimGraspMetadataTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        self.grasp = directory / "grasp_1_optimized.txt"
        self.metadata = directory / "grasp_1_metadata.json"
        np.savetxt(self.grasp, np.array([1, 0, 0, 0, 0, 0, .1] + [0] * 16))

    def test_missing_companion_has_early_named_error_but_generic_loader_stays_optional(self):
        with self.assertRaisesRegex(FileNotFoundError, "grasp_1_metadata.json.*actuation_finger_indices"):
            load_sim_grasp_config(str(self.grasp), np.zeros(3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, act_idx = load_grasp_config(str(self.grasp), offset=np.zeros(3))
        self.assertIsNone(act_idx)

    def test_valid_single_finger_metadata_loads(self):
        self.metadata.write_text(json.dumps({"actuation_finger_indices": [3]}))
        joints, pose, act_idx = load_sim_grasp_config(str(self.grasp), np.zeros(3))
        self.assertEqual(joints.shape, (16,))
        self.assertEqual(pose.shape, (4, 4))
        self.assertEqual(act_idx, [3])

    def test_invalid_companion_is_named(self):
        for payload in ({}, {"actuation_finger_indices": []},
                        {"actuation_finger_indices": [0, 1]},
                        {"actuation_finger_indices": [True]},
                        {"actuation_finger_indices": [4]},
                        {"actuation_finger_indices": ["0"]}, []):
            with self.subTest(payload=payload):
                self.metadata.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "grasp_1_metadata.json.*actuation_finger_indices"):
                    load_sim_grasp_config(str(self.grasp), np.zeros(3))

    def test_malformed_companion_is_named(self):
        self.metadata.write_text("{")
        with self.assertRaisesRegex(ValueError, "Invalid actuation metadata companion.*grasp_1_metadata.json"):
            load_sim_grasp_config(str(self.grasp), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
