"""GPU regression: every z-separated sub-scene has its own physical floor."""

import os
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien
import torch

from codex.codex_train import eval as ev


@unittest.skipUnless(os.environ.get("CODEX_RUN_GPU_FLOOR_TEST") == "1", "GPU floor test is opt-in")
class LayeredFloorTest(unittest.TestCase):
    def test_off_table_objects_land_on_local_floor_in_every_layer(self):
        scene = Path("assets/scenes/flash_light_sloth")
        config = ev.build_env_config(None, scene, None)
        env = gym.make(ev.ENV_ID, **dict(config, num_envs=27, sim_backend="physx_cuda"))
        try:
            base = env.unwrapped
            env.reset(seed=22001)
            offsets = base.scene.scene_offsets[:, 2].cpu().numpy()
            self.assertEqual(len(np.unique(offsets)), 3)
            base.obj.set_pose(sapien.Pose(p=[1.6, 0, 0.2]))
            base.obj.set_linear_velocity(torch.zeros((27, 3), device=base.device))
            base.obj.set_angular_velocity(torch.zeros((27, 3), device=base.device))
            base.scene._gpu_apply_all()
            for _ in range(300):
                base.scene.step()
                base.scene._gpu_fetch_all()
            heights = base.obj.pose.p[:, 2].cpu().numpy()
            for layer in np.unique(offsets):
                local = heights[offsets == layer]
                self.assertTrue(np.all(-1.0 < local) and -1.0 < np.median(local) < -0.5,
                                f"layer {layer}: local object heights {local}")
        finally:
            env.close()

    def test_far_ejected_objects_still_land_on_their_layer_floor(self):
        scene = Path("assets/scenes/flash_light_sloth")
        config = ev.build_env_config(None, scene, None)
        env = gym.make(ev.ENV_ID, **dict(config, num_envs=27, sim_backend="physx_cuda"))
        try:
            base = env.unwrapped
            env.reset(seed=22001)
            offsets = base.scene.scene_offsets[:, 2].cpu().numpy()
            base.obj.set_pose(sapien.Pose(p=[20.0, 5.0, 0.2]))
            base.obj.set_linear_velocity(torch.zeros((27, 3), device=base.device))
            base.obj.set_angular_velocity(torch.zeros((27, 3), device=base.device))
            base.scene._gpu_apply_all()
            for _ in range(300):
                base.scene.step()
                base.scene._gpu_fetch_all()
            heights = base.obj.pose.p[:, 2].cpu().numpy()
            for layer in np.unique(offsets):
                local = heights[offsets == layer]
                self.assertTrue(np.all(-1.0 < local) and -1.0 < np.median(local) < -0.5,
                                f"layer {layer}: local object heights {local}")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
