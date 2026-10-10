"""Teleport must update physical joints before the next pose read/control."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from codex.codex_train.envs.primitive_env import PrimitiveGraspEnv


class TeleportTests(unittest.TestCase):
    def fixture(self, gpu):
        staged = {'pose': 'old', 'fingers': 'old'}
        physical = staged.copy()

        def set_hand_qpos(**values):
            for key in ('pose', 'finger_joints'):
                if key in values:
                    staged['fingers' if key == 'finger_joints' else key] = values[key]
            if not gpu:
                physical.update(staged)

        def apply():
            physical.update(staged)

        def render():
            self.assertEqual(physical, {'pose': 'requested', 'fingers': 'requested'})

        px = SimpleNamespace(gpu_apply_articulation_qpos=Mock(side_effect=apply),
                             gpu_update_articulation_kinematics=Mock())
        scene = SimpleNamespace(px=px, _gpu_fetch_all=Mock(),
                                step=Mock(side_effect=AssertionError('Teleport must not advance physics')))
        env = SimpleNamespace(gpu_sim_enabled=gpu, set_hand_qpos=set_hand_qpos,
                              scene=scene, _maybe_render=render)
        return env

    def test_gpu_requested_pose_and_fingers_are_physical_before_render(self):
        env = self.fixture(True)
        PrimitiveGraspEnv.move_hand(env, target_pose_mat='requested',
                                   target_finger_joints='requested', teleport=True)
        env.scene.px.gpu_apply_articulation_qpos.assert_called_once()
        env.scene.px.gpu_update_articulation_kinematics.assert_called_once()
        env.scene._gpu_fetch_all.assert_called_once()
        env.scene.step.assert_not_called()

    def test_cpu_teleport_keeps_immediate_setter_semantics(self):
        env = self.fixture(False)
        PrimitiveGraspEnv.move_hand(env, target_pose_mat='requested',
                                   target_finger_joints='requested', teleport=True)
        env.scene.px.gpu_apply_articulation_qpos.assert_not_called()
        env.scene.px.gpu_update_articulation_kinematics.assert_not_called()
        env.scene._gpu_fetch_all.assert_not_called()
        env.scene.step.assert_not_called()


if __name__ == '__main__':
    unittest.main()
