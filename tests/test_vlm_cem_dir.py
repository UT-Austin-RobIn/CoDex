"""Distance-axis CEM contracts, with deterministic VLM and render doubles."""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from codex.goal_gen import vlm_cem_dir as cem

SCRATCH = Path(tempfile.gettempdir())


class GoalFixture:
    def __init__(self, root, module=cem):
        self.module = module
        self.root = Path(root)
        self.output = self.root / 'goal'
        self.keypoints = self.root / 'keypoints'
        self.keypoints.mkdir()
        (self.keypoints / 'fn_pt_mesh.json').write_text('[0.02, 0.03, 0.04]')
        (self.keypoints / 'fn_key_pts.json').write_text(json.dumps({
            'fn_pts': [[25, 50]], 'target_pts': [[60, 40]],
            'act_contact': {'dir': [0.1, 0.2, -0.9]}, 'fn_direction': [0, 0, -1],
        }))
        self.base = np.eye(4)
        self.base[:3, 3] = [0.3, 0.4, 0.5]
        np.savetxt(self.root / 'pose.txt', self.base)
        self.mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
        target = SimpleNamespace(name='target', mask=np.ones((80, 100), bool),
                                 get_pcd=lambda: SimpleNamespace(points=np.array([[1, 2, 3], [3, 4, 5]])))
        self.scene = SimpleNamespace(
            image=Image.new('RGB', (100, 80)), cam_intrinsics=np.eye(3),
            cam_extrinsics=np.diag([1., -1., -1., 1.]),
            points=np.zeros((80, 100, 3)), objects=[SimpleNamespace(name='bottle'), target],
            get_pcd=lambda **kwargs: SimpleNamespace(points=np.zeros((1, 3))),
        )
        self.calls = []
        self.poses = []

    def query(self, images, prompt, **kwargs):
        self.calls.append((len(images), prompt, kwargs))
        if 'Please propose' in prompt:
            return json.dumps({str(i): dict(dist=0.1 * i, yaw=90, pitch=0, roll=0) for i in (1, 2)})
        return json.dumps({str(img.info['id']): {
            'score': img.info['id'] / 10, 'description': f"candidate {img.info['id']}",
        } for img in images})

    def render(self, mesh, pose, scene, candidate_id, **kwargs):
        self.poses.append((candidate_id, pose.tolist(), kwargs))
        image = Image.new('RGB', (8, 8)); image.info['id'] = candidate_id
        return image

    def run(self, cached=False, views=3):
        self.module._candidate_id = 0
        random.seed(37)
        with contextlib.ExitStack() as stack:
            for name, value in {
                'load_scene_meta': lambda path: {'task': 'use bottle', 'functional_object': 'bottle', 'target_object': 'target'},
                'load_mesh': lambda path: (self.mesh, {}),
                'lift_effect_point': lambda *args, **kwargs: np.array([0.4, 0.2, 0.6]),
                'estimate_fn_point_3d_from_pixel': lambda *args, **kwargs: np.array([0.32, 0.43, 0.54]),
                'render_candidate': self.render,
                'VLMInterface': lambda: None,
                'VLMSession': lambda *args, **kwargs: SimpleNamespace(query=self.query),
            }.items():
                stack.enter_context(patch.object(self.module, name, value))
            stack.enter_context(patch.object(self.module.ImageScene, 'load', return_value=self.scene))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.module.generate_3d_goal_dist(scene_dir=str(self.root), output_dir=str(self.output),
                                              iters=2, cands=4, load_cached=cached, views=views)
        return json.loads((self.output / 'best_candidate.json').read_text())


class DistanceGoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def test_search_preserves_pose_and_output_contract(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            fixture = GoalFixture(root)
            result = fixture.run()
            self.assertEqual(result['id'], 8)
            self.assertEqual(result['score'], 0.8)
            self.assertEqual([n for n, _, _ in fixture.calls], [4, 0, 5])
            self.assertIn('description=candidate 4', fixture.calls[1][1])
            self.assertIn('turned 90 degrees around the vertical line through the effect', fixture.calls[0][1])
            self.assertIn('target_forward_distance=-0.2000', fixture.calls[2][1])
            self.assertIn('held by a robot gripper', fixture.calls[2][1])
            expected = np.array([[0., -1., 0., 0.43], [1., 0., 0., 0.18],
                                 [0., 0., 1., 0.36], [0., 0., 0., 1.]])
            np.testing.assert_allclose(result['transform'], expected, atol=1e-12)
            np.testing.assert_allclose(result['transform_world'], fixture.scene.cam_extrinsics @ expected, atol=1e-12)
            np.testing.assert_allclose(result['relative_transform'], np.linalg.inv(fixture.base) @ expected, atol=1e-12)
            self.assertEqual(result['fn_pt_mesh'], [0.02, 0.03, 0.04])
            self.assertEqual(result['target_obj_center'], [2, 3, 4])
            self.assertEqual(len(list((fixture.output / 'cache').glob('*.jpg'))), 9)
            self.assertTrue((fixture.output / 'best_candidate.jpg').is_file())
            self.assertEqual(len(result['search_trace']), 2)
            self.assertEqual(len(result['search_trace'][1]['candidates']), 5)
            self.assertEqual(result['search_trace'][1]['candidates'][-1]['id'], 4)
            self.assertTrue((fixture.output / 'search_trace.json').is_file())
            self.assertTrue(all(kwargs['multi_view'] for _, _, kwargs in fixture.poses))

    def test_cached_result_makes_no_vlm_requests(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            fixture = GoalFixture(root)
            initial = fixture.run()
            fixture.calls.clear()
            cached = fixture.run(cached=True)
            self.assertEqual(fixture.calls, [])
            self.assertEqual(cached['transform'], initial['transform'])
            self.assertEqual(cached['id'], initial['id'])
            self.assertEqual(cached['table_penetration'], initial['table_penetration'])

    def test_legacy_fn_pixel_is_lifted_and_cached(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            fixture = GoalFixture(root)
            (fixture.keypoints / 'fn_pt_mesh.json').unlink()
            result = fixture.run()
            np.testing.assert_allclose(result['fn_pt_mesh'], [0.02, 0.03, 0.04])
            self.assertTrue((fixture.keypoints / 'fn_pt_mesh.json').is_file())

    def test_direction_ablation_rejects_missing_functional_direction(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            fixture = GoalFixture(root)
            path = fixture.keypoints / 'fn_key_pts.json'
            data = json.loads(path.read_text())
            del data['fn_direction']
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'requires fn_direction'):
                fixture.run()
            self.assertEqual(fixture.calls, [])

    def test_functional_direction_is_not_snapped_to_actuation_axis(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            fixture = GoalFixture(root)
            path = fixture.keypoints / 'fn_key_pts.json'
            data = json.loads(path.read_text())
            data['fn_direction'] = [1, 2, 3]
            path.write_text(json.dumps(data))
            with patch.object(cem, 'lift_effect_point', return_value=np.ones(3)):
                _, _, axis = cem._load_keypoints(fixture.scene, fixture.mesh, fixture.base,
                                                str(fixture.keypoints), 'target')
            np.testing.assert_allclose(axis, np.array([1, 2, 3]) / np.sqrt(14))

    def test_invalid_proposal_fills_with_finite_random_candidate(self):
        history = [cem.Candidate3D(0, 0, 0, 0, score=0.5)]
        vlm = SimpleNamespace(query=lambda *args, **kwargs: 'not JSON')
        with self.assertWarnsRegex(UserWarning, 'filling with random candidates'):
            result = cem._propose_candidates(vlm, history, 'use bottle', 'bottle', 1, 2, 2)
        self.assertEqual(len(result), 1)
        self.assertNotEqual(result[0].id, history[0].id)
        self.assertTrue(np.isfinite(result[0].params).all())

    def test_single_view_and_same_round_incumbent_rescoring(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            fixture = GoalFixture(root)
            def query(images, prompt, **kwargs):
                fixture.calls.append((len(images), prompt, kwargs))
                if 'Please propose' in prompt:
                    return json.dumps({'1': dict(dist=0.1, yaw=90, pitch=0, roll=0),
                                       '2': dict(dist=0.2, yaw=90, pitch=0, roll=0)})
                round2 = len(images) == 5
                return json.dumps({str(im.info['id']): {
                    'score': (0.95 if im.info['id'] == 4 else 0.2) if round2
                             else (0.8 if im.info['id'] == 4 else 0.1),
                    'description': 'same-round review'} for im in images})
            fixture.query = query
            result = fixture.run(views=1)
            self.assertEqual(result['id'], 4)
            self.assertEqual(result['score'], 0.95)
            self.assertEqual([n for n, _, _ in fixture.calls], [4, 0, 5])
            self.assertIn('original camera view', fixture.calls[0][1])
            self.assertTrue(all(not kwargs['multi_view'] for _, _, kwargs in fixture.poses))

    def test_zero_score_tie_preserves_incumbent(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            fixture = GoalFixture(root)
            def query(images, prompt, **kwargs):
                fixture.calls.append((len(images), prompt, kwargs))
                if 'Please propose' in prompt:
                    return json.dumps({'1': dict(dist=0.1, yaw=0, pitch=0, roll=0),
                                       '2': dict(dist=0.2, yaw=0, pitch=0, roll=0)})
                return json.dumps({str(im.info['id']): {
                    'score': 0 if len(images) == 5 else (0.9 if im.info['id'] == 4 else 0.1),
                    'description': 'reviewed'} for im in images})
            fixture.query = query
            result = fixture.run()
            self.assertEqual(result['id'], 4)
            self.assertEqual(result['score'], 0)
            self.assertEqual(result['search_trace'][1]['incumbent_id'], 4)

    def test_forward_distance_matches_rotated_outward_axis(self):
        base = cem.make_T(cem.euler_to_matrix(31, -12, 20), np.array([.2, .3, .5]))
        point = np.array([.02, -.01, .03])
        axis = np.array([0., -1., 0.])
        target = np.array([.1, -.4, .6])
        for distance in (-.12, 0., .12):
            pose = cem.place_pose(base, point, distance, 40, -25, 11, axis, target)
            ray = pose[:3, :3] @ axis
            offset = target - (pose[:3, :3] @ point + pose[:3, 3])
            self.assertAlmostEqual(float(offset @ ray), -distance)
            np.testing.assert_allclose(offset - (offset @ ray) * ray, 0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
