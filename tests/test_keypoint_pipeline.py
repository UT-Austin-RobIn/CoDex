"""Multi-view behavior and downstream keypoint-file compatibility."""
import json
from pathlib import Path
import tempfile
import threading
import time
import warnings
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import trimesh

from codex.keypoint_estimation.mesh_keypoints import predict_views, estimate_mesh_keypoints
from codex.keypoint_estimation.estimate_actuation_point import estimate_contact_direction
from codex.keypoint_estimation.estimate_keypoint import run_query, project_fn_to_scene

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(tempfile.gettempdir())


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def test_parallel_views_are_joint_and_hidden_points_stay_empty(self):
        lock = threading.Lock()
        state = dict(active=0, peak=0, calls=0)

        class Backend:
            concurrent_safe = True

            def predict(self, image, **kwargs):
                self_test.assertEqual(set(kwargs['descriptions']), {'fn', 'act'})
                with lock:
                    state['active'] += 1
                    state['peak'] = max(state['peak'], state['active'])
                    state['calls'] += 1
                time.sleep(.01)
                with lock:
                    state['active'] -= 1
                return {'fn': {'visible': False, 'point': None},
                        'act': {'visible': True, 'point': [25, 75]}}

        self_test = self
        views = {i: np.zeros((12, 20, 3), dtype=np.uint8) for i in range(6)}
        results = predict_views(views, Backend(), 'spray', 'bottle', workers=2)
        self.assertEqual(state['calls'], len(views))
        self.assertEqual(state['peak'], 2)
        self.assertTrue(all(r['fn']['point'] is None for r in results.values()))

    def test_local_backend_stays_serial_and_failures_are_not_hidden(self):
        backend = SimpleNamespace(concurrent_safe=False, predict=mock.Mock(return_value={
            'fn': {'visible': False, 'point': None}, 'act': {'visible': False, 'point': None}}))
        views = {i: np.zeros((12, 20, 3), dtype=np.uint8) for i in range(4)}
        with mock.patch('codex.keypoint_estimation.mesh_keypoints.ThreadPoolExecutor') as executor:
            result = predict_views(views, backend, 'spray', 'bottle')
            executor.assert_not_called()
        self.assertEqual(len(result), 4)
        self.assertEqual(backend.predict.call_count, 4)  # No all-hidden fallback.
        backend.predict.side_effect = RuntimeError('API unavailable')
        with self.assertRaisesRegex(RuntimeError, 'API unavailable'):
            predict_views(views, backend, 'spray', 'bottle')

    def test_stage_replaces_stale_fn_cache_and_preserves_contact_contract(self):
        image = Image.new('RGB', (200, 100))
        target_obj = SimpleNamespace(name='plant', get_cropped=lambda: image,
                                     map_coords=lambda pts: [[p[0]/2, p[1]/2] for p in pts])
        scene = SimpleNamespace(image=image, cam_intrinsics=None,
                                objects=[SimpleNamespace(name='bottle'), target_obj])
        backend = SimpleNamespace(predict=mock.Mock(return_value={
            'target': {'visible': True, 'point': [40, 60]}}))
        contact = {'fn': {'pos': [0.1, 0.2, 0.3], 'dir': [0, 0, 1]},
                   'act': {'pos': [0.04, 0.01, 0.03], 'dir': [1, 0, 0]}}
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            path = Path(directory)
            (path/'fn_pt_mesh.json').write_text('[999,999,999]')
            with mock.patch('codex.keypoint_estimation.estimate_keypoint.estimate_mesh_keypoints', return_value=contact) as estimate:
                run_query(None, scene, 'spray plant', scene_dir=directory, save_dir=directory, backend=backend)
            self.assertEqual(estimate.call_args.kwargs['workers'], 16)
            self.assertEqual(estimate.call_args.kwargs['point_query_model'], 'sol')
            self.assertEqual(json.loads((path/'fn_pt_mesh.json').read_text()), contact['fn']['pos'])
            data = json.loads((path/'fn_key_pts.json').read_text())
            self.assertEqual(data['act_contact'], contact['act'])
            self.assertEqual(data['fn_contact'], contact['fn'])
            self.assertEqual(data['fn_direction'], [0, 0, 1])
            self.assertEqual(data['fn_pts_3d'], [contact['fn']['pos']])
            self.assertEqual(data['target_pts'], [[20, 30]])
            self.assertEqual(data['fn_pts'], [])  # No registration: no invented 2D point.

    def test_functional_direction_is_outward_opposite_of_press_at_same_surface(self):
        mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        point = np.array([0.0, 0.0, 0.05])
        inward = estimate_contact_direction(mesh, point)
        outward = estimate_contact_direction(mesh, point, outward=True)
        np.testing.assert_allclose(outward, -inward, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(outward)))
        self.assertAlmostEqual(np.linalg.norm(outward), 1.0)
        np.testing.assert_allclose(outward, [0, 0, 1], atol=1e-12)

    def test_degenerate_nearby_faces_do_not_produce_nonfinite_direction(self):
        box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        vertices = np.vstack([box.vertices, [0, 0, 0.05]])
        center_index = len(vertices) - 1
        faces = np.vstack([box.faces, [center_index] * 3, [center_index] * 3])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        point = np.array([0., 0., 0.05])
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            for radius in (0.01, 0.05):
                inward = estimate_contact_direction(mesh, point, radius=radius)
                outward = estimate_contact_direction(mesh, point, outward=True,
                                                     radius=radius)
                self.assertTrue(np.all(np.isfinite(outward)))
                self.assertAlmostEqual(np.linalg.norm(outward), 1.0)
                np.testing.assert_allclose(outward, -inward)

    def test_shared_mesh_pipeline_requests_outward_fn_and_inward_act(self):
        class Mesh:
            def compute_vertex_normals(self):
                pass

            def compute_triangle_normals(self):
                pass

        class Estimator:
            mesh_name = 'mesh'
            cache_dir = 'cache'
            camera_data = {}

            def render_mesh_views(self):
                return {0: np.zeros((8, 8, 3), dtype=np.uint8)}

        result = {0: {
            'fn': {'visible': True, 'point': [25, 40]},
            'act': {'visible': True, 'point': [70, 30]},
        }}
        contacts = [
            {'pos': [0, 0, 0.05], 'dir': None},
            {'pos': [0, 0, -0.05], 'dir': [0, 0, 1]},
        ]
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            mesh_path = Path(directory) / 'mesh.glb'
            mesh_path.write_bytes(b'mesh fixture')
            (Path(directory) / 'view_predictions' / 'sol').mkdir(parents=True)
            fn_cache = Path(directory) / 'fn_point_cache'
            fn_cache.mkdir()
            for suffix in ('', '_single', '_all'):
                (fn_cache / f'mesh_fn{suffix}.json').write_text(json.dumps({
                    'actuation_contacts': [{'pos': [0, 0, 0.05], 'dir': None}],
                }))
            with mock.patch('codex.mesh.load_mesh.load_mesh', return_value=(object(), {})), \
                 mock.patch('codex.mesh.load_mesh.trimesh_to_o3d', return_value=Mesh()), \
                 mock.patch('codex.keypoint_estimation.triangulation.point3d_estimator.Point3DEstimator', return_value=Estimator()), \
                 mock.patch('codex.keypoint_estimation.mesh_keypoints.predict_views', return_value=result), \
                 mock.patch('codex.keypoint_estimation.estimate_actuation_point.reconstruct_contact', side_effect=contacts) as reconstruct, \
                 mock.patch('codex.keypoint_estimation.functional_direction.select_functional_direction',
                            return_value={'axis': '-X', 'direction': [-1, 0, 0]}) as select:
                output = estimate_mesh_keypoints(mesh_path, 'use bottle', 'bottle', directory,
                                                  backend=object())
            for suffix in ('', '_single', '_all'):
                saved = json.loads((fn_cache / f'mesh_fn{suffix}.json').read_text())
                self.assertEqual(saved['actuation_contacts'][0]['dir'], [-1, 0, 0])
                self.assertEqual(saved['actuation_contacts'][0]['direction_source'],
                                 'sol_canonical_axis')
        self.assertEqual(output['fn']['dir'], [-1, 0, 0])
        self.assertEqual(output['fn']['direction_axis'], '-X')
        self.assertEqual(output['fn']['direction_source'], 'sol_canonical_axis')
        self.assertEqual(output['act'], contacts[1])
        self.assertEqual(reconstruct.call_count, 2)
        self.assertFalse(reconstruct.call_args_list[0].kwargs['include_direction'])
        self.assertTrue(reconstruct.call_args_list[1].kwargs['include_direction'])
        self.assertEqual(select.call_count, 1)
        self.assertEqual(select.call_args.kwargs['visible_view_ids'], [0])

    def test_project_fn_uses_camera_intrinsics_and_rejects_behind_camera(self):
        scene = SimpleNamespace(image=Image.new('RGB', (200,100)),
                                cam_intrinsics=np.array([[100,0,100],[0,100,50],[0,0,1]]))
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            np.savetxt(Path(directory)/'pose.txt', np.eye(4))
            self.assertEqual(project_fn_to_scene([.2,.1,1],scene,directory), [[60,60]])
            self.assertEqual(project_fn_to_scene([.2,.1,-1],scene,directory), [])

    def test_target_abstention_does_not_publish_or_query_mesh(self):
        image = Image.new('RGB', (200,100))
        target = SimpleNamespace(name='plant', get_cropped=lambda:image)
        scene = SimpleNamespace(objects=[SimpleNamespace(name='bottle'),target])
        backend = SimpleNamespace(predict=lambda *a,**kw:{'target':{'visible':False,'point':None}})
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            with mock.patch('codex.keypoint_estimation.estimate_keypoint.estimate_mesh_keypoints') as estimate:
                with self.assertRaisesRegex(RuntimeError,'target is not visible'):
                    run_query(None,scene,'spray plant',scene_dir=directory,backend=backend)
                estimate.assert_not_called()
            self.assertFalse((Path(directory)/'keypoints/fn_key_pts.json').exists())


if __name__ == '__main__':
    unittest.main()
