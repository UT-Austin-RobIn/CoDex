"""Functional axis choice is a single multi-view Sol request, with no normal fallback."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import trimesh

from codex.keypoint_estimation.functional_direction import (
    AXES, annotate_axes, project_mesh_point, select_functional_direction,
)


SCRATCH = Path(tempfile.gettempdir())


class DirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        self.mesh.apply_translation([0, 0, 1])
        self.point = [0, 0, 1]
        intrinsic = np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]])
        self.views = {i: np.full((200, 200, 3), 180, dtype=np.uint8) for i in range(3)}
        self.camera_data = {}
        for i in range(3):
            extrinsic = np.eye(4)
            extrinsic[0, 3] = i * 0.1
            self.camera_data[i] = {'extrinsic': extrinsic.tolist(),
                                   'intrinsic': intrinsic.tolist()}

    def test_projection_uses_cached_opencv_world_to_camera_matrices(self):
        camera = self.camera_data[1]
        self.assertEqual(project_mesh_point([0, 0, 1], camera['extrinsic'],
                                            camera['intrinsic']), (110., 100.))
        self.assertEqual(project_mesh_point([0.1, 0, 1], camera['extrinsic'],
                                            camera['intrinsic']), (120., 100.))
        self.assertEqual(project_mesh_point([0, 0.1, 1], camera['extrinsic'],
                                            camera['intrinsic']), (110., 110.))
        self.assertIsNone(project_mesh_point([0, 0, -1], camera['extrinsic'],
                                             camera['intrinsic']))
        image = annotate_axes(self.views[0], self.camera_data[0], self.point,
                              0.1, view_id=0)
        self.assertNotEqual(image.getpixel((100, 100)), (180, 180, 180))
        self.assertNotEqual(image.getpixel((108, 100)), (180, 180, 180))

    def test_one_structured_multi_image_request_and_saved_choice(self):
        usage = SimpleNamespace(model_dump=lambda: {'input_tokens': 100, 'output_tokens': 10})
        response = SimpleNamespace(status='completed', output_text=json.dumps({
            'axis': '+Y', 'reason': 'The outlet acts along green +Y.'}),
            model='gpt-6-sol', usage=usage)
        create = mock.Mock(return_value=response)
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        result = select_functional_direction(
            self.mesh, self.point, 'spray a plant', 'spray bottle', self.views,
            self.camera_data, self.temp.name, visible_view_ids=[0, 1, 2],
            client=client, model='gpt-6-sol', max_views=3)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(result['axis'], '+Y')
        self.assertEqual(result['direction'], AXES['+Y'])
        self.assertEqual(result['usage']['input_tokens'], 100)
        request = create.call_args.kwargs
        self.assertEqual(request['model'], 'gpt-6-sol')
        self.assertEqual(request['text']['format']['schema']['properties']['axis']['enum'], list(AXES))
        content = request['input'][0]['content']
        self.assertEqual(len(content), 7)  # One text plus clean/annotated pairs.
        self.assertIn('not shown', content[0]['text'])
        self.assertIn('First identify the physical outlet', content[0]['text'])
        self.assertTrue(all(item['image_url'].startswith('data:image/png;base64,')
                            for item in content[1:]))
        saved = json.loads((Path(self.temp.name) / 'selection.json').read_text())
        self.assertEqual(saved['direction'], [0, 1, 0])
        self.assertEqual(saved['reason'], result['reason'])
        self.assertEqual(saved['usage'], {'input_tokens': 100, 'output_tokens': 10})
        self.assertEqual(len(list(Path(self.temp.name).glob('view_*_axes.png'))), 3)
        self.assertEqual(len(list(Path(self.temp.name).glob('view_*_clean.png'))), 3)

    def test_invalid_axis_fails_without_geometric_fallback(self):
        response = SimpleNamespace(status='completed', output_text='{"axis":"+Q","reason":"guess"}',
                                   model='gpt-6-sol', usage=None)
        client = SimpleNamespace(responses=SimpleNamespace(create=mock.Mock(return_value=response)))
        with self.assertRaisesRegex(ValueError, 'valid axis'):
            select_functional_direction(self.mesh, self.point, 'spray', 'bottle',
                                        self.views, self.camera_data, self.temp.name,
                                        client=client)
        self.assertFalse((Path(self.temp.name) / 'selection.json').exists())


if __name__ == '__main__':
    unittest.main()
