import unittest

import numpy as np
from PIL import Image

from codex.mesh.scene_axes import draw_axes_overlay, project_axes


class SceneAxesTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])

    def test_projects_eight_centimeter_axes_in_camera_view(self):
        segments = project_axes(
            [0, 0, 1], np.eye(3), np.eye(4), self.K,
        )

        np.testing.assert_allclose(segments[0, 0], [50, 40])
        np.testing.assert_allclose(segments[0, 1], [58, 40])
        np.testing.assert_allclose(segments[1, 1], [50, 48])
        np.testing.assert_allclose(segments[2, 1], [50, 40])

    def test_projects_using_oblique_camera_extrinsic(self):
        angle = np.deg2rad(30)
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ]
        segments = project_axes(
            [0, 0, 1], np.eye(3), world_to_camera, self.K,
        )

        self.assertTrue(np.isfinite(segments).all())
        self.assertNotAlmostEqual(segments[0, 1, 0], 58.0)
        self.assertNotAlmostEqual(segments[2, 1, 0], 50.0)

    def test_overlay_draws_axes_without_mutating_input(self):
        image = Image.new("RGB", (100, 80), (128, 128, 128))
        segments = project_axes([0, 0, 1], np.eye(3), np.eye(4), self.K)

        result = draw_axes_overlay(image, segments)

        self.assertEqual(image.getpixel((50, 40)), (128, 128, 128))
        self.assertNotEqual(result.getpixel((54, 40)), (128, 128, 128))
        self.assertNotEqual(result.getpixel((15, 15)), (128, 128, 128))


if __name__ == "__main__":
    unittest.main()
