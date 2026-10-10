from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from codex.mesh.scene_surface import build_scene_surface


def make_scene(h=3, w=3, z=1.0, extrinsics=None):
    yy, xx = np.mgrid[:h, :w]
    points = np.stack((xx * 0.01, yy * 0.01, np.full((h, w), z)), axis=-1)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    return SimpleNamespace(
        points=points,
        image=Image.fromarray(rgb),
        objects=[],
        cam_extrinsics=extrinsics,
    )


class SceneSurfaceTests(unittest.TestCase):
    def test_builds_two_triangles_per_valid_plane_cell_with_rgb_colors(self):
        mesh = build_scene_surface(make_scene())

        self.assertEqual(np.asarray(mesh.vertices).shape, (9, 3))
        self.assertEqual(np.asarray(mesh.triangles).shape, (8, 3))
        np.testing.assert_allclose(
            np.asarray(mesh.vertex_colors), np.tile([1, 0, 0], (9, 1))
        )


    def test_depth_discontinuity_rejects_triangles_across_jump(self):
        scene = make_scene(h=2, w=2)
        scene.points[:, 1, 2] = 1.1

        mesh = build_scene_surface(scene, max_edge_m=0.2, max_depth_jump_m=0.02)

        self.assertEqual(len(mesh.triangles), 0)


    def test_excluded_mask_pixels_are_not_vertices_or_triangle_vertices(self):
        scene = make_scene(h=2, w=2)
        mask = np.zeros((2, 2), dtype=np.uint8)
        mask[0, 0] = 1
        scene.objects = [SimpleNamespace(mask=mask)]

        mesh = build_scene_surface(scene, exclude_objs=[0])

        self.assertEqual(len(mesh.vertices), 3)
        self.assertEqual(len(mesh.triangles), 0)
        self.assertFalse(
            np.any(np.all(np.isclose(np.asarray(mesh.vertices), [0, 0, 1]), axis=1))
        )


    def test_transforms_vertices_from_camera_to_world(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        mesh = build_scene_surface(make_scene(h=2, w=2, extrinsics=transform))

        np.testing.assert_allclose(np.asarray(mesh.vertices)[0], [1, 2, 4])
        self.assertEqual(len(mesh.triangles), 2)


if __name__ == "__main__":
    unittest.main()
