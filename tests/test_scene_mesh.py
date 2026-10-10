"""Stage 0 object mesh: importing a user mesh and rebuilding the mesh of a scene."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from codex.scene import create_scene as cs


class BuildObjectMeshTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.scene = os.path.join(self.dir.name, "scene")
        for d in ("mesh", "mesh_coacd", "keypoints", "grasps", "rl_runs"):
            os.makedirs(os.path.join(self.scene, d))
        open(os.path.join(self.scene, "mesh", "mesh.obj"), "w").close()   # an earlier mesh
        Path(self.scene, "obj_0.png").write_bytes(b"crop fixture")
        self.registered = []
        estimator = mock.MagicMock()
        estimator.return_value.register_and_canonicalize.side_effect = \
            lambda scene, path, idx: (self.registered.append(path), (np.eye(4), "+z"))[1]
        def metadata(mesh_path):
            path = Path(mesh_path)
            (path.parent / "metadata.json").write_text('{}')
            collision = path.parent.parent / "mesh_coacd"
            collision.mkdir()
            (collision / "mesh.glb").write_bytes(path.read_bytes())
            (collision / "metadata.json").write_text('{}')
        def generate(scene_dir, api_key):
            trimesh.creation.box().export(Path(scene_dir) / "mesh" / "mesh.glb")
        self.patches = [mock.patch.object(cs, "MeshPoseEstimator", estimator),
                        mock.patch.object(cs, "post_mesh_metadata", side_effect=metadata),
                        mock.patch.object(cs, "image_to_mesh", side_effect=generate)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.dir.cleanup()

    def test_imported_mesh_is_registered_and_old_outputs_archived(self):
        src = os.path.join(self.dir.name, "object.obj")
        trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(src)   # arbitrary scale is fine
        cs.build_object_mesh(self.scene, mesh_file=src, scene=object())
        glb = os.path.join(self.scene, "mesh", "mesh.glb")
        self.assertEqual(len(self.registered), 1)
        self.assertNotEqual(self.registered[0], glb)
        self.assertEqual(set(os.listdir(os.path.join(self.scene, "mesh"))), {"mesh.glb", "metadata.json"})
        self.assertAlmostEqual(trimesh.load(glb, force="mesh").extents.max(), 3.0, places=4)
        self.assertTrue(os.path.exists(os.path.join(self.scene, "pose.txt")))
        archive = next(Path(self.scene, ".history").iterdir())
        for d in ("keypoints", "grasps"):
            self.assertFalse(os.path.exists(os.path.join(self.scene, d)))
            self.assertTrue((archive / d).exists())
        self.assertTrue((archive / "mesh" / "mesh.obj").exists())
        self.assertTrue(os.path.isdir(os.path.join(self.scene, "rl_runs")))   # run history is kept
        cs.image_to_mesh.assert_not_called()

    def test_without_a_file_the_mesh_is_generated(self):
        cs.build_object_mesh(self.scene, api_key="k", scene=object())
        self.assertEqual(cs.image_to_mesh.call_count, 1)
        self.assertEqual(cs.image_to_mesh.call_args.kwargs["api_key"], "k")
        self.assertNotEqual(cs.image_to_mesh.call_args.kwargs["scene_dir"], self.scene)

    def test_unsupported_file_is_rejected_before_anything_is_removed(self):
        with self.assertRaises(ValueError):
            cs.build_object_mesh(self.scene, mesh_file="object.fbx", scene=object())
        self.assertTrue(os.path.isdir(os.path.join(self.scene, "keypoints")))


if __name__ == "__main__":
    unittest.main()
