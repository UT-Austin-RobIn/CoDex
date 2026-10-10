"""LEAP palm force-closure reference agrees with the physical hand surface."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
import trimesh
from scipy.spatial.transform import Rotation

from codex.grasp_gen.batched_grasp_solver import (
    LEAP_PALM_CONTACT_OFFSET,
    palm_reference_in_base,
)


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/models/leap_rh/leap_solver.urdf"


def box_signed_distance(point, center, rotation, half_extent):
    delta = np.abs(rotation.T @ (point - center)) - half_extent
    return np.linalg.norm(np.maximum(delta, 0)) + min(float(delta.max()), 0.0)


class PalmReferenceTests(unittest.TestCase):
    def test_reference_is_exposed_collision_and_visual_face(self):
        palm = ET.parse(URDF).getroot().find(".//link[@name='leap_rh_palm']")
        point = np.asarray(LEAP_PALM_CONTACT_OFFSET)
        boxes = []
        for collision in palm.findall("collision"):
            origin = collision.find("origin")
            box = collision.find("geometry/box")
            center = np.fromstring(origin.attrib["xyz"], sep=" ")
            rotation = Rotation.from_euler(
                "xyz", np.fromstring(origin.attrib["rpy"], sep=" ")
            ).as_matrix()
            half_extent = np.fromstring(box.attrib["size"], sep=" ") / 2
            boxes.append((center, rotation, half_extent))
        self.assertEqual(len(boxes), 10)
        distances = np.asarray([box_signed_distance(point, *box) for box in boxes])
        self.assertAlmostEqual(float(distances[6]), 0.0, places=8)
        self.assertTrue(np.all(np.delete(distances, 6) >= 0.008 - 1e-8))
        outward = np.array([0.0, 0.0, -0.001])
        self.assertTrue(all(box_signed_distance(point + outward, *box) > 0 for box in boxes))

        visual = palm.find("visual")
        visual_origin = visual.find("origin")
        visual_center = np.fromstring(visual_origin.attrib["xyz"], sep=" ")
        visual_rotation = Rotation.from_euler(
            "xyz", np.fromstring(visual_origin.attrib["rpy"], sep=" ")
        ).as_matrix()
        mesh_file = visual.find("geometry/mesh").attrib["filename"]
        mesh = trimesh.load(URDF.parent / mesh_file, force="mesh")
        mesh.apply_transform(
            np.block([[visual_rotation, visual_center[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
        )
        _, visual_distance, faces = trimesh.proximity.closest_point(mesh, point[None, :])
        self.assertLess(float(visual_distance[0]), 0.0003)
        np.testing.assert_allclose(mesh.face_normals[faces[0]], [0, 0, -1], atol=1e-6)

    def test_fk_derived_seed_matches_certificate_reference(self):
        chain = pk.build_chain_from_urdf(URDF.read_text())
        palm_fk = chain.forward_kinematics(torch.zeros(2, 16))["leap_rh_palm"].get_matrix()
        offset = torch.tensor(LEAP_PALM_CONTACT_OFFSET, dtype=torch.float32)
        base_point = palm_reference_in_base(palm_fk, offset)
        expected = torch.tensor([0.0345, 0.002, 0.052]).expand(2, -1)
        self.assertTrue(torch.allclose(base_point, expected, atol=2e-6))


if __name__ == "__main__":
    unittest.main()
