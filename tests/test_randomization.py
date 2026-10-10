"""Per-environment physical parameters: ranges, pinned values and pair friction."""

import unittest

import numpy as np

from codex.codex_train.configs.primitive import DOMAIN_RANDOMIZATION, PRIMITIVE_ENV_CONFIG
from codex.codex_train.envs import pick_mesh as pm


class RandomizationTest(unittest.TestCase):
    def test_ranges_and_pinned_values(self):
        np.random.seed(0)
        v = pm.draw((0.4, 1.0), 1000)
        self.assertTrue(((v >= 0.4) & (v <= 1.0)).all())
        self.assertGreater(v.std(), 0.1)
        np.testing.assert_array_equal(pm.draw(0.7, 3), [0.7, 0.7, 0.7])
        np.testing.assert_array_equal(pm.draw(np.float64(0.7), 2), [0.7, 0.7])

    def test_pair_friction_is_the_average_with_the_object(self):
        m = pm.pair_material(0.6)
        self.assertAlmostEqual((m.static_friction + pm.OBJECT_MATERIAL_FRICTION) / 2, 0.6, places=6)
        with self.assertRaises(ValueError):
            pm.pair_material(pm.OBJECT_MATERIAL_FRICTION / 2 - 0.01)

    def test_configured_ranges_are_representable_and_shared(self):
        for key in ("hand_friction", "table_friction"):
            pm.pair_material(min(DOMAIN_RANDOMIZATION[key]))
        for key, value in DOMAIN_RANDOMIZATION.items():
            self.assertEqual(PRIMITIVE_ENV_CONFIG[key], value)

    def test_hand_has_no_per_link_urdf_materials(self):
        # A URDF-config material on the fingertip links removed table/object
        # friction in the first GPU sub-scene (its object slid off on its own);
        # hand friction is set per environment by pick_mesh instead.
        from codex.codex_train.agents.leap_rh.floating_leap_right import FloatingLeapHandRight
        config = FloatingLeapHandRight.urdf_config
        self.assertNotIn("_materials", config)
        self.assertFalse(any("material" in v for v in config.get("link", {}).values()))


if __name__ == "__main__":
    unittest.main()
