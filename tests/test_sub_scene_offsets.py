"""GPU sub-scenes are spread over all three axes (see pick_mesh.sub_scene_offsets)."""
import unittest
from collections import Counter

import numpy as np

from codex.codex_train.envs.pick_mesh import sub_scene_offsets


class SubSceneOffsetsTest(unittest.TestCase):
    def test_distinct_spaced_and_spread_on_every_axis(self):
        for n in (1, 2, 27, 100, 512, 2048, 4096):
            offsets = np.array(sub_scene_offsets(n, 5.0))
            self.assertEqual(len(offsets), n)
            self.assertEqual(len({tuple(o) for o in offsets}), n)       # no two sub-scenes coincide
            self.assertTrue(np.allclose(offsets / 5.0, np.round(offsets / 5.0)))
            side = int(np.ceil(n ** (1 / 3) - 1e-9))
            for axis in range(3):                                      # no axis value shared by more than side^2 envs
                self.assertLessEqual(max(Counter(offsets[:, axis]).values()), side * side)

    def test_4096_is_a_16_cube(self):
        offsets = np.array(sub_scene_offsets(4096, 5.0))
        for axis in range(3):
            self.assertEqual(len(np.unique(offsets[:, axis])), 16)


if __name__ == "__main__":
    unittest.main()
