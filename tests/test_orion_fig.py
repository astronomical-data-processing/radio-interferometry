import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "4_Visibility_Space" / "figures" / "orion_fig.py"
SPEC = importlib.util.spec_from_file_location("orion_fig", MODULE_PATH)
orion_fig = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(orion_fig)


class OrionFigureTests(unittest.TestCase):
    def test_phase_center_has_zero_direction_cosines(self):
        l, m = orion_fig.direction_cosines(orion_fig.RA_HOURS, orion_fig.DEC_DEGREES)
        np.testing.assert_allclose([l[0], m[0]], 0.0, atol=1e-15)

    def test_direction_cosines_stay_on_visible_hemisphere(self):
        l, m = orion_fig.direction_cosines(orion_fig.RA_HOURS, orion_fig.DEC_DEGREES)
        self.assertTrue(np.all(l**2 + m**2 <= 1.0))


if __name__ == "__main__":
    unittest.main()
