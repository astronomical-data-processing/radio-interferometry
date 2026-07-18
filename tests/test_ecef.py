import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "3_Positional_Astronomy" / "ecef.py"
SPEC = importlib.util.spec_from_file_location("ecef", MODULE_PATH)
ecef = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ecef)


class EcefTests(unittest.TestCase):
    def test_geodetic_round_trip(self):
        expected = np.array([51.143833512, -1.433500703, 176.028])
        xyz = ecef.geodetic2ecef(*expected)
        actual = np.array(ecef.ecef2geodetic(*xyz))
        np.testing.assert_allclose(actual[:2], expected[:2], atol=1e-9)
        self.assertAlmostEqual(actual[2], expected[2], delta=1e-3)

    def test_poles_are_finite(self):
        latitude = np.array([-90.0, 90.0])
        longitude = np.array([0.0, 123.0])
        altitude = np.array([0.0, 1000.0])
        recovered = ecef.ecef2geodetic(
            *ecef.geodetic2ecef(latitude, longitude, altitude)
        )
        np.testing.assert_allclose(recovered[0], latitude, atol=1e-10)
        np.testing.assert_allclose(recovered[2], altitude, atol=1e-3)

    def test_earth_centre_is_rejected(self):
        with self.assertRaises(ValueError):
            ecef.ecef2geodetic(0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
