import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "7_Observing_Systems"))

import system_checks


class SystemCheckTests(unittest.TestCase):
    def test_sefd_and_image_noise_match_problem_set(self):
        sefd = system_checks.sefd_jy(15.0, 0.70, 30.0)
        noise = system_checks.image_noise_jy(
            sefd, 27, 2, 1e9, 3600.0, system_efficiency=0.90
        )

        self.assertAlmostEqual(sefd, 669.68, places=2)
        self.assertAlmostEqual(noise * 1e6, 10.47, places=2)

    def test_correlator_output_rate_matches_problem_set(self):
        baselines, visibility_rate, byte_rate = system_checks.correlator_output_rates(
            64, 4096, 4, 1.0
        )

        self.assertEqual(baselines, 2016)
        self.assertEqual(visibility_rate, 33_030_144)
        self.assertAlmostEqual(byte_rate / 1e6, 264.241152)

    def test_dterm_crosshand_ratio_matches_problem_set(self):
        leakage = system_checks.dterm_crosshand_ratio(
            0.020, 0.010 * np.exp(1j * np.deg2rad(30.0))
        )

        np.testing.assert_allclose(leakage, 0.028660254 - 0.005j)
        self.assertAlmostEqual(abs(leakage), 0.029093, places=6)
        self.assertAlmostEqual(np.rad2deg(np.angle(leakage)), -9.896, places=3)

    def test_primary_beam_spectral_bias_matches_problem_set(self):
        offset = np.deg2rad(12.0 / 60.0)
        fwhm_low, beam_low = system_checks.gaussian_power_beam(
            offset, 1.4e9, 25.0
        )
        fwhm_high, beam_high = system_checks.gaussian_power_beam(
            offset, 2.8e9, 25.0
        )
        alpha = system_checks.apparent_spectral_index(
            -0.7, 1.4e9, 2.8e9, beam_low, beam_high
        )

        self.assertAlmostEqual(np.rad2deg(fwhm_low) * 60, 30.03, places=2)
        self.assertAlmostEqual(np.rad2deg(fwhm_high) * 60, 15.02, places=2)
        self.assertAlmostEqual(beam_low, 0.642, places=3)
        self.assertAlmostEqual(beam_high, 0.170, places=3)
        self.assertAlmostEqual(alpha, -2.62, places=2)


if __name__ == "__main__":
    unittest.main()
