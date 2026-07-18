import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "9_Practical" / "practical_metrics.py"
SPEC = importlib.util.spec_from_file_location("practical_metrics", MODULE_PATH)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class PracticalMetricsTests(unittest.TestCase):
    def test_beam_area_and_background_subtracted_aperture_flux(self):
        beam_pixels = metrics.pixels_per_beam(4.0, 2.0, 1.0)
        self.assertAlmostEqual(beam_pixels, np.pi * 8 / (4 * np.log(2)))
        image = np.full((4, 4), 3.0)
        flux = metrics.aperture_flux_jy(image, np.ones_like(image, bool), 4, 1.0)
        self.assertAlmostEqual(flux, 8.0)

    def test_averaging_attenuation_uses_normalized_sinc(self):
        self.assertAlmostEqual(metrics.averaging_attenuation(0.0, 10.0), 1.0)
        self.assertAlmostEqual(metrics.averaging_attenuation(0.25, 4.0), 0.0)

    def test_polarization_angle_preserves_quadrant_and_debiases(self):
        self.assertAlmostEqual(metrics.polarization_angle(-1.0, 0.0), np.pi / 2)
        self.assertAlmostEqual(
            metrics.debiased_linear_polarization(3.0, 4.0, 1.0), np.sqrt(24)
        )

    def test_rm_synthesis_recovers_faraday_thin_peak(self):
        lambda_squared = np.linspace(0.04, 0.16, 128)
        true_depth = 23.5
        polarization = np.exp(2j * true_depth * lambda_squared)
        depths = np.linspace(-100, 100, 401)
        spectrum, rmsf, lambda0_squared = metrics.rm_synthesis(
            polarization, lambda_squared, depths
        )
        self.assertAlmostEqual(depths[np.argmax(np.abs(spectrum))], true_depth)
        self.assertAlmostEqual(np.max(np.abs(rmsf)), 1.0)
        self.assertTrue(lambda_squared.min() < lambda0_squared < lambda_squared.max())

    def test_primary_beam_mosaic_recovers_common_sky(self):
        beams = np.array([[[0.5]], [[1.0]]])
        apparent = 4.0 * beams
        mosaic, uncertainty = metrics.primary_beam_mosaic(apparent, beams, [1.0, 2.0])
        self.assertAlmostEqual(mosaic.item(), 4.0)
        self.assertAlmostEqual(uncertainty.item(), np.sqrt(2.0))

    def test_hi_mass_uses_luminosity_distance_and_redshift(self):
        self.assertAlmostEqual(metrics.hi_mass_solar(2.0, 10.0, 1.0), 2.356e7)


if __name__ == "__main__":
    unittest.main()
