import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "6_Deconvolution"))

from clean_demo import (
    gaussian_clean_beam,
    hogbom_clean,
    load_clean_data,
    restore_image,
    subtract_psf,
)


class CleanDemoTests(unittest.TestCase):
    def test_missing_fits_files_use_reproducible_synthetic_data(self):
        first = load_clean_data("missing-dirty.fits", "missing-psf.fits")
        second = load_clean_data("missing-dirty.fits", "missing-psf.fits")

        dirty, psf, source = first
        self.assertEqual(dirty.shape, (101, 101))
        self.assertEqual(psf.shape, dirty.shape)
        self.assertEqual(source, "deterministic synthetic data")
        self.assertAlmostEqual(psf[50, 50], 1.0)
        self.assertGreater(dirty.max(), 10.0)
        np.testing.assert_array_equal(dirty, second[0])

    def test_psf_subtraction_clips_instead_of_wrapping_at_an_edge(self):
        image = np.zeros((7, 7))
        psf = np.arange(25, dtype=float).reshape(5, 5) + 1.0
        psf[2, 2] = 100.0
        result = subtract_psf(image, psf, (0, 0), 2.0)
        self.assertEqual(result[-1, -1], 0.0)
        self.assertEqual(result[0, 0], -2.0)
        self.assertEqual(np.count_nonzero(result), 9)

    def test_hogbom_recovers_positive_and_negative_components(self):
        psf = np.zeros((9, 9))
        psf[4, 4] = 1.0
        dirty = np.zeros_like(psf)
        dirty[2, 3] = 4.0
        dirty[6, 7] = -2.0
        result = hogbom_clean(dirty, psf, gain=1.0, niter=4, threshold=0.0)
        np.testing.assert_allclose(result["model"], dirty)
        np.testing.assert_allclose(result["residual"], 0.0)
        self.assertEqual(result["iterations"], 2)

    def test_clean_mask_limits_where_components_can_be_added(self):
        psf = np.zeros((7, 7))
        psf[3, 3] = 1.0
        dirty = np.zeros_like(psf)
        dirty[1, 1] = 3.0
        dirty[5, 5] = 8.0
        mask = np.zeros_like(dirty, dtype=bool)
        mask[:3, :3] = True
        result = hogbom_clean(dirty, psf, gain=1.0, niter=3, mask=mask)
        self.assertEqual(result["model"][1, 1], 3.0)
        self.assertEqual(result["model"][5, 5], 0.0)
        self.assertEqual(result["residual"][5, 5], 8.0)

    def test_restoration_preserves_sign_and_scales_residual(self):
        model = np.zeros((9, 9))
        model[4, 4] = -2.0
        residual = np.ones((9, 9))
        beam = np.zeros((5, 5))
        beam[2, 2] = 1.0
        restored = restore_image(model, residual, beam, residual_scale=0.25)
        self.assertAlmostEqual(restored[4, 4], -1.75)
        self.assertEqual(restored[0, 0], 0.25)

    def test_gaussian_clean_beam_is_centered_and_peak_normalized(self):
        _, psf, _ = load_clean_data("missing-dirty.fits", "missing-psf.fits")
        beam = gaussian_clean_beam(psf)
        center = tuple(np.array(beam.shape) // 2)
        self.assertAlmostEqual(beam[center], 1.0)
        np.testing.assert_allclose(beam, beam[::-1, ::-1], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
