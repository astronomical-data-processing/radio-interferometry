import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "8_Calibration"))

import calibration_solver as calibration


class CalibrationSolverTests(unittest.TestCase):
    def test_noiseless_gains_are_recovered_up_to_global_phase(self):
        sample = calibration.generate_synthetic_data(noise_std=0.0)
        solved = calibration.solve_gains(sample["data"], sample["model"])
        expected = sample["true_gains"] * np.exp(
            -1j * np.angle(sample["true_gains"][:, :1])
        )
        np.testing.assert_allclose(solved, expected, atol=1e-8)

    def test_corrected_visibilities_do_not_depend_on_reference_antenna(self):
        sample = calibration.generate_synthetic_data()
        gain_0 = calibration.solve_gains(sample["data"], sample["model"], reference=0)
        gain_2 = calibration.solve_gains(sample["data"], sample["model"], reference=2)
        corrected_0 = calibration.correct_visibilities(sample["data"], gain_0)
        corrected_2 = calibration.correct_visibilities(sample["data"], gain_2)
        np.testing.assert_allclose(corrected_0, corrected_2, atol=1e-8)

    def test_incomplete_model_leaves_a_larger_sky_residual(self):
        sample = calibration.generate_synthetic_data()
        full_gain = calibration.solve_gains(sample["data"], sample["model"])
        incomplete_gain = calibration.solve_gains(sample["data"], sample["incomplete_model"])
        full_residual = calibration.rms_residual(
            calibration.correct_visibilities(sample["data"], full_gain), sample["model"]
        )
        incomplete_residual = calibration.rms_residual(
            calibration.correct_visibilities(sample["data"], incomplete_gain), sample["model"]
        )
        self.assertGreater(incomplete_residual, 2.0 * full_residual)


if __name__ == "__main__":
    unittest.main()
