import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "8_Calibration"))

import calibration_solver as calibration


class CalibrationSolverTests(unittest.TestCase):
    def test_point_source_model_uses_p_minus_q_baselines(self):
        frequency = 1.0e9
        actual = calibration.point_source_model(
            np.array([0.0, calibration.C / frequency]),
            np.array([0.0]),
            np.array([0.25]),
            np.array([2.0]),
            frequency_hz=frequency,
        )
        np.testing.assert_allclose(actual[0, 0, 1], 2.0j, atol=1e-14)
        np.testing.assert_allclose(actual[0, 1, 0], -2.0j, atol=1e-14)

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

    def test_zero_weight_excludes_a_corrupted_baseline(self):
        sample = calibration.generate_synthetic_data(n_times=1, noise_std=0.0)
        data = sample["data"].copy()
        data[0, 0, 1] = 1e6 + 2e6j
        data[0, 1, 0] = data[0, 0, 1].conjugate()
        weights = np.ones(data.shape)
        weights[0, 0, 1] = weights[0, 1, 0] = 0.0

        solved = calibration.solve_gains(data, sample["model"], weights=weights)
        expected = sample["true_gains"] * np.exp(
            -1j * np.angle(sample["true_gains"][:, :1])
        )
        np.testing.assert_allclose(solved, expected, atol=1e-8)

    def test_disconnected_weighted_baseline_graph_is_rejected(self):
        model = np.ones((4, 4), dtype=complex)
        data = model.copy()
        weights = np.zeros((4, 4))
        weights[0, 1] = weights[1, 0] = 1.0
        weights[2, 3] = weights[3, 2] = 1.0

        with self.assertRaisesRegex(ValueError, "connect every antenna"):
            calibration.solve_gains(data, model, weights=weights)

    def test_nonconverged_solution_is_rejected(self):
        sample = calibration.generate_synthetic_data(n_times=1, noise_std=0.0)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            calibration.solve_gains(
                sample["data"],
                sample["model"],
                max_iterations=1,
                tolerance=1e-15,
            )

    def test_diagnostics_report_holdout_and_reference_invariance(self):
        sample = calibration.generate_synthetic_data()
        fit_weights = np.ones(sample["data"].shape)
        validation_weights = np.zeros(sample["data"].shape)
        fit_weights[:, 0, 1] = fit_weights[:, 1, 0] = 0.0
        validation_weights[:, 0, 1] = validation_weights[:, 1, 0] = 1.0

        gains, metrics = calibration.calibration_diagnostics(
            sample["data"],
            sample["model"],
            fit_weights=fit_weights,
            validation_weights=validation_weights,
            reference=0,
            comparison_reference=2,
        )

        self.assertEqual(gains.shape, sample["true_gains"].shape)
        self.assertLess(metrics["fit_rms_after"], metrics["fit_rms_before"])
        self.assertLess(
            metrics["validation_rms_after"], metrics["validation_rms_before"]
        )
        self.assertLess(metrics["reference_rms_difference"], 1e-8)

    def test_weighted_rms_ignores_flagged_nonfinite_values(self):
        model = np.ones((3, 3), dtype=complex)
        data = model.copy()
        data[0, 1] = data[1, 0] = np.nan
        data[0, 2] += 3.0
        data[2, 0] = data[0, 2].conjugate()
        weights = np.ones((3, 3))
        weights[0, 1] = weights[1, 0] = 0.0
        weights[1, 2] = weights[2, 1] = 0.0

        self.assertAlmostEqual(
            calibration.rms_residual(data, model, weights=weights), 3.0
        )


if __name__ == "__main__":
    unittest.main()
