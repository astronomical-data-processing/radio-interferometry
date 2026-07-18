import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "9_Practical"
    / "sample_packages"
    / "bima_ngc4826_ms_replay"
    / "analyze_ms.py"
)
SPEC = importlib.util.spec_from_file_location("bima_ms_sample", MODULE_PATH)
SAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAMPLE)


class BimaMeasurementSetTests(unittest.TestCase):
    def test_checksums(self):
        self.assertEqual(len(SAMPLE.verify_checksums()), 7)

    def test_archive_has_required_subtables(self):
        subtables = SAMPLE.archive_subtables()
        self.assertTrue({"ANTENNA", "FIELD", "SPECTRAL_WINDOW"} <= subtables)

    def test_visibility_extracts(self):
        summary = SAMPLE.visibility_summary()
        self.assertEqual(summary["calibrator_rows"], 2925)
        self.assertEqual(summary["target_channels"], 64)
        self.assertEqual(summary["target_rows"], 360)

    def test_mosaic_coverage_extract(self):
        summary = SAMPLE.coverage_summary()
        self.assertEqual(summary["target_rows"], 9720)
        self.assertEqual(summary["target_fields"], [2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(summary["data_descriptions"], [2, 3, 4, 5])
        self.assertAlmostEqual(summary["mean_row_flag_fraction"], 0.2)
        self.assertGreater(summary["calibrator_target_gap_s"], 900)

    def test_real_calibrator_solution_reduces_residual(self):
        summary = SAMPLE.calibration_summary()
        self.assertEqual(summary["active_antennas"], [0, 1, 2, 3, 4, 6, 7])
        self.assertLess(summary["rms_ratio"], 0.02)

    def test_gain_table_matches_recomputed_solution(self):
        path = MODULE_PATH.parent / "derived" / "relative_gain_table.npz"
        with np.load(path) as archive:
            stored = {key: archive[key] for key in archive.files}
        recomputed = SAMPLE.calibration_table()
        self.assertEqual(set(stored), set(recomputed))
        for key, expected in recomputed.items():
            np.testing.assert_allclose(stored[key], expected, err_msg=key)

        self.assertEqual(stored["schema_version"], 1)
        self.assertEqual(stored["gain"].shape, (65, 7))
        self.assertEqual(stored["training_baseline_pairs"].shape, (14, 2))
        self.assertEqual(stored["holdout_baseline_pairs"].shape, (7, 2))
        self.assertFalse(stored["gain_flag"].any())
        self.assertTrue((stored["input_weight"] > 0).all())
        np.testing.assert_allclose(np.angle(stored["gain"][:, 0]), 0.0, atol=1e-12)

    def test_baseline_holdout_prefers_per_integration_solutions(self):
        summaries = SAMPLE.calibration_interval_summary()
        self.assertEqual([item["solutions"] for item in summaries], [65, 13, 1])
        self.assertLess(summaries[0]["holdout_rms"], 0.25)
        self.assertLess(summaries[0]["holdout_rms"], summaries[1]["holdout_rms"])


if __name__ == "__main__":
    unittest.main()
