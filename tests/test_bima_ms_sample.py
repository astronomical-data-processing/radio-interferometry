import importlib.util
import unittest
from pathlib import Path


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
        self.assertEqual(len(SAMPLE.verify_checksums()), 6)

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


if __name__ == "__main__":
    unittest.main()
