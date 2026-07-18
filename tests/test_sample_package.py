import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = (
    ROOT
    / "9_Practical"
    / "sample_packages"
    / "continuum_lightweight_replay"
    / "analyze_sample.py"
)
SPEC = importlib.util.spec_from_file_location("sample_analysis", ANALYSIS_PATH)
sample_analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sample_analysis)


class SamplePackageTests(unittest.TestCase):
    def test_manifest_checksums(self):
        self.assertEqual(sample_analysis.verify_checksums(), {})

    def test_measurement_summary_is_physically_well_formed(self):
        summary = sample_analysis.measurement_summary()
        self.assertEqual(summary["checksum_validation"], "passed")
        self.assertGreater(summary["analysis_pixels"], 0)
        self.assertGreater(summary["pixels_per_beam"], 1.0)
        self.assertGreater(summary["independent_beams"], 1.0)
        self.assertGreater(summary["background_independent_beams"], 1.0)
        self.assertEqual(summary["noise_model"], "independent_synthesized_beams")
        self.assertGreater(summary["background_uncertainty_jy"], 0.0)
        self.assertGreater(
            summary["total_uncertainty_jy"], summary["random_uncertainty_jy"]
        )
        self.assertGreater(summary["total_uncertainty_jy"], 0.0)
        self.assertGreater(summary["three_sigma_sensitivity_jy"], 0.0)
        self.assertTrue(0.0 < summary["signal_to_noise"] < 100.0)


if __name__ == "__main__":
    unittest.main()
