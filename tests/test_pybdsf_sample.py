import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "9_Practical"
    / "sample_packages"
    / "pybdsf_abell2255_replay"
    / "analyze_products.py"
)
SPEC = importlib.util.spec_from_file_location("pybdsf_sample", MODULE_PATH)
SAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAMPLE)


class PybdsfSampleTests(unittest.TestCase):
    def test_checksums(self):
        self.assertEqual(len(SAMPLE.verify_checksums()), 7)

    def test_product_contracts(self):
        summary = SAMPLE.product_summary()
        self.assertLess(summary["frequency_average_max_error"], 2e-8)
        self.assertLess(summary["model_residual_max_error"], 2e-8)

    def test_catalogue_summary(self):
        summary = SAMPLE.product_summary()
        self.assertEqual(summary["catalog_sources"], 43)
        self.assertEqual(summary["source_codes"], {"C": 8, "M": 4, "S": 31})


if __name__ == "__main__":
    unittest.main()
