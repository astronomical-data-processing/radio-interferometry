import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data/scripts/plotUVcoverage.py"
SPEC = importlib.util.spec_from_file_location("plot_uv_coverage", SCRIPT)
plot_uv_coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_uv_coverage)


class PlotUVCoverageTests(unittest.TestCase):
    def test_uv_samples_include_conjugates(self):
        uvw = np.array([[3.0, 4.0, 2.0], [-6.0, 8.0, 1.0]])
        samples = plot_uv_coverage.uv_samples(uvw)
        expected = np.array([[3.0, 4.0], [-6.0, 8.0], [-3.0, -4.0], [6.0, -8.0]])
        np.testing.assert_allclose(samples, expected)

    def test_frequency_scaling_uses_wavelengths(self):
        uvw = np.array([[plot_uv_coverage.LIGHT_SPEED, 0.0, 0.0]])
        distances = plot_uv_coverage.uv_distances(uvw, [1.0, 2.0])
        np.testing.assert_allclose(distances, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
