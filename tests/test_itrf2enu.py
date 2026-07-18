import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "4_Visibility_Space" / "itrf2enu.py"
SPEC = importlib.util.spec_from_file_location("itrf2enu", MODULE_PATH)
itrf2enu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(itrf2enu)


class ITRFToENUTests(unittest.TestCase):
    def test_standard_output_name(self):
        source = Path("configs/vlaa.itrf.txt")
        self.assertEqual(itrf2enu._enu_output_path(source), Path("configs/vlaa.enu.txt"))

    def test_output_stays_next_to_input(self):
        source = Path("/tmp/array.txt")
        self.assertEqual(itrf2enu._enu_output_path(source), Path("/tmp/array.enu.txt"))

    def test_ecef_rotation_at_equator(self):
        a = itrf2enu.WGS84_A
        xyz = np.array([[a, 0.0, 0.0], [a + 3000.0, 1000.0, 2000.0]])
        enu = itrf2enu._ecef_offsets_to_enu(xyz)
        np.testing.assert_allclose(enu[1], [1000.0, 2000.0, 3000.0], atol=1e-9)

    def test_headerless_input_keeps_first_station(self):
        a = itrf2enu.WGS84_A
        xyz = np.array([[a, 0.0, 0.0], [a, 10.0, 0.0], [a, 20.0, 0.0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.itrf.txt"
            np.savetxt(path, xyz)
            east, north = itrf2enu.convert(path, save_enu=False)
        self.assertEqual(len(east), 3)
        np.testing.assert_allclose([east[0], north[0]], 0.0, atol=1e-15)

    def test_generated_tables_keep_every_station(self):
        config_dir = ROOT / "4_Visibility_Space" / "configs"
        for source in config_dir.glob("*.itrf.txt"):
            with self.subTest(source=source.name):
                xyz = np.loadtxt(source, comments="#", usecols=(0, 1, 2), ndmin=2)
                en = np.loadtxt(itrf2enu._enu_output_path(source), ndmin=2)
                self.assertEqual(len(en), len(xyz))


if __name__ == "__main__":
    unittest.main()
