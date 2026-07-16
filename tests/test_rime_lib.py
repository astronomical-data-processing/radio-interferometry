import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "7_Observing_Systems"))

import rime_lib


class RimeLibraryTests(unittest.TestCase):
    def test_negative_sexagesimal_degrees(self):
        expected = -(30.0 + 43.0 / 60.0 + 17.34 / 3600.0)
        self.assertAlmostEqual(rime_lib.dec_degrees("-30:43:17.34"), expected)

    def test_equatorial_coordinates_to_direction_cosines(self):
        ra = np.array([10.0, 11.0])
        dec = np.array([-30.0, -29.5])

        actual = rime_lib.lm_2_rad(ra, dec)

        delta_ra = np.deg2rad(ra - ra[0])
        dec_rad = np.deg2rad(dec)
        dec_0 = dec_rad[0]
        expected = np.column_stack(
            (
                np.cos(dec_rad) * np.sin(delta_ra),
                np.sin(dec_rad) * np.cos(dec_0)
                - np.cos(dec_rad) * np.sin(dec_0) * np.cos(delta_ra),
            )
        )
        np.testing.assert_allclose(actual, expected)

    def test_phase_uses_both_direction_cosines(self):
        lm = np.array([[0.1, 0.2]])
        uvw = np.zeros((1, 1, 3))
        uvw[0, 0, 1] = 100.0
        frequency = np.array([1.4e9])

        actual = rime_lib.phase(lm, uvw, frequency)[0, 0, 0, 0]
        expected = np.exp(
            -2j * np.pi * lm[0, 1] * uvw[0, 0, 1] * frequency[0] / rime_lib.C
        )
        np.testing.assert_allclose(actual, expected)

    def test_antenna_phase_cancels_on_autocorrelations(self):
        antenna_uvw = np.array(
            [
                [[0.0, 0.0, 0.0], [25.0, -4.0, 2.0], [-10.0, 31.0, -1.0]],
                [[0.0, 0.0, 0.0], [28.0, -3.0, 2.0], [-9.0, 35.0, -1.0]],
            ]
        )
        sources = np.array(
            [
                [10.0, -30.0, 1.0, 0.1, 0.2, 0.05],
                [10.2, -29.8, 2.0, 0.2, 0.1, -0.03],
            ]
        )
        frequencies = np.array([1.2e9, 1.5e9])

        with contextlib.redirect_stdout(io.StringIO()):
            actual = rime_lib.rime(antenna_uvw, sources, frequencies)

        brightness = rime_lib.brightness(*sources[:, 2:].T).sum(axis=0)
        antenna_pairs = np.array(np.triu_indices(antenna_uvw.shape[1])).T
        autocorrelations = np.flatnonzero(antenna_pairs[:, 0] == antenna_pairs[:, 1])
        actual_autocorrelations = actual[:, autocorrelations, :, :, :]
        expected = np.broadcast_to(brightness, actual_autocorrelations.shape)
        np.testing.assert_allclose(actual_autocorrelations, expected)


if __name__ == "__main__":
    unittest.main()
