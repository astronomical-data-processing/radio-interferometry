import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "7_Observing_Systems"))

import rime_lib


class RimeLibraryTests(unittest.TestCase):
    def test_antenna_pair_index_maps_all_optional_axes(self):
        values = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
        actual = values[rime_lib.ap_index(nsrc=2, ntime=3, na=3, nchan=2)]
        pairs = np.array(np.triu_indices(3)).T

        self.assertEqual(actual.shape, (2, 2, 3, 6, 2))
        for baseline, (ant_p, ant_q) in enumerate(pairs):
            np.testing.assert_array_equal(
                actual[0, :, :, baseline, :], values[:, :, ant_p, :]
            )
            np.testing.assert_array_equal(
                actual[1, :, :, baseline, :], values[:, :, ant_q, :]
            )

    def test_brightness_uses_total_stokes_i_normalization(self):
        actual = rime_lib.brightness(
            np.array([10.0]),
            np.array([2.0]),
            np.array([4.0]),
            np.array([6.0]),
        )
        expected = np.array([[[6.0, 2.0 + 3.0j], [2.0 - 3.0j, 4.0]]])

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual, actual.conj().transpose(0, 2, 1))
        np.testing.assert_allclose(np.trace(actual, axis1=1, axis2=2), [10.0])

    def test_unpolarized_source_splits_i_between_parallel_hands(self):
        actual = rime_lib.brightness(
            np.array([8.0]),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
        )[0]
        np.testing.assert_allclose(actual, np.diag([4.0, 4.0]))

    def test_brightness_rejects_mismatched_stokes_arrays(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            rime_lib.brightness(
                np.ones(2), np.ones(1), np.ones(2), np.ones(2)
            )

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

    def test_explicit_phase_centre_is_not_tied_to_first_source(self):
        actual = rime_lib.lm_2_rad(
            np.array([10.0, 11.0]),
            np.array([-30.0, -29.5]),
            phase_centre=(9.0, -31.0),
        )

        self.assertFalse(np.allclose(actual[0], 0.0))
        ra = np.deg2rad([10.0, 11.0])
        dec = np.deg2rad([-30.0, -29.5])
        ra0, dec0 = np.deg2rad([9.0, -31.0])
        expected = np.column_stack(
            (
                np.cos(dec) * np.sin(ra - ra0),
                np.sin(dec) * np.cos(dec0)
                - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0),
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

    def test_phase_rejects_directions_beyond_the_visible_hemisphere(self):
        with self.assertRaisesRegex(ValueError, "direction cosines"):
            rime_lib.phase(
                np.array([[0.8, 0.8]]),
                np.zeros((1, 1, 3)),
                np.array([1.4e9]),
            )

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

        actual = rime_lib.rime(antenna_uvw, sources, frequencies)

        brightness = rime_lib.brightness(*sources[:, 2:].T).sum(axis=0)
        antenna_pairs = np.array(np.triu_indices(antenna_uvw.shape[1])).T
        autocorrelations = np.flatnonzero(antenna_pairs[:, 0] == antenna_pairs[:, 1])
        actual_autocorrelations = actual[:, autocorrelations, :, :, :]
        expected = np.broadcast_to(brightness, actual_autocorrelations.shape)
        np.testing.assert_allclose(actual_autocorrelations, expected)

    def test_two_antenna_baseline_uses_p_minus_q_convention(self):
        frequency = np.array([1.0e9])
        wavelength = rime_lib.C / frequency[0]
        antenna_uvw = np.array([[[0.0, 0.0, 0.0], [wavelength, 0.0, 0.0]]])
        sources = np.array([[30.0, 0.0, 2.0, 0.0, 0.0, 0.0]])

        actual = rime_lib.rime(
            antenna_uvw,
            sources,
            frequency,
            phase_centre=(0.0, 0.0),
        )

        # Upper-triangle baseline 1 is (p, q) = (0, 1), so u_pq = -lambda.
        expected_phase = np.exp(-2j * np.pi * (-1.0) * 0.5)
        np.testing.assert_allclose(actual[0, 1, 0], expected_phase * np.eye(2))

    def test_kat7_antenna_coordinates_reconstruct_all_baselines(self):
        antenna_uvw = rime_lib.KAT7_antenna_uvw(hour_angle_start=0.0, ref_dec=-30.0)
        pairs = np.triu_indices(antenna_uvw.shape[1])
        reconstructed = antenna_uvw[:, pairs[0]] - antenna_uvw[:, pairs[1]]
        expected = rime_lib.sim_uv(
            0.0,
            -30.0,
            12.0,
            3.0,
            rime_lib.KAT7_ants,
            rime_lib.KAT7_location[0],
        ).reshape(reconstructed.shape)
        np.testing.assert_allclose(reconstructed, expected)


if __name__ == "__main__":
    unittest.main()
