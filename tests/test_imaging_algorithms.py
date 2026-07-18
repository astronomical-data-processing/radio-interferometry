import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5_Imaging"))

from AA_filter import AA_filter
from array_coordinates import ecef_to_enu
from convolutional_degridder import fft_degrid
from convolutional_gridder import grid_ifft
import jvla_a_constants
import jvla_d_constants
from track_simulator import sim_uv
import westerbork_92m_constants


class ArrayCoordinateTests(unittest.TestCase):
    def test_ecef_to_enu_rotation_at_equator(self):
        xyz = np.array([[10.0, 0.0, 0.0], [13.0, 1.0, 2.0]])
        enu = ecef_to_enu(xyz, longitude_deg=0.0, latitude_deg=0.0)
        np.testing.assert_allclose(enu[1], [1.0, 2.0, 3.0])

    def test_ecef_to_enu_rotation_at_north_pole(self):
        xyz = np.array([[0.0, 0.0, 10.0], [2.0, 3.0, 14.0]])
        enu = ecef_to_enu(xyz, longitude_deg=0.0, latitude_deg=90.0)
        np.testing.assert_allclose(enu[1], [3.0, -2.0, 4.0], atol=1e-15)

    def test_array_tables_preserve_ecef_baseline_lengths(self):
        for module in (jvla_a_constants, jvla_d_constants, westerbork_92m_constants):
            with self.subTest(module=module.__name__):
                ecef_offsets = module.ANTENNA_POSITIONS - module.ANTENNA_POSITIONS[0]
                np.testing.assert_allclose(
                    np.linalg.norm(module.ENU, axis=1),
                    np.linalg.norm(ecef_offsets, axis=1),
                    rtol=1e-12,
                    atol=1e-9,
                )
                np.testing.assert_allclose(module.ENU[0], 0.0, atol=1e-12)


class AntiAliasingFilterTests(unittest.TestCase):
    def test_sampled_kernels_are_centered_and_symmetric(self):
        for filter_type, half_support in (
            ("box", 1),
            ("sinc", 3),
            ("gaussian_sinc", 3),
        ):
            kernel = AA_filter(half_support, 63, filter_type)
            with self.subTest(filter_type=filter_type):
                np.testing.assert_allclose(kernel.filter_taps, kernel.filter_taps[::-1])
                center = (kernel.no_taps - 1) // 2
                self.assertEqual(kernel.filter_taps[center], np.max(kernel.filter_taps))

    def test_nearest_neighbor_kernel_uses_one_grid_point_at_integer_uv(self):
        kernel = AA_filter(1, 63, "box")
        offsets = np.arange(-1, 2) * kernel.oversample + 2 * kernel.oversample
        np.testing.assert_array_equal(kernel.filter_taps[offsets], [0.0, 1.0, 0.0])

    def test_fractional_coordinate_uses_nearest_oversampled_tap(self):
        kernel = AA_filter(3, 8, "sinc")
        nearest, actual = kernel.sample(0.18)
        center = int(np.rint((kernel.half_sup + 1 - 0.18) * kernel.oversample))
        expected = kernel.filter_taps[kernel.offsets * kernel.oversample + center]
        self.assertEqual(nearest, 0)
        np.testing.assert_array_equal(actual, expected)


class GriddingTests(unittest.TestCase):
    def test_centered_point_source_degrids_to_unit_visibilities(self):
        size = 8
        image = np.zeros((1, size, size))
        image[0, size // 2, size // 2] = 1.0
        uvw = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        actual = fft_degrid(image, uvw, np.array([1.0]), size, size, AA_filter(1, 63, "box"))
        np.testing.assert_allclose(actual, 1.0)

    def test_center_visibility_transforms_to_constant_dirty_image_and_psf(self):
        size = 8
        dirty, psf = grid_ifft(
            np.ones((1, 1, 1), complex),
            np.zeros((1, 3)),
            np.array([1.0]),
            size,
            size,
            AA_filter(1, 63, "box"),
        )
        np.testing.assert_allclose(dirty, 1.0 / size**2)
        np.testing.assert_allclose(psf, 1.0 / size**2)
        self.assertEqual(psf.shape, (size, size))

    def test_sinc_degridding_improves_fractional_point_source_phases(self):
        size = 32
        image = np.zeros((1, size, size))
        image[0, size // 2 + 3, size // 2 - 4] = 1.0
        uvw = np.column_stack(
            (np.linspace(-9.4, 9.4, 41), np.linspace(8.7, -8.7, 41), np.zeros(41))
        )
        expected = np.exp(
            -2j * np.pi * (uvw[:, 0] * -4 + uvw[:, 1] * 3) / size
        )
        box = fft_degrid(image, uvw, np.array([1.0]), size, size, AA_filter(1, 63, "box"))[:, 0, 0]
        sinc = fft_degrid(image, uvw, np.array([1.0]), size, size, AA_filter(3, 63, "sinc"))[:, 0, 0]
        self.assertLess(np.linalg.norm(sinc - expected), np.linalg.norm(box - expected))

    def test_gridder_and_degridder_are_adjoint_up_to_ifft_normalization(self):
        rng = np.random.default_rng(8)
        size = 16
        image = rng.normal(size=(1, size, size)) + 1j * rng.normal(size=(1, size, size))
        vis = rng.normal(size=(5, 1, 1)) + 1j * rng.normal(size=(5, 1, 1))
        uvw = np.column_stack(
            (rng.uniform(-4.0, 4.0, 5), rng.uniform(-4.0, 4.0, 5), np.zeros(5))
        )
        kernel = AA_filter(3, 31, "gaussian_sinc")
        predicted = fft_degrid(image, uvw, np.array([1.0]), size, size, kernel)
        dirty, _ = grid_ifft(vis, uvw, np.array([1.0]), size, size, kernel)
        np.testing.assert_allclose(
            np.vdot(dirty, image),
            np.vdot(vis, predicted) / size**2,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_gridder_applies_measurement_weights_to_image_and_psf(self):
        size = 8
        args = (
            np.ones((1, 1, 1), complex),
            np.zeros((1, 3)),
            np.array([1.0]),
            size,
            size,
            AA_filter(1, 63, "box"),
        )
        dirty, psf = grid_ifft(*args, weights=np.array([[2.5]]))
        np.testing.assert_allclose(dirty, 2.5 / size**2)
        np.testing.assert_allclose(psf, 2.5 / size**2)

    def test_out_of_grid_convolution_footprint_is_rejected(self):
        size = 8
        image = np.zeros((1, size, size))
        uvw = np.array([[3.5, 0.0, 0.0]])
        kernel = AA_filter(1, 63, "box")
        with self.assertRaisesRegex(ValueError, "extends beyond"):
            fft_degrid(image, uvw, np.array([1.0]), size, size, kernel)
        with self.assertRaisesRegex(ValueError, "extends beyond"):
            grid_ifft(
                np.ones((1, 1, 1)), uvw, np.array([1.0]), size, size, kernel
            )


class TrackSimulatorTests(unittest.TestCase):
    def test_two_antenna_east_west_baseline_at_zero_hour_angle(self):
        enu = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        uvw = sim_uv(0.0, -30.0, 1.0, 1.0, enu, -30.0)
        expected = np.array([[0.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(uvw, expected, atol=1e-12)

    def test_invalid_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            sim_uv(0.0, 0.0, 1.0, 1.0, np.zeros((3, 2)), 0.0)
        with self.assertRaises(ValueError):
            sim_uv(0.0, 0.0, 1.0, 1.0, np.empty((0, 3)), 0.0)

    def test_hour_angle_advances_fifteen_degrees_per_hour(self):
        enu = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        uvw = sim_uv(0.0, 0.0, 2.0, 1.0, enu, 0.0).reshape(2, 3, 3)
        np.testing.assert_allclose(uvw[1, 1, 0], -10.0 * np.cos(np.deg2rad(15.0)))

    def test_cross_correlation_mode_excludes_zero_length_autocorrelations(self):
        enu = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        uvw = sim_uv(
            0.0,
            -30.0,
            1.0,
            1.0,
            enu,
            -30.0,
            include_autocorrelations=False,
        )
        self.assertEqual(uvw.shape, (3, 3))
        self.assertFalse(np.any(np.all(np.isclose(uvw, 0.0), axis=1)))


if __name__ == "__main__":
    unittest.main()
