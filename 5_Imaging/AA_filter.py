import numpy as np


class AA_filter:
    """Sample a one-dimensional anti-aliasing convolution kernel."""

    def __init__(self, filter_half_support, filter_oversampling_factor, filter_type):
        if not isinstance(filter_half_support, (int, np.integer)) or filter_half_support < 0:
            raise ValueError("filter half support must be a non-negative integer")
        if (
            not isinstance(filter_oversampling_factor, (int, np.integer))
            or filter_oversampling_factor < 1
        ):
            raise ValueError("filter oversampling factor must be a positive integer")

        self.half_sup = filter_half_support
        self.oversample = filter_oversampling_factor
        self.offsets = np.arange(-self.half_sup, self.half_sup + 1)
        self.full_sup_wo_padding = 2 * filter_half_support + 1
        self.full_sup = self.full_sup_wo_padding + 2
        self.no_taps = (self.full_sup - 1) * self.oversample + 1
        radius = (self.full_sup - 1) / 2
        taps = np.linspace(-radius, radius, self.no_taps)

        if filter_type == "box":
            self.filter_taps = (np.abs(taps) <= 0.5).astype(float)
        elif filter_type == "sinc":
            self.filter_taps = np.sinc(taps)
        elif filter_type == "gaussian_sinc":
            alpha_1 = 1.55
            alpha_2 = 2.52
            self.filter_taps = (
                np.sinc(taps / alpha_1)
                * np.exp(-(taps / alpha_2) ** 2)
                / alpha_1
            )
        else:
            raise ValueError("expected 'box', 'sinc', or 'gaussian_sinc'")

    def sample(self, coordinate):
        """Return the nearest grid point and kernel weights for one coordinate."""
        nearest = int(np.rint(coordinate))
        center = (self.half_sup + 1 + nearest - coordinate) * self.oversample
        indices = self.offsets * self.oversample + int(np.rint(center))
        return nearest, self.filter_taps[indices]
