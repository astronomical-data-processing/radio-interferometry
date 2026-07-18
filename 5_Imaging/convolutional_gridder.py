import numpy as np


def grid_ifft(vis, uvw, ref_lda, nx, ny, convolution_filter, weights=None):
    """Grid continuum visibilities and form a dirty image and matching PSF."""
    vis = np.asarray(vis)
    uvw = np.asarray(uvw)
    ref_lda = np.asarray(ref_lda)
    if vis.ndim != 3 or uvw.shape != (vis.shape[0], 3):
        raise ValueError("vis and uvw must have shapes (row, channel, pol) and (row, 3)")
    if ref_lda.shape != (vis.shape[1],) or np.any(ref_lda <= 0):
        raise ValueError("ref_lda must contain one positive wavelength per channel")
    if not np.all(np.isfinite(uvw)):
        raise ValueError("uvw must contain only finite coordinates")
    if weights is None:
        weights = np.ones(vis.shape[:2], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != vis.shape[:2] or np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite, non-negative, and have shape (row, channel)")

    measurement_grid = np.zeros((vis.shape[2], ny, nx), complex)
    sampling_grid = np.zeros((ny, nx), complex)
    offsets = convolution_filter.offsets

    for row, coordinate in enumerate(uvw):
        for channel, wavelength in enumerate(ref_lda):
            u, v = coordinate[:2] / wavelength
            grid_u, taps_u = convolution_filter.sample(u)
            grid_v, taps_v = convolution_filter.sample(v)
            positions_u = grid_u + offsets + nx // 2
            positions_v = grid_v + offsets + ny // 2
            if (
                positions_u[0] < 0
                or positions_u[-1] >= nx
                or positions_v[0] < 0
                or positions_v[-1] >= ny
            ):
                raise ValueError("a convolution footprint extends beyond the UV grid")

            kernel = np.outer(taps_v, taps_u)
            weighted_kernel = weights[row, channel] * np.conj(kernel)
            region = np.ix_(positions_v, positions_u)
            sampling_grid[region] += weighted_kernel
            measurement_grid[:, region[0], region[1]] += (
                vis[row, channel, :, None, None] * weighted_kernel
            )

    dirty = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(measurement_grid, axes=(-2, -1))),
        axes=(-2, -1),
    )
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampling_grid)))
    return dirty, psf
