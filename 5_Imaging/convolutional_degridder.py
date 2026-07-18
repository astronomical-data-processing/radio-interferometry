import numpy as np


def fft_degrid(model_image, uvw, ref_lda, nx, ny, convolution_filter):
    """FFT a model image and interpolate it at irregular UV coordinates."""
    model_image = np.asarray(model_image)
    uvw = np.asarray(uvw)
    ref_lda = np.asarray(ref_lda)
    if model_image.ndim != 3 or model_image.shape[1:] != (ny, nx):
        raise ValueError("model_image must have shape (pol, ny, nx)")
    if uvw.ndim != 2 or uvw.shape[1] != 3:
        raise ValueError("uvw must have shape (row, 3)")
    if ref_lda.ndim != 1 or np.any(ref_lda <= 0):
        raise ValueError("ref_lda must be a one-dimensional array of positive wavelengths")
    if not np.all(np.isfinite(uvw)):
        raise ValueError("uvw must contain only finite coordinates")

    model_grid = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(model_image, axes=(-2, -1))),
        axes=(-2, -1),
    )
    vis = np.zeros((uvw.shape[0], ref_lda.size, model_image.shape[0]), complex)
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

            weights = np.outer(taps_v, taps_u)
            vis[row, channel] = np.sum(
                model_grid[:, positions_v[:, None], positions_u] * weights,
                axis=(-2, -1),
            )

    return vis
