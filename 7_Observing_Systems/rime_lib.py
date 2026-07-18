import sys
from pathlib import Path

import numpy as np


# Chapter directories are not Python packages, so resolve the shared simulator
# relative to this file rather than the caller's working directory.
IMAGING_DIR = Path(__file__).resolve().parents[1] / "5_Imaging"
if str(IMAGING_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGING_DIR))

from track_simulator import sim_uv


C = 299_792_458.0

ALL_SLICE = slice(None)
SRC_SLICE = (np.newaxis, ALL_SLICE, np.newaxis, np.newaxis, np.newaxis)
TIME_SLICE = (np.newaxis, np.newaxis, ALL_SLICE, np.newaxis, np.newaxis)
ANT_SLICE = (ALL_SLICE, np.newaxis, ALL_SLICE, ALL_SLICE, np.newaxis)
CHAN_SLICE = (np.newaxis, np.newaxis, np.newaxis, np.newaxis, ALL_SLICE)


def ap_index(nsrc=0, ntime=1, na=1, nchan=0):
    """Return indices that map per-antenna arrays onto upper-triangle baselines.

    For an input with shape ``(nsrc, ntime, na, nchan)``, the returned tuple
    selects an array with shape ``(2, nsrc, ntime, nbl, nchan)``. Source and
    channel axes are omitted when their corresponding sizes are zero.
    """
    if ntime < 1:
        raise ValueError("ntime must be at least 1")
    if na < 1:
        raise ValueError("na must be at least 1")
    if nsrc < 0 or nchan < 0:
        raise ValueError("nsrc and nchan cannot be negative")

    needed = (True, nsrc > 0, True, True, nchan > 0)
    nbl = na * (na + 1) // 2
    pairs = np.asarray(np.triu_indices(na), dtype=np.intp)
    antenna_pairs = np.broadcast_to(pairs[:, np.newaxis, :], (2, ntime, nbl))
    indices = []

    if nsrc > 0:
        source_slice = tuple(s for s, keep in zip(SRC_SLICE, needed) if keep)
        indices.append(np.arange(nsrc)[source_slice])

    time_slice = tuple(s for s, keep in zip(TIME_SLICE, needed) if keep)
    indices.append(np.arange(ntime)[time_slice])

    antenna_slice = tuple(s for s, keep in zip(ANT_SLICE, needed) if keep)
    indices.append(antenna_pairs[antenna_slice])

    if nchan > 0:
        channel_slice = tuple(s for s, keep in zip(CHAN_SLICE, needed) if keep)
        indices.append(np.arange(nchan)[channel_slice])

    return tuple(indices)


def brightness(I, Q, U, V):
    """Return linear-basis brightness matrices for arrays of Stokes values.

    This chapter uses ``B = 1/2 [[I+Q, U+iV], [U-iV, I-Q]]``. Consequently,
    an unpolarized source of total flux density ``I`` contributes ``I/2`` to
    each ideal parallel-hand correlation.
    """
    stokes = [np.asarray(component) for component in (I, Q, U, V)]
    if any(component.ndim != 1 for component in stokes):
        raise ValueError("Stokes parameters must be one-dimensional")
    if any(component.shape != stokes[0].shape for component in stokes[1:]):
        raise ValueError("I, Q, U, and V must have the same shape")
    if any(np.iscomplexobj(component) for component in stokes):
        raise ValueError("Stokes parameters must be real")

    I, Q, U, V = (component.astype(float, copy=False) for component in stokes)
    if not all(np.isfinite(component).all() for component in (I, Q, U, V)):
        raise ValueError("Stokes parameters must be finite")

    result = np.empty((I.size, 2, 2), dtype=np.complex128)
    result[:, 0, 0] = 0.5 * (I + Q)
    result[:, 0, 1] = 0.5 * (U + 1j * V)
    result[:, 1, 0] = 0.5 * (U - 1j * V)
    result[:, 1, 1] = 0.5 * (I - Q)
    return result


def lm_2_rad(ra, dec, phase_centre=None):
    """Convert equatorial coordinates in degrees to direction cosines ``l,m``.

    The returned values are dimensionless direction cosines, despite this
    historical function name. By default the first coordinate is the phase
    centre. Pass ``phase_centre=(ra0, dec0)`` in degrees to set it explicitly.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if ra.ndim != 1 or dec.ndim != 1:
        raise ValueError("ra and dec must be one-dimensional")
    if ra.shape != dec.shape:
        raise ValueError("ra and dec must have the same shape")
    if ra.size == 0:
        raise ValueError("ra and dec must contain at least one source")
    if not np.isfinite(ra).all() or not np.isfinite(dec).all():
        raise ValueError("ra and dec must be finite")
    if np.any((dec < -90.0) | (dec > 90.0)):
        raise ValueError("declinations must lie between -90 and 90 degrees")

    if phase_centre is None:
        ra0, dec0 = ra[0], dec[0]
    else:
        centre = np.asarray(phase_centre, dtype=float)
        if centre.shape != (2,) or not np.isfinite(centre).all():
            raise ValueError("phase_centre must be a finite (ra, dec) pair")
        ra0, dec0 = centre
        if not -90.0 <= dec0 <= 90.0:
            raise ValueError("phase-centre declination must be between -90 and 90")

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    delta_ra = ra_rad - np.deg2rad(ra0)
    dec0_rad = np.deg2rad(dec0)

    l = np.cos(dec_rad) * np.sin(delta_ra)
    m = (
        np.sin(dec_rad) * np.cos(dec0_rad)
        - np.cos(dec_rad) * np.sin(dec0_rad) * np.cos(delta_ra)
    )
    return np.column_stack((l, m))


def phase(lm, uvw, frequency):
    """Compute per-source, per-antenna geometric phase terms.

    ``lm`` contains dimensionless direction cosines, ``uvw`` is in metres,
    and ``frequency`` is in Hz. The result has shape
    ``(nsrc, ntime, na, nchan)`` and uses the negative-exponent convention.
    """
    lm = np.asarray(lm, dtype=float)
    uvw = np.asarray(uvw, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    if lm.ndim != 2 or lm.shape[1] != 2:
        raise ValueError("lm must have shape (nsrc, 2)")
    if uvw.ndim != 3 or uvw.shape[2] != 3:
        raise ValueError("uvw must have shape (ntime, na, 3)")
    if frequency.ndim != 1:
        raise ValueError("frequency must have shape (nchan,)")
    if not all(np.isfinite(value).all() for value in (lm, uvw, frequency)):
        raise ValueError("lm, uvw, and frequency must be finite")

    radius_squared = np.sum(lm**2, axis=1)
    if np.any(radius_squared > 1.0 + 1e-12):
        raise ValueError("direction cosines must satisfy l**2 + m**2 <= 1")

    l = lm[:, 0, np.newaxis, np.newaxis]
    m = lm[:, 1, np.newaxis, np.newaxis]
    n_minus_one = (
        np.sqrt(np.clip(1.0 - radius_squared, 0.0, None)) - 1.0
    )[:, np.newaxis, np.newaxis]
    geometric_delay = (
        l * uvw[np.newaxis, :, :, 0]
        + m * uvw[np.newaxis, :, :, 1]
        + n_minus_one * uvw[np.newaxis, :, :, 2]
    )
    return np.exp(
        -2j
        * np.pi
        * geometric_delay[:, :, :, np.newaxis]
        * frequency[np.newaxis, np.newaxis, np.newaxis, :]
        / C
    )


def dec_degrees(degree_str):
    """Convert ``DD:MM:SS.SS`` to decimal degrees."""
    parts = degree_str.strip().split(":")
    if len(parts) != 3:
        raise ValueError("angle must use DD:MM:SS.SS format")
    degree, minute, second = (float(value) for value in parts)
    if not 0.0 <= minute < 60.0 or not 0.0 <= second < 60.0:
        raise ValueError("minutes and seconds must lie in [0, 60)")
    sign = -1.0 if degree_str.strip().startswith("-") else 1.0
    return sign * (abs(degree) + minute / 60.0 + second / 3600.0)


KAT7_location = [
    dec_degrees("-30:43:17.34"),
    dec_degrees("21:24:38.46"),
    1038.0,
]

# East, North, Up offsets from the nominal array location, in metres.
KAT7_ants = np.array(
    [
        [25.095, -9.095, 0.045],
        [90.284, 26.380, -0.226],
        [3.985, 26.893, 0.000],
        [-21.605, 25.494, 0.019],
        [-38.272, -2.592, 0.391],
        [-61.595, -79.699, 0.702],
        [-87.988, 75.754, 0.138],
    ],
    dtype=np.float64,
)


def KAT7_antenna_uvw(hour_angle_start=60, ref_dec=45):
    """Return KAT-7 antenna UVW coordinates relative to antenna zero."""
    baseline_uvw = sim_uv(
        hour_angle_start=hour_angle_start,
        ref_dec=ref_dec,
        observation_length_in_hrs=12,
        integration_length=3,
        enu_coords=KAT7_ants,
        latitude=KAT7_location[0],
    )

    na = KAT7_ants.shape[0]
    nbl = na * (na + 1) // 2
    if baseline_uvw.shape[0] % nbl:
        raise RuntimeError("sim_uv returned an incomplete time sample")
    ntime = baseline_uvw.shape[0] // nbl
    baseline_uvw = baseline_uvw.reshape(ntime, nbl, 3)

    # The first na upper-triangle baselines are (0,0), (0,1), ..., (0,na-1).
    antenna_uvw = -baseline_uvw[:, :na, :]
    mapped = antenna_uvw[ap_index(ntime=ntime, na=na)]
    if not np.allclose(mapped[0] - mapped[1], baseline_uvw):
        raise RuntimeError("antenna coordinates do not reconstruct the baselines")
    return antenna_uvw


def rime(ant_uvw, sources, frequencies, phase_centre=None, verbose=False):
    """Predict ideal point-source visibility matrices without Jones corruptions.

    ``sources`` has columns ``[ra_deg, dec_deg, I, Q, U, V]``. By default its
    first row defines the phase centre; an explicit ``(ra_deg, dec_deg)`` pair
    may be supplied instead. Baselines use ``b_pq = r_p - r_q`` and include
    autocorrelations. The output shape is ``(ntime, nbl, nchan, 2, 2)``.
    """
    ant_uvw = np.asarray(ant_uvw, dtype=float)
    sources = np.asarray(sources)
    frequencies = np.asarray(frequencies, dtype=float)
    if ant_uvw.ndim != 3 or ant_uvw.shape[2] != 3:
        raise ValueError("ant_uvw must have shape (ntime, na, 3)")
    if sources.ndim != 2 or sources.shape[1] != 6 or sources.shape[0] == 0:
        raise ValueError("sources must have shape (nsrc, 6) with nsrc >= 1")
    if frequencies.ndim != 1:
        raise ValueError("frequencies must have shape (nchan,)")
    if not np.isrealobj(sources):
        raise ValueError("source coordinates and Stokes parameters must be real")
    sources = sources.astype(float, copy=False)
    if not all(np.isfinite(value).all() for value in (ant_uvw, sources, frequencies)):
        raise ValueError("all RIME inputs must be finite")

    ra, dec, I, Q, U, V = sources.T
    lm = lm_2_rad(ra, dec, phase_centre=phase_centre)
    source_brightness = brightness(I, Q, U, V)
    antenna_phase = phase(lm, ant_uvw, frequencies)

    na = ant_uvw.shape[1]
    ant_p, ant_q = np.triu_indices(na)
    baseline_phase = antenna_phase[:, :, ant_p] * np.conj(
        antenna_phase[:, :, ant_q]
    )
    visibilities = np.einsum(
        "stbc,sij->tbcij", baseline_phase, source_brightness, optimize=True
    )

    if verbose:
        ntime, nbl, nchan = visibilities.shape[:3]
        print(
            f"RIME dimensions: nsrc={sources.shape[0]}, ntime={ntime}, "
            f"na={na}, nbl={nbl}, nchan={nchan}"
        )
    return visibilities
