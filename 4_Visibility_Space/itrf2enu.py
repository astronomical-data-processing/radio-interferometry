from pathlib import Path

import numpy as np


WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563


def _enu_output_path(path):
    path = Path(path)
    stem = path.name.removesuffix(".itrf.txt")
    if stem == path.name:
        stem = path.stem
    return path.with_name(f"{stem}.enu.txt")


def _geodetic_lat_lon(ecef):
    """Return WGS84 geodetic latitude and longitude for one ECEF position."""
    x, y, z = np.asarray(ecef, dtype=float)
    p = np.hypot(x, y)
    if p == 0 and z == 0:
        raise ValueError("geodetic coordinates are undefined at the Earth's centre")

    b = WGS84_A * (1 - WGS84_F)
    e2 = 1 - (b / WGS84_A) ** 2
    ep2 = (WGS84_A / b) ** 2 - 1
    theta = np.arctan2(z * WGS84_A, p * b)
    lat = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3,
        p - e2 * WGS84_A * np.cos(theta) ** 3,
    )
    return lat, np.arctan2(y, x)


def _ecef_offsets_to_enu(xyz):
    """Rotate ECEF positions to local ENU offsets from the first position."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("xyz must have shape (n, 3)")

    lat, lon = _geodetic_lat_lon(xyz[0])
    rotation = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )
    return (xyz - xyz[0]) @ rotation.T


def convert(fileitrf, save_enu=True, plot=False):
    """Convert an ITRF table to local east/north offsets in kilometres."""
    input_path = Path(fileitrf)
    xyz = np.loadtxt(input_path, comments="#", usecols=(0, 1, 2), ndmin=2)
    enu_m = _ecef_offsets_to_enu(xyz)

    if save_enu:
        np.savetxt(_enu_output_path(input_path), enu_m[:, :2])
    if plot:
        import matplotlib.pyplot as plt

        plt.plot(enu_m[:, 0] / 1e3, enu_m[:, 1] / 1e3, "rx")
    return enu_m[:, 0] / 1e3, enu_m[:, 1] / 1e3
