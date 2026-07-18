"""WGS84 geodetic and Earth-centred, Earth-fixed coordinate conversions."""

import numpy as np


SEMI_MAJOR_AXIS_M = 6_378_137.0
FLATTENING = 1.0 / 298.257223563
SEMI_MINOR_AXIS_M = SEMI_MAJOR_AXIS_M * (1.0 - FLATTENING)
ECCENTRICITY_SQUARED = FLATTENING * (2.0 - FLATTENING)
SECOND_ECCENTRICITY_SQUARED = (
    SEMI_MAJOR_AXIS_M**2 / SEMI_MINOR_AXIS_M**2 - 1.0
)


def geodetic2ecef(latitude, longitude, altitude, degrees=True):
    """Convert WGS84 geodetic coordinates to ECEF metres."""
    latitude, longitude, altitude = np.broadcast_arrays(
        np.asarray(latitude, dtype=float),
        np.asarray(longitude, dtype=float),
        np.asarray(altitude, dtype=float),
    )
    if degrees:
        latitude, longitude = np.deg2rad([latitude, longitude])

    sin_latitude = np.sin(latitude)
    radius = SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - ECCENTRICITY_SQUARED * sin_latitude**2
    )
    x = (radius + altitude) * np.cos(latitude) * np.cos(longitude)
    y = (radius + altitude) * np.cos(latitude) * np.sin(longitude)
    z = (radius * (1.0 - ECCENTRICITY_SQUARED) + altitude) * sin_latitude
    return x, y, z


def ecef2geodetic(x, y, z, degrees=True):
    """Convert ECEF metres to WGS84 latitude, longitude and altitude."""
    x, y, z = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    horizontal_radius = np.hypot(x, y)
    if np.any((horizontal_radius == 0.0) & (z == 0.0)):
        raise ValueError("geodetic coordinates are undefined at the Earth's centre")

    auxiliary = np.arctan2(
        z * SEMI_MAJOR_AXIS_M,
        horizontal_radius * SEMI_MINOR_AXIS_M,
    )
    latitude = np.arctan2(
        z
        + SECOND_ECCENTRICITY_SQUARED
        * SEMI_MINOR_AXIS_M
        * np.sin(auxiliary) ** 3,
        horizontal_radius
        - ECCENTRICITY_SQUARED
        * SEMI_MAJOR_AXIS_M
        * np.cos(auxiliary) ** 3,
    )
    longitude = np.arctan2(y, x)
    radius = SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - ECCENTRICITY_SQUARED * np.sin(latitude) ** 2
    )
    altitude = np.where(
        horizontal_radius > 1e-9,
        horizontal_radius / np.cos(latitude) - radius,
        np.abs(z) - SEMI_MINOR_AXIS_M,
    )

    if degrees:
        latitude, longitude = np.rad2deg([latitude, longitude])
    return latitude, longitude, altitude
