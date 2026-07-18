import numpy as np


def ecef_to_enu(xyz, longitude_deg, latitude_deg, reference=0):
    """Convert ECEF station coordinates to ENU offsets from one station."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (n_stations, 3)")
    if not 0 <= reference < len(xyz):
        raise IndexError("reference station is outside the coordinate table")

    longitude, latitude = np.deg2rad([longitude_deg, latitude_deg])
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    rotation = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )
    return (xyz - xyz[reference]) @ rotation.T
