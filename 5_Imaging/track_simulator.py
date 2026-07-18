import numpy as np


def sim_uv(
    hour_angle_start,
    ref_dec,
    observation_length_in_hrs,
    integration_length,
    enu_coords,
    latitude,
    plot_on=False,
    same_scales_plot=False,
    plot_channel=299_792_458.0 / 1e9,
    include_autocorrelations=True,
):
    """Simulate UVW coordinates from an ENU layout and hour angle in degrees."""
    enu_coords = np.asarray(enu_coords, dtype=float)
    if enu_coords.ndim != 2 or enu_coords.shape[1] != 3:
        raise ValueError("enu_coords must have shape (n_antennas, 3)")
    if enu_coords.shape[0] == 0:
        raise ValueError("enu_coords must contain at least one antenna")
    if integration_length <= 0 or observation_length_in_hrs < integration_length:
        raise ValueError("observation length must include at least one positive integration")

    diagonal = 0 if include_autocorrelations else 1
    ant_p, ant_q = np.triu_indices(enu_coords.shape[0], k=diagonal)
    east, north, up = (enu_coords[ant_p] - enu_coords[ant_q]).T
    latitude = np.deg2rad(latitude)
    xyz = np.column_stack(
        (
            np.cos(latitude) * up - np.sin(latitude) * north,
            east,
            np.sin(latitude) * up + np.cos(latitude) * north,
        )
    )

    n_times = int(observation_length_in_hrs / integration_length)
    hour_angle = np.deg2rad(
        hour_angle_start + 15.0 * integration_length * np.arange(n_times)
    )
    declination = np.deg2rad(ref_dec)
    sin_h, cos_h = np.sin(hour_angle)[:, None], np.cos(hour_angle)[:, None]
    x, y, z = xyz.T
    uvw = np.stack(
        (
            -sin_h * x + cos_h * y,
            -np.sin(declination) * cos_h * x
            - np.sin(declination) * sin_h * y
            + np.cos(declination) * z,
            np.cos(declination) * cos_h * x
            + np.cos(declination) * sin_h * y
            + np.sin(declination) * z,
        ),
        axis=-1,
    ).reshape(-1, 3)

    if plot_on:
        from matplotlib import pyplot as plt

        hrs = int(observation_length_in_hrs)
        mins = int(observation_length_in_hrs * 60 - hrs * 60)
        scaled_uv = uvw[:, :2] / plot_channel / 1e3
        plt.figure(figsize=(8, 8))
        plt.title(
            f"UV COVERAGE ({hrs:d}h:{mins:d}m @ HA0={hour_angle_start:f}, DEC={ref_dec:f})"
        )
        plt.plot(scaled_uv[:, 0], scaled_uv[:, 1], "r.", label="Baselines")
        plt.plot(-scaled_uv[:, 0], -scaled_uv[:, 1], "b.", label="Conjugate Baselines")
        plt.xlabel(r"u ($k\lambda$)")
        plt.ylabel(r"v ($k\lambda$)")
        plt.legend(bbox_to_anchor=(1.75, 1.0))
        if same_scales_plot:
            max_value = np.max(np.abs(scaled_uv))
            plt.xlim([-max_value, max_value])
            plt.ylim([-max_value, max_value])
        plt.show()
    return uvw
