#!/usr/bin/env python3
"""Plot uv samples or their radial distribution from a Measurement Set."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LIGHT_SPEED = 299_792_458.0


def read_measurement_set(path):
    """Return UVW coordinates and channel frequencies from a Measurement Set."""
    try:
        from casacore.tables import table
    except ImportError as error:
        raise ImportError(
            "plotUVcoverage.py requires the optional python-casacore package"
        ) from error

    path = Path(path)
    main = table(str(path), readonly=True, ack=False)
    uvw = main.getcol("UVW")
    main.close()
    spectral_window = table(str(path / "SPECTRAL_WINDOW"), readonly=True, ack=False)
    frequencies = spectral_window.getcol("CHAN_FREQ")[0]
    spectral_window.close()
    return uvw, frequencies


def uv_samples(uvw, frequencies=None):
    """Return conjugate-symmetric uv samples in metres or wavelengths."""
    uv = np.asarray(uvw, dtype=float)[:, :2]
    if frequencies is None:
        samples = uv[np.newaxis, :, :]
    else:
        wavelengths = LIGHT_SPEED / np.asarray(frequencies, dtype=float)
        samples = uv[np.newaxis, :, :] / wavelengths[:, np.newaxis, np.newaxis]
    return np.concatenate((samples, -samples), axis=1).reshape(-1, 2)


def uv_distances(uvw, frequencies=None):
    """Return radial uv distances without duplicating conjugate samples."""
    radius_m = np.linalg.norm(np.asarray(uvw, dtype=float)[:, :2], axis=1)
    if frequencies is None:
        return radius_m
    wavelengths = LIGHT_SPEED / np.asarray(frequencies, dtype=float)
    return (radius_m[np.newaxis, :] / wavelengths[:, np.newaxis]).ravel()


def plot_coverage(uvw, frequencies, include_frequencies=False, histogram=False, limit=None):
    selected_frequencies = frequencies if include_frequencies else None
    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    if histogram:
        ax.hist(uv_distances(uvw, selected_frequencies), bins=50, alpha=0.65)
        ax.set(xlabel=r"uv distance ($\lambda$)" if include_frequencies else "uv distance (m)", ylabel="samples", title="uv distribution")
    else:
        samples = uv_samples(uvw, selected_frequencies)
        ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.25, edgecolors="none")
        unit = r"$\lambda$" if include_frequencies else "m"
        ax.set(xlabel=f"u ({unit})", ylabel=f"v ({unit})", title="uv coverage", aspect="equal")
        ax.grid(alpha=0.3)
        if limit is not None:
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument("-d", "--uvdist", action="store_true", help="plot a radial sample histogram")
    parser.add_argument("-f", "--freqs", action="store_true", help="express coordinates in wavelengths for every channel")
    parser.add_argument("-l", "--limit", type=float, help="set symmetric uv plot limits")
    parser.add_argument("-s", "--savefig", type=Path, help="write the figure instead of showing it")
    return parser.parse_args()


def main():
    args = parse_args()
    uvw, frequencies = read_measurement_set(args.measurement_set)
    fig = plot_coverage(uvw, frequencies, args.freqs, args.uvdist, args.limit)
    if args.savefig:
        fig.savefig(args.savefig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
