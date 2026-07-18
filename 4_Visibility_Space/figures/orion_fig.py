from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NAMES = np.array(
    ["Center", "Betelgeuse", "Rigel", "Bellatrix", "Mintaka", "Alnilam", "Alnitak", "Saiph"]
)
RA_HOURS = np.array(
    [
        5 + 30 / 60,
        5 + 55 / 60 + 10.3053 / 3600,
        5 + 14 / 60 + 32.272 / 3600,
        5 + 25 / 60 + 7.9 / 3600,
        5 + 32 / 60 + 0.4 / 3600,
        5 + 36 / 60 + 12.8 / 3600,
        5 + 40 / 60 + 45.5 / 3600,
        5 + 47 / 60 + 45.4 / 3600,
    ]
)
DEC_DEGREES = np.array(
    [
        0,
        7 + 24 / 60 + 25.426 / 3600,
        -(8 + 12 / 60 + 5.91 / 3600),
        6 + 20 / 60 + 59 / 3600,
        -(17 / 60 + 57 / 3600),
        -(1 + 12 / 60 + 6.9 / 3600),
        -(1 + 56 / 60 + 34 / 3600),
        -(9 + 40 / 60 + 11 / 3600),
    ]
)


def direction_cosines(ra_hours, dec_degrees, phase_center=0):
    """Return l and m relative to one phase centre."""
    ra = np.deg2rad(np.asarray(ra_hours) * 15.0)
    dec = np.deg2rad(np.asarray(dec_degrees))
    delta_ra = ra - ra[phase_center]
    dec_0 = dec[phase_center]
    l = np.cos(dec) * np.sin(delta_ra)
    m = np.sin(dec) * np.cos(dec_0) - np.cos(dec) * np.sin(dec_0) * np.cos(delta_ra)
    return l, m


def draw_orion(output=Path(__file__).with_suffix(".png")):
    """Draw the Orion coordinates used by the visibility-space problem set."""
    fig, ax = plt.subplots(figsize=(5, 7))
    colors = ["tab:red", "tab:cyan"] + ["tab:green"] * 5
    ax.scatter(RA_HOURS[0], DEC_DEGREES[0], c="tab:blue", marker="x", s=55)
    ax.scatter(RA_HOURS[1:], DEC_DEGREES[1:], c=colors)
    ax.annotate("Phase center", (RA_HOURS[0], DEC_DEGREES[0]), xytext=(8, 10), textcoords="offset points")
    label_offsets = {"Mintaka": (-8, 8), "Alnilam": (-8, -4)}
    for name, x, y in zip(NAMES[1:], RA_HOURS[1:], DEC_DEGREES[1:]):
        offset = label_offsets.get(name, (4, 4))
        alignment = "right" if name in label_offsets else "left"
        ax.annotate(
            name,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            horizontalalignment=alignment,
        )
    ax.set(xlim=(5, 6), ylim=(-11, 11), xlabel="Right Ascension [h]", ylabel="Declination [deg]")
    ax.invert_xaxis()
    ax.grid()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    draw_orion()
