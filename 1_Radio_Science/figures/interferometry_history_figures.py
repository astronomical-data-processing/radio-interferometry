"""Generate original schematics for the interferometry history section."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Polygon, Rectangle


FIGURE_DIR = Path(__file__).resolve().parent
COLORS = {
    "ink": "#243447",
    "blue": "#277da1",
    "green": "#4d908e",
    "gold": "#f9c74f",
    "red": "#d1495b",
    "gray": "#6c757d",
    "light": "#edf2f4",
}


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def double_slit():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set(xlim=(-0.5, 9.3), ylim=(-2.7, 2.7))
    ax.axis("off")
    ax.add_patch(Circle((0.2, 0), 0.14, color=COLORS["gold"]))
    ax.text(0.2, -0.38, "source", ha="center", color=COLORS["ink"])
    for y in (-0.55, 0.55):
        ax.plot([0.35, 3.0], [0, y], color=COLORS["blue"], lw=2)
        ax.plot([3.0, 7.8], [y, 1.15], color=COLORS["blue"], lw=2)
    for y0, y1 in [(-2.4, -0.72), (-0.38, 0.38), (0.72, 2.4)]:
        ax.add_patch(Rectangle((2.92, y0), 0.16, y1 - y0, color=COLORS["ink"]))
    ax.plot([7.8, 7.8], [-2.4, 2.4], color=COLORS["ink"], lw=4)
    y = np.linspace(-2.2, 2.2, 500)
    fringe = 0.75 * np.cos(3.6 * y) ** 2
    ax.plot(7.85 + fringe, y, color=COLORS["red"], lw=2.5)
    ax.scatter([7.8], [1.15], s=45, color=COLORS["red"], zorder=4)
    ax.text(3.0, -2.62, "two slits", ha="center", color=COLORS["ink"])
    ax.text(8.35, -2.62, "screen and intensity", ha="center", color=COLORS["ink"])
    ax.text(5.25, 1.55, "path difference -> phase difference", ha="center", color=COLORS["red"])
    finish(fig, "double_slit_schematic.png")


def stellar_interferometer():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set(xlim=(-4.5, 4.5), ylim=(-0.7, 5.2))
    ax.axis("off")
    for y in (4.0, 4.45):
        ax.plot([-4.0, 4.0], [y, y], color=COLORS["gold"], lw=2)
    ax.text(0, 4.75, "plane wave from a star", ha="center", color=COLORS["ink"])
    collectors = [(-3.1, 2.8), (3.1, 2.8)]
    for x, y in collectors:
        ax.add_patch(Rectangle((x - 0.45, y - 0.08), 0.9, 0.16, angle=12, color=COLORS["blue"]))
        ax.annotate(
            "",
            xy=(x, y + 0.1),
            xytext=(x, 4.0),
            arrowprops={"arrowstyle": "->", "color": COLORS["blue"], "lw": 2},
        )
        ax.plot([x, 0], [y, 0.75], color=COLORS["green"], lw=2.2)
    ax.annotate(
        "",
        xy=collectors[1],
        xytext=collectors[0],
        arrowprops={"arrowstyle": "<->", "color": COLORS["red"], "lw": 1.8},
    )
    ax.text(0, 3.0, "baseline B", ha="center", color=COLORS["red"])
    ax.add_patch(Circle((0, 0.65), 0.23, color=COLORS["red"]))
    ax.text(0, 0.15, "beam combiner / detector", ha="center", color=COLORS["ink"])
    finish(fig, "stellar_interferometer_schematic.png")


def hooker_experiment():
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.set(xlim=(-4.6, 4.6), ylim=(-0.6, 5.4))
    ax.axis("off")
    ax.add_patch(Arc((0, 0.9), 5.8, 2.5, theta1=8, theta2=172, lw=5, color=COLORS["gray"]))
    ax.add_patch(Rectangle((-0.38, 1.55), 0.76, 2.0, color=COLORS["light"], ec=COLORS["ink"], lw=1.5))
    ax.plot([-3.4, 3.4], [4.0, 4.0], color=COLORS["ink"], lw=5)
    ax.scatter([-3.1, 3.1], [4.0, 4.0], s=150, marker="s", color=COLORS["blue"])
    ax.plot([-3.1, -0.2], [4.0, 2.55], color=COLORS["green"], lw=2)
    ax.plot([3.1, 0.2], [4.0, 2.55], color=COLORS["green"], lw=2)
    ax.annotate(
        "",
        xy=(3.1, 4.35),
        xytext=(-3.1, 4.35),
        arrowprops={"arrowstyle": "<->", "color": COLORS["red"], "lw": 1.8},
    )
    ax.text(0, 4.55, "adjustable stellar-interferometer baseline", ha="center", color=COLORS["red"])
    ax.text(0, 0.05, "100-inch telescope", ha="center", color=COLORS["ink"])
    ax.annotate(
        "combined focus",
        xy=(0, 2.35),
        xytext=(1.25, 1.75),
        color=COLORS["ink"],
        arrowprops={"arrowstyle": "->", "color": COLORS["ink"], "lw": 1.2},
    )
    finish(fig, "hooker_interferometer_schematic.png")


def optical_array():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set(xlim=(-5.2, 5.2), ylim=(-1.0, 5.5))
    ax.axis("off")
    ax.scatter(0, 5.0, s=170, marker="*", color=COLORS["gold"])
    ax.text(0, 5.25, "target star", ha="center", color=COLORS["ink"])
    positions = (-4.0, -1.4, 1.4, 4.0)
    for index, x in enumerate(positions, start=1):
        ax.add_patch(Rectangle((x - 0.45, 1.2), 0.9, 1.25, fc=COLORS["light"], ec=COLORS["ink"], lw=1.4))
        ax.add_patch(Arc((x, 2.45), 0.9, 0.8, theta1=0, theta2=180, lw=2.2, color=COLORS["ink"]))
        ax.plot([0, x], [4.9, 2.75], color=COLORS["blue"], lw=1.5)
        ax.plot([x, 0], [1.2, 0.2], color=COLORS["green"], lw=1.8)
        ax.text(x, 0.9, f"UT{index}", ha="center", color=COLORS["ink"])
    ax.add_patch(Rectangle((-1.35, -0.25), 2.7, 0.8, fc="white", ec=COLORS["red"], lw=2))
    ax.text(0, 0.15, "delay lines + beam combiner", ha="center", va="center", color=COLORS["ink"])
    ax.text(0, -0.72, "Geometric delay is equalized before optical interference is detected.", ha="center", color=COLORS["gray"])
    finish(fig, "optical_interferometer_array.png")


def sea_cliff():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set(xlim=(-0.8, 10), ylim=(-0.8, 5.6))
    ax.axis("off")
    ax.fill_between([-0.8, 10], -0.8, 0, color="#9bd3e5")
    cliff = Polygon([[6.0, 0], [10, 0], [10, 3.3], [7.0, 3.3]], color="#8d6e63")
    ax.add_patch(cliff)
    source = (0.2, 5.0)
    antenna = (7.2, 3.65)
    reflection = (4.1, 0)
    ax.scatter(*source, s=160, marker="*", color=COLORS["gold"])
    ax.scatter(*antenna, s=120, marker="^", color=COLORS["ink"])
    ax.plot([source[0], antenna[0]], [source[1], antenna[1]], color=COLORS["blue"], lw=2.4, label="direct path")
    ax.plot(
        [source[0], reflection[0], antenna[0]],
        [source[1], reflection[1], antenna[1]],
        color=COLORS["red"],
        lw=2.4,
        label="sea-reflected path",
    )
    ax.scatter(*reflection, s=45, color=COLORS["red"])
    ax.text(source[0], source[1] + 0.35, "radio source", ha="center", color=COLORS["ink"])
    ax.text(antenna[0] + 0.3, antenna[1] + 0.1, "antenna", color=COLORS["ink"])
    ax.text(4.2, -0.5, "sea", ha="center", color=COLORS["blue"])
    ax.legend(loc="upper center", frameon=False, ncol=2)
    finish(fig, "sea_cliff_schematic.png")


def dish(ax, x, y, scale=1.0):
    ax.add_patch(Arc((x, y), 0.8 * scale, 0.45 * scale, theta1=200, theta2=340, lw=2.5, color=COLORS["ink"]))
    ax.plot([x, x], [y - 0.2 * scale, y - 0.65 * scale], color=COLORS["ink"], lw=2)
    ax.plot([x - 0.25 * scale, x + 0.25 * scale], [y - 0.65 * scale, y - 0.65 * scale], color=COLORS["ink"], lw=2)


def connected_array():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set(xlim=(-0.5, 10.5), ylim=(-0.5, 5.2))
    ax.axis("off")
    positions = [(1.0, 4.2), (1.0, 2.6), (1.0, 1.0), (3.0, 3.4), (3.0, 1.8)]
    for index, (x, y) in enumerate(positions, start=1):
        dish(ax, x, y, 0.9)
        ax.plot([x + 0.4, 5.0], [y - 0.25, 2.5], color=COLORS["blue"], lw=1.5)
        ax.text(x - 0.65, y, f"A{index}", ha="right", va="center", color=COLORS["ink"])
    ax.add_patch(Rectangle((5.0, 1.65), 2.0, 1.7, fc=COLORS["light"], ec=COLORS["red"], lw=2))
    ax.text(6.0, 2.5, "delay model\nchannelizer\ncorrelator", ha="center", va="center", color=COLORS["ink"])
    ax.annotate(
        "",
        xy=(9.2, 2.5),
        xytext=(7.0, 2.5),
        arrowprops={"arrowstyle": "->", "color": COLORS["green"], "lw": 2.5},
    )
    ax.text(9.25, 2.5, r"$V_{pq}(\nu,t)$", va="center", color=COLORS["ink"], fontsize=14)
    ax.text(2.0, 4.85, "independent voltage streams", ha="center", color=COLORS["blue"])
    finish(fig, "connected_array_correlator.png")


def core_and_arms():
    rng = np.random.default_rng(13)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    core_r = 0.85 * np.sqrt(rng.random(42))
    core_phi = 2 * np.pi * rng.random(42)
    ax.scatter(core_r * np.cos(core_phi), core_r * np.sin(core_phi), s=32, color=COLORS["blue"], label="compact core")
    for arm_index, angle in enumerate(np.deg2rad([15, 135, 255])):
        radius = np.linspace(1.0, 4.7, 9)
        jitter = rng.normal(0, 0.06, radius.size)
        x = radius * np.cos(angle) - jitter * np.sin(angle)
        y = radius * np.sin(angle) + jitter * np.cos(angle)
        label = "long-baseline stations" if arm_index == 0 else None
        ax.scatter(x, y, s=38, color=COLORS["red"], label=label)
    ax.plot([-4.8, -2.8], [-4.6, -4.6], color=COLORS["ink"], lw=3)
    ax.text(-3.8, -4.35, "long-baseline scale", ha="center", color=COLORS["ink"])
    ax.set(
        aspect="equal",
        xlim=(-5.2, 5.2),
        ylim=(-5.2, 5.2),
        xlabel="east-west position",
        ylabel="north-south position",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper right")
    finish(fig, "array_core_and_arms.png")


def main():
    double_slit()
    stellar_interferometer()
    hooker_experiment()
    optical_array()
    sea_cliff()
    connected_array()
    core_and_arms()


if __name__ == "__main__":
    main()
