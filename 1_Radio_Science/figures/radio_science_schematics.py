"""Generate original radio-science schematics for Chapter 1."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, Polygon, Rectangle


FIGURE_DIR = Path(__file__).resolve().parent
COLORS = {
    "ink": "#243447",
    "blue": "#277da1",
    "green": "#4d908e",
    "gold": "#f9c74f",
    "red": "#d1495b",
    "violet": "#6a4c93",
    "gray": "#6c757d",
    "light": "#edf2f4",
}


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def clean(ax, xlim=(-5, 5), ylim=(-3, 3)):
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax.axis("off")


def galaxy(ax, center, width, height, angle=0, color="gold", alpha=0.7):
    ax.add_patch(
        Ellipse(
            center,
            width,
            height,
            angle=angle,
            fc=COLORS[color],
            ec=COLORS["ink"],
            lw=1.2,
            alpha=alpha,
        )
    )
    ax.add_patch(Circle(center, 0.08 * width, color=COLORS["red"], zorder=3))


def draw_dish(ax, center=(0, 0), scale=1.0, kind="prime", beams=False):
    cx, cy = center
    x = np.linspace(-1, 1, 120)
    y = 0.34 * x**2
    if kind == "offset":
        mask = x > -0.45
        x, y = x[mask] - 0.28, y[mask]
    ax.plot(cx + scale * x, cy + scale * y, color=COLORS["ink"], lw=3)
    ax.plot([cx, cx], [cy, cy - 0.75 * scale], color=COLORS["gray"], lw=3)
    ax.plot(
        [cx - 0.45 * scale, cx + 0.45 * scale],
        [cy - 0.75 * scale, cy - 0.75 * scale],
        color=COLORS["gray"],
        lw=3,
    )

    if kind == "prime":
        feed = (cx, cy + 0.58 * scale)
        ax.scatter(*feed, s=45 * scale, color=COLORS["red"], zorder=4)
        ax.plot([cx - 0.75 * scale, feed[0]], [cy + 0.19 * scale, feed[1]], color=COLORS["blue"], lw=1.4)
        ax.plot([cx + 0.75 * scale, feed[0]], [cy + 0.19 * scale, feed[1]], color=COLORS["blue"], lw=1.4)
    elif kind == "cassegrain":
        ax.add_patch(Arc((cx, cy + 0.58 * scale), 0.38 * scale, 0.18 * scale, theta1=20, theta2=160, lw=2, color=COLORS["red"]))
        ax.scatter(cx, cy + 0.03 * scale, s=35 * scale, color=COLORS["green"], zorder=4)
        ax.plot([cx - 0.72 * scale, cx], [cy + 0.18 * scale, cy + 0.58 * scale], color=COLORS["blue"], lw=1.2)
        ax.plot([cx + 0.72 * scale, cx], [cy + 0.18 * scale, cy + 0.58 * scale], color=COLORS["blue"], lw=1.2)
    elif kind == "offset":
        ax.add_patch(Arc((cx + 0.15 * scale, cy + 0.64 * scale), 0.42 * scale, 0.2 * scale, theta1=15, theta2=160, lw=2, color=COLORS["red"]))
        ax.scatter(cx - 0.3 * scale, cy + 0.06 * scale, s=35 * scale, color=COLORS["green"], zorder=4)
        ax.plot([cx + 0.55 * scale, cx + 0.15 * scale], [cy + 0.11 * scale, cy + 0.64 * scale], color=COLORS["blue"], lw=1.2)
    elif kind == "paf":
        for dx in (-0.14, 0, 0.14):
            for dy in (-0.05, 0.08):
                ax.scatter(cx + dx * scale, cy + (0.56 + dy) * scale, s=22 * scale, color=COLORS["red"])

    if beams:
        for dx, color in [(-0.8, "blue"), (0, "green"), (0.8, "red")]:
            ax.plot([cx, cx + dx * scale], [cy + 0.6 * scale, cy + 2.0 * scale], color=COLORS[color], lw=1.8)


def cygnus_a_optical():
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    clean(ax, (-5, 5), (-3, 3))
    ax.scatter(rng.uniform(-4.7, 4.7, 80), rng.uniform(-2.7, 2.7, 80), s=rng.uniform(3, 18, 80), color=COLORS["gray"], alpha=0.45)
    galaxy(ax, (0, 0), 2.4, 1.0, angle=18)
    ax.text(0, 1.25, "optical host galaxy", ha="center", color=COLORS["ink"])
    ax.text(0, -2.5, "Starlight traces the compact host; extended radio plasma is not shown.", ha="center", color=COLORS["gray"])
    finish(fig, "cygnus_a_optical_schematic.png")


def cygnus_a_radio():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    clean(ax, (-5, 5), (-3, 3))
    galaxy(ax, (0, 0), 1.3, 0.55, angle=18, alpha=0.5)
    for x, sign in [(-2.8, -1), (2.8, 1)]:
        ax.add_patch(Ellipse((x, 0), 2.5, 1.7, fc=COLORS["blue"], ec=COLORS["ink"], alpha=0.35, lw=1.5))
        for width in (1.1, 1.6, 2.1):
            ax.add_patch(Ellipse((x, 0), width, 0.7 * width, fill=False, ec=COLORS["blue"], lw=1.2))
        ax.scatter(x + 0.65 * sign, 0, s=55, color=COLORS["red"], zorder=4)
    ax.plot([-2.2, 2.2], [0, 0], color=COLORS["red"], lw=2.2)
    ax.text(0, 0.35, "jets", ha="center", color=COLORS["red"])
    ax.text(-2.8, 1.2, "radio lobe", ha="center", color=COLORS["blue"])
    ax.text(2.8, 1.2, "radio lobe", ha="center", color=COLORS["blue"])
    ax.text(0, -2.5, "Synchrotron emission traces relativistic particles and magnetic fields.", ha="center", color=COLORS["gray"])
    finish(fig, "cygnus_a_radio_schematic.png")


def interacting_galaxies():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax in axes:
        clean(ax, (-4, 4), (-3, 3))
    centers = [(-2.0, 0.6), (1.1, -0.2), (2.5, 1.25)]
    sizes = [(2.2, 0.8, 25), (3.0, 1.1, -18), (1.2, 0.5, 5)]
    for center, (width, height, angle) in zip(centers, sizes):
        galaxy(axes[0], center, width, height, angle=angle)
        galaxy(axes[1], center, width, height, angle=angle, alpha=0.35)
    axes[0].set_title("Starlight", color=COLORS["ink"])
    t = np.linspace(0, 1, 160)
    axes[1].plot(-2 + 4.5 * t, 0.6 + 0.5 * np.sin(np.pi * t), color=COLORS["blue"], lw=11, alpha=0.28)
    axes[1].plot(-1.8 + 4.6 * t, 0.3 - 1.3 * t + 0.35 * np.sin(2 * np.pi * t), color=COLORS["blue"], lw=7, alpha=0.24)
    axes[1].text(0, 1.9, "H I bridge and tidal tails", ha="center", color=COLORS["blue"])
    axes[1].set_title("21-cm H I", color=COLORS["ink"])
    finish(fig, "interacting_galaxies_hi_schematic.png")


def rotation_curve():
    radius = np.linspace(0.05, 5, 300)
    stellar = 1.75 * radius / (1 + radius**2) ** 0.75
    gas = 0.45 * radius / (1 + (radius / 2.5) ** 2)
    baryonic = np.sqrt(stellar**2 + gas**2)
    halo = 0.95 * np.tanh(radius / 1.4)
    observed = np.sqrt(baryonic**2 + halo**2)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(radius, observed, color=COLORS["red"], lw=3, label="flat outer rotation curve")
    ax.plot(radius, baryonic, color=COLORS["blue"], lw=2.5, ls="--", label="visible matter only")
    ax.axvspan(2.6, 5.0, color=COLORS["light"], label="H I extends beyond bright stellar disk")
    ax.set(xlabel="galactocentric radius (schematic units)", ylabel="circular speed (schematic units)", xlim=(0, 5), ylim=(0, 1.7))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    finish(fig, "rotation_curve_schematic.png")


def atmospheric_windows():
    frequency = np.logspace(-3, 3.5, 1800)
    low_cutoff = 1 / (1 + np.exp(-8 * (np.log10(frequency) + 2)))
    opacity = 0.06 * (frequency / 80) ** 0.7
    for center, strength, width in [(22, 0.25, 0.05), (60, 1.2, 0.07), (118, 0.8, 0.04), (183, 1.7, 0.045), (325, 0.8, 0.04), (557, 1.5, 0.05), (752, 1.2, 0.04)]:
        opacity += strength * np.exp(-0.5 * ((np.log10(frequency) - np.log10(center)) / width) ** 2)
    transmission = low_cutoff * np.exp(-opacity)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.semilogx(frequency, transmission, color=COLORS["blue"], lw=2.6)
    ax.axvspan(1e-3, 1e-2, color=COLORS["violet"], alpha=0.13, label="ionospheric cutoff")
    ax.axvspan(100, 3000, color=COLORS["gold"], alpha=0.14, label="water vapour and molecular lines")
    ax.set(xlabel="frequency (GHz)", ylabel="approximate zenith transmission", xlim=(1e-3, 3000), ylim=(0, 1.05))
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, loc="lower left")
    finish(fig, "atmospheric_radio_windows.png")


def radio_galaxy():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    clean(ax, (-5, 5), (-3, 3))
    galaxy(ax, (0, 0), 1.4, 0.55, angle=12, alpha=0.55)
    ax.plot([-3.2, 3.2], [0.15, -0.15], color=COLORS["red"], lw=2.5)
    for x in (-3.3, 3.3):
        ax.add_patch(Ellipse((x, 0), 2.4, 1.5, fc=COLORS["blue"], ec=COLORS["ink"], alpha=0.32))
        for scale in (0.55, 0.8, 1.05):
            ax.add_patch(Ellipse((x, 0), 2.0 * scale, 1.25 * scale, fill=False, ec=COLORS["blue"], lw=1.2))
    ax.text(0, 0.55, "active nucleus", ha="center", color=COLORS["red"])
    ax.text(0, -2.4, "Double-lobed radio galaxy: core + jets + synchrotron lobes", ha="center", color=COLORS["ink"])
    finish(fig, "radio_galaxy_lobes_schematic.png")


def starburst_radio():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    clean(ax, (-5, 5), (-3, 3.4))
    ax.add_patch(Ellipse((0, 0), 6.5, 1.15, fc=COLORS["gold"], ec=COLORS["ink"], alpha=0.65))
    for x in np.linspace(-2.3, 2.3, 8):
        ax.add_patch(Circle((x, 0.05 * np.sin(2 * x)), 0.16, color=COLORS["red"], alpha=0.8))
    for sign in (-1, 1):
        ax.add_patch(Polygon([(-1.5, 0.2 * sign), (1.5, 0.2 * sign), (0.8, 2.6 * sign), (-0.8, 2.6 * sign)], fc=COLORS["blue"], alpha=0.18, ec=COLORS["blue"]))
    ax.text(-2.4, -0.65, "synchrotron from SNRs", ha="center", color=COLORS["blue"])
    ax.text(2.3, 0.7, "free-free from H II regions", ha="center", color=COLORS["red"])
    ax.text(0, 2.9, "magnetized outflow", ha="center", color=COLORS["blue"])
    finish(fig, "starburst_radio_schematic.png")


def hii_region_freefree():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8))
    for ax in axes:
        clean(ax, (-3, 3), (-2.7, 2.7))
        ax.add_patch(Circle((0, 0), 1.75, fc=COLORS["red"], ec=COLORS["ink"], alpha=0.12))
        ax.scatter(0, 0, marker="*", s=260, color=COLORS["gold"], ec=COLORS["ink"], zorder=4)

    axes[0].add_patch(Rectangle((-2.7, -0.5), 5.4, 1.0, fc=COLORS["gray"], alpha=0.65))
    axes[0].text(0, 2.2, "Optical view", ha="center", color=COLORS["ink"])
    axes[0].text(0, -2.2, "Dust can hide ionized gas", ha="center", color=COLORS["gray"])

    for radius in (0.55, 0.95, 1.35, 1.7):
        axes[1].add_patch(Circle((0, 0), radius, fill=False, ec=COLORS["blue"], lw=1.6))
    axes[1].text(0, 2.2, "Radio free-free view", ha="center", color=COLORS["ink"])
    axes[1].text(0, -2.2, "Contours trace emission measure", ha="center", color=COLORS["blue"])
    finish(fig, "hii_region_freefree_schematic.png")


def dish_figure(name, kind, title, notes, beams=False):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    clean(ax, (-4.5, 4.5), (-2.7, 3.4))
    draw_dish(ax, scale=2.0, kind=kind, beams=beams)
    ax.text(0, 2.9, title, ha="center", color=COLORS["ink"], fontsize=13)
    ax.text(0, -2.15, notes, ha="center", color=COLORS["gray"])
    finish(fig, name)


def low_frequency_dipole():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
    for ax in axes:
        clean(ax, (-3, 3), (-2.5, 2.8))
    for angle in (45, 135):
        dx, dy = 1.4 * np.cos(np.deg2rad(angle)), 1.4 * np.sin(np.deg2rad(angle))
        axes[0].plot([-dx, dx], [-dy, dy], color=COLORS["ink"], lw=5)
    axes[0].scatter(0, 0, s=60, color=COLORS["red"])
    axes[0].text(0, -2.0, "crossed dual-polarization dipole", ha="center", color=COLORS["ink"])
    positions = [(x, y) for x in (-1.7, -0.55, 0.55, 1.7) for y in (-1.2, 0, 1.2)]
    axes[1].scatter(*zip(*positions), marker="x", s=65, color=COLORS["blue"])
    axes[1].add_patch(Ellipse((0, 0), 4.9, 3.7, fill=False, ls="--", ec=COLORS["green"], lw=2))
    axes[1].text(0, -2.0, "many dipoles form one station beam", ha="center", color=COLORS["ink"])
    finish(fig, "low_frequency_dipole_schematic.png")


def layout_figure(name, kind, title):
    rng = np.random.default_rng(9)
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    if kind == "compact":
        radius = 2.4 * np.sqrt(rng.random(14))
        angle = 2 * np.pi * rng.random(14)
        points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    elif kind == "core_arms":
        radius = 1.15 * np.sqrt(rng.random(34))
        angle = 2 * np.pi * rng.random(34)
        points = [*np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))]
        for arm in np.deg2rad((15, 135, 255)):
            r = np.linspace(1.4, 4.7, 7)
            points.extend(np.column_stack((r * np.cos(arm), r * np.sin(arm))))
        points = np.asarray(points)
    elif kind == "y":
        points = []
        for arm in np.deg2rad((90, 210, 330)):
            r = np.geomspace(0.35, 4.7, 9)
            points.extend(np.column_stack((r * np.cos(arm), r * np.sin(arm))))
        points = np.asarray(points)
        for scale, style in [(1.0, "-"), (0.45, "--")]:
            for arm in np.deg2rad((90, 210, 330)):
                ax.plot([0, 4.8 * scale * np.cos(arm)], [0, 4.8 * scale * np.sin(arm)], ls=style, color=COLORS["gray"], alpha=0.45)
    else:
        raise ValueError(kind)
    ax.scatter(points[:, 0], points[:, 1], s=48, color=COLORS["blue"], ec="white", lw=0.5, zorder=3)
    ax.set(aspect="equal", xlim=(-5.2, 5.2), ylim=(-5.2, 5.2), xlabel="east-west position", ylabel="north-south position", title=title)
    ax.grid(alpha=0.18)
    finish(fig, name)


def reflector_scaling():
    diameter = np.logspace(0, 3.1, 300)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for wavelength, label, color in [(0.21, "21 cm", "blue"), (0.03, "3 cm", "red")]:
        resolution = 1.22 * wavelength / diameter * 206265
        ax.loglog(diameter, resolution, lw=2.5, color=COLORS[color], label=label)
    ax.axvspan(100, 1200, color=COLORS["gold"], alpha=0.15, label="large movable structures become difficult")
    ax.set(xlabel="dish diameter D (m)", ylabel="diffraction scale 1.22 lambda / D (arcsec)", xlim=(1, 1200), ylim=(1, 1e5))
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    finish(fig, "reflector_scaling_schematic.png")


def structural_load_paths():
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    clean(ax, (-5, 5), (-3, 3))
    draw_dish(ax, scale=2.0, kind="prime")
    for x in (-1.3, -0.65, 0, 0.65, 1.3):
        ax.annotate("", xy=(x, -0.1 + 0.34 * (x / 2) ** 2), xytext=(x, 1.35), arrowprops={"arrowstyle": "->", "color": COLORS["red"], "lw": 1.8})
    ax.annotate("gravity", xy=(2.8, -0.4), xytext=(2.8, 1.3), ha="center", color=COLORS["red"], arrowprops={"arrowstyle": "->", "color": COLORS["red"], "lw": 2})
    ax.annotate("wind", xy=(-1.0, 0.8), xytext=(-4.1, 0.8), va="center", color=COLORS["blue"], arrowprops={"arrowstyle": "->", "color": COLORS["blue"], "lw": 2})
    ax.text(0, -2.25, "Surface accuracy and load paths become harder to control as D grows.", ha="center", color=COLORS["ink"])
    finish(fig, "structural_load_paths_schematic.png")


def fixed_spherical_reflector():
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    clean(ax, (-5, 5), (-3.2, 3.8))
    ax.add_patch(Arc((0, 1.0), 8.5, 5.5, theta1=200, theta2=340, lw=5, color=COLORS["ink"]))
    ax.plot([-4.1, -2.0, 0], [2.9, 1.6, 2.7], color=COLORS["gray"], lw=2)
    ax.plot([4.1, 2.0, 0], [2.9, 1.6, 2.7], color=COLORS["gray"], lw=2)
    ax.add_patch(Rectangle((-0.55, 2.45), 1.1, 0.45, fc=COLORS["red"], ec=COLORS["ink"]))
    ax.plot([0, -0.8], [2.45, -0.05], color=COLORS["blue"], lw=1.7)
    ax.plot([0, 0.8], [2.45, -0.05], color=COLORS["blue"], lw=1.7)
    ax.text(0, 3.35, "cable-suspended receiver platform", ha="center", color=COLORS["ink"])
    ax.text(0, -2.55, "Fixed spherical reflector: large area, limited steering, demanding cable loads", ha="center", color=COLORS["gray"])
    finish(fig, "fixed_spherical_reflector_schematic.png")


def focal_plane_array():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    for ax in axes:
        clean(ax, (-3, 3), (-2.7, 2.7))
    for x in np.linspace(-1.4, 1.4, 7):
        for y in np.linspace(-1.4, 1.4, 7):
            axes[0].add_patch(Rectangle((x - 0.12, y - 0.12), 0.24, 0.24, fc=COLORS["blue"], ec="white", lw=0.4))
    axes[0].set_title("focal-plane receiver elements")
    for x, y, color in [(-1.1, 0.8, "blue"), (0, 0, "green"), (1.1, -0.8, "red"), (-0.5, -1.0, "gold"), (0.7, 1.0, "violet")]:
        axes[1].add_patch(Ellipse((x, y), 1.6, 1.15, fc=COLORS[color], alpha=0.3, ec=COLORS[color], lw=1.5))
    axes[1].set_title("simultaneous beams on the sky")
    finish(fig, "focal_plane_array_schematic.png")


def aperture_array_station():
    rng = np.random.default_rng(8)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    clean(ax, (-5, 5), (-3.2, 3.4))
    points = rng.uniform([-3.6, -1.8], [0.2, 1.8], (45, 2))
    ax.scatter(points[:, 0], points[:, 1], marker="x", s=38, color=COLORS["blue"])
    for x in (1.0, 1.7, 2.4, 3.1):
        for y in (-1.2, -0.4, 0.4, 1.2):
            ax.add_patch(Rectangle((x - 0.28, y - 0.24), 0.56, 0.48, fc=COLORS["violet"], alpha=0.75))
    ax.annotate("", xy=(0.5, 0), xytext=(-0.4, 0), arrowprops={"arrowstyle": "->", "color": COLORS["red"], "lw": 2.5})
    ax.text(-1.8, 2.45, "low-band dipoles", ha="center", color=COLORS["blue"])
    ax.text(2.05, 2.45, "high-band tiles", ha="center", color=COLORS["violet"])
    ax.text(0, -2.65, "Station beamforming occurs before inter-station correlation.", ha="center", color=COLORS["ink"])
    finish(fig, "aperture_array_station_schematic.png")


def distributed_stations():
    rng = np.random.default_rng(5)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    core = rng.normal(0, 0.38, (28, 2))
    remote = np.array([[-4.1, 2.8], [-3.0, -3.6], [-1.0, 4.3], [2.7, 3.6], [4.2, 0.8], [3.7, -3.1], [0.8, -4.4]])
    ax.scatter(core[:, 0], core[:, 1], s=35, color=COLORS["blue"], label="dense core")
    ax.scatter(remote[:, 0], remote[:, 1], s=65, marker="s", color=COLORS["red"], label="remote stations")
    for point in remote:
        ax.plot([0, point[0]], [0, point[1]], color=COLORS["gray"], alpha=0.35)
    ax.set(aspect="equal", xlim=(-5, 5), ylim=(-5, 5), xlabel="east-west distance", ylabel="north-south distance")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    finish(fig, "distributed_station_layout_schematic.png")


def millimeter_dish_family():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.4))
    labels = [("12 m main array", 1.0), ("7 m compact array", 0.72), ("total-power dish", 1.0)]
    for ax, (label, scale) in zip(axes, labels):
        clean(ax, (-2.2, 2.2), (-2.2, 2.8))
        draw_dish(ax, scale=1.55 * scale, kind="cassegrain")
        ax.text(0, -1.8, label, ha="center", color=COLORS["ink"])
    finish(fig, "millimeter_dish_family_schematic.png")


def millimeter_array_layout():
    rng = np.random.default_rng(12)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    core = rng.normal(0, 0.45, (34, 2))
    compact = rng.normal(0, 0.22, (12, 2))
    arms = []
    for angle in np.deg2rad((15, 135, 255)):
        r = np.linspace(1.2, 4.4, 6)
        arms.extend(np.column_stack((r * np.cos(angle), r * np.sin(angle))))
    arms = np.asarray(arms)
    ax.scatter(core[:, 0], core[:, 1], s=38, color=COLORS["blue"], label="12 m array")
    ax.scatter(compact[:, 0], compact[:, 1], s=48, color=COLORS["gold"], ec=COLORS["ink"], lw=0.4, label="7 m compact array")
    ax.scatter(arms[:, 0], arms[:, 1], s=42, color=COLORS["red"], label="extended configuration")
    ax.scatter([4.2, 4.5], [-4.1, -3.5], marker="s", s=75, color=COLORS["green"], label="total power")
    ax.set(aspect="equal", xlim=(-5, 5), ylim=(-5, 5), xlabel="east-west position", ylabel="north-south position")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper right")
    finish(fig, "millimeter_array_layout_schematic.png")


def vlbi_network():
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    clean(ax, (-5.5, 5.5), (-3.3, 3.3))
    ax.add_patch(Ellipse((0, 0), 9.8, 5.3, fc=COLORS["light"], ec=COLORS["ink"], lw=1.5))
    stations = np.array([[-4.1, 0.7], [-2.4, 1.8], [-0.7, 1.45], [0.7, 1.7], [2.5, 1.1], [4.0, 0.2], [1.1, -1.7], [-1.9, -1.2]])
    for i, first in enumerate(stations):
        for second in stations[i + 1 :]:
            if np.linalg.norm(first - second) > 3.0:
                ax.plot([first[0], second[0]], [first[1], second[1]], color=COLORS["blue"], alpha=0.18, lw=1)
    ax.scatter(stations[:, 0], stations[:, 1], marker="^", s=85, color=COLORS["red"], zorder=4)
    ax.add_patch(Rectangle((-1.0, -2.75), 2.0, 0.55, fc="white", ec=COLORS["ink"], lw=1.4))
    ax.text(0, -2.48, "correlator + delay model", ha="center", va="center", color=COLORS["ink"])
    ax.text(0, 2.75, "independent clocks and recorded voltage streams", ha="center", color=COLORS["ink"])
    finish(fig, "vlbi_network_schematic.png")


def redundant_array():
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    points = []
    for row in range(-4, 5):
        for col in range(-4, 5):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            if x**2 + y**2 < 18:
                points.append((x, y))
    points = np.asarray(points)
    ax.scatter(points[:, 0], points[:, 1], s=70, color=COLORS["blue"], ec="white", lw=0.5)
    ax.plot([points[14, 0], points[15, 0]], [points[14, 1], points[15, 1]], color=COLORS["red"], lw=3, label="repeated baseline vector")
    ax.plot([points[32, 0], points[33, 0]], [points[32, 1], points[33, 1]], color=COLORS["red"], lw=3)
    ax.set(aspect="equal", xlabel="east-west position", ylabel="north-south position")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    finish(fig, "redundant_array_schematic.png")


def cylinder_array():
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    clean(ax, (-5, 5), (-2.8, 3.2))
    for offset in (-3.0, -1.0, 1.0, 3.0):
        ax.add_patch(Arc((offset, -0.2), 1.5, 3.5, theta1=270, theta2=90, lw=5, color=COLORS["blue"]))
        for y in np.linspace(-1.45, 1.05, 7):
            ax.scatter(offset + 0.68, y, s=24, color=COLORS["red"])
    for x in (-3.0, -1.0, 1.0, 3.0):
        ax.plot([x + 0.7, x + 0.7], [1.3, 2.6], color=COLORS["green"], lw=1.5)
    ax.text(0, 2.85, "fixed cylinders + line feeds + digital north-south beams", ha="center", color=COLORS["ink"])
    ax.text(0, -2.35, "Transit instrument optimized for mapping speed and statistical signals", ha="center", color=COLORS["gray"])
    finish(fig, "cylinder_array_schematic.png")


def main():
    cygnus_a_optical()
    cygnus_a_radio()
    interacting_galaxies()
    rotation_curve()
    atmospheric_windows()
    radio_galaxy()
    starburst_radio()
    hii_region_freefree()
    dish_figure("steerable_reflector_schematic.png", "cassegrain", "Steerable reflector antenna", "A continuous aperture measures total power and large angular scales.")
    low_frequency_dipole()
    layout_figure("compact_array_schematic.png", "compact", "Compact connected-element array")
    reflector_scaling()
    structural_load_paths()
    fixed_spherical_reflector()
    dish_figure("prime_focus_dish_schematic.png", "prime", "Prime-focus antenna", "Simple optics; feed and supports can block part of the aperture.")
    layout_figure("core_and_arms_array_schematic.png", "core_arms", "Dense core plus long-baseline arms")
    dish_figure("offset_gregorian_schematic.png", "offset", "Offset Gregorian antenna", "Unblocked aperture and shaped dual-reflector optics.")
    dish_figure("cassegrain_dish_schematic.png", "cassegrain", "Cassegrain antenna", "A secondary reflector returns the beam toward a receiver near the vertex.")
    layout_figure("reconfigurable_y_array_schematic.png", "y", "Reconfigurable Y-shaped array")
    dish_figure("paf_dish_schematic.png", "paf", "Dish with phased-array feed", "Multiple receiver elements synthesize several simultaneous primary beams.", beams=True)
    focal_plane_array()
    aperture_array_station()
    distributed_stations()
    millimeter_dish_family()
    millimeter_array_layout()
    vlbi_network()
    redundant_array()
    cylinder_array()


if __name__ == "__main__":
    main()
