"""Generate original image inputs for the spatial-frequency examples."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
SIZE = 512


def normalized(image):
    image = image - image.min()
    return image / image.max()


def radio_dish_scene():
    y, x = np.mgrid[-1:1:complex(SIZE), -1:1:complex(SIZE)]
    image = 0.08 + 0.12 * (y + 1) / 2
    image += 0.8 * np.exp(-((x + 0.62) ** 2 + (y + 0.62) ** 2) / 0.002)
    reflector = np.abs(y - (0.28 + 0.48 * x**2)) < 0.018
    reflector &= np.abs(x) < 0.72
    mast = (np.abs(x) < 0.018) & (y > 0.28) & (y < 0.82)
    support = (np.abs(y - 0.82) < 0.018) & (np.abs(x) < 0.34)
    feed = x**2 + (y - 0.02) ** 2 < 0.012
    ground = y > 0.84 + 0.025 * np.sin(10 * x)
    image[reflector | mast | support] = 0.95
    image[feed] = 0.55
    image[ground] = 0.35
    plt.imsave(
        FIGURE_DIR / "synthetic_radio_dish_scene.png",
        normalized(image),
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def spiral_galaxy():
    rng = np.random.default_rng(23)
    y, x = np.mgrid[-1:1:complex(SIZE), -1:1:complex(SIZE)]
    radius = np.hypot(x / 0.82, y)
    angle = np.arctan2(y, x / 0.82)
    disk = np.exp(-3.0 * radius)
    arms = np.exp(1.8 * np.cos(2 * (angle - 3.6 * radius))) * np.exp(-2.2 * radius)
    image = disk + 0.42 * arms + 2.3 * np.exp(-18 * radius**2)
    samples = zip(
        rng.uniform(0.18, 0.82, 45),
        rng.uniform(-np.pi, np.pi, 45),
        rng.uniform(0.15, 0.5, 45),
    )
    for radius_knot, theta, strength in samples:
        theta += 3.6 * radius_knot + rng.normal(0, 0.11)
        x0 = 0.82 * radius_knot * np.cos(theta)
        y0 = radius_knot * np.sin(theta)
        image += strength * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 0.0008)
    image += rng.normal(0, 0.012, image.shape)
    plt.imsave(
        FIGURE_DIR / "synthetic_spiral_galaxy.png",
        normalized(image),
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def main():
    radio_dish_scene()
    spiral_galaxy()


if __name__ == "__main__":
    main()
