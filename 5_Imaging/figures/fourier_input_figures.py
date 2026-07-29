"""Generate the original image inputs and amplitude-phase comparison."""

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
    image = normalized(image)
    plt.imsave(
        FIGURE_DIR / "synthetic_radio_dish_scene.png",
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    return image


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
    image = normalized(image)
    plt.imsave(
        FIGURE_DIR / "synthetic_spiral_galaxy.png",
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    return image


def amplitude_phase_comparison(dish, galaxy):
    fft_dish = np.fft.fftshift(np.fft.fft2(dish))
    fft_galaxy = np.fft.fftshift(np.fft.fft2(galaxy))
    amp_dish, phase_dish = np.abs(fft_dish), np.angle(fft_dish)
    amp_galaxy, phase_galaxy = np.abs(fft_galaxy), np.angle(fft_galaxy)

    items = [
        (
            "phase: galaxy, amplitude: dish",
            np.fft.ifft2(np.fft.ifftshift(amp_dish * np.exp(1j * phase_galaxy))).real,
        ),
        (
            "phase: dish, amplitude: galaxy",
            np.fft.ifft2(np.fft.ifftshift(amp_galaxy * np.exp(1j * phase_dish))).real,
        ),
        (
            "galaxy reconstructed from phase only",
            np.fft.ifft2(np.fft.ifftshift(np.exp(1j * phase_galaxy))).real,
        ),
        (
            "galaxy reconstructed from amplitude only",
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(amp_galaxy))).real,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.0))
    for ax, (title, image) in zip(axes.flat, items):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase_amplitude_role.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    dish = radio_dish_scene()
    galaxy = spiral_galaxy()
    amplitude_phase_comparison(dish, galaxy)


if __name__ == "__main__":
    main()
