"""Synthetic images with known ground truth (self-test and unit tests)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def translocation_field(
    shape=(512, 512),
    ratio: float = 2.0,
    cytoplasm: float = 40.0,
    background: float = 6.0,
    nuclear_stain: float = 90.0,
    nucleus_radius: float = 9.0,
    cell_radius: float = 26.0,
    spacing: int = 64,
    noise: float = 2.0,
    blur: float = 1.0,
    seed: int = 0,
):
    """Two uint8 channels of a cell field and the true background-subtracted N/C ratio.

    Nuclei are ellipses inside larger round cell bodies on a jittered grid.
    Target intensity: background outside cells, ``cytoplasm`` in cell bodies and
    ``background + ratio * (cytoplasm - background)`` in nuclei, so that the true
    background-subtracted ratio is exactly ``ratio``.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    nuc = np.zeros(shape, bool)
    cell = np.zeros(shape, bool)
    centres = []
    for cy in range(spacing // 2, h - spacing // 4, spacing):
        for cx in range(spacing // 2, w - spacing // 4, spacing):
            y = cy + rng.uniform(-6, 6)
            x = cx + rng.uniform(-6, 6)
            a = nucleus_radius * rng.uniform(1.0, 1.3)
            b = nucleus_radius * rng.uniform(0.8, 1.0)
            th = rng.uniform(0, np.pi)
            dy, dx = yy - y, xx - x
            u = dx * np.cos(th) + dy * np.sin(th)
            v = -dx * np.sin(th) + dy * np.cos(th)
            nuc |= (u / a) ** 2 + (v / b) ** 2 <= 1
            r = cell_radius * rng.uniform(0.85, 1.1)
            cell |= dy**2 + dx**2 <= r**2
            centres.append((y, x))
    cyt_level = cytoplasm
    nuc_level = background + ratio * (cytoplasm - background)
    t = np.full(shape, background, np.float64)
    t[cell] = cyt_level
    t[nuc] = nuc_level
    n = np.full(shape, 3.0)
    n[nuc] = nuclear_stain
    if blur:
        t = ndi.gaussian_filter(t, blur)
        n = ndi.gaussian_filter(n, blur)
    t += rng.normal(0, noise, shape)
    n += rng.normal(0, noise, shape)
    return (
        np.clip(np.round(n), 0, 255).astype(np.uint8),
        np.clip(np.round(t), 0, 255).astype(np.uint8),
        {"ratio": ratio, "n_cells": len(centres), "background": background},
    )


def sem_image(shape=(300, 400), porosity: float = 0.15, seed: int = 0):
    """Grey SEM-like image: bright textured solid with dark round pores."""
    rng = np.random.default_rng(seed)
    h, w = shape
    solid = np.ones(shape, bool)
    yy, xx = np.mgrid[0:h, 0:w]
    while (~solid).mean() < porosity:
        r = rng.uniform(2.5, 7.0)
        y, x = rng.uniform(0, h), rng.uniform(0, w)
        solid &= (yy - y) ** 2 + (xx - x) ** 2 > r * r
    img = np.where(solid, 170.0, 45.0)
    img += ndi.gaussian_filter(rng.normal(0, 25, shape), 1.5)
    img = ndi.gaussian_filter(img, 0.7)
    return np.clip(np.round(img), 0, 255).astype(np.uint8), float((~solid).mean())
