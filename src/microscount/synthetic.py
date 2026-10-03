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


def _smooth_field(rng, shape, sigma):
    f = ndi.gaussian_filter(rng.normal(0, 1, shape), sigma, mode="reflect")
    f -= f.mean()
    return f / (np.abs(f).max() + 1e-12)


def sem_surface(shape=(480, 640), porosity: float = 0.05, seed: int = 0, style: str = "cored",
                radius=(2.5, 22.0), shading: float = 0.0, blotches: float = 0.0, gain: float = 1.0,
                offset: float = 0.0, noise: float = 8.0, blur: float = 0.8, figure: bool = False,
                brightness: float = 150.0):
    """SEM-like membrane surface with known pores: (uint8 image, true pore mask).

    Pore radii are log-normal between ``radius`` (pixels); pores do not touch. ``style``:
    ``"flat"`` - uniformly dark pores; ``"cored"`` - as large pores look in secondary-electron
    images: a dark core, often off-centre, inside grey sloping walls, and a bright rim (the edge
    effect); pores under 4 px are dark throughout. The true pore is the whole opening.

    Faults of real micrographs: ``shading`` - a smooth brightness change across the image, as a
    fraction of the brightness; ``blotches`` - dark charging patches, as a fraction; ``gain`` and
    ``offset`` - the exposure; ``noise`` (grey levels) and ``blur`` (pixels); ``figure`` - resized
    to 70% and saved as JPEG, as in a published figure (the mask is resized with it).
    """
    import io
    import math

    from PIL import Image

    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = brightness + 10 * _smooth_field(rng, shape, 3) + 6 * _smooth_field(rng, shape, 12)
    truth = np.zeros(shape, bool)
    rmin, rmax = radius
    mu, sig = math.log(math.sqrt(rmin * rmax)), math.log(rmax / rmin) / 4
    placed: list[tuple[float, float, float]] = []
    area, tries = 0.0, 0
    while area < porosity * h * w and tries < 200000:
        tries += 1
        r = float(np.clip(rng.lognormal(mu, sig), rmin, rmax))
        rim = max(1.5, 0.15 * r)
        y, x = rng.uniform(-r / 2, h + r / 2), rng.uniform(-r / 2, w + r / 2)
        if placed:
            p = np.asarray(placed)
            if np.any(np.hypot(p[:, 0] - y, p[:, 1] - x) < p[:, 2] + r + rim + 2):
                continue
        placed.append((y, x, r + rim))
        y0, y1 = int(max(0, y - r - rim - 2)), int(min(h, y + r + rim + 3))
        x0, x1 = int(max(0, x - r - rim - 2)), int(min(w, x + r + rim + 3))
        if y0 >= y1 or x0 >= x1:
            continue
        d = np.hypot(yy[y0:y1, x0:x1] - y, xx[y0:y1, x0:x1] - x)
        sub = img[y0:y1, x0:x1]
        inside = d < r
        dark = rng.uniform(20, 45)
        if style == "flat" or r < 4:
            prof = np.full(d.shape, dark)
        else:
            c = rng.uniform(0.5, 0.9) if r < 6 else rng.uniform(0.15, 0.6)  # core share of the radius
            wall = brightness * rng.uniform(0.6, 0.85)
            ang, off = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.6) * (1 - c) * r
            dc = np.hypot(yy[y0:y1, x0:x1] - y - off * np.sin(ang), xx[y0:y1, x0:x1] - x - off * np.cos(ang))
            t = np.clip((dc - c * r) / max(1.0, (1 - c) * r), 0, 1)
            prof = dark + (wall - dark) * t * t * (3 - 2 * t)
            ring = (d >= r) & (d < r + rim)
            sub[ring] += rng.uniform(20, 45) * (1 - (d[ring] - r) / rim)
        sub[inside] = prof[inside] + 4 * rng.normal(0, 1, int(inside.sum()))
        truth[y0:y1, x0:x1] |= inside
        area += float(inside.sum())
    if shading:
        s = _smooth_field(rng, shape, 0.2 * min(h, w))
        grad = (xx / w - 0.5) * rng.uniform(-1, 1) + (yy / h - 0.5) * rng.uniform(-1, 1)
        s = 0.7 * s + 0.3 * grad / (np.abs(grad).max() + 1e-12)
        img *= 1 + shading * s / (np.abs(s).max() + 1e-12)
    if blotches:
        f = ndi.gaussian_filter(rng.normal(0, 1, shape), 0.08 * min(h, w))
        m = ndi.gaussian_filter((f > np.percentile(f, 72)).astype(float), 0.02 * min(h, w))
        img *= 1 - blotches * m
    if blur:
        img = ndi.gaussian_filter(img, blur)
    img = gain * img + offset + rng.normal(0, noise, shape)
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    if figure:
        size = (int(w * 0.7), int(h * 0.7))
        buf = io.BytesIO()
        Image.fromarray(img).resize(size, Image.BILINEAR).save(buf, "JPEG", quality=75)
        img = np.asarray(Image.open(io.BytesIO(buf.getvalue())))
        truth = np.asarray(Image.fromarray(truth.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) >= 128
    return img, truth
