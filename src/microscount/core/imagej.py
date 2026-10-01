"""ImageJ routines used by the lab protocol, reimplemented from ImageJ's source.

* ``subtract_background`` - *Process > Subtract Background...* with the rolling-ball
  algorithm (``ij.plugin.filter.BackgroundSubtracter``, ImageJ 1.54), dialog defaults:
  dark background, no sliding paraboloid, smoothing enabled. The image is pre-smoothed
  with ImageJ's 3x3 mean (float32), shrunk by block minima, a trimmed ball patch is rolled
  over it (a grey opening with a non-flat structuring element), the background is
  enlarged back with ImageJ's bilinear interpolation, and subtracted with rounding and
  clipping exactly as ImageJ does for 8- and 16-bit images.
* ``traced_perimeter`` / ``circularity`` - *Analyze Particles*: particle outlines are
  traced with the legacy 8-connected ``Wand`` and measured with
  ``PolygonRoi.getTracedPerimeter``; circularity = 4 pi area / perimeter^2 (capped at 1).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

F32 = np.float32
THIRD = np.float32(0.333333333)  # the float literal ImageJ uses in filter3x3


# ---------------------------------------------------------------------- rolling ball


def _mean3_lines(a: np.ndarray, axis: int) -> np.ndarray:
    """ImageJ BackgroundSubtracter.filter3 (MEAN) along one axis, edges replicated."""
    a = np.moveaxis(a, axis, -1)
    left = np.concatenate([a[..., :1], a[..., :-1]], axis=-1)
    right = np.concatenate([a[..., 1:], a[..., -1:]], axis=-1)
    out = ((left + a) + right) * THIRD  # float32 arithmetic, same order as Java
    return np.moveaxis(out.astype(F32), -1, axis)


def _ball(radius: float) -> tuple[np.ndarray, int]:
    """RollingBall: patch heights and the shrink factor."""
    if radius <= 10:
        shrink, trim = 1, 24
    elif radius <= 30:
        shrink, trim = 2, 24
    elif radius <= 100:
        shrink, trim = 4, 32
    else:
        shrink, trim = 8, 40
    small_r = radius / shrink
    if small_r < 1:
        small_r = 1.0
    rsq = small_r * small_r
    xtrim = int(trim * small_r) // 100
    half = int(math.floor(small_r - xtrim + 0.5))  # Java Math.round
    w = 2 * half + 1
    yy, xx = np.mgrid[0:w, 0:w] - half
    temp = rsq - xx * xx - yy * yy
    data = np.where(temp > 0, np.sqrt(np.maximum(temp, 0)), 0.0).astype(F32)
    return data, shrink


def _shrink(img: np.ndarray, f: int) -> np.ndarray:
    h, w = img.shape
    sh, sw = (h + f - 1) // f, (w + f - 1) // f
    pad = np.full((sh * f, sw * f), np.float32(np.finfo(np.float32).max), F32)
    pad[:h, :w] = img
    return pad.reshape(sh, f, sw, f).min(axis=(1, 3))


def _roll_ball(small: np.ndarray, ball: np.ndarray) -> np.ndarray:
    """Grey opening with the non-flat ball, ball centres allowed up to one radius outside."""
    r = ball.shape[0] // 2
    big = np.float32(np.finfo(np.float32).max)
    padded = np.pad(small, r, mode="constant", constant_values=big)
    # erosion: z(c) = min over the patch of (image - ball); positions whose patch lies
    # wholly outside never occur (every centre keeps at least one image pixel)
    z = ndi.grey_erosion(padded, structure=ball, mode="constant", cval=big)
    z = z.astype(F32)
    # dilation back: background(p) = max over centres of (z + ball)
    lowest = np.float32(-np.finfo(np.float32).max)
    bg = ndi.grey_dilation(z, structure=ball, mode="constant", cval=lowest).astype(F32)
    return bg[r:-r, r:-r] if r else bg


def _interp_arrays(length: int, small_len: int, f: int):
    idx = np.empty(length, np.int64)
    wts = np.empty(length, F32)
    for i in range(length):
        v = i - f // 2
        si = int(v / f)  # Java integer division truncates towards zero
        if si >= small_len - 1:
            si = small_len - 2
        idx[i] = si
        distance = np.float32(np.float32(i + np.float32(0.5)) / np.float32(f)) - np.float32(si + np.float32(0.5))
        wts[i] = np.float32(1.0) - np.float32(distance)
    return idx, wts


def _enlarge(small: np.ndarray, shape: tuple[int, int], f: int) -> np.ndarray:
    h, w = shape
    sh, sw = small.shape
    xi, xw = _interp_arrays(w, sw, f)
    yi, yw = _interp_arrays(h, sh, f)
    lines = small[:, xi] * xw[None, :] + small[:, xi + 1] * (np.float32(1) - xw)[None, :]
    lines = lines.astype(F32)
    out = lines[yi, :] * yw[:, None] + lines[yi + 1, :] * (np.float32(1) - yw)[:, None]
    return out.astype(F32)


def rolling_ball_background(img: np.ndarray, radius: float = 50.0, smooth: bool = True,
                            light_background: bool = False) -> np.ndarray:
    """Background estimated by ImageJ's rolling ball (float32 array)."""
    fp = np.asarray(img, dtype=F32).copy()
    if light_background:
        fp = -fp
    if smooth:
        fp = _mean3_lines(fp, 1)  # rows first, then columns (ImageJ order)
        fp = _mean3_lines(fp, 0)
    ball, f = _ball(float(radius))
    small = _shrink(fp, f) if f > 1 else fp
    small = _roll_ball(small, ball)
    bg = _enlarge(small, fp.shape, f) if f > 1 else small
    return -bg if light_background else bg


def subtract_background(img: np.ndarray, radius: float = 50.0, smooth: bool = True) -> np.ndarray:
    """*Process > Subtract Background* (rolling ball, dark background) with ImageJ's rounding."""
    a = np.asarray(img)
    bg = rolling_ball_background(a, radius, smooth)
    if a.dtype == np.uint8:
        v = (a.astype(F32) - bg) + np.float32(0.5)
        return np.floor(np.clip(v, 0, 255)).astype(np.uint8)
    if a.dtype == np.uint16:
        v = (a.astype(F32) - bg) + np.float32(0.5)
        return np.floor(np.clip(v, 0, 65535)).astype(np.uint16)
    return (a.astype(F32) - bg).astype(F32)


# ---------------------------------------------------------------------- particle outlines


def trace_outline(mask: np.ndarray, x0: int, y0: int) -> tuple[list[int], list[int]]:
    """ImageJ Wand, legacy thresholded mode (8-connected), started as Analyze Particles does.

    ``(x0, y0)`` is the particle's first pixel in raster order. Returns polygon vertices.
    """
    h, w = mask.shape

    def inside(x, y):
        return 0 <= x < w and 0 <= y < h and bool(mask[y, x])

    def inside_dir(x, y, d):
        d &= 3
        if d == 0:
            return inside(x, y)
        if d == 1:
            return inside(x, y - 1)
        if d == 2:
            return inside(x - 1, y - 1)
        return inside(x - 1, y)

    x, y = x0, y0
    while inside(x, y):  # find the border to the right of the seed run
        x += 1
    start_x, start_y = x, y
    if inside(start_x, start_y):
        start_dir = 1
    else:
        start_dir = 3
        start_y += 1
    x, y, direction = start_x, start_y, start_dir
    xs: list[int] = []
    ys: list[int] = []
    while True:
        new_dir = direction + 1
        while True:
            if inside_dir(x, y, new_dir):
                break
            new_dir -= 1
            if new_dir < direction:
                break
        if new_dir != direction:
            xs.append(x)
            ys.append(y)
        d = new_dir & 3
        if d == 0:
            x += 1
        elif d == 1:
            y -= 1
        elif d == 2:
            x -= 1
        else:
            y += 1
        direction = new_dir
        if x == start_x and y == start_y and (direction & 3) == start_dir:
            break
    if xs and xs[0] != x:
        xs.append(x)
        ys.append(y)
    return xs, ys


def traced_perimeter(xs: list[int], ys: list[int]) -> float:
    """PolygonRoi.getTracedPerimeter (uncalibrated)."""
    n = len(xs)
    if n < 4:
        return 0.0
    sumdx = sumdy = ncorners = 0
    dx1 = xs[0] - xs[n - 1]
    dy1 = ys[0] - ys[n - 1]
    side1 = abs(dx1) + abs(dy1)
    corner = False
    for i in range(n):
        nxt = i + 1 if i + 1 < n else 0
        dx2 = xs[nxt] - xs[i]
        dy2 = ys[nxt] - ys[i]
        sumdx += abs(dx1)
        sumdy += abs(dy1)
        side2 = abs(dx2) + abs(dy2)
        if side1 > 1 or not corner:
            corner = True
            ncorners += 1
        else:
            corner = False
        dx1, dy1, side1 = dx2, dy2, side2
    return sumdx + sumdy - ncorners * (2.0 - math.sqrt(2.0))


def particle_shape(labels: np.ndarray, n: int | None = None) -> dict[int, dict]:
    """Area (pixel count), traced perimeter and circularity of every label, ImageJ style."""
    n = int(labels.max()) if n is None else n
    out: dict[int, dict] = {}
    if n == 0:
        return out
    objs = ndi.find_objects(labels)
    for lab in range(1, n + 1):
        sl = objs[lab - 1]
        if sl is None:
            continue
        sub = labels[sl] == lab
        area = int(sub.sum())
        ys_, xs_ = np.nonzero(sub)
        k = np.lexsort((xs_, ys_))[0]  # first pixel in raster order
        vx, vy = trace_outline(sub, int(xs_[k]), int(ys_[k]))
        per = traced_perimeter(vx, vy)
        circ = 0.0 if per == 0 else 4.0 * math.pi * (area / (per * per))
        out[lab] = {"area": area, "perimeter": per, "circularity": min(circ, 1.0)}
    return out
