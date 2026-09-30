"""Automatic thresholds.

``ij_default`` is a line-by-line port of ImageJ's "Default" method
(``AutoThresholder.defaultIsoData`` -> ``IJIsoData``, ImageJ 1.54): the modal
histogram bin is clipped to 1.5x the second-highest count when it exceeds twice
that count, bins 0 and 255 are ignored, and the iterative intermeans loop is run
on the 256-bin histogram. Images that are not 8-bit are first scaled to 8 bits
exactly as ImageJ's ``convertToByte(true)`` does (min-max scaling).

All functions return a value ``t`` in the image's own intensity units with the
convention foreground = ``image > t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

METHOD_LABELS = {
    "ij_default": "ImageJ Default (IsoData, as in the paper)",
    "otsu": "Otsu",
    "li": "Li (minimum cross-entropy)",
    "triangle": "Triangle",
    "isodata": "IsoData (Ridler–Calvard)",
    "yen": "Yen",
    "manual": "Manual value",
}


@dataclass
class Threshold:
    value: float  # foreground = image > value
    method: str
    level8: int | None = None  # ImageJ 8-bit level, when applicable
    note: str = ""


# ------------------------------------------------------------------ ImageJ port


def ij_isodata(hist: np.ndarray) -> int:
    """ImageJ ``AutoThresholder.IJIsoData`` (original ImageJ IsoData)."""
    data = np.asarray(hist, dtype=np.float64).copy()
    max_value = len(data) - 1
    data[0] = 0
    data[max_value] = 0
    nz = np.nonzero(data)[0]
    if len(nz) == 0:
        return len(data) // 2
    lo, hi = int(nz[0]), int(nz[-1])
    if lo >= hi:
        return len(data) // 2
    idx = np.arange(len(data), dtype=np.float64)
    moving = lo
    while True:
        s1 = float(np.dot(idx[lo : moving + 1], data[lo : moving + 1]))
        s2 = float(data[lo : moving + 1].sum())
        s3 = float(np.dot(idx[moving + 1 : hi + 1], data[moving + 1 : hi + 1]))
        s4 = float(data[moving + 1 : hi + 1].sum())
        # Java double division: x/0.0 -> NaN/Inf; the loop then stops
        a = s1 / s2 if s2 else (math.nan if s1 == 0 else math.inf)
        b = s3 / s4 if s4 else (math.nan if s3 == 0 else math.inf)
        result = (a + b) / 2.0
        moving += 1
        if not ((moving + 1) <= result and moving < hi - 1):
            break
    if math.isnan(result):
        return 0
    return int(math.floor(result + 0.5))  # Java Math.round


def _bilevel(hist: np.ndarray, subtract_one: bool = True) -> int:
    nz = np.nonzero(hist)[0]
    if len(nz) == 1:
        return int(nz[0]) - (1 if subtract_one else 0)
    if len(nz) == 2:
        return int(nz[1]) - (1 if subtract_one else 0)
    return -1


def ij_default(hist: np.ndarray) -> int:
    """ImageJ "Default" = ``defaultIsoData`` (mode-clipped IJIsoData)."""
    hist = np.asarray(hist, dtype=np.int64)
    b = _bilevel(hist)
    if b >= 0:
        return b
    data = hist.copy()
    mode = int(np.argmax(data))  # first maximum, as in the Java loop
    max_count = int(data[mode])
    others = np.delete(data, mode)
    max2 = int(others.max()) if len(others) else 0
    if max_count > 2 * max2 and max2 != 0:
        data[mode] = int(max2 * 1.5)
    t = ij_isodata(data)
    return 0 if t == -1 else t


def imagej_to_byte(values: np.ndarray, vmin: float | None = None, vmax: float | None = None):
    """ImageJ ``TypeConverter`` scaling of 16-bit/float data to 8 bits.

    Returns (byte_values, vmin, scale) with byte = floor((v - vmin) * scale + 0.5).
    """
    v = np.asarray(values)
    if v.dtype == np.uint8:
        return v, 0.0, 1.0
    vmin = float(v.min()) if vmin is None else float(vmin)
    vmax = float(v.max()) if vmax is None else float(vmax)
    if np.issubdtype(v.dtype, np.integer):
        scale = 256.0 / (vmax - vmin + 1.0)
        b = np.floor((v.astype(np.float64) - vmin).clip(0) * scale + 0.5)
    else:  # FloatProcessor: value = (v - min) * 256/(max-min), clipped to 255
        scale = 256.0 / (vmax - vmin) if vmax > vmin else 1.0
        b = np.floor((v.astype(np.float64) - vmin) * scale)
    return np.clip(b, 0, 255).astype(np.uint8), vmin, scale


# ------------------------------------------------------------------ public API


def compute_threshold(
    image: np.ndarray,
    method: str = "ij_default",
    valid: np.ndarray | None = None,
    manual_value: float | None = None,
    legacy_inclusive: bool = False,
) -> Threshold:
    """Threshold ``image`` using only pixels where ``valid`` is True.

    ``legacy_inclusive`` reproduces ImageJ <= 1.41, whose automatic threshold
    kept pixels *equal* to the level (``>= t``); current ImageJ uses ``> t``.
    """
    method = (method or "ij_default").lower()
    vals = image[valid] if valid is not None else image.ravel()
    if vals.size == 0:
        raise ValueError("no valid pixels to threshold")
    if method == "manual":
        if manual_value is None:
            raise ValueError("manual threshold selected but no value given")
        return Threshold(float(manual_value), "manual")

    if method == "ij_default":
        b, vmin, scale = imagej_to_byte(vals)
        hist = np.bincount(b.ravel(), minlength=256)[:256]
        t8 = ij_default(hist)
        lower8 = t8 if legacy_inclusive else t8 + 1  # first foreground byte value
        if vals.dtype == np.uint8:
            value = float(lower8) - 0.5
        elif np.issubdtype(vals.dtype, np.integer):
            # smallest v with floor((v-vmin)*scale+0.5) >= lower8
            v_lo = vmin + (lower8 - 0.5) / scale
            value = float(math.ceil(v_lo - 1e-9)) - 0.5
        else:
            value = float(np.nextafter(vmin + lower8 / scale, -np.inf))
        return Threshold(value, method, level8=int(t8))

    from skimage import filters

    fn = {
        "otsu": filters.threshold_otsu,
        "li": filters.threshold_li,
        "triangle": filters.threshold_triangle,
        "isodata": filters.threshold_isodata,
        "yen": filters.threshold_yen,
    }.get(method)
    if fn is None:
        raise ValueError(f"unknown threshold method '{method}'")
    v = vals.astype(np.float64)
    if np.ptp(v) == 0:
        return Threshold(float(v[0]), method, note="image is uniform")
    t = float(np.atleast_1d(fn(v))[0])
    if legacy_inclusive and np.issubdtype(vals.dtype, np.integer):
        t = math.floor(t) - 0.5 if t == math.floor(t) else t
    return Threshold(t, method)


def multi_otsu(image: np.ndarray, n_thresholds: int, valid: np.ndarray | None = None) -> np.ndarray:
    """``n_thresholds`` multi-level Otsu thresholds (MATLAB ``multithresh``)."""
    from skimage.filters import threshold_multiotsu

    vals = image[valid] if valid is not None else image.ravel()
    return np.asarray(threshold_multiotsu(vals, classes=n_thresholds + 1), dtype=np.float64)
