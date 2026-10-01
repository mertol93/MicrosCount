"""Porosity and pore-size distribution of SEM images.

Python port of ``SEM_Porosity.m`` by Arash Rabbani (BSD-3-Clause, (c) 2020
Arash Rabbani), the MATLAB script used for the membrane SEM analysis in
Acarer-Arat et al. (2024) ACS Omega 9(41):42159-42171. Each MATLAB step is
reproduced with its MATLAB semantics:

1. ``rgb2gray`` (MATLAB coefficients, rounded to uint8);
2. ``multithresh(B, N)`` - Otsu thresholds computed on MATLAB's normalised
   256-bin histogram and mapped back to image units;
3. ``imquantize``: pores are the darkest class, ``B <= level(1)``;
4. ``bwmorph(P, 'majority', 1)`` with zero padding;
5. ``-bwdist(P, 'cityblock')``, ``medfilt2`` (zero padding) and an
   8-connected watershed from the regional minima to separate touching pores;
6. ``bwareaopen(Pr, 9, 8)``; pore radius = resolution * sqrt(area / pi);
7. porosity = 1 - mean(P), i.e. the pore fraction *before* the watershed split.

Additions (all off by default so results match the MATLAB script): cropping
of an SEM data bar, counting more than one dark class as pore, excluding
pores cut by the image edge from the size statistics, and bright pores.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy import ndimage as ndi

from ..core.imageio import LoadedImage

RGB2GRAY = (0.298936021293775, 0.587043074451121, 0.114020904255103)


@dataclass
class PorositySettings:
    pixel_size_um: float = 0.459  # MATLAB script default "Resolution" (micron/pixel)
    n_thresholds: int = 4  # MATLAB "N": number of multithresh levels
    threshold_search: str = "matlab"  # "matlab" (fminsearch, as published) | "exhaustive"
    pore_classes: int = 1  # darkest classes counted as pore (MATLAB: 1)
    majority_passes: int = 1
    split_pores: bool = True  # watershed separation of touching pores
    min_pore_area_px: int = 9  # bwareaopen size
    crop_bottom_px: int = 0  # remove an SEM information bar at the bottom
    exclude_edge_pores: bool = False  # size statistics only
    pores_are_bright: bool = False
    histogram_bins: int = 25

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PorositySettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class PorosityResult:
    name: str
    condition: str
    summary: dict
    pores: list[dict]
    warnings: list[str]
    layers: dict | None = None


# ---------------------------------------------------------------------- MATLAB equivalents


def _round_half_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def rgb2gray_matlab(rgb: np.ndarray) -> np.ndarray:
    """MATLAB ``rgb2gray`` for uint8/uint16 input, (3, Y, X) or (Y, X, 3)."""
    a = np.asarray(rgb)
    if a.shape[0] == 3 and a.ndim == 3 and a.shape[-1] != 3:
        a = np.moveaxis(a, 0, -1)
    g = a[..., 0] * RGB2GRAY[0] + a[..., 1] * RGB2GRAY[1] + a[..., 2] * RGB2GRAY[2]
    info = np.iinfo(a.dtype) if np.issubdtype(a.dtype, np.integer) else None
    if info is None:
        return g
    return np.clip(_round_half_away(g), info.min, info.max).astype(a.dtype)


def _fminsearch(fun, x0, tolx=1e-4, tolf=1e-4, maxiter=None, maxfun=None) -> np.ndarray:
    """MATLAB ``fminsearch`` (Lagarias et al. Nelder-Mead), step for step."""
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    n = x0.size
    maxiter = maxiter or 200 * n
    maxfun = maxfun or 200 * n
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    v = np.zeros((n + 1, n))
    fv = np.zeros(n + 1)
    v[0] = x0
    fv[0] = fun(x0)
    for j in range(n):
        y = x0.copy()
        y[j] = (1 + 0.05) * y[j] if y[j] != 0 else 0.00025
        v[j + 1] = y
        fv[j + 1] = fun(y)
    order = np.argsort(fv, kind="stable")
    v, fv = v[order], fv[order]
    itercount, func_evals = 1, n + 1
    while func_evals < maxfun and itercount < maxiter:
        if (np.max(np.abs(fv[0] - fv[1:])) <= max(tolf, 10 * np.spacing(fv[0]))
                and np.max(np.abs(v[1:] - v[0])) <= max(tolx, 10 * np.spacing(np.max(v[0])))):
            break
        xbar = v[:n].sum(axis=0) / n
        xr = (1 + rho) * xbar - rho * v[-1]
        fxr = fun(xr)
        func_evals += 1
        shrink = False
        if fxr < fv[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * v[-1]
            fxe = fun(xe)
            func_evals += 1
            if fxe < fxr:
                v[-1], fv[-1] = xe, fxe
            else:
                v[-1], fv[-1] = xr, fxr
        elif fxr < fv[n - 1]:
            v[-1], fv[-1] = xr, fxr
        elif fxr < fv[-1]:
            xc = (1 + psi * rho) * xbar - psi * rho * v[-1]
            fxc = fun(xc)
            func_evals += 1
            if fxc <= fxr:
                v[-1], fv[-1] = xc, fxc
            else:
                shrink = True
        else:
            xcc = (1 - psi) * xbar + psi * v[-1]
            fxcc = fun(xcc)
            func_evals += 1
            if fxcc < fv[-1]:
                v[-1], fv[-1] = xcc, fxcc
            else:
                shrink = True
        if shrink:
            for j in range(1, n + 1):
                v[j] = v[0] + sigma * (v[j] - v[0])
                fv[j] = fun(v[j])
            func_evals += n
        order = np.argsort(fv, kind="stable")
        v, fv = v[order], fv[order]
        itercount += 1
    return v[0]


def _otsu_bign_objective(p: np.ndarray):
    """MATLAB ``objCriteriaBigN``: -between-class variance for rounded thresholds."""
    nb = len(p)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, nb + 1))
    mu_t = mu[-1]

    def fun(thresh):
        b = _round_half_away(np.asarray(thresh)) + 1  # 1-based boundaries
        if not np.all(np.diff(np.r_[1, b, nb]) > 0):
            return math.inf
        b = np.r_[b, nb].astype(np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = omega[b[0] - 1] * (mu[b[0] - 1] / omega[b[0] - 1] - mu_t) ** 2
            for k in range(1, len(b)):
                om = omega[b[k] - 1] - omega[b[k - 1] - 1]
                mk = (mu[b[k] - 1] - mu[b[k - 1] - 1]) / om
                val += om * (mk - mu_t) ** 2
        return -val if math.isfinite(val) else math.inf

    return fun


def matlab_multithresh(img: np.ndarray, n: int, search: str = "matlab") -> np.ndarray:
    """``multithresh(img, n)``: Otsu thresholds in image units (MATLAB conventions).

    MATLAB normalises the image to [0, 1] with its min/max, converts with
    ``im2uint8`` and histograms 256 bins; thresholds are bin indices of the last
    bin of each lower class, mapped back as ``min + t/255*(max-min)`` and cast
    to the image class. For ``n >= 3`` MATLAB maximises the Otsu criterion with
    ``fminsearch`` from uniformly spaced starting thresholds, which can stop at a
    local optimum; ``search="matlab"`` reproduces that exactly, while
    ``search="exhaustive"`` returns the global optimum.
    """
    from skimage.filters import threshold_multiotsu

    a = np.asarray(img)
    amin, amax = float(a.min()), float(a.max())
    if amax <= amin:
        return np.full(n, amin)
    if np.issubdtype(a.dtype, np.integer):  # single((A - minA)) / single(maxA - minA)
        norm = (a.astype(np.float64) - amin).astype(np.float32) / np.float32(amax - amin)
    else:
        norm = ((a.astype(np.float64) - amin) / (amax - amin)).astype(np.float32)
    b = np.floor(norm * np.float32(255.0) + np.float32(0.5)).astype(np.int64)  # im2uint8
    counts = np.bincount(np.clip(b, 0, 255).ravel(), minlength=256)[:256].astype(np.float64)
    nvalues = int(np.count_nonzero(counts))
    if nvalues <= n:
        # too few grey levels: every occupied level but the last is a threshold
        occ = np.nonzero(counts)[0][:-1]
        t_bins = np.pad(occ, (0, n - len(occ)), mode="edge") if len(occ) else np.zeros(n)
    elif n >= 3 and search == "matlab":
        p = counts / counts.sum()
        x0 = np.linspace(0, 255, n + 2)[1:-1]
        t_bins = _round_half_away(_fminsearch(_otsu_bign_objective(p), x0, tolx=1.0))
    else:
        t_bins = threshold_multiotsu(classes=n + 1, hist=(counts, np.arange(256)))
    t = amin + np.asarray(t_bins, np.float64) / 255.0 * (amax - amin)
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        t = np.clip(_round_half_away(t), info.min, info.max)
    return t


def bwmorph_majority(mask: np.ndarray, passes: int = 1) -> np.ndarray:
    """``bwmorph(mask, 'majority', n)``: 1 if >= 5 of the 3x3 neighbourhood are 1."""
    m = mask.astype(bool)
    k = np.ones((3, 3), np.int32)
    for _ in range(max(0, int(passes))):
        count = ndi.convolve(m.astype(np.int32), k, mode="constant", cval=0)
        new = count >= 5
        if np.array_equal(new, m):
            break
        m = new
    return m


def split_pores_watershed(solid: np.ndarray) -> np.ndarray:
    """Pore pixels minus 8-connected watershed lines of the city-block distance map."""
    from skimage.morphology import local_minima
    from skimage.segmentation import watershed

    pore = ~solid
    if not pore.any():
        return pore
    if not solid.any():
        return pore
    d = -ndi.distance_transform_cdt(pore, metric="taxicab").astype(np.float64)  # -bwdist(P,'cityblock')
    b = ndi.median_filter(d, size=3, mode="constant", cval=0.0)  # medfilt2 pads with zeros
    minima = local_minima(b, connectivity=2, allow_borders=True)
    markers, _ = ndi.label(minima, structure=np.ones((3, 3)))
    ws = watershed(b, markers, connectivity=2, watershed_line=True)
    return pore & (ws != 0)


def bwareaopen(mask: np.ndarray, min_px: int, connectivity: int = 8) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove objects with fewer than ``min_px`` pixels; returns (mask, labels, n)."""
    structure = np.ones((3, 3)) if connectivity == 8 else None
    lab, n = ndi.label(mask, structure=structure)
    if n == 0:
        return mask & False, lab, 0
    areas = np.bincount(lab.ravel())
    keep = areas >= int(min_px)
    keep[0] = False
    out = keep[lab]
    lab, n = ndi.label(out, structure=structure)
    return out, lab, n


# ---------------------------------------------------------------------- analysis


def grey_from_image(img: LoadedImage) -> tuple[np.ndarray, list[str]]:
    notes = []
    if img.kind in ("single_colour", "merged_rgb") and img.n_channels == 3:
        return rgb2gray_matlab(img.data), notes
    if img.n_channels == 1:
        return img.data[0], notes
    notes.append(f"{img.name}: {img.n_channels} channels, first channel used")
    return img.data[0], notes


def analyse_sem(
    img: LoadedImage,
    settings: PorositySettings | None = None,
    condition: str = "",
    keep_layers: bool = True,
) -> PorosityResult:
    s = settings or PorositySettings()
    warns: list[str] = []
    grey, notes = grey_from_image(img)
    warns.extend(notes)
    if img.lossy:
        warns.append("JPEG input: compression can shift pore boundaries slightly")
    valid = np.ones(grey.shape, bool)
    if img.annotation_mask is not None:
        valid &= ~img.annotation_mask
    if s.crop_bottom_px and s.crop_bottom_px > 0:
        cb = min(int(s.crop_bottom_px), grey.shape[0] - 1)
        grey = grey[: grey.shape[0] - cb]
        valid = valid[: valid.shape[0] - cb]
    work = grey
    if s.pores_are_bright:
        if np.issubdtype(grey.dtype, np.integer):
            work = (np.iinfo(grey.dtype).max - grey).astype(grey.dtype)
        else:
            work = grey.max() - grey
    vals = work if valid.all() else work[valid]
    levels = matlab_multithresh(vals, int(s.n_thresholds), s.threshold_search)
    k = int(np.clip(s.pore_classes, 1, len(levels)))
    pore_level = float(levels[k - 1])
    solid = ~(work <= pore_level)  # imquantize: class 1 = B <= level(1)
    solid |= ~valid  # excluded pixels never count as pore
    solid = bwmorph_majority(solid, s.majority_passes)
    n_valid = int(valid.sum())
    porosity = float(((~solid) & valid).sum() / n_valid) if n_valid else math.nan
    pores = split_pores_watershed(solid) if s.split_pores else ~solid
    pores &= valid
    pores, labels, n = bwareaopen(pores, s.min_pore_area_px, 8)

    px = float(s.pixel_size_um)
    rows = []
    radii = []
    if n:
        areas = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.float64)
        objs = ndi.find_objects(labels)
        h, w = labels.shape
        cy = ndi.mean(np.indices(labels.shape)[0], labels, np.arange(1, n + 1))
        cx = ndi.mean(np.indices(labels.shape)[1], labels, np.arange(1, n + 1))
        for i in range(n):
            sl = objs[i]
            edge = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w
            r = px * math.sqrt(areas[i] / math.pi)
            include = not (s.exclude_edge_pores and edge)
            rows.append(
                {
                    "pore": i + 1,
                    "x": float(cx[i]),
                    "y": float(cy[i]),
                    "area_px": int(areas[i]),
                    "area_um2": float(areas[i] * px * px),
                    "equivalent_radius_um": r,
                    "touches_edge": bool(edge),
                    "included": include,
                }
            )
            if include:
                radii.append(r)
    radii = np.asarray(radii)
    summary = {
        "image": img.name,
        "condition": condition,
        "porosity": porosity,
        "porosity_percent": 100 * porosity if math.isfinite(porosity) else math.nan,
        "n_pores": int(radii.size),
        "mean_pore_radius_um": float(radii.mean()) if radii.size else math.nan,
        "sd_pore_radius_um": float(radii.std(ddof=1)) if radii.size > 1 else math.nan,
        "median_pore_radius_um": float(np.median(radii)) if radii.size else math.nan,
        "mean_pore_area_um2": float(np.mean([r["area_um2"] for r in rows if r["included"]])) if radii.size else math.nan,
        "pore_threshold": pore_level,
        "thresholds": ", ".join(f"{v:g}" for v in levels),
        "pixel_size_um": px,
        "image_height_px": int(grey.shape[0]),
        "image_width_px": int(grey.shape[1]),
        "cropped_bottom_px": int(s.crop_bottom_px or 0),
    }
    layers = None
    if keep_layers:
        layers = {"grey": grey, "solid": solid, "pores": pores, "pore_labels": labels, "valid": valid}
    return PorosityResult(img.name, condition, summary, rows, warns, layers)


def summarise_porosity(results: list[PorosityResult]) -> list[dict]:
    by: dict[str, list[PorosityResult]] = {}
    for r in results:
        by.setdefault(r.condition or "(none)", []).append(r)
    rows = []
    for cond, rs in by.items():
        por = np.array([r.summary["porosity"] for r in rs], dtype=np.float64)
        rad = np.array([p["equivalent_radius_um"] for r in rs for p in r.pores if p["included"]])
        mr = np.array([r.summary["mean_pore_radius_um"] for r in rs], dtype=np.float64)
        rows.append(
            {
                "condition": cond,
                "n_images": len(rs),
                "porosity_mean": float(np.nanmean(por)),
                "porosity_sd": float(np.nanstd(por, ddof=1)) if len(rs) > 1 else math.nan,
                "mean_pore_radius_um_mean": float(np.nanmean(mr)),
                "mean_pore_radius_um_sd": float(np.nanstd(mr, ddof=1)) if len(rs) > 1 else math.nan,
                "pooled_n_pores": int(rad.size),
                "pooled_mean_pore_radius_um": float(rad.mean()) if rad.size else math.nan,
                "pooled_sd_pore_radius_um": float(rad.std(ddof=1)) if rad.size > 1 else math.nan,
            }
        )
    return rows
