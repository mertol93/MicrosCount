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

Around the MATLAB core, for images straight from the microscope:

* the pixel size of every image: typed, read from the SEM metadata (FEI / Thermo
  Fisher, Zeiss) or the TIFF calibration, or a default for all images;
* the SEM data bar: its height from the FEI / Thermo Fisher metadata, or found as the
  block of flat graphic rows at the bottom of the image, is left out;
* per pore: equivalent diameter, perimeter and circularity (ImageJ's traced perimeter);
  per image: diameter statistics (mean, median, d10, d90, area-weighted mean, largest),
  pore density and quality checks.

Options (all off by default so results match the MATLAB script): a fixed pore
threshold, counting more than one dark class as pore, excluding pores cut by the
image edge from the size statistics, and bright pores.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np
from scipy import ndimage as ndi

from ..core.imageio import TIFF_RESOLUTION, LoadedImage

RGB2GRAY = (0.298936021293775, 0.587043074451121, 0.114020904255103)
DATA_BAR_MODES = ("auto", "none", "manual")


@dataclass
class PorositySettings:
    pixel_size_um: float | None = None  # default for images whose size is not known otherwise (MATLAB: 0.459)
    use_metadata_pixel_size: bool = True  # SEM metadata / TIFF calibration before the default
    n_thresholds: int = 4  # MATLAB "N": number of multithresh levels
    threshold_search: str = "matlab"  # "matlab" (fminsearch, as published) | "exhaustive"
    pore_threshold_value: float | None = None  # fixed threshold (image grey levels) instead of multithresh
    pore_classes: int = 1  # darkest classes counted as pore (MATLAB: 1)
    majority_passes: int = 1
    split_pores: bool = True  # watershed separation of touching pores
    min_pore_area_px: int = 9  # bwareaopen size
    data_bar: str = "auto"  # SEM data bar: "auto" (metadata or detected) | "none" | "manual" (crop_bottom_px)
    crop_bottom_px: int = 0  # rows removed at the bottom when data_bar is "manual"
    exclude_edge_pores: bool = False  # size statistics only
    pores_are_bright: bool = False
    histogram_bins: int = 25  # pore-size distribution bins
    # ---- experiment design (see materials/experiment.py)
    samples_from: str = "auto"  # "auto" | "name" (file names) | "folder" (folder names)
    reference_sample: str = ""  # reference for percent changes and the default comparisons
    references: dict = field(default_factory=dict)  # sample -> its own reference
    comparisons: list = field(default_factory=list)  # [[A, B], ...]; empty = each sample vs the reference
    sample_order: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PorositySettings":
        d = dict(d or {})
        if "data_bar" not in d:  # settings of version 0.2 and earlier: crop only what was asked for
            d["data_bar"] = "manual" if (d.get("crop_bottom_px") or 0) > 0 else "none"
        if "pixel_size_um" in d and not d["pixel_size_um"]:
            d["pixel_size_um"] = None
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class PorosityResult:
    name: str
    sample: str
    summary: dict
    pores: list[dict]
    warnings: list[str]
    layers: dict | None = None
    repetition: str = ""
    image_id: str = ""

    @property
    def condition(self) -> str:  # name used by version 0.2
        return self.sample


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


def meyer_watershed(image: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """8-connected watershed with lines, as MATLAB's ``watershed``: labels, 0 on the lines.

    Meyer's flooding from the labelled ``markers``: pixels leave a priority queue lowest
    first, ties first in first out, in MATLAB's column-major order (pixels and their
    neighbours). A pixel whose labelled neighbours carry one label takes it and queues its
    neighbours that are not queued yet; a pixel touching two labels is a line pixel and does
    not spread. Each pixel is queued once, so the time grows with the image size only
    (scikit-image's watershed with lines re-queues line pixels and can take minutes on a
    large flat region such as an uncropped data bar). On the MATLAB script's example
    images the pores come out pixel-identical to MATLAB's.

    The flood runs in rounds - the queue of one level at one moment. A round is decided
    at once; only neighbouring pixels of a round that would take different labels are
    settled one by one, first come first served.
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape
    s = h + 2  # padded and transposed: C order here is MATLAB's column-major order
    it = np.int32 if (w + 2) * s < 2**31 else np.int64
    levels, inv = np.unique(image.T, return_inverse=True)
    lev = np.zeros((w + 2, s), it)
    lev[1:-1, 1:-1] = inv.reshape(w, h)
    lev = lev.ravel()
    out = np.zeros((w + 2, s), it)
    out[1:-1, 1:-1] = np.asarray(markers).T
    out = out.ravel()
    queued = np.ones((w + 2, s), bool)
    queued[1:-1, 1:-1] = False
    queued = queued.ravel() | (out != 0)
    offs = np.array([-s - 1, -s, -s + 1, -1, 1, s - 1, s, s + 1], it)  # MATLAB's neighbour order
    pending: list[list[np.ndarray]] = [[] for _ in levels]
    pos = np.full(out.size, -1, it)  # place of a pixel in the current round or push

    def push(src: np.ndarray, nbrs: np.ndarray, k: int | None) -> None:
        """Queue the unqueued neighbours of ``src`` in order (pixel by pixel, neighbour by neighbour)."""
        key = np.flatnonzero(~queued[nbrs].ravel()).astype(it)
        if key.size == 0:
            return
        x = nbrs.ravel()[key]
        pos[src] = np.arange(src.size, dtype=it)
        first = np.ones(x.size, bool)  # not queued already by an earlier pixel of src
        for j in range(8):
            e = pos[x - offs[j]]
            first &= ~((e >= 0) & (e * 8 + j < key))
        pos[src] = -1
        new = x[first]
        queued[new] = True
        lv = lev[new]
        if k is not None:
            np.maximum(lv, k, out=lv)  # a lower pixel joins the current level
        lo, hi = int(lv.min()), int(lv.max())
        if lo == hi:
            pending[lo].append(new)
        else:
            order = np.argsort(lv, kind="stable")
            lvs = lv[order]
            for chunk in np.split(order, np.flatnonzero(np.diff(lvs)) + 1):
                pending[int(lv[chunk[0]])].append(new[chunk])

    seeds = np.flatnonzero(out).astype(it)
    if seeds.size:
        push(seeds, seeds[:, None] + offs, None)
    for k in range(len(levels)):
        while pending[k]:
            r = pending[k][0] if len(pending[k]) == 1 else np.concatenate(pending[k])
            pending[k] = []
            nbrs = r[:, None] + offs
            nl = out[nbrs]
            hi = nl.max(axis=1)
            single = ((nl == hi[:, None]) | (nl == 0)).all(axis=1) & (hi > 0)
            lab = np.where(single, hi, 0).astype(it)
            # neighbours within the round that would take different labels: the earlier one wins
            pos[r] = np.arange(r.size, dtype=it)
            earlier, later = [], []
            for j in range(8):
                q = pos[nbrs[:, j]]
                i = np.flatnonzero((q >= 0) & single)
                q = q[i]
                clash = single[q] & (lab[q] != lab[i]) & (q < i)
                if clash.any():
                    earlier.append(q[clash])
                    later.append(i[clash])
            pos[r] = -1
            if earlier:
                before, after = np.concatenate(earlier), np.concatenate(later)
                order = np.argsort(after, kind="stable")
                before, after = before[order].tolist(), after[order]
                nodes, starts = np.unique(after, return_index=True)
                ends = np.append(starts[1:], after.size)
                keeps = bytearray(b"\x01") * r.size
                for p, i0, i1 in zip(nodes.tolist(), starts.tolist(), ends.tolist()):
                    for t in range(i0, i1):
                        if keeps[before[t]]:
                            keeps[p] = 0
                            break
                lab[~np.frombuffer(keeps, dtype=bool)] = 0
            done = lab > 0
            out[r[done]] = lab[done]
            if done.any():
                push(r[done], nbrs[done], k)
    return out.reshape(w + 2, s)[1:-1, 1:-1].T.copy()


def split_pores_watershed(solid: np.ndarray) -> np.ndarray:
    """Pore pixels minus 8-connected watershed lines of the city-block distance map."""
    from skimage.morphology import local_minima

    pore = ~solid
    if not pore.any():
        return pore
    if not solid.any():
        return pore
    d = -ndi.distance_transform_cdt(pore, metric="taxicab").astype(np.float64)  # -bwdist(P,'cityblock')
    b = ndi.median_filter(d, size=3, mode="constant", cval=0.0)  # medfilt2 pads with zeros
    minima = local_minima(b, connectivity=2, allow_borders=True)  # imregionalmin
    markers, _ = ndi.label(minima, structure=np.ones((3, 3)))
    return pore & (meyer_watershed(b, markers) != 0)


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


def _to8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    lo, hi = float(a.min()), float(a.max())
    return np.round((a.astype(np.float64) - lo) / max(hi - lo, 1e-12) * 255).astype(np.uint8)


def _graphic_rows(a8: np.ndarray, tol: int = 3) -> np.ndarray:
    """Fraction of each row's pixels within ``tol`` grey levels of its two most common values."""
    h, w = a8.shape
    counts = np.bincount((np.arange(h)[:, None] * 256 + a8).ravel(), minlength=h * 256).reshape(h, 256)
    near = ndi.uniform_filter1d(counts.astype(np.float64), 2 * tol + 1, axis=1, mode="constant") * (2 * tol + 1)
    first = near.argmax(axis=1)
    best = near[np.arange(h), first]
    lv = np.arange(256)[None, :]
    near2 = np.where(np.abs(lv - first[:, None]) > 2 * tol, near, 0)
    return (best + near2.max(axis=1)) / w


def detect_data_bar(grey: np.ndarray, max_fraction: float = 0.35, min_rows: int = 8) -> int:
    """Height in rows of an SEM information bar at the bottom of an image (0 if there is none).

    A data bar is drawn graphics: nearly every pixel of each of its rows has one of two
    values (background and text, within 3 grey levels for JPEG), unlike the noisy
    micrograph above it. The bar is the tallest such block at the bottom, at most
    ``max_fraction`` of the image, with ordinary image rows just above it.
    """
    a = _to8(np.asarray(grey))
    h, w = a.shape
    if h < 5 * min_rows or w < 32:
        return 0
    g = _graphic_rows(a) >= 0.75
    limit = int(h * max_fraction)
    best = 0
    for k in range(min_rows, limit + 1):
        top = h - k
        if not g[top] or g[top - 1]:
            continue  # the bar's top row is graphic and the row above it is not
        if g[top:].mean() < 0.9:
            continue
        above = g[max(0, top - 8):top]
        if above.mean() <= 0.5:
            best = k
    if best:  # a data bar carries text and a scale bar: a flat strip of one tone is not one
        bar = a[h - best:]
        hist = np.bincount(bar.ravel(), minlength=256)
        if hist[np.abs(np.arange(256) - int(hist.argmax())) >= 64].sum() < 0.002 * bar.size:
            return 0
    return best


def measure_data_bar_scale(grey: np.ndarray, bar_rows: int) -> float | None:
    """Length in pixels of the scale bar drawn in an SEM data bar, or None.

    The scale bar is the longest solid horizontal bar among the bright and the dark shapes
    of the data bar: 2-20 rows thick where it is full width (end ticks allowed), between
    3% and 60% of the image width, and not a panel of the bar.
    """
    if bar_rows <= 0:
        return None
    a = _to8(np.asarray(grey))[-bar_rows:]
    h, w = a.shape
    best = 0
    for mask in (a > 127, a <= 127):
        lab, n = ndi.label(mask, structure=np.ones((3, 3)))
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None:
                continue
            ys, xs = sl
            width, height = xs.stop - xs.start, ys.stop - ys.start
            if not 0.03 * w <= width <= 0.6 * w or height > max(25, 0.5 * h):
                continue
            comp = lab[sl] == i
            body = int((comp.sum(axis=1) >= 0.9 * width).sum())  # rows the bar fills from end to end
            if 2 <= body <= 20:
                best = max(best, width)
    return float(best) if best else None


def _percentile(v: np.ndarray, q: float) -> float:
    return float(np.percentile(v, q)) if v.size else math.nan


def _full_scale(grey: np.ndarray, img: LoadedImage) -> float:
    """The grey level of a saturated pixel: 255, 65535, or e.g. 4095 for 12-bit data in a 16-bit file."""
    if not np.issubdtype(grey.dtype, np.integer):
        return float(grey.max() or 1) if grey.size else 1.0
    top = float(np.iinfo(grey.dtype).max)
    sat = getattr(img, "saturation_value", None)
    if sat and sat < top:
        return float(sat)
    if grey.dtype == np.uint16 and grey.size:  # the bit depth the values need
        return float(min(top, 2 ** max(8, int(grey.max()).bit_length()) - 1))
    return top


def analyse_sem(
    img: LoadedImage,
    settings: PorositySettings | None = None,
    sample: str = "",
    keep_layers: bool = True,
    pixel_size_um: float | None = None,
    repetition: str = "",
    image_id: str = "",
) -> PorosityResult:
    """Porosity and pores of one SEM image. ``pixel_size_um`` overrides everything else for this image."""
    s = settings or PorositySettings()
    warns: list[str] = []
    grey, notes = grey_from_image(img)
    warns.extend(notes)
    if img.lossy:
        warns.append("JPEG input: compression can shift pore boundaries slightly")
    meta = dict(getattr(img, "metadata", {}) or {})

    # pixel size of this image: typed, then the file's calibration, then the default; a bare TIFF
    # resolution tag (written by many programs) only when there is no default
    file_px = getattr(img, "pixel_size_um", None) if s.use_metadata_pixel_size else None
    weak = getattr(img, "pixel_size_source", None) == TIFF_RESOLUTION
    if pixel_size_um:
        px, px_source = float(pixel_size_um), "typed for this image"
    elif file_px and not (weak and s.pixel_size_um):
        px, px_source = float(file_px), img.pixel_size_source or "file metadata"
    elif s.pixel_size_um:
        px, px_source = float(s.pixel_size_um), "default setting"
    else:
        px, px_source = math.nan, "unknown"
    known = math.isfinite(px)
    if not known:
        warns.append("pixel size unknown: pore sizes are in pixels (set it, or measure the scale bar)")

    # data bar
    full = grey
    detected = detect_data_bar(grey)
    if s.data_bar == "manual":
        crop, crop_source = int(s.crop_bottom_px or 0), "manual"
    elif s.data_bar == "none":
        crop, crop_source = 0, "none"
        if detected:
            warns.append(f"the bottom {detected} rows look like an SEM data bar but are analysed (data bar: none)")
    elif meta.get("data_bar_px"):
        crop, crop_source = int(meta["data_bar_px"]), f"{meta.get('instrument', 'file')} metadata"
    else:
        crop, crop_source = detected, "detected" if detected else "none found"
    crop = int(np.clip(crop, 0, grey.shape[0] - 1))
    valid = np.ones(grey.shape, bool)
    if img.annotation_mask is not None:
        valid &= ~img.annotation_mask
    if crop:
        grey = grey[: grey.shape[0] - crop]
        valid = valid[: valid.shape[0] - crop]

    work, top = grey, None  # bright pores: thresholds are found on the inverted image, top - grey
    if s.pores_are_bright:
        if np.issubdtype(grey.dtype, np.integer):
            top = float(np.iinfo(grey.dtype).max)
            work = (np.iinfo(grey.dtype).max - grey).astype(grey.dtype)
        else:
            top = float(grey.max())
            work = grey.max() - grey

    def grey_level(v: float) -> float:  # a level of `work` on the image's own grey scale
        return top - v if top is not None else v

    vals = work if valid.all() else work[valid]
    levels = matlab_multithresh(vals, int(s.n_thresholds), s.threshold_search)
    if s.pore_threshold_value is not None:
        pore_level, thr_source = grey_level(float(s.pore_threshold_value)), "fixed value"
    else:
        k = int(np.clip(s.pore_classes, 1, len(levels)))
        pore_level, thr_source = float(levels[k - 1]), f"multithresh, N = {int(s.n_thresholds)}"
    if top is not None:
        thr_source += "; pores at or above it"
    solid = ~(work <= pore_level)  # imquantize: class 1 = B <= level(1)
    solid |= ~valid  # excluded pixels never count as pore
    solid = bwmorph_majority(solid, s.majority_passes)
    n_valid = int(valid.sum())
    porosity = float(((~solid) & valid).sum() / n_valid) if n_valid else math.nan
    pores = split_pores_watershed(solid) if s.split_pores else ~solid
    pores &= valid
    pores, labels, n = bwareaopen(pores, s.min_pore_area_px, 8)

    from ..core.imagej import particle_shape

    shapes = particle_shape(labels, n) if n else {}
    rows = []
    if n:
        areas = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.float64)
        objs = ndi.find_objects(labels)
        h, w = labels.shape
        idx = np.arange(1, n + 1)
        cy = ndi.mean(np.arange(h)[:, None] * np.ones((1, w)), labels, idx)
        cx = ndi.mean(np.ones((h, 1)) * np.arange(w)[None, :], labels, idx)
        for i in range(n):
            sl = objs[i]
            edge = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w
            r_px = math.sqrt(areas[i] / math.pi)
            sh = shapes.get(i + 1, {})
            rows.append({
                "pore": i + 1,
                "x": float(cx[i]),
                "y": float(cy[i]),
                "area_px": int(areas[i]),
                "area_um2": float(areas[i] * px * px) if known else math.nan,
                "equivalent_radius_um": px * r_px if known else math.nan,
                "equivalent_diameter_um": 2 * px * r_px if known else math.nan,
                "equivalent_diameter_px": 2 * r_px,
                "perimeter_px": float(sh.get("perimeter", math.nan)),
                "circularity": float(sh.get("circularity", math.nan)),
                "touches_edge": bool(edge),
                "included": not (s.exclude_edge_pores and edge),
            })
    inc = [r for r in rows if r["included"]]
    d_px = np.array([r["equivalent_diameter_px"] for r in inc])
    a_px = np.array([r["area_px"] for r in inc], dtype=np.float64)
    f = px if known else math.nan
    radii = d_px / 2 * f
    diam = d_px * f
    area_um2 = n_valid * px * px if known else math.nan
    aw = float((d_px * a_px).sum() / a_px.sum()) if a_px.size else math.nan
    summary = {
        "image": img.name,
        "image_id": image_id,
        "sample": sample,
        "repetition": repetition,
        "porosity": porosity,
        "porosity_percent": 100 * porosity if math.isfinite(porosity) else math.nan,
        "n_pores": int(len(inc)),
        "n_pores_edge": int(sum(1 for r in rows if r["touches_edge"])),
        "mean_pore_diameter_um": float(diam.mean()) if diam.size else math.nan,
        "sd_pore_diameter_um": float(diam.std(ddof=1)) if diam.size > 1 else math.nan,
        "median_pore_diameter_um": float(np.median(diam)) if diam.size else math.nan,
        "d10_pore_diameter_um": _percentile(diam, 10),
        "d90_pore_diameter_um": _percentile(diam, 90),
        "area_weighted_mean_pore_diameter_um": aw * f,
        "largest_pore_diameter_um": float(diam.max()) if diam.size else math.nan,
        "pore_density_per_um2": len(rows) / area_um2 if known and area_um2 > 0 else math.nan,
        "mean_pore_radius_um": float(radii.mean()) if radii.size else math.nan,
        "sd_pore_radius_um": float(radii.std(ddof=1)) if radii.size > 1 else math.nan,
        "median_pore_radius_um": float(np.median(radii)) if radii.size else math.nan,
        "mean_pore_area_um2": float(a_px.mean() * px * px) if a_px.size and known else math.nan,
        "mean_circularity": float(np.nanmean([r["circularity"] for r in inc])) if inc else math.nan,
        "mean_pore_diameter_px": float(d_px.mean()) if d_px.size else math.nan,
        "sd_pore_diameter_px": float(d_px.std(ddof=1)) if d_px.size > 1 else math.nan,
        "median_pore_diameter_px": float(np.median(d_px)) if d_px.size else math.nan,
        "d10_pore_diameter_px": _percentile(d_px, 10),
        "d90_pore_diameter_px": _percentile(d_px, 90),
        "area_weighted_mean_pore_diameter_px": aw,
        "largest_pore_diameter_px": float(d_px.max()) if d_px.size else math.nan,
        "pore_density_per_mpx": len(rows) / n_valid * 1e6 if n_valid else math.nan,
        "pixel_size_um": px if known else math.nan,
        "pixel_size_source": px_source,
        "analysed_area_um2": area_um2,
        "analysed_area_px": n_valid,
        "data_bar_px": crop,
        "data_bar_source": crop_source,
        "pore_threshold": grey_level(pore_level),
        "threshold_source": thr_source,
        "thresholds": ", ".join(f"{v:g}" for v in sorted(grey_level(float(v)) for v in levels)),
        "image_height_px": int(grey.shape[0]),
        "image_width_px": int(grey.shape[1]),
        "cropped_bottom_px": crop,
    }
    for k in ("instrument", "voltage_kv", "working_distance_mm", "detector", "magnification"):
        if k in meta:
            summary[k] = meta[k]

    # quality checks
    if math.isfinite(porosity) and (porosity > 0.6 or porosity < 0.005):
        warns.append(f"porosity {100 * porosity:.1f}%: check the overlay - are the pores the darkest class "
                     "(otherwise tick 'Pores are bright') and is the number of thresholds right?")
    if 0 < len(inc) < 20:
        warns.append(f"only {len(inc)} pores: pore-size statistics are uncertain")
    if len(rows) >= 10 and summary["n_pores_edge"] / len(rows) > 0.25:
        warns.append(f"{100 * summary['n_pores_edge'] / len(rows):.0f}% of the pores are cut by the image edge: "
                     "their sizes are underestimated (consider a lower magnification or leaving them out of the "
                     "size statistics)")
    if a_px.size and n_valid and a_px.max() / n_valid > 0.1:
        warns.append(f"one pore covers {100 * a_px.max() / n_valid:.0f}% of the image: a defect, or the threshold "
                     "joined separate regions")
    scale = _full_scale(grey, img)
    if np.issubdtype(grey.dtype, np.integer):
        sat = float((grey[valid] >= scale).mean()) if n_valid else 0.0
        summary["saturated_fraction"] = sat
        if sat > 0.02:
            warns.append(f"{100 * sat:.1f}% of the pixels are saturated (charging or too much contrast)")
    lo, hi = np.percentile(grey[valid], [1, 99]) if n_valid else (0, 0)
    if n_valid and (hi - lo) < 0.1 * scale:
        warns.append("low contrast: pores and solid may not separate well")

    layers = None
    if keep_layers:
        layers = {"grey": grey, "solid": solid, "pores": pores, "pore_labels": labels, "valid": valid,
                  "grey_full": full, "crop": crop}
    return PorosityResult(img.name, sample, summary, rows, warns, layers, repetition, image_id)


def summarise_porosity(results: list[PorosityResult]) -> list[dict]:
    """Per sample: mean ± SD over images (kept from version 0.2; see ``materials.experiment``)."""
    by: dict[str, list[PorosityResult]] = {}
    for r in results:
        by.setdefault(r.sample or "(none)", []).append(r)
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
                "mean_pore_radius_um_mean": float(np.nanmean(mr)) if np.isfinite(mr).any() else math.nan,
                "mean_pore_radius_um_sd": float(np.nanstd(mr, ddof=1)) if len(rs) > 1 else math.nan,
                "pooled_n_pores": int(rad.size),
                "pooled_mean_pore_radius_um": float(np.nanmean(rad)) if rad.size else math.nan,
                "pooled_sd_pore_radius_um": float(np.nanstd(rad, ddof=1)) if rad.size > 1 else math.nan,
            }
        )
    return rows
