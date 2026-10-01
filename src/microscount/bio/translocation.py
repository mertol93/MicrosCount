"""Nuclear translocation: nuclear/cytoplasmic (N/C) intensity ratios.

Paper method (always computed)
    Noursadeghi M. et al. (2008) J Immunol Methods 329:194-200, step by step:

    1. for each high-power field, a 3x3 median filter is applied to the DAPI and
       rel A (target) images;
    2. each filtered image is converted to a binary mask by ImageJ's automatic
       IsoData threshold (ImageJ 1.39 "Auto", pixels >= level), keeping the
       fluorescence above background;
    3. nuclear ROI = DAPI mask; cytoplasmic ROI = rel A mask minus DAPI mask
       (ImageJ image calculator, 8-bit subtraction);
    4. both masks are applied to the *original* rel A image, the ImageJ histograms
       of the two ROIs are normalised by their number of data points (the zero bin,
       which holds everything outside the mask, is not a data point) and the sums
       of the normalised staining intensities are compared: N/C = mean nuclear rel A
       / mean cytoplasmic rel A over the non-zero ROI pixels;
    5. per condition, mean +/- SD over (five) high-power fields.

Per-cell lab protocol (``method="per_cell"``, the default; adds to the paper method)
    The laboratory's ImageJ protocol, automated for every cell:

    1. nuclei (blue): the nuclear stain is thresholded (after light Gaussian smoothing),
       holes are filled, touching nuclei are split (watershed), and the particles are
       filtered as by *Analyze Particles* - Size 0-Infinity px^2, Circularity 0.2-1.0,
       with ImageJ's traced perimeter so that the values match ImageJ;
    2. target (green): *Process > Subtract Background*, rolling ball radius 50 px
       (30-50 px in the protocol), ImageJ's algorithm reproduced exactly;
    3. mean grey values of the corrected target in each nucleus (Nuc_Mean), in a
       perinuclear cytoplasmic ring (Cyto_Mean) and in the cell-free area of the field
       (Background_Mean); the mean grey value of the nuclear stain is reported too;
    4. Nuc_corr = Nuc_Mean - Background_Mean, Cyto_corr = Cyto_Mean - Background_Mean,
       N/C = Nuc_corr / Cyto_corr, and its inverse C/N = Cyto_corr / Nuc_corr.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Callable

import numpy as np
from scipy import ndimage as ndi

from ..core.imageio import ChannelError, LoadedImage
from ..core.thresholds import compute_threshold

NUCLEAR_TOKENS = ("dapi", "hoechst", "nuclear", "nuclei", "nucleus", "draq5", "h33342", "sytox", "pi")


# ---------------------------------------------------------------------- settings


@dataclass
class TranslocationSettings:
    """Defaults: the paper method (exact) plus the per-cell lab protocol.

    ``paper()`` runs the published method only; the paper ratio is identical either way.
    """

    method: str = "per_cell"  # "per_cell" = paper + per-cell lab protocol | "paper" = paper only
    nuclear_channel: str = "auto"
    target_channel: str = "auto"
    pixel_size_um: float | None = None
    exclude_annotations: bool = True
    exclude_rects: list = field(default_factory=list)  # [[x0, y0, x1, y1], ...] pixels
    # ---- paper method (Noursadeghi et al. 2008); the thresholds also find the nuclei
    median_size: int = 3
    nuclear_threshold: str = "ij139_auto"
    nuclear_threshold_value: float | None = None
    target_threshold: str = "ij139_auto"
    target_threshold_value: float | None = None
    legacy_inclusive_threshold: bool = False  # only affects the ImageJ 1.42+ variant
    exclude_zero_pixels: bool = True  # the paper's histograms drop the zero bin
    fields_per_condition: int = 5  # the paper's sampling criteria (checked, not enforced)
    min_cells_per_condition: int = 500
    # ---- per-cell lab protocol: nuclei (blue) as ImageJ Analyze Particles
    segmentation_smoothing_px: float = 2.0  # Gaussian sigma before thresholding the nuclear stain
    split_touching: bool = True  # watershed, as Process > Binary > Watershed
    split_sensitivity: float = 0.10  # neck depth needed to split, x nucleus diameter
    nucleus_diameter_px: float | None = None  # None = estimated from the image
    size_min_px2: float = 0.0
    size_max_px2: float | None = None  # None = Infinity
    circularity_min: float = 0.2
    circularity_max: float = 1.0
    exclude_border_cells: bool = True  # as "Exclude on edges"
    # ---- per-cell lab protocol: target (green)
    rolling_ball_radius: float = 50.0  # Subtract Background radius in px; 0 = off
    background: str = "auto"  # Background_Mean: "auto" (cell-free area) | "manual" | "none"
    background_value: float = 0.0
    cytoplasm: str = "ring"  # "ring" | "territory"
    ring_gap_px: float = 0.0
    ring_width_px: float | None = None  # None = 0.3 x nucleus diameter
    territory_px: float | None = None  # None = 1.0 x nucleus diameter
    restrict_to_cells: bool = True  # drop ring pixels at background level (off the cell)
    cell_detection_sigmas: float = 3.0
    nucleus_erode_px: int = 0
    min_cytoplasm_pixels: int = 15
    exclude_saturated: bool = False
    max_saturated_fraction: float = 0.05
    max_area_fraction: float | None = None  # optional: exclude nuclei > this x typical area
    min_solidity: float | None = None  # optional: exclude nuclei less convex than this
    responder_ratio: float | None = 1.0  # per field: fraction of cells with N/C above this
    # ---- experiment design (see bio/experiment.py)
    conditions_from: str = "auto"  # "auto" | "name" (file names) | "folder" (folder names)
    control_condition: str = ""  # control: responder cut-off, default fold reference and comparisons
    fold_references: dict = field(default_factory=dict)  # condition -> its reference for the fold change
    comparisons: list = field(default_factory=list)  # [[A, B], ...]; empty = each condition vs the control
    condition_order: list = field(default_factory=list)
    responder_percentile: float | None = 95.0  # responders: above this percentile of the control's cells
    exclude_failed_paper_fields: bool = True  # leave fields with a failed automatic threshold out of paper means

    @classmethod
    def paper(cls) -> "TranslocationSettings":
        """Noursadeghi et al. (2008) as published, without the per-cell step."""
        return cls(method="paper")

    @classmethod
    def per_cell(cls) -> "TranslocationSettings":
        """The paper method plus the per-cell lab protocol (the defaults)."""
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TranslocationSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


# ---------------------------------------------------------------------- results


@dataclass
class FieldResult:
    field_id: str
    condition: str
    nuclear_file: str
    target_file: str
    method: str
    summary: dict
    cells: list[dict]
    warnings: list[str]
    layers: dict | None = None  # arrays for overlays / preview
    histograms: dict | None = None  # normalised ROI histograms (paper Fig. 2B)
    repetition: str = ""

    @property
    def ratio(self) -> float:
        """Headline ratio of this field: median per-cell N/C, or the paper ratio."""
        key = "paper_ratio" if self.method == "paper" else "median_nc"
        return float(self.summary.get(key, math.nan))


# ---------------------------------------------------------------------- helpers


def role_channel(img: LoadedImage, spec: str | int | None, role: str) -> int:
    """Channel index for the nuclear or target role, resolving ``"auto"``."""
    if spec not in (None, "", "auto"):
        return img.resolve_channel(spec)
    if img.n_channels == 1:
        return 0
    if img.kind == "single_colour" and img.signal_channel is not None:
        return img.signal_channel
    if img.kind == "merged_rgb":
        hi = [float(np.percentile(img.data[c, ::8, ::8], 99.5)) for c in range(3)]
        if role == "nuclear":
            return 2
        # target: brightest non-blue channel (green preferred on ties)
        return 1 if hi[1] >= hi[0] else 0
    names = [n.lower() for n in img.channel_names]
    nuc = [i for i, n in enumerate(names) if any(t in n for t in NUCLEAR_TOKENS)]
    if role == "nuclear":
        if nuc:
            return nuc[0]
        raise ChannelError(f"{img.name}: choose the nuclear-stain channel ({', '.join(img.channel_names)})")
    others = [i for i in range(img.n_channels) if i not in nuc]
    if nuc and len(others) == 1:
        return others[0]
    raise ChannelError(f"{img.name}: choose the target channel ({', '.join(img.channel_names)})")


def _disk(radius: float) -> np.ndarray:
    r = int(math.ceil(radius))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= radius * radius + 1e-9


def _median(img: np.ndarray, size: int) -> np.ndarray:
    """Median filter with edge replication (ImageJ "Median..." radius 1 = 3x3)."""
    if size is None or size <= 1:
        return img
    if int(size) == 3:
        return median3x3(img)
    return ndi.median_filter(img, size=int(size), mode="nearest")


def median3x3(img: np.ndarray) -> np.ndarray:
    """Exact 3x3 median (edges replicated) with a 19-comparator sorting network.

    Identical to ``ndimage.median_filter(img, 3, mode="nearest")`` but ~5x faster.
    """
    a = np.pad(img, 1, mode="edge")
    h, w = img.shape
    p = [a[dy : dy + h, dx : dx + w].copy() for dy in range(3) for dx in range(3)]

    def s(i, j):
        lo = np.minimum(p[i], p[j])
        np.maximum(p[i], p[j], out=p[j])
        p[i] = lo

    for i, j in ((1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8), (0, 3), (5, 8),
                 (4, 7), (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4), (4, 2)):
        s(i, j)
    return p[4]


def _dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Binary dilation by a Euclidean disk (distance-transform based for large radii)."""
    if radius <= 0:
        return mask
    if radius <= 1.5:
        return ndi.binary_dilation(mask, structure=_disk(radius))
    if not mask.any():
        return mask.copy()
    return ndi.distance_transform_edt(~mask) <= radius


def valid_mask(shape, images: list[LoadedImage], s: TranslocationSettings) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    if s.exclude_annotations:
        for im in images:
            if im.annotation_mask is not None:
                valid &= ~im.annotation_mask
    for rect in s.exclude_rects or []:
        x0, y0, x1, y1 = (int(round(v)) for v in rect)
        valid[max(0, min(y0, y1)) : max(y0, y1), max(0, min(x0, x1)) : max(x0, x1)] = False
    return valid


def estimate_background(img_f: np.ndarray, region: np.ndarray) -> tuple[float, float]:
    """Mode and left-side spread (sigma) of the intensity histogram in ``region``."""
    vals = img_f[region].astype(np.float64)
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [0.1, 99.0])
    if np.issubdtype(img_f.dtype, np.integer) and hi - lo <= 4096:
        edges = np.arange(math.floor(lo), math.ceil(hi) + 2) - 0.5
    else:
        edges = np.linspace(lo, hi if hi > lo else lo + 1, 513)
    h, edges = np.histogram(vals, bins=edges)
    hs = ndi.gaussian_filter1d(h.astype(np.float64), sigma=max(1.0, len(h) / 256.0))
    k = int(np.argmax(hs))
    mode = 0.5 * (edges[k] + edges[k + 1])
    left = vals[vals <= mode]
    bw = edges[1] - edges[0]
    if left.size > 50:
        sigma = math.sqrt(float(np.mean((left - mode) ** 2)))
    else:
        sigma = float(np.std(vals))
    return float(mode), max(sigma, 0.5 * bw, 1e-6)


def channel_shift(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel shift (dy, dx) of ``b`` relative to ``a`` and a similarity score."""
    from skimage.registration import phase_cross_correlation

    f = 2 if min(a.shape) >= 512 else 1
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    v = valid[:h, :w].reshape(h // f, f, w // f, f).all(axis=(1, 3))

    def bandpass(x):
        x = x[:h, :w].astype(np.float32).reshape(h // f, f, w // f, f).mean(axis=(1, 3))
        x = np.where(v, x, np.median(x[v]) if v.any() else 0)
        return ndi.gaussian_filter(x, 1.5 / f) - ndi.gaussian_filter(x, 12.0 / f)

    fa, fb = bandpass(a), bandpass(b)
    try:
        shift, _, _ = phase_cross_correlation(fa, fb, upsample_factor=20, normalization=None)
    except Exception:  # noqa: BLE001
        return math.nan, math.nan, math.nan
    r = float(np.corrcoef(fa[v].ravel(), fb[v].ravel())[0, 1]) if v.sum() > 10 else math.nan
    return float(shift[0]) * f, float(shift[1]) * f, r


def _erode_labels(labels: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return labels
    from skimage.segmentation import find_boundaries

    out = labels.copy()
    for _ in range(int(r)):
        out[find_boundaries(out, mode="inner")] = 0
    return out


# ---------------------------------------------------------------------- segmentation


def _label_max(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Maximum of ``values`` within each label 1..max (fast sort/reduceat)."""
    sel = labels > 0
    lab = labels[sel]
    val = values[sel].astype(np.float64)
    order = np.argsort(lab, kind="stable")
    lab, val = lab[order], val[order]
    starts = np.r_[0, np.nonzero(np.diff(lab))[0] + 1]
    return np.maximum.reduceat(val, starts) if val.size else np.zeros(0)


def estimate_nucleus_diameter(labels: np.ndarray, dist: np.ndarray | None = None) -> float:
    """Typical nucleus diameter from a labelled nuclear mask.

    Uses the maximum of the distance transform inside each object (its inscribed
    radius), which is the same for a single nucleus and for a clump of touching
    ones, so clumps do not inflate the estimate and splitting cannot feed back
    into it. Objects smaller than 3x3 are ignored as noise.
    """
    if dist is None:
        dist = ndi.distance_transform_edt(labels > 0)
    n = int(labels.max())
    if n == 0:
        return 10.0
    rmax = _label_max(dist, labels)
    rmax = rmax[rmax >= 2.0]
    if rmax.size == 0:
        return 4.0
    # inscribed radius underestimates the equivalent radius of an ellipse by
    # sqrt(a/b); 1.15 corrects for the typical 1.3:1 aspect of nuclei
    return float(2.0 * 1.15 * np.median(rmax))


def segment_nuclei(
    nuc_f: np.ndarray,
    mask: np.ndarray,
    diameter: float | None = None,
    split: bool = True,
    sensitivity: float = 0.10,
    min_area_fraction: float = 0.30,
    opening: bool = True,
    connectivity: int = 1,
) -> tuple[np.ndarray, dict]:
    """Label individual nuclei in a binary nuclear mask.

    Holes are filled; ``opening`` removes one-pixel spurs and specks; ``connectivity`` 2
    joins diagonal neighbours into one particle, as ImageJ's Analyze Particles does.
    """
    from skimage.morphology import h_maxima
    from skimage.segmentation import relabel_sequential, watershed

    m = ndi.binary_fill_holes(mask)
    if opening:
        m = ndi.binary_opening(m)
    lab0, n0 = ndi.label(m, structure=np.ones((3, 3)) if connectivity == 2 else None)
    info = {"components": int(n0), "diameter": diameter, "diameter_estimated": diameter is None}
    if n0 == 0:
        info["diameter"] = float(diameter) if diameter else math.nan
        return np.zeros(mask.shape, np.int32), info
    dist = ndi.distance_transform_edt(m)
    if diameter is None:
        diameter = estimate_nucleus_diameter(lab0, dist)
    a_ref = math.pi * (diameter / 2.0) ** 2
    labels = lab0.astype(np.int32)
    n_split = 0
    if split:
        # only objects larger than one typical nucleus can hold several nuclei
        areas0 = np.bincount(lab0.ravel())
        clumps = np.nonzero(areas0[1:] > 1.2 * a_ref)[0] + 1
        slices = ndi.find_objects(lab0)
        sigma = max(0.5, 0.05 * diameter)
        h = max(0.5, sensitivity * diameter)
        next_label = n0 + 1
        hh, ww = m.shape
        for c in clumps:
            sy, sx = slices[c - 1]
            sl = (slice(max(0, sy.start - 3), min(hh, sy.stop + 3)), slice(max(0, sx.start - 3), min(ww, sx.stop + 3)))
            sub = lab0[sl] == c
            dsub = ndi.gaussian_filter(np.where(sub, dist[sl], 0.0), sigma)
            peaks = h_maxima(dsub, h) & sub
            markers, k = ndi.label(peaks, structure=np.ones((3, 3)))
            if k <= 1:
                continue
            ws = watershed(-dsub, markers, mask=sub)
            region = labels[sl]
            for j in range(2, k + 1):
                region[ws == j] = next_label
                next_label += 1
            n_split += k - 1
    areas = np.bincount(labels.ravel())[1:]
    small = np.nonzero(areas < min_area_fraction * a_ref)[0] + 1 if min_area_fraction > 0 else np.zeros(0, int)
    if small.size:
        labels = np.where(np.isin(labels, small), 0, labels)
    labels, _, _ = relabel_sequential(labels.astype(np.int32))
    info.update(diameter=float(diameter), removed_small=int(small.size), n_nuclei=int(labels.max()), splits=n_split)
    return labels.astype(np.int32), info


# ---------------------------------------------------------------------- analysis


def analyse_field(
    nuc: LoadedImage,
    tgt: LoadedImage,
    settings: TranslocationSettings | None = None,
    field_id: str = "",
    condition: str = "",
    keep_layers: bool = True,
    repetition: str = "",
) -> FieldResult:
    s = settings or TranslocationSettings()
    per_cell = s.method != "paper"
    warns: list[str] = []
    n_idx = role_channel(nuc, s.nuclear_channel, "nuclear")
    t_idx = role_channel(tgt, s.target_channel, "target")
    n_raw = nuc.data[n_idx]
    t_raw = tgt.data[t_idx]
    if n_raw.shape != t_raw.shape:
        raise ValueError(
            f"image sizes differ: {nuc.name} {n_raw.shape[::-1]} vs {tgt.name} {t_raw.shape[::-1]}"
        )
    same_file = nuc.path == tgt.path
    if same_file and n_idx == t_idx:
        raise ChannelError(f"{nuc.name}: nuclear and target channels are the same")
    for im in {id(nuc): nuc, id(tgt): tgt}.values():
        if im.lossy:
            warns.append(f"{im.name}: JPEG compression alters intensities")
    valid = valid_mask(n_raw.shape, [nuc, tgt], s)
    if valid.sum() < 100:
        raise ValueError("almost no analysable pixels left after exclusions")

    # ---- 1. paper method, exactly as published (always computed). The ROI masks are
    # applied to the original rel A image; ImageJ histograms of the masked images are
    # normalised by their number of data points - the zero bin holds everything outside
    # the mask, so zero-valued pixels are not data points - and the sums of normalised
    # intensity (= mean intensities) compared. No background, no saturation handling.
    n_f = _median(n_raw, s.median_size)
    t_f = _median(t_raw, s.median_size)
    thr_n = compute_threshold(n_f, s.nuclear_threshold, valid, s.nuclear_threshold_value, s.legacy_inclusive_threshold)
    thr_t = compute_threshold(t_f, s.target_threshold, valid, s.target_threshold_value, s.legacy_inclusive_threshold)
    mask_n = (n_f > thr_n.value) & valid
    mask_t = (t_f > thr_t.value) & valid
    paper_meas = valid & (t_raw > 0) if s.exclude_zero_pixels else valid
    nuc_roi = mask_n & paper_meas
    cyto_roi = mask_t & ~mask_n & paper_meas
    t_val = t_raw.astype(np.float64)
    p_n = float(t_val[nuc_roi].mean()) if nuc_roi.any() else math.nan
    p_c = float(t_val[cyto_roi].mean()) if cyto_roi.any() else math.nan
    paper_ratio = p_n / p_c if (p_c and math.isfinite(p_c) and p_c > 0) else math.nan
    histograms = roi_histograms(t_raw, nuc_roi, cyto_roi)
    n_valid = max(1, int(valid.sum()))
    frac_n = float(mask_n.sum()) / n_valid
    frac_t = float(mask_t.sum()) / n_valid
    for thr, name in ((thr_n, "nuclear"), (thr_t, "target")):
        if thr.note:
            warns.append(f"{name} threshold: {thr.note}")
    paper_ok = 0.002 <= frac_n <= 0.6 and 0.005 <= frac_t <= 0.97
    if frac_n > 0.6 or frac_n < 0.002:
        warns.append(f"the nuclear mask covers {100 * frac_n:.1f}% of the field: the automatic threshold has "
                     "probably failed (check the overlay; try another ImageJ threshold variant)")
    if frac_t > 0.97 or frac_t < 0.005:
        warns.append(f"the target mask covers {100 * frac_t:.1f}% of the field: check the target threshold")

    sat = tgt.saturation_value
    saturated = (t_raw >= sat) if sat is not None else np.zeros(t_raw.shape, bool)

    # ---- 2. individual nuclei: particles for the per-cell step, or a count for the paper's
    # ">= 500 cells" criterion. Smoothing keeps dim, noisy nuclei whole.
    sigma = float(s.segmentation_smoothing_px or 0)
    n_seg = ndi.gaussian_filter(n_raw.astype(np.float32), sigma) if sigma > 0 else n_f
    thr_seg = compute_threshold(n_seg, s.nuclear_threshold, valid, s.nuclear_threshold_value,
                                s.legacy_inclusive_threshold)
    seg_mask = (n_seg > thr_seg.value) & valid
    if per_cell:
        # as ImageJ: threshold, fill holes, (watershed), 8-connected particles, no size cut here
        labels, seg = segment_nuclei(n_seg, seg_mask, s.nucleus_diameter_px, s.split_touching, s.split_sensitivity,
                                     min_area_fraction=0.0, opening=False, connectivity=2)
    else:
        labels, seg = segment_nuclei(n_seg, seg_mask, s.nucleus_diameter_px, True, s.split_sensitivity, 0.30)
    diameter = seg.get("diameter") or s.nucleus_diameter_px
    if not diameter or not math.isfinite(diameter):
        diameter = 10.0

    dy, dx, rcorr = channel_shift(n_raw, t_raw, valid)
    if math.isfinite(dy) and math.hypot(dy, dx) > 1.5 and abs(rcorr) > 0.2:
        warns.append(f"channels offset by ({dy:.1f}, {dx:.1f}) px; check registration")

    summary = {
        "field": field_id,
        "condition": condition,
        "repetition": repetition,
        "nuclear_file": nuc.name,
        "target_file": tgt.name,
        "nuclear_channel": nuc.channel_names[n_idx],
        "target_channel": tgt.channel_names[t_idx],
        "method": s.method,
        "threshold_method": s.nuclear_threshold if s.nuclear_threshold == s.target_threshold
        else f"{s.nuclear_threshold} / {s.target_threshold}",
        "paper_ratio": paper_ratio,
        "paper_nuclear_mean": p_n,
        "paper_cytoplasm_mean": p_c,
        "paper_nuclear_area_px": int(nuc_roi.sum()),
        "paper_cytoplasm_area_px": int(cyto_roi.sum()),
        "nuclear_threshold": thr_n.value,
        "target_threshold": thr_t.value,
        "nuclear_threshold_ij8": thr_n.level8,
        "target_threshold_ij8": thr_t.level8,
        "nuclei_count": int(labels.max()),
        "criterion_fields": int(s.fields_per_condition),
        "criterion_cells": int(s.min_cells_per_condition),
        "nuclear_mask_fraction": frac_n,
        "target_mask_fraction": frac_t,
        "paper_ok": paper_ok,
        "nucleus_diameter_px": float(diameter),
        "saturated_fraction_nuclear": float(saturated[mask_n].mean()) if mask_n.any() else 0.0,
        "saturated_fraction_cytoplasm": float(saturated[mask_t & ~mask_n].mean())
        if (mask_t & ~mask_n).any()
        else 0.0,
        "channel_shift_dy": dy,
        "channel_shift_dx": dx,
        "channel_similarity": rcorr,
        "valid_fraction": float(valid.mean()),
    }
    if s.pixel_size_um:
        summary["pixel_size_um"] = float(s.pixel_size_um)
        summary["nucleus_diameter_um"] = float(diameter) * float(s.pixel_size_um)
    if summary["saturated_fraction_nuclear"] > 0.01:
        warns.append(
            f"{100 * summary['saturated_fraction_nuclear']:.1f}% of nuclear pixels are saturated "
            "(ratios are underestimated)"
        )

    layers = None
    if keep_layers:
        layers = {
            "nuclear": n_raw,
            "target": t_raw,
            "valid": valid,
            "paper_nuclear": nuc_roi,
            "paper_cytoplasm": cyto_roi,
        }

    cells: list[dict] = []
    if per_cell:
        summary["segmentation_threshold"] = thr_seg.value
        cells, cell_layers, cell_summary, cell_warns = _per_cell(
            labels, seg_mask, n_raw, t_raw, valid, saturated, diameter, s
        )
        summary.update(cell_summary)
        summary["nuclei_count"] = cell_summary["n_nuclei"]
        warns.extend(cell_warns)
        if layers is not None:
            layers.update(cell_layers)
    for c in cells:
        c["field"] = field_id
        c["condition"] = condition
        c["repetition"] = repetition
    res = FieldResult(field_id, condition, nuc.name, tgt.name, s.method, summary, cells, warns, layers,
                      repetition=repetition)
    res.histograms = histograms
    return res


def roi_histograms(values: np.ndarray, nuc_roi: np.ndarray, cyto_roi: np.ndarray) -> dict:
    """Normalised intensity histograms (frequency %, as Fig. 2B of the paper)."""
    v = np.asarray(values)
    if v.dtype == np.uint8:
        edges = np.arange(257) - 0.5
    else:
        hi = float(v.max()) if v.size else 1.0
        edges = np.linspace(0, hi if hi > 0 else 1.0, 257)
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = {"intensity": centres}
    for key, roi in (("nuclear_pct", nuc_roi), ("cytoplasm_pct", cyto_roi)):
        h, _ = np.histogram(v[roi], bins=edges)
        tot = h.sum()
        out[key] = 100.0 * h / tot if tot else np.zeros(len(centres))
    return out


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    areas = np.bincount(lab.ravel())
    keep = areas >= min_size
    keep[0] = False
    return keep[lab]


def _stats(v: np.ndarray, prefix: str) -> dict:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    nan = math.nan
    return {
        f"median_{prefix}": float(np.median(v)) if v.size else nan,
        f"q1_{prefix}": float(np.percentile(v, 25)) if v.size else nan,
        f"q3_{prefix}": float(np.percentile(v, 75)) if v.size else nan,
        f"mean_{prefix}": float(np.mean(v)) if v.size else nan,
        f"sd_{prefix}": float(np.std(v, ddof=1)) if v.size > 1 else nan,
    }


def _per_cell(labels, nuclear_mask, n_raw, t_raw, valid, saturated, d, s):
    """The lab protocol: particle filters, rolling ball, background-corrected means per cell."""
    from skimage.measure import regionprops_table

    from ..core.imagej import particle_shape, subtract_background

    warns: list[str] = []
    h, w = labels.shape
    n = int(labels.max())
    a_ref = math.pi * (d / 2.0) ** 2

    # target: Process > Subtract Background (rolling ball), measured without further filtering
    radius = float(s.rolling_ball_radius or 0)
    corrected = subtract_background(t_raw, radius) if radius > 0 else t_raw
    g = corrected.astype(np.float64)
    g_f = _median(corrected, s.median_size)  # only for deciding which pixels belong to cells

    # where the cells are: target above the background noise, or nuclear stain
    nuclear_any = nuclear_mask | (labels > 0)
    grow = max(2, int(round(0.5 * d)))
    outside = valid & ~_dilate(nuclear_any, grow)
    if outside.sum() < max(1000, 0.005 * valid.sum()):
        outside = valid & ~nuclear_any
    bg_mode, bg_sigma = estimate_background(g_f, outside if outside.any() else valid)
    cell_thr = bg_mode + s.cell_detection_sigmas * bg_sigma
    on_cells = (g_f > cell_thr) & valid

    # Background_Mean: mean grey value of the corrected target in the cell-free area
    cellish = _remove_small(on_cells, max(9, int(0.1 * a_ref))) | nuclear_any
    free = valid & ~_dilate(cellish, max(2, int(round(0.15 * d))))
    n_free = int(free.sum())
    if s.background == "none":
        bgm, source = 0.0, "none"
    elif s.background == "manual":
        bgm, source = float(s.background_value), "manual"
    elif n_free >= max(1000, 0.002 * valid.sum()):
        bgm, source = float(g[free].mean()), "mean of the cell-free area"
    else:
        bgm = float(estimate_background(corrected, outside if outside.any() else valid)[0])
        source = "most common value outside nuclei"
        warns.append("almost no cell-free area in this field: Background_Mean was taken as the most common "
                     "target value outside nuclei; consider a manual value")

    # cytoplasm: a ring around each nucleus (or its territory), split between neighbours
    gap = float(s.ring_gap_px or 0)
    if s.cytoplasm == "territory":
        reach = float(s.territory_px) if s.territory_px else float(d)
    else:
        reach = float(s.ring_width_px) if s.ring_width_px else max(2.0, float(round(0.3 * d)))
    if n:
        dist_bg, (iy, ix) = ndi.distance_transform_edt(labels == 0, return_indices=True)
        nearest = labels[iy, ix]
        inner = np.where(dist_bg <= gap, nearest, 0) if gap > 0 else labels
        outer = np.where(dist_bg <= gap + reach, nearest, 0)
        del iy, ix, dist_bg
    else:
        inner = outer = labels
    blocked = _dilate(nuclear_any, gap)
    cyto = np.where((inner == 0) & ~blocked & valid, outer, 0)
    cyto_before = cyto.copy()
    if s.restrict_to_cells:
        cyto = np.where(on_cells, cyto, 0)

    nuc_meas = _erode_labels(labels, s.nucleus_erode_px)
    counts_er = np.bincount(nuc_meas.ravel(), minlength=n + 1)
    lost = np.nonzero(counts_er[1:] == 0)[0] + 1
    if lost.size:  # fall back to the full nucleus where trimming removed everything
        nuc_meas = np.where(np.isin(labels, lost), labels, nuc_meas)
    measurable = valid & ~saturated if s.exclude_saturated else valid

    def sums(lab, values):
        sel = measurable & (lab > 0)
        idx = lab[sel]
        cnt = np.bincount(idx, minlength=n + 1).astype(np.float64)
        tot = np.bincount(idx, weights=values[sel], minlength=n + 1)
        return cnt, tot

    n_cnt, n_sum = sums(nuc_meas, g)
    c_cnt, c_sum = sums(cyto, g)
    sat_f = saturated.ravel().astype(np.float64)
    n_sat = np.bincount(labels.ravel(), weights=sat_f, minlength=n + 1)
    n_all = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    c_sat = np.bincount(cyto.ravel(), weights=sat_f, minlength=n + 1)
    c_all = np.bincount(cyto.ravel(), minlength=n + 1).astype(np.float64)
    c_before = np.bincount(cyto_before.ravel(), minlength=n + 1).astype(np.float64)
    dapi_sum = np.bincount(labels.ravel(), weights=n_raw.ravel().astype(np.float64), minlength=n + 1)
    if (~valid).any():
        near_invalid = ndi.binary_dilation(~valid, iterations=2)
        touches_invalid = np.bincount(labels[near_invalid].ravel(), minlength=n + 1) > 0
    else:
        touches_invalid = np.zeros(n + 1, bool)

    shapes = particle_shape(labels, n)  # ImageJ area, traced perimeter, circularity
    want = ("label", "centroid", "bbox") + (("solidity",) if s.min_solidity else ())
    if n:
        props = regionprops_table(labels, properties=want)
    else:
        props = {k: np.array([]) for k in ("label", "centroid-0", "centroid-1", "bbox-0", "bbox-1", "bbox-2", "bbox-3")}
    size_min = float(s.size_min_px2 or 0)
    size_max = float(s.size_max_px2) if s.size_max_px2 else math.inf
    px = float(s.pixel_size_um) if s.pixel_size_um else None
    status = np.array(["ok"] * (n + 1), dtype=object)
    cells = []
    n_particles_ok = 0
    for i in range(len(props["label"])):
        lab = int(props["label"][i])
        sh = shapes.get(lab, {"area": 0, "perimeter": 0.0, "circularity": 0.0})
        area, circ = sh["area"], sh["circularity"]
        border = props["bbox-0"][i] == 0 or props["bbox-1"][i] == 0 or props["bbox-2"][i] == h or props["bbox-3"][i] == w
        nm = n_sum[lab] / n_cnt[lab] if n_cnt[lab] else math.nan
        cm = c_sum[lab] / c_cnt[lab] if c_cnt[lab] else math.nan
        nuc_corr, cyto_corr = nm - bgm, cm - bgm
        nc = nuc_corr / cyto_corr if cyto_corr > 0 else math.nan
        cn = cyto_corr / nuc_corr if nuc_corr > 0 else math.nan
        nsf = n_sat[lab] / n_all[lab] if n_all[lab] else 0.0
        csf = c_sat[lab] / c_all[lab] if c_all[lab] else 0.0
        in_filter = size_min <= area <= size_max and s.circularity_min <= circ <= s.circularity_max
        n_particles_ok += in_filter
        reason = "ok"
        if not size_min <= area <= size_max:
            reason = "size outside the particle filter"
        elif not s.circularity_min <= circ <= s.circularity_max:
            reason = "circularity outside the particle filter"
        elif s.exclude_border_cells and border:
            reason = "touches image edge"
        elif touches_invalid[lab]:
            reason = "touches excluded region"
        elif s.max_area_fraction and area > s.max_area_fraction * a_ref:
            reason = "too large (clump?)"
        elif s.min_solidity and props["solidity"][i] < s.min_solidity:
            reason = "irregular shape (merged nuclei?)"
        elif c_cnt[lab] < s.min_cytoplasm_pixels:
            reason = "too little cytoplasm"
        elif s.exclude_saturated and max(nsf, csf) > s.max_saturated_fraction:
            reason = "saturated"
        elif not cyto_corr > 0:
            reason = "cytoplasm at background level (Cyto_corr <= 0)"
        status[lab] = reason
        row = {
            "cell": lab,
            "x": float(props["centroid-1"][i]),
            "y": float(props["centroid-0"][i]),
            "included": reason == "ok",
            "exclusion_reason": "" if reason == "ok" else reason,
            "nucleus_area_px": int(area),
            "nucleus_perimeter_px": float(sh["perimeter"]),
            "circularity": float(circ),
            "dapi_mean": float(dapi_sum[lab] / n_all[lab]) if n_all[lab] else math.nan,
            "nuc_mean": float(nm),
            "cyto_mean": float(cm),
            "background_mean": bgm,
            "nuc_corr": float(nuc_corr),
            "cyto_corr": float(cyto_corr),
            "nc_ratio": float(nc),
            "cn_ratio": float(cn),
            "log2_nc": math.log2(nc) if nc > 0 else math.nan,
            "nuclear_pixels": int(n_cnt[lab]),
            "cytoplasm_pixels": int(c_cnt[lab]),
            "cytoplasm_coverage": float(c_all[lab] / c_before[lab]) if c_before[lab] else 0.0,
            "saturated_fraction_nucleus": float(nsf),
            "saturated_fraction_cytoplasm": float(csf),
        }
        if px:
            row["nucleus_area_um2"] = area * px * px
            row["nucleus_perimeter_um"] = float(sh["perimeter"]) * px
        cells.append(row)

    inc = [c for c in cells if c["included"]]
    summ = {
        "n_particles": int(n),
        "n_nuclei": int(n_particles_ok),
        "n_cells_analysed": len(inc),
        **_stats([c["nc_ratio"] for c in inc], "nc"),
        **{k: v for k, v in _stats([c["cn_ratio"] for c in inc], "cn").items() if k.startswith(("median", "mean"))},
        "mean_nuc_corr": float(np.mean([c["nuc_corr"] for c in inc])) if inc else math.nan,
        "mean_cyto_corr": float(np.mean([c["cyto_corr"] for c in inc])) if inc else math.nan,
        "background_mean": bgm,
        "background_source": source,
        "background_area_px": n_free,
        "rolling_ball_radius": radius,
        "cell_threshold": float(cell_thr),
        "ring_width_px": float(reach) if s.cytoplasm != "territory" else math.nan,
        "territory_px": float(reach) if s.cytoplasm == "territory" else math.nan,
        "ring_gap_px": gap,
    }
    nc_inc = np.array([c["nc_ratio"] for c in inc], dtype=np.float64)
    if s.responder_ratio is not None and nc_inc.size:
        summ["responder_fraction"] = float(np.mean(nc_inc > s.responder_ratio))
        summ["responder_cutoff"] = float(s.responder_ratio)
    if not inc and n > 0:
        warns.append("no cell passed the filters")
    if inc:
        cut = max(2, int(round(0.1 * a_ref)))
        small = sum(1 for c in inc if c["nucleus_area_px"] < cut)
        if small:
            warns.append(f"{small} measured particle(s) are smaller than a tenth of a typical nucleus (< {cut} px², "
                         f"probably specks of debris or noise) but pass the Size {size_min:g}–"
                         f"{'Infinity' if math.isinf(size_max) else f'{size_max:g}'} px² filter; a minimum size "
                         f"of about {cut} px² would exclude them")
        cyto_level = float(np.median([c["cyto_mean"] for c in inc]))
        if s.background == "auto" and bgm > 0.5 * cyto_level:
            warns.append("Background_Mean is more than half the cytoplasmic level: the field may be confluent; "
                         "consider a manual background value")
    if not s.exclude_saturated and inc:
        sat_cells = sum(1 for c in inc if max(c["saturated_fraction_nucleus"], c["saturated_fraction_cytoplasm"])
                        > s.max_saturated_fraction)
        if sat_cells:
            warns.append(f"{sat_cells} measured cell(s) have more than {100 * s.max_saturated_fraction:.0f}% "
                         "saturated pixels (their means are underestimated)")
    reasons: dict[str, int] = {}
    for c in cells:
        if not c["included"]:
            reasons[c["exclusion_reason"]] = reasons.get(c["exclusion_reason"], 0) + 1
    summ["excluded_cells"] = "; ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))
    layers = {"nuclear_labels": labels, "cytoplasm_labels": cyto, "cell_status": status,
              "target_corrected": corrected, "background_region": free}
    return cells, layers, summ, warns


# ---------------------------------------------------------------------- batches


@dataclass
class FieldSpec:
    """One field of view: where the nuclear and target channels come from."""

    nuclear_path: str
    target_path: str
    condition: str = ""
    field_id: str = ""
    repetition: str = ""


def analyse_fields(
    specs: list[FieldSpec],
    settings: TranslocationSettings,
    progress: Callable[[int, int, str], None] | None = None,
    keep_layers: bool = False,
    cancel: Callable[[], bool] | None = None,
    loader=None,
) -> tuple[list[FieldResult], list[dict]]:
    """Run a batch; returns (results, errors)."""
    from ..core.imageio import load_image

    loader = loader or load_image
    results, errors = [], []
    for i, spec in enumerate(specs):
        if cancel and cancel():
            break
        fid = spec.field_id or f"field_{i + 1:03d}"
        if progress:
            progress(i, len(specs), fid)
        try:
            nuc = loader(spec.nuclear_path)
            tgt = nuc if spec.target_path == spec.nuclear_path else loader(spec.target_path)
            res = analyse_field(nuc, tgt, settings, fid, spec.condition, keep_layers=keep_layers,
                                repetition=spec.repetition)
            results.append(res)
        except Exception as exc:  # noqa: BLE001 - keep going, report per field
            errors.append({"field": fid, "nuclear_file": spec.nuclear_path, "target_file": spec.target_path, "error": str(exc)})
    if progress:
        progress(len(specs), len(specs), "done")
    return results, errors


def summarise_conditions(results: list[FieldResult], settings: TranslocationSettings | None = None) -> list[dict]:
    """Per repetition and condition: paper ratio and per-cell statistics (see ``bio.experiment``)."""
    from .experiment import ExperimentDesign, summarise_experiment

    s = settings or TranslocationSettings()
    fields_ = [dict(r.summary, repetition=r.repetition or r.summary.get("repetition") or "1") for r in results]
    cells = [dict(c, repetition=r.repetition or "1") for r in results for c in r.cells]
    crit = (int(results[0].summary.get("criterion_fields", 5)), int(results[0].summary.get("criterion_cells", 500))) \
        if results else (5, 500)
    return summarise_experiment(fields_, cells, ExperimentDesign.from_settings(s), crit).conditions
