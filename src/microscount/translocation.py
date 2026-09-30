"""Nuclear translocation: nuclear/cytoplasmic (N/C) intensity ratio.

Two methods are provided.

``paper``
    Noursadeghi M. et al. (2008) J Immunol Methods 329:194-200, as published:
    3x3 median filter on both channels, ImageJ Default (IsoData) auto-threshold of
    each filtered channel, nuclear ROI = nuclear-stain mask, cytoplasmic ROI =
    target mask minus nuclear mask, and the ratio of mean *unfiltered* target
    intensity in the two ROIs, pooled over the whole field.

``per_cell``
    Nuclei are segmented individually (touching nuclei split by a distance-
    transform watershed), each nucleus gets its own perinuclear cytoplasmic ring
    (or territory) that does not depend on the target intensity, and the ratio
    is computed per cell after background subtraction. The paper metric is still
    reported for every field so the two can be compared.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Callable

import numpy as np
from scipy import ndimage as ndi

from .imageio import ChannelError, LoadedImage
from .thresholds import compute_threshold

NUCLEAR_TOKENS = ("dapi", "hoechst", "nuclear", "nuclei", "nucleus", "draq5", "h33342", "sytox", "pi")


# ---------------------------------------------------------------------- settings


@dataclass
class TranslocationSettings:
    method: str = "per_cell"  # "per_cell" | "paper"
    nuclear_channel: str = "auto"
    target_channel: str = "auto"
    median_size: int = 3
    nuclear_threshold: str = "ij_default"
    nuclear_threshold_value: float | None = None
    target_threshold: str = "ij_default"
    target_threshold_value: float | None = None
    legacy_inclusive_threshold: bool = False
    background: str = "auto"  # "none" | "auto" | "manual"
    background_value: float = 0.0
    exclude_annotations: bool = True
    exclude_rects: list = field(default_factory=list)  # [[x0, y0, x1, y1], ...] pixels
    exclude_saturated: bool = True
    exclude_zero_pixels: bool = False
    # per-cell
    segmentation_smoothing_px: float = 2.0  # Gaussian sigma for nucleus detection
    nucleus_diameter_px: float | None = None  # None = estimate from the image
    split_touching: bool = True
    split_sensitivity: float = 0.10  # watershed h as a fraction of nucleus diameter
    min_area_fraction: float = 0.30
    max_area_fraction: float = 4.0
    min_solidity: float = 0.80
    exclude_border_cells: bool = True
    nucleus_erode_px: int = 1
    cytoplasm: str = "ring"  # "ring" | "territory"
    ring_gap_px: float | None = None  # None = 1 px
    ring_width_px: float | None = None  # None = 0.3 x nucleus diameter
    territory_px: float | None = None  # None = 1.0 x nucleus diameter
    restrict_to_cells: bool = True
    cell_detection_sigmas: float = 3.0
    min_cytoplasm_pixels: int = 15
    max_saturated_fraction: float = 0.05
    responder_ratio: float | None = 1.0
    pixel_size_um: float | None = None

    @classmethod
    def paper(cls) -> "TranslocationSettings":
        """Noursadeghi et al. (2008) exactly as published."""
        return cls(
            method="paper",
            median_size=3,
            nuclear_threshold="ij_default",
            target_threshold="ij_default",
            background="none",
            exclude_saturated=False,
            exclude_zero_pixels=False,
        )

    @classmethod
    def per_cell(cls) -> "TranslocationSettings":
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

    @property
    def ratio(self) -> float:
        """Headline ratio of this field for the chosen method."""
        key = "paper_ratio" if self.method == "paper" else "median_ratio"
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
) -> tuple[np.ndarray, dict]:
    """Label individual nuclei in a binary nuclear mask."""
    from skimage.morphology import h_maxima
    from skimage.segmentation import relabel_sequential, watershed

    m = ndi.binary_fill_holes(mask)
    m = ndi.binary_opening(m)
    lab0, n0 = ndi.label(m)
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
    small = np.nonzero(areas < min_area_fraction * a_ref)[0] + 1
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
) -> FieldResult:
    s = settings or TranslocationSettings()
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

    n_f = _median(n_raw, s.median_size)
    t_f = _median(t_raw, s.median_size)
    thr_n = compute_threshold(n_f, s.nuclear_threshold, valid, s.nuclear_threshold_value, s.legacy_inclusive_threshold)
    thr_t = compute_threshold(t_f, s.target_threshold, valid, s.target_threshold_value, s.legacy_inclusive_threshold)
    mask_n = (n_f > thr_n.value) & valid
    mask_t = (t_f > thr_t.value) & valid

    sat = tgt.saturation_value
    saturated = (t_raw >= sat) if sat is not None else np.zeros(t_raw.shape, bool)
    measurable = valid.copy()
    if s.exclude_saturated:
        measurable &= ~saturated
    if s.exclude_zero_pixels:
        measurable &= t_raw > 0
    t_val = t_raw.astype(np.float64)

    # ---- per-cell segmentation (needed for the background region as well)
    labels = None
    seg: dict = {}
    region_mask = mask_n  # nuclear pixels used to shape background / cytoplasm regions
    thr_seg = None
    if s.method == "per_cell":
        sigma = float(s.segmentation_smoothing_px or 0)
        n_seg = ndi.gaussian_filter(n_raw.astype(np.float32), sigma) if sigma > 0 else n_f
        thr_seg = compute_threshold(
            n_seg, s.nuclear_threshold, valid, s.nuclear_threshold_value, s.legacy_inclusive_threshold
        )
        region_mask = (n_seg > thr_seg.value) & valid
        labels, seg = segment_nuclei(
            n_seg, region_mask, s.nucleus_diameter_px, s.split_touching, s.split_sensitivity, s.min_area_fraction
        )
    diameter = seg.get("diameter") or s.nucleus_diameter_px
    if not diameter or not math.isfinite(diameter):
        smooth = ndi.gaussian_filter(n_raw.astype(np.float32), 2.0)
        thr_tmp = compute_threshold(smooth, s.nuclear_threshold, valid, s.nuclear_threshold_value)
        lab_tmp, _ = ndi.label(ndi.binary_opening(ndi.binary_fill_holes((smooth > thr_tmp.value) & valid)))
        diameter = estimate_nucleus_diameter(lab_tmp)

    # ---- background of the target channel
    grow = max(2, int(round(0.5 * diameter)))
    bg_region = valid & ~_dilate(region_mask, grow)
    source = "outside nuclei"
    if bg_region.sum() < max(1000, 0.005 * valid.sum()):
        bg_region, source = valid, "whole field"
    bg_mode, bg_sigma = estimate_background(t_f, bg_region)
    if s.background == "none":
        bg = 0.0
    elif s.background == "manual":
        bg, source = float(s.background_value), "manual"
    else:
        bg = bg_mode
    t_val -= bg

    # ---- paper metric (whole field). In per-cell mode it is always computed exactly as
    # published (raw intensities, no background subtraction, no saturation filter) so the
    # "paper-method ratio" column means the same thing whichever method is selected.
    if s.method == "per_cell":
        paper_vals = t_raw.astype(np.float64)
        paper_meas = valid & (t_raw > 0) if s.exclude_zero_pixels else valid
    else:
        paper_vals, paper_meas = t_val, measurable
    nuc_roi = mask_n & paper_meas
    cyto_roi = mask_t & ~mask_n & paper_meas
    p_n = float(paper_vals[nuc_roi].mean()) if nuc_roi.any() else math.nan
    p_c = float(paper_vals[cyto_roi].mean()) if cyto_roi.any() else math.nan
    paper_ratio = p_n / p_c if (p_c and math.isfinite(p_c) and p_c > 0) else math.nan
    lab_all, n_comp = ndi.label(mask_n)
    comp_areas = np.bincount(lab_all.ravel())[1:]
    n_comp_real = int((comp_areas >= max(4, 0.3 * math.pi * (diameter / 2) ** 2)).sum())

    dy, dx, rcorr = channel_shift(n_raw, t_raw, valid)
    if math.isfinite(dy) and math.hypot(dy, dx) > 1.5 and abs(rcorr) > 0.2:
        warns.append(f"channels offset by ({dy:.1f}, {dx:.1f}) px; check registration")

    summary = {
        "field": field_id,
        "condition": condition,
        "nuclear_file": nuc.name,
        "target_file": tgt.name,
        "nuclear_channel": nuc.channel_names[n_idx],
        "target_channel": tgt.channel_names[t_idx],
        "method": s.method,
        "paper_ratio": paper_ratio,
        "paper_nuclear_mean": p_n,
        "paper_cytoplasm_mean": p_c,
        "paper_nuclear_area_px": int(nuc_roi.sum()),
        "paper_cytoplasm_area_px": int(cyto_roi.sum()),
        "nuclear_threshold": thr_n.value,
        "target_threshold": thr_t.value,
        "nuclear_threshold_ij8": thr_n.level8,
        "target_threshold_ij8": thr_t.level8,
        "background": bg,
        "background_source": source if s.background != "none" else "none",
        "background_estimate": bg_mode,
        "background_sigma": bg_sigma,
        "nuclear_components": n_comp_real,
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
    if s.method == "per_cell":
        summary["segmentation_threshold"] = thr_seg.value
        cells, cell_layers, cell_summary, cell_warns = _per_cell(
            labels, region_mask, t_val, t_raw, n_raw, t_f, measurable, saturated, valid, bg, bg_mode, bg_sigma, diameter, s
        )
        summary.update(cell_summary)
        warns.extend(cell_warns)
        if layers is not None:
            layers.update(cell_layers)
    for c in cells:
        c["field"] = field_id
        c["condition"] = condition
    return FieldResult(field_id, condition, nuc.name, tgt.name, s.method, summary, cells, warns, layers)


def _per_cell(labels, mask_n, t_val, t_raw, n_raw, t_f, measurable, saturated, valid, bg, bg_mode, bg_sigma, d, s):
    from skimage.measure import regionprops_table

    warns: list[str] = []
    h, w = labels.shape
    n = int(labels.max())
    gap = 1.0 if s.ring_gap_px is None else float(s.ring_gap_px)
    if s.cytoplasm == "territory":
        reach = float(s.territory_px) if s.territory_px else float(d)
    else:
        reach = float(s.ring_width_px) if s.ring_width_px else max(2.0, round(0.3 * d))
    if n:
        dist_bg, (iy, ix) = ndi.distance_transform_edt(labels == 0, return_indices=True)
        nearest = labels[iy, ix]
        inner = np.where(dist_bg <= gap, nearest, 0) if gap > 0 else labels
        outer = np.where(dist_bg <= gap + reach, nearest, 0)
        del iy, ix
    else:
        inner = outer = labels
    all_nuclear = mask_n | (labels > 0)
    blocked = _dilate(all_nuclear, gap)
    cyto = np.where((inner == 0) & ~blocked & valid, outer, 0)
    cyto_before = cyto.copy()
    cell_thr = bg_mode + s.cell_detection_sigmas * bg_sigma
    if s.restrict_to_cells:
        cyto = np.where(t_f > cell_thr, cyto, 0)

    nuc_meas = _erode_labels(labels, s.nucleus_erode_px)
    # fall back to the full nucleus where erosion removed everything
    counts_er = np.bincount(nuc_meas.ravel(), minlength=n + 1)
    lost = np.nonzero(counts_er[1:] == 0)[0] + 1
    if lost.size:
        nuc_meas = np.where(np.isin(labels, lost), labels, nuc_meas)

    def sums(lab):
        sel = measurable & (lab > 0)
        idx = lab[sel]
        cnt = np.bincount(idx, minlength=n + 1).astype(np.float64)
        tot = np.bincount(idx, weights=t_val[sel], minlength=n + 1)
        return cnt, tot

    n_cnt, n_sum = sums(nuc_meas)
    c_cnt, c_sum = sums(cyto)
    n_sat = np.bincount(labels.ravel(), weights=saturated.ravel().astype(np.float64), minlength=n + 1)
    n_all = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    c_sat = np.bincount(cyto.ravel(), weights=saturated.ravel().astype(np.float64), minlength=n + 1)
    c_all = np.bincount(cyto.ravel(), minlength=n + 1).astype(np.float64)
    c_before = np.bincount(cyto_before.ravel(), minlength=n + 1).astype(np.float64)
    dapi_sum = np.bincount(labels.ravel(), weights=n_raw.ravel().astype(np.float64), minlength=n + 1)
    if (~valid).any():
        near_invalid = ndi.binary_dilation(~valid, iterations=2)
        touches_invalid = np.bincount(labels[near_invalid].ravel(), minlength=n + 1) > 0
    else:
        touches_invalid = np.zeros(n + 1, bool)

    status = np.array(["ok"] * (n + 1), dtype=object)
    if n:
        props = regionprops_table(labels, properties=("label", "area", "centroid", "bbox", "solidity"))
    else:
        props = {k: np.array([]) for k in ("label", "area", "centroid-0", "centroid-1", "bbox-0", "bbox-1", "bbox-2", "bbox-3", "solidity")}
    a_ref = math.pi * (d / 2.0) ** 2
    cells = []
    ratios = []
    for i in range(len(props["label"])):
        lab = int(props["label"][i])
        area = float(props["area"][i])
        border = props["bbox-0"][i] == 0 or props["bbox-1"][i] == 0 or props["bbox-2"][i] == h or props["bbox-3"][i] == w
        nm = n_sum[lab] / n_cnt[lab] if n_cnt[lab] else math.nan
        cm = c_sum[lab] / c_cnt[lab] if c_cnt[lab] else math.nan
        nsf = n_sat[lab] / n_all[lab] if n_all[lab] else 0.0
        csf = c_sat[lab] / c_all[lab] if c_all[lab] else 0.0
        reason = "ok"
        if s.exclude_border_cells and border:
            reason = "touches image edge"
        elif touches_invalid[lab]:
            reason = "touches excluded region"
        elif area > s.max_area_fraction * a_ref:
            reason = "too large (clump?)"
        elif props["solidity"][i] < s.min_solidity:
            reason = "irregular shape (merged nuclei?)"
        elif c_cnt[lab] < s.min_cytoplasm_pixels:
            reason = "too little cytoplasm"
        elif s.exclude_saturated and max(nsf, csf) > s.max_saturated_fraction:
            reason = "saturated"
        elif not (cm > 0):
            reason = "cytoplasm at background level"
        ratio = nm / cm if (reason == "ok") else (nm / cm if (cm and cm > 0) else math.nan)
        status[lab] = reason
        row = {
            "cell": lab,
            "x": float(props["centroid-1"][i]),
            "y": float(props["centroid-0"][i]),
            "nucleus_area_px": area,
            "nucleus_solidity": float(props["solidity"][i]),
            "nuclear_mean": nm,
            "cytoplasm_mean": cm,
            "ratio": ratio,
            "log2_ratio": math.log2(ratio) if ratio and ratio > 0 and math.isfinite(ratio) else math.nan,
            "nuclear_pixels": int(n_cnt[lab]),
            "cytoplasm_pixels": int(c_cnt[lab]),
            "cytoplasm_coverage": float(c_all[lab] / c_before[lab]) if c_before[lab] else 0.0,
            "nuclear_stain_mean": float(dapi_sum[lab] / n_all[lab]) if n_all[lab] else math.nan,
            "saturated_fraction_nucleus": float(nsf),
            "saturated_fraction_cytoplasm": float(csf),
            "included": reason == "ok",
            "exclusion_reason": "" if reason == "ok" else reason,
        }
        if s.pixel_size_um:
            row["nucleus_area_um2"] = area * float(s.pixel_size_um) ** 2
        cells.append(row)
        if reason == "ok":
            ratios.append(ratio)

    r = np.asarray(ratios, dtype=np.float64)
    r = r[np.isfinite(r) & (r > 0)]
    summ = {
        "n_nuclei_detected": int(n),
        "n_cells_analysed": int(r.size),
        "median_ratio": float(np.median(r)) if r.size else math.nan,
        "mean_ratio": float(np.mean(r)) if r.size else math.nan,
        "geomean_ratio": float(np.exp(np.mean(np.log(r)))) if r.size else math.nan,
        "sd_ratio": float(np.std(r, ddof=1)) if r.size > 1 else math.nan,
        "q1_ratio": float(np.percentile(r, 25)) if r.size else math.nan,
        "q3_ratio": float(np.percentile(r, 75)) if r.size else math.nan,
        "cell_threshold": float(cell_thr),
        "ring_width_px": float(reach) if s.cytoplasm != "territory" else math.nan,
        "territory_px": float(reach) if s.cytoplasm == "territory" else math.nan,
        "ring_gap_px": gap,
    }
    if s.responder_ratio is not None and r.size:
        summ["responder_fraction"] = float(np.mean(r > s.responder_ratio))
        summ["responder_cutoff"] = float(s.responder_ratio)
    if r.size == 0 and n > 0:
        warns.append("no cell passed the quality filters")
    if s.background == "auto" and r.size:
        cyto_level = np.nanmedian([c["cytoplasm_mean"] + bg for c in cells if c["included"]])
        if bg > 0.5 * cyto_level:
            warns.append(
                "estimated background is more than half the cytoplasmic level: the field may be "
                "confluent; consider a manual background value"
            )
    reasons = {}
    for c in cells:
        if not c["included"]:
            reasons[c["exclusion_reason"]] = reasons.get(c["exclusion_reason"], 0) + 1
    summ["excluded_cells"] = "; ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))
    layers = {"nuclear_labels": labels, "cytoplasm_labels": cyto, "cell_status": status}
    return cells, layers, summ, warns


# ---------------------------------------------------------------------- batches


@dataclass
class FieldSpec:
    """One field of view: where the nuclear and target channels come from."""

    nuclear_path: str
    target_path: str
    condition: str = ""
    field_id: str = ""


def analyse_fields(
    specs: list[FieldSpec],
    settings: TranslocationSettings,
    progress: Callable[[int, int, str], None] | None = None,
    keep_layers: bool = False,
    cancel: Callable[[], bool] | None = None,
    loader=None,
) -> tuple[list[FieldResult], list[dict]]:
    """Run a batch; returns (results, errors)."""
    from .imageio import load_image

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
            res = analyse_field(nuc, tgt, settings, fid, spec.condition, keep_layers=keep_layers)
            results.append(res)
        except Exception as exc:  # noqa: BLE001 - keep going, report per field
            errors.append({"field": fid, "nuclear_file": spec.nuclear_path, "target_file": spec.target_path, "error": str(exc)})
    if progress:
        progress(len(specs), len(specs), "done")
    return results, errors


def summarise_conditions(results: list[FieldResult]) -> list[dict]:
    """Per-condition summary: mean ± SD over fields, plus pooled per-cell stats."""
    by: dict[str, list[FieldResult]] = {}
    for r in results:
        by.setdefault(r.condition or "(none)", []).append(r)
    rows = []
    for cond, rs in by.items():
        paper = np.array([r.summary["paper_ratio"] for r in rs], dtype=np.float64)
        paper = paper[np.isfinite(paper)]
        row = {
            "condition": cond,
            "n_fields": len(rs),
            "paper_ratio_mean": float(paper.mean()) if paper.size else math.nan,
            "paper_ratio_sd": float(paper.std(ddof=1)) if paper.size > 1 else math.nan,
            "paper_ratio_sem": float(paper.std(ddof=1) / math.sqrt(paper.size)) if paper.size > 1 else math.nan,
        }
        if any(r.method == "per_cell" for r in rs):
            med = np.array([r.summary.get("median_ratio", math.nan) for r in rs], dtype=np.float64)
            med = med[np.isfinite(med)]
            pooled = np.array([c["ratio"] for r in rs for c in r.cells if c["included"]], dtype=np.float64)
            pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
            row.update(
                {
                    "n_cells": int(pooled.size),
                    "field_median_ratio_mean": float(med.mean()) if med.size else math.nan,
                    "field_median_ratio_sd": float(med.std(ddof=1)) if med.size > 1 else math.nan,
                    "pooled_median_ratio": float(np.median(pooled)) if pooled.size else math.nan,
                    "pooled_q1_ratio": float(np.percentile(pooled, 25)) if pooled.size else math.nan,
                    "pooled_q3_ratio": float(np.percentile(pooled, 75)) if pooled.size else math.nan,
                    "pooled_geomean_ratio": float(np.exp(np.log(pooled).mean())) if pooled.size else math.nan,
                }
            )
            cut = next((r.summary.get("responder_cutoff") for r in rs if "responder_cutoff" in r.summary), None)
            if cut is not None and pooled.size:
                row["responder_fraction"] = float(np.mean(pooled > cut))
                row["responder_cutoff"] = float(cut)
        rows.append(row)
    return rows
