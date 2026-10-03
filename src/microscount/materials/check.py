"""Check the SEM porosity analysis against pores traced by hand.

Next to an image ``membrane_01.tif``, a mask ``membrane_01_pores.png`` (or ``.tif``) of the same
size marks the pores the way an expert sees them: white = pore, black = solid, mid-grey = not
traced (left out of the check). The analysis is run with the given settings and compared with
the tracing:

* porosity found and traced, and their difference in percentage points;
* overlap = Jaccard index of the pore pixels (1 = identical);
* pores found = share of traced pores (of at least the minimum pore area) of which at least 30%
  is found; found in traced = share of found pores lying at least half in traced pores;
* size ratio = mean equivalent diameter of the pores found ÷ that of the traced pores;
* pieces = pores found per traced pore that is found (above 1: pores are split).

Image folders are scanned as for an analysis; the masks themselves are never analysed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..core.imageio import MASK_SUFFIX, load_image
from .porosity import PorositySettings, analyse_sem

MASK_EXTENSIONS = (".png", ".tif", ".tiff")
METRICS = ["porosity_found_percent", "porosity_traced_percent", "difference_pp", "overlap", "pores_found",
           "found_in_traced", "size_ratio", "pieces", "n_traced", "n_found"]


def mask_for(image: str | Path) -> Path | None:
    """The traced mask next to an image, if there is one."""
    p = Path(image)
    for ext in MASK_EXTENSIONS:
        for name in (p.stem + MASK_SUFFIX + ext, p.stem + MASK_SUFFIX + ext.upper()):
            q = p.with_name(name)
            if q.is_file():
                return q
    return None


def read_mask(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(pore, traced) boolean arrays of a mask: white = pore, black = solid, mid-grey = not traced."""
    from PIL import Image

    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        a = np.asarray(tifffile.imread(p))
    else:
        a = np.asarray(Image.open(p))
    a = np.squeeze(a)
    if a.ndim == 3:  # colour: the brightest channel
        a = a[..., :3].max(axis=-1) if a.shape[-1] in (3, 4) else a.max(axis=0)
    a = a.astype(np.float64)
    top = 1.0 if a.max() <= 1 else (255.0 if a.max() <= 255 else float(np.iinfo(np.uint16).max))
    a = a / top
    return a >= 0.75, (a >= 0.75) | (a <= 0.25)


def score(found: np.ndarray, labels: np.ndarray, traced: np.ndarray, region: np.ndarray, min_area: int = 9) -> dict:
    """Compare found pores (mask and labelled pores) with traced pores inside ``region``."""
    found = found & region
    traced = traced & region
    t_lab, nt = ndi.label(traced, structure=np.ones((3, 3)))
    t_area = np.bincount(t_lab.ravel(), minlength=nt + 1)
    big = np.flatnonzero(t_area >= min_area)
    big = big[big > 0]
    hit = np.bincount(t_lab[found].ravel(), minlength=nt + 1)
    got = hit[big] >= 0.3 * t_area[big]
    labels = np.where(region, labels, 0)
    n = int(labels.max())
    p_area = np.bincount(labels.ravel(), minlength=n + 1)
    present = np.flatnonzero(p_area)
    present = present[present > 0]
    inside = np.bincount(labels[traced].ravel(), minlength=n + 1)
    good = inside[present] >= 0.5 * p_area[present]
    both = (t_lab > 0) & (labels > 0)
    pairs = np.unique(np.stack([t_lab[both], labels[both]]), axis=1) if both.any() else np.zeros((2, 0), int)
    pieces = np.bincount(pairs[0], minlength=nt + 1)[big[got]] if big.size else np.array([], int)
    pieces = pieces[pieces > 0]  # a found pore can vanish under the minimum pore area
    d_traced = 2 * np.sqrt(t_area[big] / np.pi)
    d_found = 2 * np.sqrt(p_area[present] / np.pi)
    n_region = int(region.sum())
    union = int((found | traced).sum())
    pf = found.sum() / n_region if n_region else math.nan
    pt = traced.sum() / n_region if n_region else math.nan
    return {
        "porosity_found_percent": 100 * pf,
        "porosity_traced_percent": 100 * pt,
        "difference_pp": 100 * (pf - pt),
        "overlap": float((found & traced).sum() / union) if union else 1.0,
        "pores_found": float(got.mean()) if got.size else math.nan,
        "found_in_traced": float(good.mean()) if good.size else math.nan,
        "size_ratio": float(d_found.mean() / d_traced.mean()) if d_found.size and d_traced.size else math.nan,
        "pieces": float(pieces.mean()) if len(pieces) else math.nan,
        "n_traced": int(big.size),
        "n_found": int(present.size),
    }


def check_image(path: str | Path, settings: PorositySettings | None = None, mask: str | Path | None = None) -> dict:
    """Analyse one image and score it against its traced mask."""
    s = settings or PorositySettings()
    img = load_image(path)
    m = Path(mask) if mask else mask_for(path)
    if m is None:
        raise FileNotFoundError(f"no traced mask next to {Path(path).name} (expected {Path(path).stem}{MASK_SUFFIX}.png)")
    pore, traced = read_mask(m)
    full = img.data.shape[-2:]
    if pore.shape != full:
        raise ValueError(f"{m.name} is {pore.shape[1]} × {pore.shape[0]} px but the image is {full[1]} × {full[0]} px")
    res = analyse_sem(img, s, keep_layers=True)
    L = res.layers
    h = L["grey"].shape[0]
    found = ~L["solid"] & L["valid"]
    row = {"image": img.name, "mask": m.name}
    row.update(score(found, L["pore_labels"], pore[:h], traced[:h] & L["valid"], s.min_pore_area_px))
    row["warnings"] = "; ".join(res.warnings)
    return row


def check_images(paths: list[str | Path], settings: PorositySettings | None = None, progress=None,
                 cancel=None) -> tuple[list[dict], dict]:
    """Rows of the images that have a traced mask, and their means."""
    rows = []
    with_mask = [p for p in paths if mask_for(p)]
    for i, p in enumerate(with_mask):
        if cancel and cancel():
            break
        if progress:
            progress(i, len(with_mask), Path(p).name)
        rows.append(check_image(p, settings))
    mean = {"image": "mean", "mask": f"{len(rows)} image(s)"}
    for k in METRICS:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        mean[k] = float(v.mean()) if v.size else math.nan
    return rows, mean
