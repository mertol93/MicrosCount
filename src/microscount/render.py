"""Display images and quality-control overlays (RGB uint8 arrays)."""

from __future__ import annotations

import numpy as np

CYAN = (0, 230, 255)
RED = (255, 70, 70)
MAGENTA = (255, 0, 200)
YELLOW = (255, 220, 0)
EXCLUDED_TINT = (40, 40, 160)


def stretch(img: np.ndarray, valid: np.ndarray | None = None, lo: float = 0.5, hi: float = 99.7, gamma: float = 1.0) -> np.ndarray:
    """Percentile contrast stretch to uint8 (display only, never used for measuring)."""
    a = img.astype(np.float32)
    sample = a[valid] if valid is not None and valid.any() else a
    if sample.size > 400_000:
        sample = sample[:: max(1, sample.size // 400_000)]
    p0, p1 = np.percentile(sample, [lo, hi]) if sample.size else (0.0, 1.0)
    if p1 <= p0:
        p1 = p0 + 1.0
    out = np.clip((a - p0) / (p1 - p0), 0, 1)
    if gamma != 1.0:
        out = out**gamma
    return (out * 255 + 0.5).astype(np.uint8)


def grey_rgb(img8: np.ndarray) -> np.ndarray:
    return np.repeat(img8[..., None], 3, axis=2)


def composite(nuclear: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Nuclear stain in blue/cyan, target in green."""
    n8 = stretch(nuclear, valid).astype(np.uint16)
    t8 = stretch(target, valid).astype(np.uint16)
    rgb = np.zeros(nuclear.shape + (3,), np.uint16)
    rgb[..., 0] = n8 * 0.25
    rgb[..., 1] = np.maximum(t8, n8 * 0.35)
    rgb[..., 2] = n8
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _blend(rgb: np.ndarray, mask: np.ndarray, colour, alpha: float) -> None:
    if not mask.any():
        return
    c = np.asarray(colour, np.float32)
    px = rgb[mask].astype(np.float32)
    rgb[mask] = np.clip(px * (1 - alpha) + c * alpha, 0, 255).astype(np.uint8)


def _outline(labels_or_mask: np.ndarray) -> np.ndarray:
    from skimage.segmentation import find_boundaries

    return find_boundaries(labels_or_mask, mode="inner")


def translocation_overlay(layers: dict, base: str = "target", show_cytoplasm: bool = True, show_nuclei: bool = True,
                          show_excluded: bool = True) -> np.ndarray:
    """Target (or nuclear/composite) image with segmentation drawn on top."""
    valid = layers.get("valid")
    if base == "nuclear":
        rgb = grey_rgb(stretch(layers["nuclear"], valid))
    elif base == "composite":
        rgb = composite(layers["nuclear"], layers["target"], valid)
    else:
        rgb = grey_rgb(stretch(layers["target"], valid))
    if valid is not None and show_excluded:
        _blend(rgb, ~valid, EXCLUDED_TINT, 0.55)

    labels = layers.get("nuclear_labels")
    if labels is None:  # paper method: whole-field ROIs
        if show_cytoplasm:
            _blend(rgb, layers["paper_cytoplasm"], MAGENTA, 0.35)
        if show_nuclei:
            rgb[_outline(layers["paper_nuclear"])] = CYAN
        return rgb

    status = layers.get("cell_status")
    ok = np.zeros(int(labels.max()) + 1, bool)
    if status is not None:
        ok = np.array([s == "ok" for s in status], bool)
    ok[0] = False
    if show_cytoplasm:
        cyto = layers["cytoplasm_labels"]
        _blend(rgb, ok[cyto], MAGENTA, 0.35)
    if show_nuclei:
        edge = _outline(labels)
        good = edge & ok[labels]
        rgb[good] = CYAN
        if show_excluded:
            rgb[edge & ~ok[labels] & (labels > 0)] = RED
    return rgb


def porosity_overlay(grey: np.ndarray, pores_labels: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    rgb = grey_rgb(stretch(grey, valid, 0.1, 99.9))
    _blend(rgb, pores_labels > 0, (255, 60, 0), 0.45)
    rgb[_outline(pores_labels)] = YELLOW
    if valid is not None:
        _blend(rgb, ~valid, EXCLUDED_TINT, 0.55)
    return rgb


def label_colours(labels: np.ndarray, seed: int = 0, background=(255, 255, 255)) -> np.ndarray:
    """Random distinct colour per label (MATLAB ``label2rgb(..., 'shuffle')``)."""
    import matplotlib

    n = int(labels.max())
    cmap = matplotlib.colormaps["jet"].resampled(max(n, 1))
    cols = (cmap(np.arange(max(n, 1)))[:, :3] * 255).astype(np.uint8)
    rng = np.random.default_rng(seed)
    cols = cols[rng.permutation(len(cols))]
    lut = np.vstack([np.array(background, np.uint8)[None], cols])
    return lut[labels]


def jet_map(grey: np.ndarray) -> np.ndarray:
    """MATLAB ``label2rgb(I)`` of an intensity image: jet(max(I)) colours, 0 = white."""
    import matplotlib

    g = np.asarray(grey)
    n = int(g.max())
    cmap = matplotlib.colormaps["jet"].resampled(max(n, 1))
    cols = (cmap(np.arange(max(n, 1)))[:, :3] * 255 + 0.5).astype(np.uint8)
    lut = np.vstack([np.array([255, 255, 255], np.uint8)[None], cols])
    return lut[np.clip(g.astype(np.int64), 0, n)]


def save_png(path, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb).save(path, optimize=False, compress_level=3)
