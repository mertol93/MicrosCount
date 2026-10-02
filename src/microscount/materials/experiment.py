"""SEM experiments: images, samples, repetitions, a reference sample and comparisons.

Every SEM image gives one value per measure: porosity, mean and median equivalent pore
diameter, pore density. Images of one sample in one repetition (one membrane, one batch)
are technical replicates; repetitions (independent membranes or batches) are the
independent replicates. ``summarise_samples`` reports, for every measure:

* per repetition and sample: mean ± SD over images and the change against the reference
  sample of the same repetition;
* per sample: mean ± SD over repetitions (over images when there is one repetition);
* comparisons A vs B (by default every sample against the reference): Welch's t-test on
  the image values within a repetition (exploratory) and, across repetitions, the change
  in each repetition, whether its direction agrees, and a paired t-test when there are
  at least three repetitions;
* the pore-size distribution of every sample (number and area fractions, cumulative).

Samples and repetitions are read from the names (``core.naming``): a folder whose file
names name several samples (``CA_5kx_01.tif``, ``CA-CNF1_5kx_01.tif``) is one repetition;
otherwise each folder is a sample and its parent folder the repetition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.experiment import (
    GroupDesign, GroupSummary, Measure, combine_repetitions, mean_sd, read_csv, summarise_groups, to_float, truthy,
)
from ..core.naming import GROUP_SOURCES, assign_groups, group_key, label_items

SAMPLE_SOURCES = GROUP_SOURCES
MEASURES_UM = [
    Measure("porosity_percent", "Porosity", "%"),
    Measure("mean_pore_diameter_um", "Mean pore diameter", "µm"),
    Measure("median_pore_diameter_um", "Median pore diameter", "µm"),
    Measure("pore_density_per_um2", "Pore density", "pores/µm²"),
]
MEASURES_PX = [
    Measure("porosity_percent", "Porosity", "%"),
    Measure("mean_pore_diameter_px", "Mean pore diameter", "px"),
    Measure("median_pore_diameter_px", "Median pore diameter", "px"),
    Measure("pore_density_per_mpx", "Pore density", "pores/Mpx"),
]


@dataclass
class ImageSpec:
    """One SEM image and what is known about it before the analysis."""

    path: str
    sample: str = ""
    repetition: str = ""
    image_id: str = ""
    field_no: int | None = None
    pixel_size_um: float | None = None  # typed for this image: overrides everything else
    file_pixel_size_um: float | None = None  # recorded in the file (SEM metadata, calibration)
    file_pixel_size_source: str = ""
    data_bar_px: int | None = None  # recorded in the file (FEI / Thermo Fisher)
    note: str = ""


def file_pixel_size(path: str | Path) -> tuple[float | None, str, dict]:
    """Pixel size recorded in an image file without reading its pixels: (µm or None, source, SEM metadata)."""
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        return None, "", {}
    import tifffile

    from ..core.imageio import _sem_metadata, _tiff_pixel_size

    try:
        with tifffile.TiffFile(path) as tif:
            meta = _sem_metadata(tif)
            if meta.get("pixel_size_um"):
                px = (meta["pixel_size_um"], f"{meta['instrument']} metadata")
            else:
                px = _tiff_pixel_size(tif)
            height = tif.pages[0].shape[0] if tif.pages[0].shape else None
    except Exception:  # noqa: BLE001 - unreadable files are reported when they are analysed
        return None, "", {}
    if "scan_height_px" in meta and height:
        bar = int(height) - int(meta["scan_height_px"])
        if 0 < bar < 0.5 * height:
            meta["data_bar_px"] = bar
    return (px[0], px[1], meta) if px else (None, "", meta)


def scan_sem_files(paths: list[str | Path], progress=None, cancel=None, samples_from: str = "auto") -> list[ImageSpec]:
    """Image specs with the pixel size each file records, samples, repetitions and image ids."""
    specs = []
    for i, p in enumerate(paths):
        if cancel and cancel():
            break
        if progress:
            progress(i, len(paths), Path(p).name)
        px, src, meta = file_pixel_size(p)
        specs.append(ImageSpec(str(p), file_pixel_size_um=px, file_pixel_size_source=src,
                               data_bar_px=meta.get("data_bar_px")))
    if specs:
        assign_samples(specs, samples_from)
        label_images(specs)
    return specs


def assign_samples(specs: list[ImageSpec], source: str = "auto") -> str:
    """Sample, repetition and image number of every image from its names (see ``core.naming``)."""
    return assign_groups(specs, lambda s: s.path, None, source, "sample")


def label_images(specs: list[ImageSpec]) -> None:
    """Image ids: ``"CA #1"``, or ``"2 · CA #1"`` when there are several repetitions."""
    label_items(specs, lambda s: s.path, "sample", "image_id", "(no sample)")


def design_from_settings(s) -> GroupDesign:
    return GroupDesign(
        reference=getattr(s, "reference_sample", "") or "",
        references=dict(getattr(s, "references", {}) or {}),
        comparisons=[list(c) for c in (getattr(s, "comparisons", []) or [])],
        order=list(getattr(s, "sample_order", []) or []),
    )


# ---------------------------------------------------------------------- summary


@dataclass
class MaterialsSummary:
    groups: GroupSummary  # long tables: per repetition, comparisons, across repetitions
    summary: list[dict]  # one row per sample (the headline numbers)
    samples: list[dict]  # one row per repetition and sample, with pooled pore statistics
    distribution: list[dict]  # pore-size distribution per repetition and sample, and pooled
    measures: list[Measure]
    size_unit: str  # "µm" or "px"
    notes: list[str] = field(default_factory=list)

    @property
    def order(self) -> list[str]:
        return self.groups.order

    @property
    def repetitions(self) -> list[str]:
        return self.groups.repetitions

    @property
    def comparisons(self) -> list[dict]:
        return self.groups.comparisons


def choose_measures(image_rows: list[dict]) -> list[Measure]:
    """Sizes in µm when any image has a pixel size, otherwise in pixels."""
    known = any(math.isfinite(to_float(r.get("pixel_size_um"))) for r in image_rows)
    return MEASURES_UM if known else MEASURES_PX


def _nice_ceiling(x: float) -> float:
    if not math.isfinite(x) or x <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * e >= x:
            return m * e
    return 10 * e


def pore_size_distribution(pore_rows: list[dict], image_rows: list[dict], key: str, unit: str, bins: int,
                           order: list[str], reps: list[str]) -> list[dict]:
    """Number and area fractions of pores per diameter bin, per repetition and sample and pooled."""
    sample_of = {}
    for r in image_rows:
        sample_of[(str(r.get("repetition") or "1"), str(r.get("image_id") or r.get("image")))] = r.get("sample")
    canon = {group_key(g): g for g in order}
    by: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for p in pore_rows:
        if not truthy(p.get("included", True)):
            continue
        d = to_float(p.get(key))
        a = d * d  # pore area up to π/4, in the unit of d: right for images of different pixel sizes
        if not math.isfinite(d):
            continue
        rep = str(p.get("repetition") or "1")
        g = canon.get(group_key(str(p.get("sample") or sample_of.get((rep, str(p.get("image_id") or p.get("image"))))
                                    or "(none)")))
        if g is None:
            continue
        by.setdefault((rep, g), []).append((d, a))
        by.setdefault(("all", g), []).append((d, a))
    all_d = np.array([d for (rep, _), v in by.items() if rep == "all" for d, _ in v])
    if all_d.size == 0:
        return []
    top = _nice_ceiling(float(np.percentile(all_d, 99.5)))
    edges = np.linspace(0.0, top, int(bins) + 1)
    rows = []
    for rep in (reps + ["all"] if len(reps) > 1 else ["all"]):
        for g in order:
            v = by.get((rep, g))
            if not v:
                continue
            d = np.array([x[0] for x in v])
            a = np.array([x[1] for x in v])
            idx = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, len(edges) - 2)  # the last bin holds larger pores
            n = np.bincount(idx, minlength=len(edges) - 1)
            area = np.bincount(idx, weights=a, minlength=len(edges) - 1)
            cum = np.cumsum(n)
            for i in range(len(edges) - 1):
                rows.append({
                    "repetition": rep if len(reps) > 1 else (reps[0] if reps else "1"),
                    "sample": g, "bin": i + 1, f"diameter_from_{unit}": float(edges[i]),
                    f"diameter_to_{unit}": float(edges[i + 1]),  # the last bin also holds the few larger pores
                    "n_pores": int(n[i]), "pores_pct": 100.0 * n[i] / d.size,
                    "cumulative_pct": 100.0 * cum[i] / d.size, "area_pct": 100.0 * area[i] / a.sum() if a.sum() else math.nan,
                })
    return rows


def _pooled(pores: list[tuple[float, float]]) -> dict:
    if not pores:
        return {"pooled_n_pores": 0}
    d = np.array([p[0] for p in pores])
    a = np.array([p[1] for p in pores])
    return {
        "pooled_n_pores": int(d.size),
        "pooled_median_diameter": float(np.median(d)),
        "pooled_d10_diameter": float(np.percentile(d, 10)),
        "pooled_d90_diameter": float(np.percentile(d, 90)),
        "pooled_area_weighted_mean_diameter": float((d * a).sum() / a.sum()) if a.sum() else math.nan,
        "largest_pore_diameter": float(d.max()),
    }


def summarise_samples(image_rows: list[dict], pore_rows: list[dict] | None = None, design: GroupDesign | None = None,
                      bins: int = 25) -> MaterialsSummary:
    """Values per repetition and sample, changes against the reference, comparisons and distributions."""
    design = design or GroupDesign()
    pore_rows = pore_rows or []
    for r in image_rows:
        if not str(r.get("repetition", "") or "").strip():
            r["repetition"] = "1"
        r["repetition"] = str(r["repetition"])
    measures = choose_measures(image_rows)
    unit = "µm" if measures is MEASURES_UM else "px"
    size_key = "equivalent_diameter_um" if unit == "µm" else "equivalent_diameter_px"
    gs = summarise_groups(image_rows, measures, design, group="sample", unit="image")
    notes = list(gs.notes)
    if unit == "µm":
        unknown = [r for r in image_rows if not math.isfinite(to_float(r.get("pixel_size_um")))]
        if unknown:
            notes.append(f"{len(unknown)} image(s) without a pixel size are left out of the pore sizes and densities")
    else:
        notes.append("no image has a pixel size: pore sizes are in pixels and pore densities per megapixel")
    canon = {group_key(g): g for g in gs.order}

    # pores pooled per repetition and sample
    sample_of = {(str(r["repetition"]), str(r.get("image_id") or r.get("image"))): r.get("sample") for r in image_rows}
    pooled: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for p in pore_rows:
        if not truthy(p.get("included", True)):
            continue
        d = to_float(p.get(size_key))
        if not math.isfinite(d):
            continue
        rep = str(p.get("repetition") or "1")
        g = canon.get(group_key(str(p.get("sample") or sample_of.get((rep, str(p.get("image_id") or p.get("image"))))
                                    or "(none)")))
        if g:
            pooled.setdefault((rep, g), []).append((d, d * d))  # area weight up to π/4, as in the distribution

    # one row per repetition and sample
    per = {(r["repetition"], r["sample"], r["key"]): r for r in gs.per_repetition}
    samples = []
    for rep in gs.repetitions:
        for g in gs.order:
            imgs = [r for r in image_rows if r["repetition"] == rep and canon.get(group_key(str(r.get("sample") or "(none)"))) == g]
            if not imgs:
                continue
            pxs = [to_float(r.get("pixel_size_um")) for r in imgs]
            pxs = [x for x in pxs if math.isfinite(x)]
            row = {"repetition": rep, "sample": g, "n_images": len(imgs),
                   "n_pores": int(sum(to_float(r.get("n_pores")) for r in imgs if math.isfinite(to_float(r.get("n_pores"))))),
                   "pixel_size_min_um": min(pxs) if pxs else math.nan, "pixel_size_max_um": max(pxs) if pxs else math.nan}
            ref = ""
            for m in measures:
                pr = per.get((rep, g, m.key))
                if pr is None:
                    continue
                ref = pr["reference"]
                row[f"{m.key}_mean"], row[f"{m.key}_sd"] = pr["mean"], pr["sd"]
                row[f"{m.key}_change_pct"] = pr["change_pct"]
            row["reference"] = ref
            row.update({f"{k}_{unit}" if k != "pooled_n_pores" else k: v
                        for k, v in _pooled(pooled.get((rep, g), [])).items()})
            samples.append(row)
            if pxs and max(pxs) / min(pxs) > 1.05:
                notes.append(f"repetition {rep}, {g}: pixel sizes from {min(pxs):.4g} to {max(pxs):.4g} µm "
                             "(different magnifications); pore sizes depend on resolution")

    # one row per sample: across repetitions, or over images when there is one repetition
    several = len(gs.repetitions) > 1
    over = {(r["sample"], r["key"]): r for r in gs.overall}
    summary = []
    for g in gs.order:
        rows_g = [r for r in samples if r["sample"] == g]
        if not rows_g:
            continue
        row = {"sample": g, "reference": rows_g[0].get("reference", ""), "n_repetitions": len(rows_g),
               "n_images": sum(r["n_images"] for r in rows_g), "n_pores": sum(r["n_pores"] for r in rows_g),
               "replicates": "repetitions" if several else "images"}
        for m in measures:
            o = over.get((g, m.key))
            if o is None:
                continue
            if several:
                row[f"{m.key}_mean"], row[f"{m.key}_sd"] = o["mean"], o["sd"]
            else:
                row[f"{m.key}_mean"], row[f"{m.key}_sd"] = rows_g[0].get(f"{m.key}_mean"), rows_g[0].get(f"{m.key}_sd")
            row[f"{m.key}_change_pct"] = o["change_pct"]
        allp = [x for (rep, gg), v in pooled.items() if gg == g for x in v]
        row.update({f"{k}_{unit}" if k != "pooled_n_pores" else k: v for k, v in _pooled(allp).items()})
        summary.append(row)

    dist = pore_size_distribution(pore_rows, image_rows, size_key, unit, bins, gs.order, gs.repetitions)
    return MaterialsSummary(gs, summary, samples, dist, measures, unit, notes)


# ---------------------------------------------------------------------- saved results


def load_result_folder(folder: str | Path, repetition: str | None = None) -> tuple[list[dict], list[dict]]:
    """Image and pore rows of a MicrosCount SEM result folder (``per_image.csv``, ``per_pore.csv``)."""
    folder = Path(folder)
    ir = read_csv(folder / "per_image.csv")
    pr = read_csv(folder / "per_pore.csv") if (folder / "per_pore.csv").exists() else []
    for rows in (ir, pr):
        for r in rows:
            if repetition is not None:
                r["repetition"] = repetition
            elif not str(r.get("repetition", "")).strip():
                r["repetition"] = "1"
    return ir, pr


def combine_result_folders(folders: list[str | Path], design: GroupDesign | None = None,
                           repetitions: list[str] | None = None, bins: int = 25):
    """Summarise several result folders (e.g. one per repetition) as one experiment."""
    folders = [Path(f) for f in folders]
    loaded = [load_result_folder(f) for f in folders]
    notes = combine_repetitions(folders, [ir for ir, _ in loaded], [pr for _, pr in loaded], "sample", repetitions,
                                "image_id")
    all_i = [r for ir, _ in loaded for r in ir]
    all_p = [r for _, pr in loaded for r in pr]
    ms = summarise_samples(all_i, all_p, design, bins)
    ms.notes[:0] = notes
    return ms, all_i, all_p


__all__ = ["ImageSpec", "MaterialsSummary", "SAMPLE_SOURCES", "assign_samples", "choose_measures",
           "combine_result_folders", "design_from_settings", "file_pixel_size", "label_images", "load_result_folder",
           "mean_sd", "pore_size_distribution", "scan_sem_files", "summarise_samples"]
