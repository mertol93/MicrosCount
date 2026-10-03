"""Batch runs, tables and figures for the SEM porosity tool.

A run analyses every image, summarises the experiment (samples per repetition, changes
against the reference sample, comparisons, pore-size distributions; see
``materials.experiment``) and writes:

``results.xlsx``                Read me, Summary, Comparisons, Samples, Images, Pores, Distribution,
                                Settings, How to cite
``summary.csv``                 one row per sample: mean ± SD of every measure, change against the reference
``comparisons.csv``             each comparison of each measure, per repetition and across repetitions
``per_sample.csv``              one row per repetition and sample, with pooled pore statistics
``per_image.csv``               one row per image (pixel size and its source, data bar, warnings)
``per_pore.csv``                one row per pore
``pore_size_distribution.csv``  number and area fractions per diameter bin
``summary.png``, ``distribution.png``, ``images/`` (the MATLAB-style images of every micrograph)
``settings.yaml``               everything needed to repeat the run; ``CITATION.txt``

``combine_results`` does the same summary for result folders analysed separately.
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .. import __version__
from ..citation import citation_text
from ..core.experiment import GroupDesign, to_float
from ..core.imageio import load_image
from ..core.naming import group_key
from ..core.render import jet_map, label_colours, porosity_overlay, save_png
from ..core.report import (  # noqa: F401 - default_output_dir is re-exported
    default_output_dir,
    INK, INK2, SERIES, SURFACE, Progress, _figure, _safe, _settings_rows, _style, write_csv,
    write_settings_yaml, write_xlsx,
)
from .experiment import ImageSpec, MaterialsSummary, combine_result_folders, design_from_settings, summarise_samples
from .porosity import PorosityResult, PorositySettings, analyse_sem

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4 of the reference palette

IMAGE_FIRST = [
    "image_id", "image", "sample", "repetition", "porosity_percent", "n_pores", "mean_pore_diameter_um",
    "sd_pore_diameter_um", "median_pore_diameter_um", "d10_pore_diameter_um", "d90_pore_diameter_um",
    "area_weighted_mean_pore_diameter_um", "largest_pore_diameter_um", "pore_density_per_um2", "mean_circularity",
    "mean_pore_radius_um", "sd_pore_radius_um", "median_pore_radius_um", "pixel_size_um", "pixel_size_source",
    "data_bar_px", "data_bar_source", "analysed_area_um2", "n_pores_edge", "pore_threshold", "threshold_source",
    "thresholds", "porosity_N2_percent", "porosity_N3_percent", "porosity_N4_percent", "porosity_N5_percent",
    "porosity_N6_percent",
    "porosity_evened_percent", "evened_overlap", "warnings",
]
PORE_FIRST = ["image_id", "image", "sample", "repetition", "pore", "equivalent_diameter_um", "equivalent_radius_um",
              "area_um2", "area_px", "equivalent_diameter_px", "perimeter_px", "circularity", "touches_edge", "included",
              "x", "y"]
COMP_FIRST = ["measure", "unit", "sample_a", "sample_b", "repetition", "n_a", "n_b", "n_repetitions", "mean_a", "mean_b",
              "difference", "difference_pct", "per_repetition", "same_direction", "welch_t", "welch_p", "paired_t",
              "paired_p", "test"]


# ---------------------------------------------------------------------- figures


def pore_histogram_figure(path: Path, diameters: np.ndarray, bins: int = 25, title: str = "",
                          unit: str = "µm") -> Path | None:
    d = diameters[np.isfinite(diameters)]
    fig, axes = _figure(1, 5.0, 3.4)
    ax = axes[0]
    _style(ax)
    if d.size:
        ax.hist(d, bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=1.0, zorder=2)
        m = d.mean()
        ax.axvline(m, color=INK2, linewidth=0.8, zorder=3)
        ax.annotate(f"mean {m:.3g} {unit}", xy=(m, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel(f"Equivalent pore diameter ({unit})", color=INK, fontsize=10)
    ax.set_ylabel("Number of pores", color=INK, fontsize=10)
    ax.set_title(title or "Pore size distribution", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def summary_figure(path: Path, ms: MaterialsSummary, image_rows: list[dict]) -> Path | None:
    """Porosity and mean pore diameter per sample: image values (coloured by repetition) and mean ± SD."""
    if not ms.summary:
        return None
    order, reps = ms.order, ms.repetitions
    show = [m for m in ms.measures if m.key in ("porosity_percent", "mean_pore_diameter_um", "mean_pore_diameter_px")]
    width = max(4.2, 0.6 * len(order) + 1.8)
    fig, axes = _figure(len(show), width, 3.8)
    by_rep = 1 < len(reps) <= 3
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, show):
        _style(ax)
        for j, rep in enumerate(reps if by_rep else [None]):
            for i, g in enumerate(order):
                v = np.array([to_float(r.get(m.key)) for r in image_rows
                              if group_key(str(r.get("sample") or "(none)")) == group_key(g)
                              and (rep is None or str(r.get("repetition")) == rep)])
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                off = (j - (len(reps) - 1) / 2) * 0.1 if by_rep else 0.0
                ax.scatter(i + off + rng.uniform(-0.04, 0.04, v.size), v, s=26,
                           color=PALETTE[j] if by_rep else SERIES, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                           label=f"repetition {rep}" if (by_rep and i == 0) else None)
        for i, row in enumerate(ms.summary):
            mu, sd = to_float(row.get(f"{m.key}_mean")), to_float(row.get(f"{m.key}_sd"))
            if math.isfinite(mu):
                ax.errorbar(i + 0.32, mu, yerr=sd if math.isfinite(sd) else 0.0, fmt="_", color=INK, markersize=12,
                            capsize=3, linewidth=1.0, zorder=4)
        ax.set_xticks(range(len(order)))
        long = max((len(g) for g in order), default=0) > 8
        ax.set_xticklabels(order, rotation=20 if long else 0, ha="right" if long else "center", color=INK)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(f"{m.label} ({m.unit})", color=INK, fontsize=10)
        over = "repetitions" if len(reps) > 1 else "images"
        ax.set_title(f"{m.label} · mean ± SD of {over}", color=INK, fontsize=10, loc="left")
        if by_rep:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def distribution_figure(path: Path, ms: MaterialsSummary) -> Path | None:
    """Pore-size distribution per sample (all repetitions): frequency and cumulative, or small multiples."""
    unit = ms.size_unit
    rows = [r for r in ms.distribution if r["repetition"] == "all" or len(ms.repetitions) == 1]
    samples = [g for g in ms.order if any(r["sample"] == g for r in rows)]
    if not samples:
        return None
    lo_key, hi_key = f"diameter_from_{unit}", f"diameter_to_{unit}"

    def series(g, key):
        rr = sorted((r for r in rows if r["sample"] == g), key=lambda r: r["bin"])
        x = [rr[0][lo_key]] + [r[hi_key] for r in rr]
        return np.array(x), np.array([r[key] for r in rr])

    if len(samples) <= 4:
        fig, axes = _figure(2, 4.6, 3.5)
        ax1, ax2 = axes
        for ax in axes:
            _style(ax)
        for i, g in enumerate(samples):
            x, y = series(g, "pores_pct")
            ax1.stairs(y, x, color=PALETTE[i], linewidth=1.6, label=g)
            x, c = series(g, "cumulative_pct")
            ax2.plot(x, np.r_[0.0, c], color=PALETTE[i], linewidth=1.6, label=g)
        ax1.set_ylabel("Pores (%)", color=INK, fontsize=10)
        ax1.set_title("Pore-size distribution", color=INK, fontsize=10, loc="left")
        ax2.set_ylabel("Cumulative pores (%)", color=INK, fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.set_title("Cumulative", color=INK, fontsize=10, loc="left")
        for ax in axes:
            ax.set_xlabel(f"Equivalent pore diameter ({unit})", color=INK, fontsize=10)
            ax.set_xlim(left=0)
        if len(samples) > 1:
            ax1.legend(frameon=False, fontsize=8, labelcolor=INK)
    else:
        cols = 4
        nrow = math.ceil(len(samples) / cols)
        from matplotlib.figure import Figure

        fig = Figure(figsize=(3.2 * cols, 2.6 * nrow), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        axs = fig.subplots(nrow, cols, squeeze=False, sharex=True, sharey=True)
        for k, ax in enumerate(axs.ravel()):
            if k >= len(samples):
                ax.set_visible(False)
                continue
            _style(ax)
            x, y = series(samples[k], "pores_pct")
            ax.stairs(y, x, color=SERIES, linewidth=1.6)
            ax.set_title(samples[k], color=INK, fontsize=9, loc="left")
            if k % cols == 0:
                ax.set_ylabel("Pores (%)", color=INK, fontsize=9)
            if k >= len(samples) - cols:
                ax.set_xlabel(f"Diameter ({unit})", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


# ---------------------------------------------------------------------- tables


def readme_text(ms: MaterialsSummary, design: GroupDesign, n_images: int, n_warned: int, source: str = "") -> str:
    reps = ms.repetitions
    u = ms.size_unit
    lines = [
        f"MicrosCount {__version__} · Materials & Mechanics · SEM porosity",
        f"Created {_dt.datetime.now().strftime('%d %B %Y %H:%M')}" + (f" · {source}" if source else ""),
        f"{n_images} image(s) of {len(ms.order)} sample(s) in {len(reps)} repetition(s) ({', '.join(reps)}); "
        f"samples: {', '.join(ms.order)}.",
        "",
        "SHEETS",
        "Summary        one row per sample: mean ± SD of every measure (over repetitions when there are several, "
        "otherwise over images), change against the reference, pooled pore statistics",
        "Comparisons    each comparison of each measure, in each repetition and across repetitions",
        "Samples        one row per repetition and sample",
        "Images         one row per image: every measure, pixel size and where it came from, data bar, warnings",
        "Pores          one row per pore",
        "Distribution   pore-size distribution per sample: number and area fractions per diameter bin",
        "Settings       every setting used",
        "",
        "DEFINITIONS",
        "Porosity = pore fraction of the analysed area (MATLAB SEM_Porosity: darkest multithresh class, majority "
        "filter), before touching pores are separated.",
        f"Pores: separated by a watershed of the distance map, at least the minimum pore area; equivalent diameter = "
        f"2 × √(area / π) ({u}).",
        "Mean and median pore diameter: over the pores of an image; d10 / d90: 10th and 90th percentiles; "
        "area-weighted mean: Σ d·A / Σ A.",
        "Pore density = number of pores / analysed area (pores cut by the image edge included).",
        "Circularity = 4π × area / perimeter² (ImageJ's traced perimeter); 1 for a circle.",
        "Analysed area = the image without the SEM data bar and burned-in annotations.",
        "Checks (Images sheet): porosity_N2..N6_percent = the porosity with 2 to 6 thresholds; porosity_evened_percent "
        "and evened_overlap = the porosity, and the overlap (Jaccard index) of the pores, after the large-scale "
        "brightness is evened out (rolling ball, radius 15% of the image). A low overlap means the pores follow uneven "
        "brightness or an unstable threshold.",
        "Sample value in a repetition = mean ± SD of its images (images of one sample are technical replicates).",
        "Across repetitions = mean ± SD of the repetition values (repetitions are the independent replicates).",
        f"Reference sample: {design.reference or '(none set)'}. Change = 100 × (value / reference value − 1) in the "
        "same repetition; for porosity this is a relative change, not percentage points (see 'difference').",
        "Comparisons within a repetition: Welch t-test on the image values. Exploratory: images of one sample are not "
        "independent experiments.",
        "Comparisons across repetitions: change in each repetition, whether its direction agrees, and a paired t-test "
        "on the repetition values when there are at least 3 repetitions.",
        "Distribution: common bins for all samples up to about the 99.5th percentile of all pores; the last bin also "
        "holds the few larger pores.",
        "",
        "NOTES",
    ]
    notes = list(ms.notes)
    if n_warned:
        notes.append(f"{n_warned} image(s) have warnings: see the warnings column of the Images sheet.")
    lines += notes or ["none"]
    return "\n".join(lines)


def _summary_first(ms: MaterialsSummary) -> list[str]:
    cols = ["sample", "reference", "n_repetitions", "n_images", "n_pores", "replicates"]
    for m in ms.measures:
        cols += [f"{m.key}_mean", f"{m.key}_sd", f"{m.key}_change_pct"]
    return cols


def write_experiment(out: Path, ms: MaterialsSummary, image_rows: list[dict], pore_rows: list[dict],
                     design: GroupDesign, settings_rows: list[dict] | None = None, source: str = "",
                     errors: list[dict] | None = None) -> dict:
    """Tables, workbook and figures of a materials experiment; returns the files written."""
    first = _summary_first(ms)
    sample_first = ["repetition", "sample", "n_images", "n_pores", "pixel_size_min_um", "pixel_size_max_um"] + first[6:]
    files = {
        "summary": write_csv(out / "summary.csv", ms.summary, first),
        "per_sample": write_csv(out / "per_sample.csv", ms.samples, sample_first),
        "per_image": write_csv(out / "per_image.csv", image_rows, IMAGE_FIRST),
        "per_pore": write_csv(out / "per_pore.csv", pore_rows, PORE_FIRST),
    }
    if ms.comparisons:
        files["comparisons"] = write_csv(out / "comparisons.csv", ms.comparisons, COMP_FIRST)
    if ms.distribution:
        files["distribution"] = write_csv(out / "pore_size_distribution.csv", ms.distribution)
    n_warned = sum(1 for r in image_rows if str(r.get("warnings") or "").strip())
    readme = readme_text(ms, design, len(image_rows), n_warned, source)
    cite = citation_text("porosity")
    sheets = {"Read me": [], "Summary": ms.summary, "Comparisons": ms.comparisons, "Samples": ms.samples,
              "Images": image_rows, "Pores": pore_rows, "Distribution": ms.distribution}
    if not ms.comparisons:
        sheets.pop("Comparisons")
    if errors:
        sheets["Errors"] = errors
    if settings_rows:
        sheets["Settings"] = settings_rows
    sheets["How to cite"] = []
    x = write_xlsx(out / "results.xlsx", sheets,
                   {"Summary": first, "Comparisons": COMP_FIRST, "Samples": sample_first, "Images": IMAGE_FIRST,
                    "Pores": PORE_FIRST}, notes={"Read me": readme, "How to cite": cite})
    if x:
        files["excel"] = x
    (out / "CITATION.txt").write_text(cite, encoding="utf-8")
    files["citation"] = out / "CITATION.txt"
    try:
        f = summary_figure(out / "summary.png", ms, image_rows)
        if f:
            files["figure"] = f
        f = distribution_figure(out / "distribution.png", ms)
        if f:
            files["distribution_figure"] = f
    except Exception as exc:  # noqa: BLE001 - a plotting failure must not lose the results
        if errors is not None:
            errors.append({"image": "", "error": f"figure failed: {exc}"})
    return files


def save_porosity_images(out: Path, stem: str, res: PorosityResult) -> dict:
    L = res.layers
    files = {}
    files["binary"] = out / f"{stem}_binary_segmentation.png"
    save_png(files["binary"], np.repeat((L["solid"].astype(np.uint8) * 255)[..., None], 3, axis=2))
    files["depth"] = out / f"{stem}_depth_map.png"
    save_png(files["depth"], jet_map(L["grey"]))
    files["pores"] = out / f"{stem}_pore_space_segmentation.png"
    save_png(files["pores"], label_colours(L["pore_labels"]))
    files["overlay"] = out / f"{stem}_overlay.png"
    save_png(files["overlay"], porosity_overlay(L["grey"], L["pore_labels"], L["valid"]))
    return files


def _as_specs(items) -> list[ImageSpec]:
    """ImageSpecs from specs or (path, sample) pairs (version 0.2)."""
    out = []
    for it in items:
        if isinstance(it, ImageSpec):
            out.append(it)
        else:
            path, sample = (list(it) + [""])[:2]
            out.append(ImageSpec(str(path), sample=sample or ""))
    return out


# ---------------------------------------------------------------------- batch


def run_porosity(
    items,
    settings: PorositySettings,
    out_dir: str | Path,
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
    save_images: bool = True,
) -> dict:
    """Analyse SEM images (``ImageSpec`` objects or ``(path, sample)`` pairs) and write the results."""
    specs = _as_specs(items)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    if save_images:
        img_dir.mkdir(exist_ok=True)
    results: list[PorosityResult] = []
    errors: list[dict] = []
    n = len(specs)
    for i, sp in enumerate(specs):
        if cancel and cancel():
            errors.append({"image": "", "error": "cancelled by user"})
            break
        iid = sp.image_id or Path(sp.path).stem
        if progress:
            progress(i, n, f"{iid}: {Path(sp.path).name}")
        try:
            img = load_image(sp.path)
            r = analyse_sem(img, settings, sp.sample, keep_layers=save_images, pixel_size_um=sp.pixel_size_um,
                            repetition=sp.repetition or "1", image_id=iid)
            if save_images:
                stem = f"{i + 1:03d}_{_safe(Path(sp.path).stem)}"
                save_porosity_images(img_dir, stem, r)
                key = "equivalent_diameter_um" if math.isfinite(r.summary["pixel_size_um"]) else "equivalent_diameter_px"
                pore_histogram_figure(img_dir / f"{stem}_pore_size_distribution.png",
                                      np.array([p[key] for p in r.pores if p["included"]]), settings.histogram_bins,
                                      Path(sp.path).name, "µm" if key.endswith("um") else "px")
            r.layers = None
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            errors.append({"image": sp.path, "error": str(exc)})
    if progress:
        progress(n, n, "writing results")
    image_rows = []
    for r in results:
        row = dict(r.summary)
        row["warnings"] = " | ".join(r.warnings)
        image_rows.append(row)
    pore_rows = [{"image_id": r.image_id, "image": r.name, "sample": r.sample, "repetition": r.repetition, **p}
                 for r in results for p in r.pores]
    design = design_from_settings(settings)
    ms = summarise_samples(image_rows, pore_rows, design, settings.histogram_bins)
    files = write_experiment(out, ms, image_rows, pore_rows, design, _settings_rows(settings), errors=errors)
    if errors:
        files["errors"] = write_csv(out / "errors.csv", errors)
    inputs = [{"image": sp.path, "sample": sp.sample, "repetition": sp.repetition, "image_id": sp.image_id,
               "pixel_size_um": sp.pixel_size_um} for sp in specs]
    files["settings"] = write_settings_yaml(out / "settings.yaml", "porosity", settings, inputs)
    return {"results": results, "errors": errors, "experiment": ms, "summary": ms.summary, "image_rows": image_rows,
            "out_dir": out, "files": files}


def combine_results(folders: list[str | Path], design: GroupDesign | None, out_dir: str | Path,
                    repetitions: list[str] | None = None, bins: int = 25) -> dict:
    """One experiment from SEM result folders analysed separately (e.g. one per repetition)."""
    import yaml

    design = design or GroupDesign()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ms, image_rows, pore_rows = combine_result_folders(folders, design, repetitions, bins)
    source = "combined from " + ", ".join(Path(f).name for f in folders)
    errors: list[dict] = []
    files = write_experiment(out, ms, image_rows, pore_rows, design, source=source, errors=errors)
    if errors:
        files["errors"] = write_csv(out / "errors.csv", errors)
    doc = {"microscount_version": __version__, "analysis": "porosity",
           "combined_from": [str(Path(f).resolve()) for f in folders], "repetitions": repetitions,
           "design": design.to_dict(), "histogram_bins": bins}
    with open(out / "combine.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    files["settings"] = out / "combine.yaml"
    return {"experiment": ms, "summary": ms.summary, "image_rows": image_rows, "out_dir": out, "files": files,
            "errors": errors}
