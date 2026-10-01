"""Batch runs, tables and figures for the nuclear translocation tool.

A run analyses every field, summarises the experiment (conditions per repetition,
fold changes, responders, comparisons; see ``bio.experiment``) and writes:

``results.xlsx``       Read me, Summary, Comparisons, Conditions, Fields, Cells, Settings, How to cite
``summary.csv``        one row per condition across repetitions
``comparisons.csv``    each comparison per repetition and across repetitions
``per_condition.csv``  one row per repetition and condition
``per_field.csv``      one row per field (image pair)
``per_cell.csv``       one row per nucleus, measured or not
``histograms.csv``     the paper method's normalised ROI histograms
``summary.png``, ``cells.png``, ``histograms.png``, ``overlays/``
``settings.yaml``      everything needed to repeat the run; ``CITATION.txt``

``combine_results`` does the same summary for result folders analysed separately
(for example one per repetition).
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .. import __version__
from ..citation import citation_text
from ..core.imageio import load_image
from ..core.render import save_png, translocation_overlay
from ..core.report import (  # noqa: F401 - default_output_dir is re-exported
    default_output_dir,
    INK, INK2, SERIES, SERIES_2, SURFACE, Progress, _figure, _points_with_mean, _ref_line, _safe, _settings_rows,
    _style, write_csv, write_settings_yaml, write_xlsx,
)
from .experiment import ExperimentDesign, ExperimentSummary, combine_result_folders, summarise_experiment
from .translocation import FieldResult, FieldSpec, TranslocationSettings, analyse_field

REPETITION_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (validated for scatter)
NUCLEAR_COLOUR = SERIES  # categorical slot 1
CYTO_COLOUR = SERIES_2  # categorical slot 2

FIELD_FIRST = [
    "field", "condition", "repetition", "method", "nuclear_file", "target_file",
    # per-cell lab protocol
    "n_cells_analysed", "median_nc", "q1_nc", "q3_nc", "mean_nc", "sd_nc", "median_cn", "mean_cn",
    "mean_nuc_corr", "mean_cyto_corr", "background_mean", "background_source", "rolling_ball_radius",
    "responder_fraction", "n_particles", "n_nuclei", "excluded_cells", "nucleus_diameter_px",
    # paper method
    "paper_ratio", "paper_ok", "paper_nuclear_mean", "paper_cytoplasm_mean", "paper_nuclear_area_px",
    "paper_cytoplasm_area_px", "nuclear_threshold", "target_threshold", "threshold_method", "nuclei_count",
    "nuclear_mask_fraction", "target_mask_fraction", "warnings",
]
CELL_FIRST = [
    "field", "condition", "repetition", "cell", "included", "exclusion_reason", "nc_ratio", "cn_ratio", "nuc_mean",
    "cyto_mean", "background_mean", "nuc_corr", "cyto_corr", "log2_nc", "dapi_mean", "nucleus_area_px",
    "nucleus_area_um2", "nucleus_perimeter_px", "circularity", "x", "y",
]
COND_FIRST = [
    "repetition", "condition", "n_fields", "n_cells", "field_median_nc_mean", "field_median_nc_sd",
    "field_median_nc_sem", "fold_change", "fold_reference", "responder_fraction", "responder_cutoff", "responder_rule",
    "median_nc", "q1_nc", "q3_nc", "mean_nc", "sd_nc", "median_cn", "mean_cn", "mean_nuc_corr", "mean_cyto_corr",
    "paper_ratio_mean", "paper_ratio_sd", "paper_ratio_sem", "paper_fields_used", "paper_fields_excluded",
    "nuclei_count", "criteria_notes",
]
SUMMARY_FIRST = ["condition", "n_repetitions", "n_fields", "n_cells", "mean", "sd", "sem"]
COMP_FIRST = ["repetition", "condition_a", "condition_b", "n_a", "n_b", "n_repetitions", "mean_a", "mean_b",
              "difference", "difference_pct", "per_repetition", "same_direction", "welch_t", "welch_p", "paired_t",
              "paired_p", "test"]


# ---------------------------------------------------------------------- figures


def _values_for(field_rows: list[dict], cond: str, rep: str | None, per_cell: bool, exclude_failed: bool) -> np.ndarray:
    from .experiment import _f, _true
    from .pairing import condition_key

    k = condition_key(cond)
    out = []
    for r in field_rows:
        if condition_key(str(r.get("condition") or "(none)")) != k or (rep is not None and str(r.get("repetition")) != rep):
            continue
        if per_cell:
            out.append(_f(r.get("median_nc")))
        elif not exclude_failed or _true(r.get("paper_ok", True)):
            out.append(_f(r.get("paper_ratio")))
    return np.array(out, dtype=np.float64)


def experiment_figure(path: Path, exp: ExperimentSummary, field_rows: list[dict],
                      exclude_failed: bool = True) -> Path | None:
    """Field values per condition in each repetition (mean ± SD), then the repetitions together."""
    if not exp.conditions:
        return None
    per_cell = exp.measure.startswith("median")
    reps = exp.repetitions
    shown = reps if len(reps) <= 4 else []
    n = len(shown) + (1 if len(reps) > 1 else 0) or 1
    width = max(3.6, 0.55 * len(exp.order) + 1.6)
    fig, axes = _figure(n, width, 3.8)
    rng = np.random.default_rng(0)
    ylab = "Median N/C per field" if per_cell else "Paper N/C per field"
    panels = [(ax, rep) for ax, rep in zip(axes, shown or reps[:1])]
    for ax, rep in panels:
        _style(ax)
        vals = [_values_for(field_rows, c, rep, per_cell, exclude_failed) for c in exp.order]
        _points_with_mean(ax, exp.order, vals, rng)
        _ref_line(ax, 1.0, "N = C")
        ax.set_ylabel(ylab, color=INK, fontsize=10)
        ax.set_title(f"Repetition {rep} (fields; mean ± SD)" if len(reps) > 1 else "Fields (mean ± SD)",
                     color=INK, fontsize=10, loc="left")
    if len(reps) > 1:
        ax = axes[-1]
        _style(ax)
        by = {(r["repetition"], r["condition"]): r for r in exp.conditions}
        key = "field_median_nc_mean" if per_cell else "paper_ratio_mean"
        for j, rep in enumerate(reps):
            xs, ys = [], []
            for i, c in enumerate(exp.order):
                row = by.get((rep, c))
                if row is not None and math.isfinite(row.get(key, math.nan)):
                    xs.append(i + (j - (len(reps) - 1) / 2) * 0.08)
                    ys.append(row[key])
            colour = REPETITION_COLOURS[j] if len(reps) <= 3 else SERIES
            ax.scatter(xs, ys, s=34, color=colour, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label=f"repetition {rep}" if len(reps) <= 3 else None)
        for i, o in enumerate(exp.overall):
            if math.isfinite(o["mean"]):
                sd = o["sd"] if math.isfinite(o["sd"]) else 0.0
                ax.errorbar(exp.order.index(o["condition"]) + 0.3, o["mean"], yerr=sd, fmt="_", color=INK,
                            markersize=12, capsize=3, linewidth=1.0, zorder=4)
        ax.set_xticks(range(len(exp.order)))
        long = max((len(c) for c in exp.order), default=0) > 10
        ax.set_xticklabels(exp.order, rotation=20 if long else 0, ha="right" if long else "center", color=INK)
        ax.set_xlim(-0.6, len(exp.order) - 0.4)
        _ref_line(ax, 1.0, "N = C")
        ax.set_title("Repetitions (mean ± SD across repetitions)", color=INK, fontsize=10, loc="left")
        if len(reps) <= 3:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def cells_figure(path: Path, field_rows: list[dict], cell_rows: list[dict], order: list[str],
                 exclude_failed: bool = True) -> Path | None:
    """Per-cell N/C per condition (all repetitions), and the paper-method ratio per field."""
    from .experiment import _f, _true
    from .pairing import condition_key

    if not field_rows:
        return None
    per_cell = bool(cell_rows)
    rng = np.random.default_rng(0)
    width = max(3.6, 0.55 * len(order) + 1.6)
    fig, axes = _figure(2 if per_cell else 1, width, 3.8)
    k = 0
    if per_cell:
        ax = axes[0]
        _style(ax)
        data = []
        for c in order:
            key = condition_key(c)
            v = np.array([_f(x.get("nc_ratio")) for x in cell_rows
                          if _true(x.get("included", True)) and condition_key(str(x.get("condition") or "(none)")) == key])
            data.append(v[np.isfinite(v) & (v > 0)])
        for i, v in enumerate(data):
            if v.size == 0:
                continue
            show = v if v.size <= 1500 else rng.choice(v, 1500, replace=False)
            ax.scatter(i + rng.uniform(-0.25, 0.25, show.size), show, s=4, color=SERIES, alpha=0.25, linewidth=0, zorder=2)
        ax.boxplot([d if d.size else [np.nan] for d in data], positions=range(len(order)), widths=0.55,
                   whis=(5, 95), showfliers=False, patch_artist=False,
                   medianprops={"color": INK, "linewidth": 1.6},
                   boxprops={"color": INK, "linewidth": 0.9}, whiskerprops={"color": INK, "linewidth": 0.9},
                   capprops={"color": INK, "linewidth": 0.9})
        for i, v in enumerate(data):
            if v.size:
                ax.annotate(f"{np.median(v):.2f}", xy=(i, np.percentile(v, 95)), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(len(order)))
        long = max((len(c) for c in order), default=0) > 10
        ax.set_xticklabels(order, rotation=20 if long else 0, ha="right" if long else "center", color=INK)
        _ref_line(ax, 1.0, "N = C")
        ax.set_ylabel("N/C = Nuc_corr / Cyto_corr", color=INK, fontsize=10)
        ax.set_title("Per cell, all repetitions (box: IQR, whiskers: 5–95%)", color=INK, fontsize=10, loc="left")
        k = 1
    ax = axes[k]
    _style(ax)
    vals = [_values_for(field_rows, c, None, False, exclude_failed) for c in order]
    _points_with_mean(ax, order, vals, rng)
    _ref_line(ax, 1.0, "N = C")
    ax.set_ylabel("N/C ratio per field", color=INK, fontsize=10)
    ax.set_title("Per field, Noursadeghi et al. 2008 (mean ± SD)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def translocation_figure(path: Path, results: list[FieldResult]) -> Path | None:
    """Per-cell and per-field figure of a list of field results (kept from version 0.1)."""
    rows = [dict(r.summary) for r in results]
    cells = [c for r in results for c in r.cells]
    order = list(dict.fromkeys(r.condition or "(none)" for r in results))
    return cells_figure(path, rows, cells, order)


def condition_histograms(results: list[FieldResult]) -> dict[str, dict]:
    """Mean normalised ROI histograms per condition (as Fig. 2B of the paper)."""
    out: dict[str, dict] = {}
    for r in results:
        h = getattr(r, "histograms", None)
        if not h:
            continue
        c = r.condition or "(none)"
        d = out.setdefault(c, {"intensity": h["intensity"], "nuclear_pct": [], "cytoplasm_pct": []})
        if len(d["intensity"]) != len(h["intensity"]) or not np.allclose(d["intensity"], h["intensity"]):
            continue  # different bit depths within one condition: keep the first kind
        d["nuclear_pct"].append(h["nuclear_pct"])
        d["cytoplasm_pct"].append(h["cytoplasm_pct"])
    for c, d in out.items():
        d["nuclear_pct"] = np.mean(d["nuclear_pct"], axis=0)
        d["cytoplasm_pct"] = np.mean(d["cytoplasm_pct"], axis=0)
    return out


def histogram_figure(path: Path, per_condition: dict[str, dict]) -> Path | None:
    if not per_condition:
        return None
    conds = list(per_condition)[:6]
    fig, axes = _figure(len(conds), 4.0, 3.3)
    for ax, c in zip(axes, conds):
        _style(ax)
        d = per_condition[c]
        x = np.asarray(d["intensity"])
        keep = x >= (1 if x[1] - x[0] >= 1 else x[1])  # the zero bin is not a data point
        ax.plot(x[keep], d["nuclear_pct"][keep], color=NUCLEAR_COLOUR, linewidth=1.6, label="Nuclear")
        ax.plot(x[keep], d["cytoplasm_pct"][keep], color=CYTO_COLOUR, linewidth=1.6, label="Cytoplasmic")
        ax.set_xlabel("Target intensity", color=INK, fontsize=9)
        ax.set_ylabel("Frequency (%)", color=INK, fontsize=9)
        ax.set_title(c, color=INK, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


# ---------------------------------------------------------------------- tables


def readme_text(exp: ExperimentSummary, design: ExperimentDesign, n_fields: int, n_warned: int,
                source: str = "") -> str:
    reps = exp.repetitions
    per_cell = exp.measure.startswith("median")
    rule = next((r.get("responder_rule") for r in exp.conditions if r.get("responder_rule")), "")
    lines = [
        f"MicrosCount {__version__} · Bio & Cells · nuclear translocation",
        f"Created {_dt.datetime.now().strftime('%d %B %Y %H:%M')}" + (f" · {source}" if source else ""),
        f"{n_fields} field(s) in {len(reps)} repetition(s) ({', '.join(reps)}); conditions: {', '.join(exp.order)}.",
        "",
        "SHEETS",
        "Summary        one row per condition: its value in each repetition, mean ± SD across repetitions, "
        "fold change and responders",
        "Comparisons    each comparison in each repetition, then across repetitions",
        "Conditions     one row per repetition and condition: field values, pooled cells, fold change, responders, "
        "paper-method ratio",
        "Fields         one row per field (image pair), with its warnings",
    ]
    if per_cell:
        lines.append("Cells          one row per nucleus found, measured or not (with the reason)")
    lines += [
        "Settings       every setting used",
        "",
        "DEFINITIONS",
    ]
    if per_cell:
        lines += [
            "N/C per cell = Nuc_corr / Cyto_corr, Nuc_corr = Nuc_Mean − Background_Mean, Cyto_corr = Cyto_Mean − "
            "Background_Mean (target channel after the rolling ball; Background_Mean from the cell-free area).",
            "Field value = median N/C of the cells measured in the field.",
        ]
    else:
        lines.append("Field value = paper-method N/C (Noursadeghi et al. 2008): mean target intensity in the nuclear "
                     "mask / in the target mask minus the nuclear mask.")
    lines += [
        "Condition value = mean ± SD of its field values within a repetition (fields are technical replicates).",
        "Across repetitions = mean ± SD of the repetition values (the repetitions are the independent replicates).",
        f"Control: {design.control or '(none set)'}.",
        "Fold change = condition value / value of its reference condition in the same repetition "
        "(the control unless another reference was set).",
    ]
    if per_cell:
        lines.append(f"Responders = share of measured cells {rule or 'above the cut-off'} (per repetition).")
    lines += [
        "Comparisons within a repetition: Welch t-test on the field values. Exploratory: fields of one dish are not "
        "independent experiments.",
        "Comparisons across repetitions: difference in each repetition, whether its direction agrees, and a paired "
        "t-test on the repetition values when there are at least 3 repetitions.",
    ]
    if design.exclude_failed_paper_fields:
        lines.append("Paper-method means leave out fields whose automatic threshold failed (Fields sheet: paper_ok = "
                     "False; nuclear mask above 60% or below 0.2% of the field, or target mask above 97% or below "
                     "0.5%).")
    lines += ["", "NOTES"]
    notes = list(exp.notes)
    if n_warned:
        notes.append(f"{n_warned} field(s) have warnings: see the warnings column of the Fields sheet.")
    lines += notes or ["none"]
    return "\n".join(lines)


def write_experiment(out: Path, exp: ExperimentSummary, field_rows: list[dict], cell_rows: list[dict],
                     design: ExperimentDesign, settings_rows: list[dict] | None = None, source: str = "",
                     errors: list[dict] | None = None) -> dict:
    """Tables, workbook and figures of an experiment summary; returns the files written."""
    files = {
        "summary": write_csv(out / "summary.csv", exp.overall, SUMMARY_FIRST),
        "per_condition": write_csv(out / "per_condition.csv", exp.conditions, COND_FIRST),
        "per_field": write_csv(out / "per_field.csv", field_rows, FIELD_FIRST),
    }
    if exp.comparisons:
        files["comparisons"] = write_csv(out / "comparisons.csv", exp.comparisons, COMP_FIRST)
    if cell_rows:
        files["per_cell"] = write_csv(out / "per_cell.csv", cell_rows, CELL_FIRST)
    n_warned = sum(1 for r in field_rows if str(r.get("warnings") or "").strip())
    readme = readme_text(exp, design, len(field_rows), n_warned, source)
    cite = citation_text("translocation")
    sheets = {"Read me": [], "Summary": exp.overall, "Comparisons": exp.comparisons, "Conditions": exp.conditions,
              "Fields": field_rows}
    if cell_rows:
        sheets["Cells"] = cell_rows
    if errors:
        sheets["Errors"] = errors
    if settings_rows:
        sheets["Settings"] = settings_rows
    sheets["How to cite"] = []
    if not exp.comparisons:
        sheets.pop("Comparisons")
    x = write_xlsx(out / "results.xlsx", sheets,
                   {"Summary": SUMMARY_FIRST, "Comparisons": COMP_FIRST, "Conditions": COND_FIRST,
                    "Fields": FIELD_FIRST, "Cells": CELL_FIRST},
                   notes={"Read me": readme, "How to cite": cite})
    if x:
        files["excel"] = x
    (out / "CITATION.txt").write_text(cite, encoding="utf-8")
    files["citation"] = out / "CITATION.txt"
    try:
        f = experiment_figure(out / "summary.png", exp, field_rows, design.exclude_failed_paper_fields)
        if f:
            files["figure"] = f
        f = cells_figure(out / "cells.png", field_rows, cell_rows, exp.order, design.exclude_failed_paper_fields)
        if f:
            files["cells_figure"] = f
    except Exception as exc:  # noqa: BLE001 - a plotting failure must not lose the results
        if errors is not None:
            errors.append({"field": "", "error": f"figure failed: {exc}"})
    return files


# ---------------------------------------------------------------------- batch: translocation


def run_translocation(
    specs: list[FieldSpec],
    settings: TranslocationSettings,
    out_dir: str | Path,
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
    save_overlays: bool = True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ov_dir = out / "overlays"
    if save_overlays:
        ov_dir.mkdir(exist_ok=True)
    results: list[FieldResult] = []
    errors: list[dict] = []
    n = len(specs)
    for i, spec in enumerate(specs):
        if cancel and cancel():
            errors.append({"field": "", "error": "cancelled by user"})
            break
        fid = spec.field_id or f"field_{i + 1:03d}"
        if progress:
            progress(i, n, f"{fid}: {Path(spec.nuclear_path).name}")
        try:
            nuc = load_image(spec.nuclear_path)
            tgt = nuc if spec.target_path == spec.nuclear_path else load_image(spec.target_path)
            r = analyse_field(nuc, tgt, settings, fid, spec.condition, keep_layers=save_overlays,
                              repetition=spec.repetition or "1")
            if save_overlays and r.layers is not None:
                save_png(ov_dir / f"{_safe(fid)}_{_safe(Path(spec.nuclear_path).stem)}_overlay.png",
                         translocation_overlay(r.layers, base="target"))
            r.layers = None
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            errors.append({"field": fid, "nuclear_file": spec.nuclear_path, "target_file": spec.target_path, "error": str(exc)})
    if progress:
        progress(n, n, "writing results")
    design = ExperimentDesign.from_settings(settings)
    field_rows = []
    for r in results:
        row = dict(r.summary)
        row["warnings"] = " | ".join(r.warnings)
        field_rows.append(row)
    cell_rows = [c for r in results for c in r.cells]
    crit = (int(settings.fields_per_condition), int(settings.min_cells_per_condition))
    exp = summarise_experiment(field_rows, cell_rows, design, crit)
    hist_rows = []
    for r in results:
        h = getattr(r, "histograms", None)
        if not h:
            continue
        for x, a, b in zip(h["intensity"], h["nuclear_pct"], h["cytoplasm_pct"]):
            hist_rows.append({"field": r.field_id, "condition": r.condition, "repetition": r.repetition,
                              "intensity": float(x), "nuclear_frequency_pct": float(a),
                              "cytoplasmic_frequency_pct": float(b)})
    files = write_experiment(out, exp, field_rows, cell_rows, design, _settings_rows(settings), errors=errors)
    if hist_rows:
        files["histograms"] = write_csv(out / "histograms.csv", hist_rows)
    if errors:
        files["errors"] = write_csv(out / "errors.csv", errors)
    inputs = [{"field": s.field_id, "condition": s.condition, "repetition": s.repetition, "nuclear": s.nuclear_path,
               "target": s.target_path} for s in specs]
    files["settings"] = write_settings_yaml(out / "settings.yaml", "translocation", settings, inputs)
    try:
        f = histogram_figure(out / "histograms.png", condition_histograms(results))
        if f:
            files["histogram_figure"] = f
    except Exception as exc:  # noqa: BLE001
        errors.append({"field": "", "error": f"figure failed: {exc}"})
    return {"results": results, "errors": errors, "conditions": exp.conditions, "experiment": exp,
            "field_rows": field_rows, "out_dir": out, "files": files}


def combine_results(folders: list[str | Path], design: ExperimentDesign | None, out_dir: str | Path,
                    repetitions: list[str] | None = None) -> dict:
    """One experiment from result folders analysed separately (e.g. one per repetition)."""
    import yaml

    design = design or ExperimentDesign()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exp, field_rows, cell_rows = combine_result_folders(folders, design, repetitions)
    source = "combined from " + ", ".join(Path(f).name for f in folders)
    files = write_experiment(out, exp, field_rows, cell_rows, design, source=source, errors=[])
    doc = {"microscount_version": __version__, "combined_from": [str(Path(f).resolve()) for f in folders],
           "repetitions": repetitions, "design": design.to_dict()}
    with open(out / "combine.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    files["settings"] = out / "combine.yaml"
    return {"experiment": exp, "conditions": exp.conditions, "field_rows": field_rows, "out_dir": out, "files": files,
            "errors": []}
