"""Batch runs, tables and figures for the SEM porosity tool."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..citation import citation_text
from ..core.imageio import load_image
from ..core.render import jet_map, label_colours, porosity_overlay, save_png
from ..core.report import (  # noqa: F401 - default_output_dir is re-exported
    default_output_dir,
    INK, INK2, SERIES, SURFACE, Progress, _figure, _points_with_mean, _safe, _settings_rows, _style, write_csv,
    write_settings_yaml, write_xlsx,
)
from .porosity import PorosityResult, PorositySettings, analyse_sem, summarise_porosity


def pore_histogram_figure(path: Path, radii: np.ndarray, bins: int = 25, title: str = "") -> Path | None:
    radii = radii[np.isfinite(radii)]
    fig, axes = _figure(1, 5.0, 3.4)
    ax = axes[0]
    _style(ax)
    if radii.size:
        ax.hist(radii, bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=1.0, zorder=2)
        m = radii.mean()
        ax.axvline(m, color=INK2, linewidth=0.8, zorder=3)
        ax.annotate(f"mean {m:.2f} µm", xy=(m, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Equivalent pore radius (µm)", color=INK, fontsize=10)
    ax.set_ylabel("Number of pores", color=INK, fontsize=10)
    ax.set_title(title or "Pore size distribution", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def porosity_figure(path: Path, results: list[PorosityResult]) -> Path | None:
    if not results:
        return None
    conds = list(dict.fromkeys(r.condition or "(none)" for r in results))
    rng = np.random.default_rng(0)
    fig, axes = _figure(2)
    _style(axes[0])
    _points_with_mean(axes[0], conds, [np.array([100 * r.summary["porosity"] for r in results if (r.condition or "(none)") == c]) for c in conds], rng)
    axes[0].set_ylabel("Porosity (%)", color=INK, fontsize=10)
    axes[0].set_title("Porosity per image (mean ± SD)", color=INK, fontsize=10, loc="left")
    _style(axes[1])
    _points_with_mean(axes[1], conds, [np.array([r.summary["mean_pore_radius_um"] for r in results if (r.condition or "(none)") == c]) for c in conds], rng)
    axes[1].set_ylabel("Mean pore radius (µm)", color=INK, fontsize=10)
    axes[1].set_title("Mean pore radius per image (mean ± SD)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


# ---------------------------------------------------------------------- batch: translocation

POROSITY_FIRST = ["image", "condition", "porosity", "porosity_percent", "n_pores", "mean_pore_radius_um", "sd_pore_radius_um"]


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


def run_porosity(
    items: list[tuple[str, str]],  # (path, condition)
    settings: PorositySettings,
    out_dir: str | Path,
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
    save_images: bool = True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    if save_images:
        img_dir.mkdir(exist_ok=True)
    results: list[PorosityResult] = []
    errors: list[dict] = []
    n = len(items)
    for i, (path, cond) in enumerate(items):
        if cancel and cancel():
            errors.append({"image": "", "error": "cancelled by user"})
            break
        if progress:
            progress(i, n, Path(path).name)
        try:
            img = load_image(path)
            r = analyse_sem(img, settings, cond, keep_layers=save_images)
            stem = f"{i + 1:03d}_{_safe(Path(path).stem)}"
            if save_images:
                save_porosity_images(img_dir, stem, r)
                radii = np.array([p["equivalent_radius_um"] for p in r.pores if p["included"]])
                pore_histogram_figure(img_dir / f"{stem}_pore_size_distribution.png", radii, settings.histogram_bins, Path(path).name)
            r.layers = None
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            errors.append({"image": path, "error": str(exc)})
    if progress:
        progress(n, n, "writing results")
    summary_rows = []
    for r in results:
        row = dict(r.summary)
        row["warnings"] = " | ".join(r.warnings)
        summary_rows.append(row)
    pore_rows = []
    for r in results:
        for p in r.pores:
            pore_rows.append({"image": r.name, "condition": r.condition, **p})
    conditions = summarise_porosity(results)
    files = {
        "summary": write_csv(out / "porosity_summary.csv", summary_rows, POROSITY_FIRST),
        "pores": write_csv(out / "pores.csv", pore_rows),
        "per_condition": write_csv(out / "porosity_by_condition.csv", conditions),
    }
    if errors:
        files["errors"] = write_csv(out / "errors.csv", errors)
    inputs = [{"image": p, "condition": c} for p, c in items]
    files["settings"] = write_settings_yaml(out / "settings.yaml", "porosity", settings, inputs)
    cite = citation_text("porosity")
    (out / "CITATION.txt").write_text(cite, encoding="utf-8")
    files["citation"] = out / "CITATION.txt"
    sheets = {"Images": summary_rows, "Conditions": conditions, "Pores": pore_rows}
    if errors:
        sheets["Errors"] = errors
    sheets["Settings"] = _settings_rows(settings)
    sheets["How to cite"] = []
    x = write_xlsx(out / "results.xlsx", sheets, {"Images": POROSITY_FIRST}, notes={"How to cite": cite})
    if x:
        files["excel"] = x
    try:
        f = porosity_figure(out / "summary.png", results)
        if f:
            files["figure"] = f
    except Exception as exc:  # noqa: BLE001
        errors.append({"image": "", "error": f"summary figure failed: {exc}"})
    return {"results": results, "errors": errors, "conditions": conditions, "out_dir": out, "files": files}
