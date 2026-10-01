"""Experiments: conditions, repetitions, a control and comparisons.

Each field gives a value per cell (N/C) and one value for the field: the median N/C of
its cells (per-cell protocol) or its paper-method ratio. Within each *repetition* (an
independent experiment) the fields are grouped by condition:

* condition value = mean ± SD of its field values (fields are technical replicates);
* fold change = condition value / the value of its reference condition in the same
  repetition (the control unless another reference is set);
* responders = share of cells above a percentile (default the 95th) of the control's
  cells in the same repetition, or above a fixed N/C when there is no control;
* comparisons A vs B within a repetition: difference of the means and Welch's t-test on
  the field values. This is exploratory: fields of one dish are not independent;
* across repetitions: mean ± SD of the repetition values and, for each comparison, the
  difference in every repetition, whether its direction agrees, and a paired t-test on
  the repetition values when there are at least three repetitions (the repetitions are
  the independent replicates).

Everything works on plain rows (dicts), so results saved by earlier runs can be combined
(``combine_result_folders``).
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .pairing import condition_key

NAN = math.nan


@dataclass
class ExperimentDesign:
    control: str = ""  # control condition: responder cut-off, default reference and comparisons
    references: dict = field(default_factory=dict)  # condition -> reference condition for its fold change
    comparisons: list = field(default_factory=list)  # [[A, B], ...]; empty = every condition vs the control
    order: list = field(default_factory=list)  # condition order in tables and figures
    responder_percentile: float | None = 95.0  # of the control's cells; None/0 = fixed cut-off
    responder_ratio: float | None = 1.0  # fixed N/C cut-off without a control
    exclude_failed_paper_fields: bool = True  # leave fields whose automatic threshold failed out of paper means

    @classmethod
    def from_settings(cls, s) -> "ExperimentDesign":
        return cls(
            control=getattr(s, "control_condition", "") or "",
            references=dict(getattr(s, "fold_references", {}) or {}),
            comparisons=[list(c) for c in (getattr(s, "comparisons", []) or [])],
            order=list(getattr(s, "condition_order", []) or []),
            responder_percentile=getattr(s, "responder_percentile", 95.0),
            responder_ratio=getattr(s, "responder_ratio", 1.0),
            exclude_failed_paper_fields=bool(getattr(s, "exclude_failed_paper_fields", True)),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentDesign":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class ExperimentSummary:
    conditions: list[dict]  # one row per repetition and condition
    comparisons: list[dict]  # one row per repetition and comparison, then one "all repetitions" row each
    overall: list[dict]  # one row per condition across repetitions
    order: list[str]
    repetitions: list[str]
    measure: str  # what the field value is
    notes: list[str]


# ---------------------------------------------------------------------- helpers


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return NAN
    return x


def _true(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _mean_sd(v) -> tuple[float, float, float]:
    v = np.asarray([x for x in v if math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return NAN, NAN, NAN
    sd = float(v.std(ddof=1)) if v.size > 1 else NAN
    return float(v.mean()), sd, (sd / math.sqrt(v.size) if v.size > 1 else NAN)


def _welch(a, b) -> tuple[float, float]:
    from scipy import stats

    a = [x for x in a if math.isfinite(x)]
    b = [x for x in b if math.isfinite(x)]
    if len(a) < 2 or len(b) < 2 or (np.ptp(a) == 0 and np.ptp(b) == 0):
        return NAN, NAN
    with warnings.catch_warnings():  # identical values in one group: scipy warns, the test is still defined
        warnings.simplefilter("ignore", RuntimeWarning)
        r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def _paired(a, b) -> tuple[float, float]:
    from scipy import stats

    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return NAN, NAN
    d = np.array([y - x for x, y in pairs])
    if np.ptp(d) == 0:
        return NAN, NAN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = stats.ttest_rel([p[1] for p in pairs], [p[0] for p in pairs])
    return float(r.statistic), float(r.pvalue)


def _stats(v, prefix: str) -> dict:
    v = np.asarray([x for x in v if math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {f"median_{prefix}": NAN, f"q1_{prefix}": NAN, f"q3_{prefix}": NAN, f"mean_{prefix}": NAN,
                f"sd_{prefix}": NAN}
    return {
        f"median_{prefix}": float(np.median(v)),
        f"q1_{prefix}": float(np.percentile(v, 25)),
        f"q3_{prefix}": float(np.percentile(v, 75)),
        f"mean_{prefix}": float(v.mean()),
        f"sd_{prefix}": float(v.std(ddof=1)) if v.size > 1 else NAN,
    }


def _signed_pct(v: float) -> str:
    return "–" if not math.isfinite(v) else f"{v:+.1f}%"


def order_conditions(names: list[str], preferred: list[str] | None = None) -> list[str]:
    """Conditions in the preferred order first (spelling-insensitive), then as met."""
    seen: dict[str, str] = {}
    for n in names:
        seen.setdefault(condition_key(n), n)
    out = []
    for p in preferred or []:
        k = condition_key(p)
        if k in seen and seen[k] not in out:
            out.append(seen[k])
    out += [n for n in seen.values() if n not in out]
    return out


def order_repetitions(reps: list[str]) -> list[str]:
    uniq = list(dict.fromkeys(reps))
    if all(r.strip().lstrip("-").replace(".", "", 1).isdigit() for r in uniq):
        return sorted(uniq, key=float)
    return uniq


# ---------------------------------------------------------------------- summary


def summarise_experiment(field_rows: list[dict], cell_rows: list[dict] | None,
                         design: ExperimentDesign | None = None, criteria: tuple[int, int] = (5, 500)) -> ExperimentSummary:
    """Condition values per repetition, fold changes, responders and comparisons."""
    design = design or ExperimentDesign()
    cell_rows = cell_rows or []
    notes: list[str] = []
    for r in field_rows:
        r.setdefault("repetition", "1")
        if not str(r.get("repetition", "")).strip():
            r["repetition"] = "1"
    names = [str(r.get("condition") or "(none)") for r in field_rows]
    order = order_conditions(names, design.order or ([design.control] if design.control else []))  # control first
    canon = {condition_key(c): c for c in order}

    def canon_of(name) -> str:
        return canon.get(condition_key(str(name or "(none)")), str(name or "(none)"))

    reps = order_repetitions([str(r["repetition"]) for r in field_rows])
    per_cell = any(math.isfinite(_f(r.get("median_nc"))) for r in field_rows) or bool(cell_rows)
    measure = "median N/C of the cells in each field" if per_cell else "paper-method N/C of each field"
    control = canon.get(condition_key(design.control), "") if design.control else ""
    if design.control and not control:
        notes.append(f"control condition '{design.control}' not found; no fold changes against it, no responder cut-off")

    # index rows
    F: dict[tuple[str, str], list[dict]] = {}
    for r in field_rows:
        F.setdefault((str(r["repetition"]), canon_of(r.get("condition"))), []).append(r)
    field_rep = {str(r.get("field")): str(r["repetition"]) for r in field_rows}
    C: dict[tuple[str, str], list[dict]] = {}
    for c in cell_rows:
        if not _true(c.get("included", True)):
            continue
        rep = str(c.get("repetition") or field_rep.get(str(c.get("field")), "1") or "1")
        C.setdefault((rep, canon_of(c.get("condition"))), []).append(c)

    def field_values(rows) -> list[float]:
        if per_cell:
            return [_f(r.get("median_nc")) for r in rows]
        return [_f(r.get("paper_ratio")) for r in rows
                if not design.exclude_failed_paper_fields or _true(r.get("paper_ok", True))]

    # responder cut-off per repetition
    cutoffs: dict[str, tuple[float, str]] = {}
    for rep in reps:
        cut, rule = NAN, ""
        if per_cell and control and design.responder_percentile:
            v = [_f(c.get("nc_ratio")) for c in C.get((rep, control), [])]
            v = [x for x in v if math.isfinite(x)]
            if v:
                cut = float(np.percentile(v, float(design.responder_percentile)))
                rule = f"above the {float(design.responder_percentile):g}th percentile of {control} (N/C {cut:.3g})"
        if not math.isfinite(cut) and per_cell and design.responder_ratio:
            cut, rule = float(design.responder_ratio), f"N/C above {float(design.responder_ratio):g}"
        cutoffs[rep] = (cut, rule)

    # ---- per repetition and condition
    rows: list[dict] = []
    value: dict[tuple[str, str], float] = {}
    for rep in reps:
        for cond in order:
            fr = F.get((rep, cond))
            if not fr:
                continue
            row = {"repetition": rep, "condition": cond, "n_fields": len(fr)}
            nuclei = int(sum(int(_f(r.get("nuclei_count")) or 0) for r in fr if math.isfinite(_f(r.get("nuclei_count")))))
            row["nuclei_count"] = nuclei
            paper_all = [_f(r.get("paper_ratio")) for r in fr]
            ok = [_true(r.get("paper_ok", True)) for r in fr]
            paper_use = [p for p, g in zip(paper_all, ok) if g or not design.exclude_failed_paper_fields]
            m, sd, sem = _mean_sd(paper_use)
            row.update(paper_ratio_mean=m, paper_ratio_sd=sd, paper_ratio_sem=sem,
                       paper_fields_used=sum(1 for p in paper_use if math.isfinite(p)),
                       paper_fields_excluded=sum(1 for g in ok if not g) if design.exclude_failed_paper_fields else 0)
            want_f, want_c = criteria
            crit = []
            if len(fr) < want_f:
                crit.append(f"{len(fr)} field(s): the paper analysed {want_f} per condition")
            if nuclei < want_c:
                crit.append(f"about {nuclei} nuclei: the paper required at least {want_c} cells")
            row["criteria_notes"] = "; ".join(crit)
            if per_cell:
                cells = C.get((rep, cond), [])
                nc = [_f(c.get("nc_ratio")) for c in cells]
                row["n_cells"] = len(cells)
                row.update(_stats(nc, "nc"))
                cn = _stats([_f(c.get("cn_ratio")) for c in cells], "cn")
                row.update({k: v for k, v in cn.items() if k.startswith(("median", "mean"))})
                row["mean_nuc_corr"] = _mean_sd([_f(c.get("nuc_corr")) for c in cells])[0]
                row["mean_cyto_corr"] = _mean_sd([_f(c.get("cyto_corr")) for c in cells])[0]
                m, sd, sem = _mean_sd(field_values(fr))
                row.update(field_median_nc_mean=m, field_median_nc_sd=sd, field_median_nc_sem=sem)
                cut, rule = cutoffs[rep]
                finite = [x for x in nc if math.isfinite(x)]
                row["responder_fraction"] = float(np.mean([x > cut for x in finite])) if finite and math.isfinite(cut) else NAN
                row["responder_cutoff"] = cut
                row["responder_rule"] = rule
                value[(rep, cond)] = row["field_median_nc_mean"]
            else:
                value[(rep, cond)] = row["paper_ratio_mean"]
            rows.append(row)

    # fold changes against each condition's reference
    refs = {condition_key(k): v for k, v in (design.references or {}).items()}
    for row in rows:
        ref_name = refs.get(condition_key(row["condition"])) or control
        ref = canon.get(condition_key(ref_name), "") if ref_name else ""
        row["fold_reference"] = ref
        v, rv = value.get((row["repetition"], row["condition"]), NAN), value.get((row["repetition"], ref), NAN)
        row["fold_change"] = v / rv if ref and math.isfinite(v) and math.isfinite(rv) and rv != 0 else NAN

    # ---- comparisons
    pairs = [tuple(c[:2]) for c in design.comparisons if len(c) >= 2]
    if not pairs and control:
        pairs = [(control, c) for c in order if c != control]
    comp_rows: list[dict] = []
    for a_name, b_name in pairs:
        a, b = canon.get(condition_key(a_name)), canon.get(condition_key(b_name))
        if not a or not b:
            notes.append(f"comparison {a_name} vs {b_name}: condition not found")
            continue
        per_rep = []
        for rep in reps:
            fa, fb = F.get((rep, a)), F.get((rep, b))
            if not fa or not fb:
                continue
            va, vb = field_values(fa), field_values(fb)
            ma, mb = _mean_sd(va)[0], _mean_sd(vb)[0]
            t, p = _welch(va, vb)
            d = mb - ma if math.isfinite(ma) and math.isfinite(mb) else NAN
            pct = 100 * (mb / ma - 1) if math.isfinite(d) and ma != 0 else NAN
            comp_rows.append({"repetition": rep, "condition_a": a, "condition_b": b, "n_a": len(va), "n_b": len(vb),
                              "mean_a": ma, "mean_b": mb, "difference": d, "difference_pct": pct, "welch_t": t,
                              "welch_p": p, "test": "Welch t-test on field values (exploratory: fields are not "
                                                    "independent experiments)"})
            per_rep.append((rep, ma, mb, pct))
        if len(reps) > 1 and per_rep:
            ma = [x[1] for x in per_rep]
            mb = [x[2] for x in per_rep]
            pcts = [x[3] for x in per_rep if math.isfinite(x[3])]
            t, p = _paired(ma, mb)
            same = len({math.copysign(1, x) for x in pcts if x != 0}) <= 1 if pcts else False
            comp_rows.append({
                "repetition": "all", "condition_a": a, "condition_b": b, "n_repetitions": len(per_rep),
                "mean_a": _mean_sd(ma)[0], "mean_b": _mean_sd(mb)[0],
                "difference_pct": float(np.mean(pcts)) if pcts else NAN,
                "per_repetition": "; ".join(f"{r}: {_signed_pct(x)}" for r, _, _, x in per_rep),
                "same_direction": same, "paired_t": t, "paired_p": p,
                "test": "paired t-test on the repetition values" if math.isfinite(p) else
                        "at least 3 repetitions are needed for a test across repetitions",
            })

    # ---- across repetitions
    overall: list[dict] = []
    for cond in order:
        rr = [r for r in rows if r["condition"] == cond]
        if not rr:
            continue
        vals = [value.get((r["repetition"], cond), NAN) for r in rr]
        m, sd, sem = _mean_sd(vals)
        o = {"condition": cond, "n_repetitions": len(rr), "n_fields": sum(r["n_fields"] for r in rr),
             "mean": m, "sd": sd, "sem": sem}
        if per_cell:
            o["n_cells"] = sum(r.get("n_cells", 0) for r in rr)
        for r, v in zip(rr, vals):
            o[f"repetition {r['repetition']}"] = v
        o["fold_change_mean"] = _mean_sd([r["fold_change"] for r in rr])[0]
        o["fold_reference"] = rr[0]["fold_reference"]
        if per_cell:
            o["responder_fraction_mean"] = _mean_sd([r.get("responder_fraction", NAN) for r in rr])[0]
        o["paper_ratio_mean"] = _mean_sd([r["paper_ratio_mean"] for r in rr])[0]
        overall.append(o)
    if len(reps) < 3 and len(reps) > 1:
        notes.append(f"{len(reps)} repetitions: differences are reported per repetition; a test across repetitions "
                     "needs at least 3")
    return ExperimentSummary(rows, comp_rows, overall, order, reps, measure, notes)


# ---------------------------------------------------------------------- saved results


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_result_folder(folder: str | Path, repetition: str | None = None) -> tuple[list[dict], list[dict]]:
    """Field and cell rows of a MicrosCount result folder (``per_field.csv``, ``per_cell.csv``)."""
    folder = Path(folder)
    fr = _read_csv(folder / "per_field.csv")
    cr = _read_csv(folder / "per_cell.csv") if (folder / "per_cell.csv").exists() else []
    for rows in (fr, cr):
        for r in rows:
            if repetition is not None:
                r["repetition"] = repetition
            elif not str(r.get("repetition", "")).strip():
                r["repetition"] = folder.name
    return fr, cr


def combine_result_folders(folders: list[str | Path], design: ExperimentDesign | None = None,
                           repetitions: list[str] | None = None) -> tuple[ExperimentSummary, list[dict], list[dict]]:
    """Summarise several result folders (e.g. one per repetition) as one experiment."""
    from .pairing import short_labels

    folders = [Path(f) for f in folders]
    labels = repetitions or [None] * len(folders)
    all_f, all_c = [], []
    for f, lab in zip(folders, labels):
        fr, cr = load_result_folder(f, lab)
        # a folder that holds a single repetition "1" (analysed alone) is its own repetition
        reps = {r["repetition"] for r in fr}
        if lab is None and reps <= {"1", ""} and len(folders) > 1:
            for r in fr + cr:
                r["repetition"] = f.name
        all_f += fr
        all_c += cr
    if repetitions is None and len(folders) > 1:
        short = short_labels([r["repetition"] for r in all_f])
        for r in all_f + all_c:
            r["repetition"] = short.get(r["repetition"], r["repetition"])
    return summarise_experiment(all_f, all_c, design), all_f, all_c
