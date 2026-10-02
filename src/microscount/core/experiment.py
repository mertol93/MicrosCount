"""Experiments shared by the modules: groups, repetitions, a reference and comparisons.

Every analysed unit (a field of cells, an SEM image) gives one value per measure. Within
each *repetition* (an independent experiment, sample or batch) the units are grouped
(conditions, samples):

* group value = mean ± SD of its unit values (units of one repetition are technical
  replicates);
* change against a reference group in the same repetition: difference, ratio and
  percent change;
* comparisons A vs B within a repetition: difference and Welch's t-test on the unit
  values (exploratory);
* across repetitions: mean ± SD of the repetition values and, for each comparison, the
  difference in every repetition, whether its direction agrees, and a paired t-test on
  the repetition values when there are at least three repetitions.

``summarise_groups`` does this for any number of measures and returns long tables (one
row per measure); the modules turn them into their own sheets. Everything works on plain
rows (dicts), so results saved by earlier runs can be combined.
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .naming import group_key

NAN = math.nan


# ---------------------------------------------------------------------- small helpers


def to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def mean_sd(v) -> tuple[float, float, float]:
    """Mean, SD (n - 1) and SEM of the finite values."""
    v = np.asarray([x for x in v if math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return NAN, NAN, NAN
    sd = float(v.std(ddof=1)) if v.size > 1 else NAN
    return float(v.mean()), sd, (sd / math.sqrt(v.size) if v.size > 1 else NAN)


def welch(a, b) -> tuple[float, float]:
    """Welch's t-test (t, p); NaN when a group has fewer than two values or no spread."""
    from scipy import stats

    a = [x for x in a if math.isfinite(x)]
    b = [x for x in b if math.isfinite(x)]
    if len(a) < 2 or len(b) < 2 or (np.ptp(a) == 0 and np.ptp(b) == 0):
        return NAN, NAN
    with warnings.catch_warnings():  # identical values in one group: scipy warns, the test is still defined
        warnings.simplefilter("ignore", RuntimeWarning)
        r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def paired(a, b) -> tuple[float, float]:
    """Paired t-test of b against a (t, p); needs at least three pairs."""
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


def describe(v, prefix: str) -> dict:
    """Median, quartiles, mean and SD of the finite values, keys suffixed with ``prefix``."""
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


def signed_pct(v: float) -> str:
    return "–" if not math.isfinite(v) else f"{v:+.1f}%"


def order_groups(names: list[str], preferred: list[str] | None = None) -> list[str]:
    """Groups in the preferred order first (spelling-insensitive), then as met."""
    seen: dict[str, str] = {}
    for n in names:
        seen.setdefault(group_key(n), n)
    out = []
    for p in preferred or []:
        k = group_key(p)
        if k in seen and seen[k] not in out:
            out.append(seen[k])
    out += [n for n in seen.values() if n not in out]
    return out


def order_repetitions(reps: list[str]) -> list[str]:
    uniq = list(dict.fromkeys(reps))
    if all(r.strip().lstrip("-").replace(".", "", 1).isdigit() for r in uniq):
        return sorted(uniq, key=float)
    return uniq


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def combine_repetitions(folders: list, rows: list[list[dict]], extra: list[list[dict]], group: str,
                        repetitions: list[str] | None = None, id_key: str = "") -> list[str]:
    """Repetition labels when result folders are put together as one experiment; returns notes.

    ``rows[i]`` and ``extra[i]`` are the unit rows (images, fields) and the object rows (pores,
    cells) of ``folders[i]``; their ``repetition`` (and the repetition part of their ``id_key``)
    is changed in place. With ``repetitions`` every folder is one repetition. Otherwise each
    folder keeps its labels, except that a label found in several folders with the same groups
    names different repetitions there: a folder analysed alone becomes a repetition named after
    the folder, and a folder that holds several repetitions prefixes its labels with the folder's
    name. A label found in several folders with different groups (samples analysed separately)
    stays one repetition.
    """
    from .naming import path_labels, short_labels

    n = len(folders)
    for i in range(n):
        for r in rows[i] + extra[i]:
            r["repetition"] = str(r.get("repetition") or "").strip() or "1"
    labs = [{r["repetition"] for r in rows[i] + extra[i]} for i in range(n)]
    final: list[dict[str, str]] = [{lab: lab for lab in labs[i]} for i in range(n)]
    notes = []
    if repetitions:
        final = [{lab: str(repetitions[i]) for lab in labs[i]} for i in range(n)]
    elif n > 1:
        paths = [str(Path(f).resolve()) for f in folders]
        flab = path_labels(paths)
        clashes = []
        for lab in order_repetitions([x for s in labs for x in sorted(s)]):
            holders = [i for i in range(n) if lab in labs[i]]
            if len(holders) < 2:
                continue
            groups = [{group_key(str(r.get(group) or "")) for r in rows[i] if r["repetition"] == lab} for i in holders]
            if all(not (groups[a] & groups[b]) for a in range(len(groups)) for b in range(a + 1, len(groups))):
                continue  # one repetition whose samples were analysed separately
            for i in holders:
                final[i][lab] = flab[paths[i]] if labs[i] == {lab} else f"{flab[paths[i]]}/{lab}"
            if any(labs[i] != {lab} for i in holders):
                clashes.append((lab, holders))
        renamed = [{lab for lab in labs[i] if final[i][lab] != lab} for i in range(n)]
        kept = {lab for i in range(n) for lab in labs[i] if lab not in renamed[i]}
        for i in range(n):
            for lab in renamed[i]:
                if final[i][lab] in kept:  # would join a repetition of another folder
                    final[i][lab] = f"{flab[paths[i]]}/{lab}"
        short = short_labels([v for f in final for v in f.values()])
        final = [{lab: short.get(v, v) for lab, v in f.items()} for f in final]
        notes = [f"repetition {lab} is in {len(h)} result folders with the same samples: kept apart as "
                 + ", ".join(final[i][lab] for i in h) for lab, h in clashes]
    several = len({v for f in final for v in f.values()}) > 1
    for i in range(n):
        for r in rows[i] + extra[i]:
            old = r["repetition"]
            new = r["repetition"] = final[i].get(old, old)
            if id_key and r.get(id_key):  # ids carry the repetition when there are several
                rid = str(r[id_key])
                rest = rid[len(old) + 3:] if rid.startswith(f"{old} · ") else rid
                r[id_key] = f"{new} · {rest}" if several else rest
    return notes


# ---------------------------------------------------------------------- design and summary


@dataclass
class Measure:
    key: str  # column holding one value per unit
    label: str
    unit: str = ""


@dataclass
class GroupDesign:
    reference: str = ""  # reference group: default reference for changes and comparisons
    references: dict = field(default_factory=dict)  # group -> its own reference
    comparisons: list = field(default_factory=list)  # [[A, B], ...]; empty = every group vs the reference
    order: list = field(default_factory=list)  # group order in tables and figures

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GroupDesign":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class GroupSummary:
    per_repetition: list[dict]  # measure x repetition x group
    comparisons: list[dict]  # measure x comparison x repetition, then an "all" row each
    overall: list[dict]  # measure x group, across repetitions
    order: list[str]
    repetitions: list[str]
    notes: list[str]
    group: str
    unit: str
    measures: list[Measure]


def summarise_groups(rows: list[dict], measures: list[Measure], design: GroupDesign | None = None,
                     group: str = "sample", unit: str = "image") -> GroupSummary:
    """Group values per repetition, changes against references and comparisons, for every measure."""
    design = design or GroupDesign()
    notes: list[str] = []
    for r in rows:
        if not str(r.get("repetition", "") or "").strip():
            r["repetition"] = "1"
        r["repetition"] = str(r["repetition"])
    names = [str(r.get(group) or "(none)") for r in rows]
    order = order_groups(names, design.order or ([design.reference] if design.reference else []))
    canon = {group_key(g): g for g in order}

    def canon_of(name) -> str:
        return canon.get(group_key(str(name or "(none)")), str(name or "(none)"))

    reps = order_repetitions([r["repetition"] for r in rows])
    reference = canon.get(group_key(design.reference), "") if design.reference else ""
    if design.reference and not reference:
        notes.append(f"reference '{design.reference}' not found among the {group}s; no changes against it")
    refs = {}
    for g, ref in (design.references or {}).items():
        cg, cr = canon.get(group_key(g)), canon.get(group_key(ref))
        if cg and cr:
            refs[cg] = cr
        elif ref:
            notes.append(f"reference for '{g}': '{ref}' not found")
    G: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        G.setdefault((r["repetition"], canon_of(r.get(group))), []).append(r)
    n_key = f"n_{unit}s"

    def values(rep, g, m) -> list[float]:
        return [x for x in (to_float(r.get(m.key)) for r in G.get((rep, g), [])) if math.isfinite(x)]

    # ---- per repetition
    per_rep: list[dict] = []
    means: dict[tuple[str, str, str], float] = {}
    for m in measures:
        for rep in reps:
            for g in order:
                if (rep, g) not in G:
                    continue
                v = values(rep, g, m)
                mu, sd, sem = mean_sd(v)
                means[(m.key, rep, g)] = mu
                per_rep.append({"measure": m.label, "unit": m.unit, "key": m.key, "repetition": rep, group: g,
                                n_key: len(v), "mean": mu, "sd": sd, "sem": sem})
    for row in per_rep:
        g = row[group]
        ref = refs.get(g) or reference
        row["reference"] = ref
        mu, rv = row["mean"], means.get((row["key"], row["repetition"], ref), NAN) if ref else NAN
        ok = math.isfinite(mu) and math.isfinite(rv)
        row["difference"] = mu - rv if ok else NAN
        row["ratio"] = mu / rv if ok and rv != 0 else NAN
        row["change_pct"] = 100 * (mu / rv - 1) if ok and rv != 0 else NAN

    # ---- comparisons
    pairs = [tuple(c[:2]) for c in design.comparisons if len(c) >= 2]
    if not pairs and reference:
        pairs = [(reference, g) for g in order if g != reference]
    resolved = []
    for a_name, b_name in pairs:
        a, b = canon.get(group_key(a_name)), canon.get(group_key(b_name))
        if not a or not b:
            notes.append(f"comparison {a_name} vs {b_name}: {group} not found")
            continue
        resolved.append((a, b))
    comps: list[dict] = []
    for m in measures:
        for a, b in resolved:
            per = []
            for rep in reps:
                if (rep, a) not in G or (rep, b) not in G:
                    continue
                va, vb = values(rep, a, m), values(rep, b, m)
                ma, mb = mean_sd(va)[0], mean_sd(vb)[0]
                t, p = welch(va, vb)
                d = mb - ma if math.isfinite(ma) and math.isfinite(mb) else NAN
                pct = 100 * (mb / ma - 1) if math.isfinite(d) and ma != 0 else NAN
                comps.append({"measure": m.label, "unit": m.unit, "key": m.key, "repetition": rep,
                              f"{group}_a": a, f"{group}_b": b, "n_a": len(va), "n_b": len(vb), "mean_a": ma,
                              "mean_b": mb, "difference": d, "difference_pct": pct, "welch_t": t, "welch_p": p,
                              "test": f"Welch t-test on {unit} values (exploratory: {unit}s of one repetition are "
                                      "not independent)"})
                per.append((rep, ma, mb, d, pct))
            if len(reps) > 1 and per:
                ma = [x[1] for x in per]
                mb = [x[2] for x in per]
                ds = [x[3] for x in per if math.isfinite(x[3])]
                pcts = [x[4] for x in per if math.isfinite(x[4])]
                t, p = paired(ma, mb)
                same = len({math.copysign(1, x) for x in ds if x != 0}) <= 1 if ds else False
                comps.append({
                    "measure": m.label, "unit": m.unit, "key": m.key, "repetition": "all", f"{group}_a": a,
                    f"{group}_b": b, "n_repetitions": len(per), "mean_a": mean_sd(ma)[0], "mean_b": mean_sd(mb)[0],
                    "difference": float(np.mean(ds)) if ds else NAN,
                    "difference_pct": float(np.mean(pcts)) if pcts else NAN,
                    "per_repetition": "; ".join(f"{r}: {signed_pct(x)}" for r, _, _, _, x in per),
                    "same_direction": same, "paired_t": t, "paired_p": p,
                    "test": "paired t-test on the repetition values" if math.isfinite(p) else
                            "at least 3 repetitions are needed for a test across repetitions",
                })

    # ---- across repetitions
    overall: list[dict] = []
    for m in measures:
        for g in order:
            rr = [r for r in per_rep if r["key"] == m.key and r[group] == g]
            if not rr:
                continue
            vals = [r["mean"] for r in rr]
            mu, sd, sem = mean_sd(vals)
            o = {"measure": m.label, "unit": m.unit, "key": m.key, group: g,
                 "n_repetitions": sum(1 for v in vals if math.isfinite(v)), n_key: sum(r[n_key] for r in rr),
                 "mean": mu, "sd": sd, "sem": sem}
            for r in rr:
                o[f"repetition {r['repetition']}"] = r["mean"]
            o["reference"] = rr[0]["reference"]
            o["change_pct"] = mean_sd([r["change_pct"] for r in rr])[0]
            o["difference"] = mean_sd([r["difference"] for r in rr])[0]
            overall.append(o)
    if 1 < len(reps) < 3:
        notes.append(f"{len(reps)} repetitions: differences are reported per repetition; a test across repetitions "
                     "needs at least 3")
    return GroupSummary(per_rep, comps, overall, order, reps, notes, group, unit, list(measures))
