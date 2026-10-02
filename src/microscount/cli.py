"""Command line.

``microscount``                              open the graphical interface
``microscount bio translocation ...``        Bio & Cells: nuclear translocation (N/C ratio)
``microscount bio combine RESULTS ...``      one experiment from result folders analysed separately
``microscount materials porosity ...``       Materials & Mechanics: SEM porosity
``microscount materials combine RESULTS ...`` one SEM experiment from result folders analysed separately
``microscount run settings.yaml``            repeat a saved analysis
``microscount modules`` / ``selftest``

``microscount translocation`` and ``microscount porosity`` remain as shortcuts.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from . import APP_NAME, __version__
from .modules import MODULES


def _expand(inputs: list[str], recursive: bool = True) -> list[Path]:
    from .core.imageio import is_supported, list_images

    out: list[Path] = []
    for s in inputs:
        p = Path(s).absolute()  # settings.yaml records where the images are, wherever it is run from
        if p.is_dir():
            out += list_images(p, recursive=recursive)
        elif p.is_file() and is_supported(p):
            out.append(p)
        else:
            print(f"skipping {s}: not a TIFF/PNG/JPEG file or folder", file=sys.stderr)
    return out


def _progress(i: int, n: int, msg: str) -> None:
    print(f"[{i}/{n}] {msg}", flush=True)


def _range(text: str, what: str) -> tuple[float, float | None]:
    """"0-Infinity", "50-inf", "0.2-1.0" -> (low, high); high None = Infinity."""
    lo, sep, hi = text.replace("–", "-").partition("-")
    try:
        low = float(lo)
        high = None if hi.strip().lower() in ("", "inf", "infinity", "∞") else float(hi)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what}: use MIN-MAX, e.g. 0-Infinity") from None
    if not sep:
        raise argparse.ArgumentTypeError(f"{what}: use MIN-MAX, e.g. 0-Infinity")
    return low, high


def _fmt(v) -> str:
    if isinstance(v, float):
        return "–" if math.isnan(v) else f"{v:.4g}"
    return str(v)


# ---------------------------------------------------------------------- Bio & Cells


def _translocation_args(t: argparse.ArgumentParser) -> None:
    t.add_argument("inputs", nargs="*", help="image files or folders (TIFF/PNG/JPEG)")
    t.add_argument("--method", choices=["per_cell", "paper"], default=None,
                   help="per_cell (default) = the paper ratio per field plus the per-cell lab protocol; "
                        "paper = Noursadeghi et al. (2008) only")
    t.add_argument("--config", help="settings.yaml saved by MicrosCount (its inputs are used unless you give images)")
    t.add_argument("--condition", help="condition label for all inputs")
    t.add_argument("--pixel-size", type=float, help="micrometres per pixel")
    g = t.add_argument_group("per-cell lab protocol")
    g.add_argument("--size", type=lambda s: _range(s, "--size"), metavar="MIN-MAX",
                   help="particle size filter for nuclei in px² (default 0-Infinity)")
    g.add_argument("--circularity", type=lambda s: _range(s, "--circularity"), metavar="MIN-MAX",
                   help="particle circularity filter (default 0.2-1.0)")
    g.add_argument("--rolling-ball", type=float, metavar="PX",
                   help="Subtract Background radius for the target channel in px (default 50; 0 = off)")
    g.add_argument("--background", metavar="auto|none|VALUE",
                   help="Background_Mean: auto = mean of the cell-free area (default), none, or a number")
    g.add_argument("--ring-width", type=float, metavar="PX", help="width of the cytoplasmic ring (default 0.3 x "
                   "nucleus diameter)")
    g.add_argument("--keep-edge-cells", action="store_true", help="measure nuclei touching the image edge too")
    t.add_argument("--out", help="output folder")
    t.add_argument("--no-subfolders", action="store_true", help="only the images directly inside the given folders")
    e = t.add_argument_group("experiment")
    e.add_argument("--conditions-from", choices=["auto", "name", "folder"],
                   help="conditions from file names or folder names (auto: file names when a folder holds several "
                        "conditions)")
    e.add_argument("--repetition", help="repetition label for all inputs (default: one per folder)")
    _design_args(e)


def _design_args(e) -> None:
    e.add_argument("--control", metavar="CONDITION", help="control condition (responders, fold change, comparisons)")
    e.add_argument("--compare", nargs=2, action="append", metavar=("A", "B"),
                   help="compare condition B with condition A (repeat for more; default: every condition vs the control)")
    e.add_argument("--reference", nargs=2, action="append", metavar=("CONDITION", "REFERENCE"),
                   help="fold change of CONDITION relative to REFERENCE instead of the control (repeatable)")
    e.add_argument("--order", nargs="+", metavar="CONDITION", help="order of the conditions in tables and figures")
    e.add_argument("--responder-percentile", type=float, metavar="P",
                   help="responders: cells above this percentile of the control's cells (default 95; 0 = N/C > 1)")
    e.add_argument("--keep-failed-paper-fields", action="store_true",
                   help="keep fields whose automatic threshold failed in the paper-method means")


def _apply_design(s, a) -> None:
    """Experiment options of the command line into settings (or an ExperimentDesign)."""
    control_attr = "control_condition" if hasattr(s, "control_condition") else "control"
    if a.control is not None:
        setattr(s, control_attr, a.control)
    if a.compare:
        s.comparisons = [list(c) for c in a.compare]
    if a.reference:
        attr = "fold_references" if hasattr(s, "fold_references") else "references"
        setattr(s, attr, {c: r for c, r in a.reference})
    if a.order:
        setattr(s, "condition_order" if hasattr(s, "condition_order") else "order", list(a.order))
    if a.responder_percentile is not None:
        s.responder_percentile = a.responder_percentile or None
    if a.keep_failed_paper_fields:
        s.exclude_failed_paper_fields = False


def cmd_translocation(a) -> int:
    from .bio.pairing import label_fields, pair_files, scan_files
    from .bio.report import default_output_dir, run_translocation
    from .bio.translocation import FieldSpec, TranslocationSettings
    from .modules import load_settings

    s = TranslocationSettings()
    specs: list[FieldSpec] = []
    if a.config:
        an, s, inputs = load_settings(a.config)
        if an.key != "translocation":
            print(f"{a.config} holds settings for {an.title}, not nuclear translocation", file=sys.stderr)
            return 2
        specs = [FieldSpec(i["nuclear"], i["target"], i.get("condition", ""), i.get("field", ""),
                           str(i.get("repetition") or "")) for i in inputs]
    if a.method:
        s.method = a.method
    if a.conditions_from:
        s.conditions_from = a.conditions_from
    _apply_design(s, a)
    if a.inputs:
        files = _expand(a.inputs, recursive=not a.no_subfolders)
        pairs, unpaired = pair_files(scan_files(files), conditions_from=s.conditions_from)
        noted: dict[str, int] = {}
        for u in unpaired:
            if u.note and not u.error:
                noted[u.note] = noted.get(u.note, 0) + 1
            else:
                print(f"unpaired: {u.path} {u.error}", file=sys.stderr)
        for note, k in noted.items():
            print(f"{k} file(s) not used: {note}", file=sys.stderr)
        if a.condition or a.repetition:
            for p in pairs:
                p.condition = a.condition or p.condition
                p.repetition = a.repetition or p.repetition
            label_fields(pairs)
        specs = [FieldSpec(p.nuclear, p.target, p.condition, p.field_id, p.repetition) for p in pairs]
        for p in pairs:
            print(f"{p.field_id}: {Path(p.nuclear).name} + {Path(p.target).name} [{p.confidence}] {p.note}")
    elif a.condition or a.repetition:
        from .core.naming import label_items, number_from_id

        for sp in specs:
            sp.condition = a.condition or sp.condition
            sp.repetition = a.repetition or sp.repetition
            sp.field_no = number_from_id(sp.field_id)
        label_items(specs, lambda sp: sp.nuclear_path, "condition", "field_id", "(no condition)")
    if not specs:
        print("no fields to analyse", file=sys.stderr)
        return 2
    if a.pixel_size:
        s.pixel_size_um = a.pixel_size
    if a.size:
        s.size_min_px2, s.size_max_px2 = a.size
    if a.circularity:
        s.circularity_min, hi = a.circularity
        s.circularity_max = 1.0 if hi is None else hi
    if a.rolling_ball is not None:
        s.rolling_ball_radius = a.rolling_ball
    if a.background:
        b = a.background.strip().lower()
        if b in ("auto", "none"):
            s.background = b
        else:
            try:
                s.background, s.background_value = "manual", float(b)
            except ValueError:
                print("--background: auto, none or a number", file=sys.stderr)
                return 2
    if a.ring_width:
        s.ring_width_px = a.ring_width
    if a.keep_edge_cells:
        s.exclude_border_cells = False
    out = Path(a.out) if a.out else default_output_dir([specs[0].nuclear_path])
    res = run_translocation(specs, s, out, progress=_progress)
    _print_experiment(res["experiment"])
    for r in res["results"]:
        for w in r.warnings:
            print(f"{r.field_id}: {w}", file=sys.stderr)
    for e in res["errors"]:
        print("error:", e, file=sys.stderr)
    print(f"results written to {out}")
    return 0 if res["results"] else 1


def _print_experiment(exp) -> None:
    several = len(exp.repetitions) > 1
    for row in exp.conditions:
        head = f"{'repetition ' + row['repetition'] + ' · ' if several else ''}{row['condition']}: {row['n_fields']} field(s)"
        if "n_cells" in row:
            line = (f"{head}; {row['n_cells']} cells; N/C {_fmt(row['field_median_nc_mean'])} ± "
                    f"{_fmt(row['field_median_nc_sd'])} (median per field, mean ± SD)")
            if math.isfinite(row.get("fold_change", math.nan)):
                line += f"; fold {_fmt(row['fold_change'])} vs {row['fold_reference']}"
        else:
            line = f"{head}; paper N/C {_fmt(row['paper_ratio_mean'])} ± {_fmt(row['paper_ratio_sd'])}"
        print(line)
    for c in exp.comparisons:
        if c["repetition"] == "all":
            print(f"{c['condition_b']} vs {c['condition_a']}, all repetitions: {_fmt(c['difference_pct'])}% "
                  f"({c['per_repetition']}); paired p {_fmt(c['paired_p'])}")
        else:
            print(f"{c['condition_b']} vs {c['condition_a']}{', repetition ' + c['repetition'] if several else ''}: "
                  f"{_fmt(c['difference_pct'])}%, Welch p {_fmt(c['welch_p'])}")
    for note in exp.notes:
        print("note:", note)


def _combine_args(c: argparse.ArgumentParser) -> None:
    c.add_argument("folders", nargs="+", help="result folders written by MicrosCount (with per_field.csv)")
    c.add_argument("--repetitions", nargs="+", metavar="LABEL", help="one repetition label per folder")
    c.add_argument("--config", help="settings.yaml or combine.yaml to take the experiment design from")
    _design_args(c)
    c.add_argument("--out", help="output folder")


def cmd_combine(a) -> int:
    from .bio.experiment import ExperimentDesign
    from .bio.report import combine_results

    design = ExperimentDesign()
    if a.config:
        from .core.report import read_settings_yaml

        doc = read_settings_yaml(a.config)
        if "design" in doc:
            design = ExperimentDesign.from_dict(doc["design"])
        elif "settings" in doc:
            from .bio.translocation import TranslocationSettings

            design = ExperimentDesign.from_settings(TranslocationSettings.from_dict(doc["settings"]))
    _apply_design(design, a)
    missing = [f for f in a.folders if not (Path(f) / "per_field.csv").exists()]
    if missing:
        print("not a MicrosCount result folder (no per_field.csv): " + ", ".join(missing), file=sys.stderr)
        return 2
    if a.repetitions and len(a.repetitions) != len(a.folders):
        print("--repetitions: give one label per folder", file=sys.stderr)
        return 2
    from .core.report import default_combined_dir

    out = Path(a.out) if a.out else default_combined_dir(a.folders)
    res = combine_results(a.folders, design, out, a.repetitions)
    _print_experiment(res["experiment"])
    print(f"results written to {out}")
    return 0


# ---------------------------------------------------------------------- Materials & Mechanics


def _porosity_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("inputs", nargs="*", help="SEM image files or folders")
    s.add_argument("--config", help="settings.yaml saved by MicrosCount (its inputs are used unless you give images)")
    s.add_argument("--pixel-size", type=float, help="micrometres per pixel for images whose file does not record it "
                   "(MATLAB 'Resolution')")
    s.add_argument("--ignore-file-pixel-size", action="store_true",
                   help="use --pixel-size for every image, even when the file records its own")
    s.add_argument("--n-thresholds", type=int, help="MATLAB 'N' (default 4)")
    s.add_argument("--threshold", type=float, metavar="GREY", help="fixed pore threshold instead of multithresh")
    s.add_argument("--data-bar", metavar="auto|none|ROWS",
                   help="SEM data bar at the bottom: auto = from the file or detected (default), none, or rows to remove")
    s.add_argument("--crop-bottom", type=int, help="rows to remove at the bottom (same as --data-bar ROWS)")
    s.add_argument("--edge-pores-out", action="store_true", help="leave pores cut by the image edge out of the sizes")
    s.add_argument("--bright-pores", action="store_true", help="pores are brighter than the solid")
    s.add_argument("--out")
    s.add_argument("--no-subfolders", action="store_true", help="only the images directly inside the given folders")
    e = s.add_argument_group("experiment")
    e.add_argument("--samples-from", choices=["auto", "name", "folder"],
                   help="samples from file names or folder names (auto: file names when a folder holds several samples)")
    e.add_argument("--sample", "--condition", dest="sample", help="sample label for all inputs")
    e.add_argument("--repetition", help="repetition label for all inputs (default: from the folders)")
    _sample_design_args(e)


def _sample_design_args(e) -> None:
    e.add_argument("--reference", metavar="SAMPLE", help="reference sample (percent changes and default comparisons)")
    e.add_argument("--compare", nargs=2, action="append", metavar=("A", "B"),
                   help="compare sample B with sample A (repeatable; default: every sample vs the reference)")
    e.add_argument("--relative-to", nargs=2, action="append", metavar=("SAMPLE", "REFERENCE"),
                   help="changes of SAMPLE relative to REFERENCE instead of the reference sample (repeatable)")
    e.add_argument("--order", nargs="+", metavar="SAMPLE", help="order of the samples in tables and figures")


def _apply_sample_design(s, a) -> None:
    """Experiment options of the command line into PorositySettings or a GroupDesign."""
    ref_attr = "reference_sample" if hasattr(s, "reference_sample") else "reference"
    if a.reference is not None:
        setattr(s, ref_attr, a.reference)
    if a.compare:
        s.comparisons = [list(c) for c in a.compare]
    if a.relative_to:
        s.references = {g: r for g, r in a.relative_to}
    if a.order:
        setattr(s, "sample_order" if hasattr(s, "sample_order") else "order", list(a.order))


def cmd_porosity(a) -> int:
    from .materials.experiment import ImageSpec, label_images, scan_sem_files
    from .materials.porosity import PorositySettings
    from .materials.report import default_output_dir, run_porosity
    from .modules import load_settings

    s = PorositySettings()
    specs: list = []
    if a.config:
        an, s, inputs = load_settings(a.config)
        if an.key != "porosity":
            print(f"{a.config} holds settings for {an.title}, not SEM porosity", file=sys.stderr)
            return 2
        specs = [ImageSpec(i["image"], sample=i.get("sample") or i.get("condition", "") or "",
                           repetition=str(i.get("repetition") or "1"), image_id=i.get("image_id") or "",
                           pixel_size_um=i.get("pixel_size_um")) for i in inputs]
    if a.samples_from:
        s.samples_from = a.samples_from
    if a.pixel_size:
        s.pixel_size_um = a.pixel_size
    if a.ignore_file_pixel_size:
        s.use_metadata_pixel_size = False
    if a.n_thresholds:
        s.n_thresholds = a.n_thresholds
    if a.threshold is not None:
        s.pore_threshold_value = a.threshold
    if a.crop_bottom is not None:
        s.data_bar, s.crop_bottom_px = ("manual", a.crop_bottom) if a.crop_bottom > 0 else ("none", 0)
    if a.data_bar:
        v = a.data_bar.strip().lower()
        if v in ("auto", "none"):
            s.data_bar = v
        elif v.isdigit():
            s.data_bar, s.crop_bottom_px = "manual", int(v)
        else:
            print("--data-bar: auto, none or a number of rows", file=sys.stderr)
            return 2
    if a.edge_pores_out:
        s.exclude_edge_pores = True
    if a.bright_pores:
        s.pores_are_bright = True
    _apply_sample_design(s, a)
    if a.inputs:
        specs = scan_sem_files(_expand(a.inputs, recursive=not a.no_subfolders), samples_from=s.samples_from)
        if a.sample or a.repetition:
            for sp in specs:
                sp.sample = a.sample or sp.sample
                sp.repetition = a.repetition or sp.repetition
            label_images(specs)
        for sp in specs:
            px = f"{sp.file_pixel_size_um:.4g} µm/px ({sp.file_pixel_size_source})" if sp.file_pixel_size_um else ""
            print(f"{sp.image_id}: {Path(sp.path).name} {px}".rstrip())
    elif a.sample or a.repetition:
        from .core.naming import number_from_id

        for sp in specs:
            sp.sample = a.sample or sp.sample
            sp.repetition = a.repetition or sp.repetition
            sp.field_no = number_from_id(sp.image_id)
        label_images(specs)
    if not specs:
        print("no images to analyse", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else default_output_dir([specs[0].path])
    res = run_porosity(specs, s, out, progress=_progress)
    _print_samples(res["experiment"])
    for r in res["results"]:
        for w in r.warnings:
            print(f"{r.image_id}: {w}", file=sys.stderr)
    for e in res["errors"]:
        print("error:", e, file=sys.stderr)
    print(f"results written to {out}")
    return 0 if res["results"] else 1


def _print_samples(ms) -> None:
    several = len(ms.repetitions) > 1
    m_por, m_d = ms.measures[0], ms.measures[1]
    for row in ms.summary:
        line = (f"{row['sample']}: {row['n_images']} image(s)"
                + (f" in {row['n_repetitions']} repetitions" if several else "")
                + f"; porosity {_fmt(row.get(m_por.key + '_mean'))} ± {_fmt(row.get(m_por.key + '_sd'))} %"
                + f"; mean pore diameter {_fmt(row.get(m_d.key + '_mean'))} ± {_fmt(row.get(m_d.key + '_sd'))} {m_d.unit}")
        ch = row.get(m_por.key + "_change_pct")
        if row.get("reference") and row["reference"] != row["sample"] and isinstance(ch, float) and math.isfinite(ch):
            line += f"; porosity {ch:+.1f}% vs {row['reference']}"
        print(line)
    for c in ms.comparisons:
        if c["key"] not in (m_por.key, m_d.key):
            continue
        change = c["difference_pct"]
        change = f"{change:+.4g}%" if isinstance(change, float) and math.isfinite(change) else "–"
        if c["repetition"] == "all":
            print(f"{c['measure']}: {c['sample_b']} vs {c['sample_a']}, all repetitions: change {change} "
                  f"({c['per_repetition']}); paired p {_fmt(c['paired_p'])}")
        else:
            print(f"{c['measure']}: {c['sample_b']} vs {c['sample_a']}{', repetition ' + c['repetition'] if several else ''}"
                  f": change {change}, Welch p {_fmt(c['welch_p'])}")
    for note in ms.notes:
        print("note:", note)


def _materials_combine_args(c: argparse.ArgumentParser) -> None:
    c.add_argument("folders", nargs="+", help="SEM result folders written by MicrosCount (with per_image.csv)")
    c.add_argument("--repetitions", nargs="+", metavar="LABEL", help="one repetition label per folder")
    c.add_argument("--config", help="settings.yaml or combine.yaml to take the experiment design from")
    _sample_design_args(c)
    c.add_argument("--bins", type=int, default=None, help="pore-size distribution bins (default 25)")
    c.add_argument("--out", help="output folder")


def cmd_materials_combine(a) -> int:
    from .core.experiment import GroupDesign
    from .materials.experiment import design_from_settings
    from .materials.report import combine_results

    design, bins = GroupDesign(), 25
    if a.config:
        from .core.report import read_settings_yaml

        doc = read_settings_yaml(a.config)
        if "design" in doc:
            design = GroupDesign.from_dict(doc["design"])
            bins = int(doc.get("histogram_bins") or bins)
        elif "settings" in doc:
            from .materials.porosity import PorositySettings

            ps = PorositySettings.from_dict(doc["settings"])
            design, bins = design_from_settings(ps), ps.histogram_bins
    _apply_sample_design(design, a)
    missing = [f for f in a.folders if not (Path(f) / "per_image.csv").exists()]
    if missing:
        print("not a MicrosCount SEM result folder (no per_image.csv): " + ", ".join(missing), file=sys.stderr)
        return 2
    if a.repetitions and len(a.repetitions) != len(a.folders):
        print("--repetitions: give one label per folder", file=sys.stderr)
        return 2
    from .core.report import default_combined_dir

    out = Path(a.out) if a.out else default_combined_dir(a.folders)
    res = combine_results(a.folders, design, out, a.repetitions, a.bins or bins)
    _print_samples(res["experiment"])
    print(f"results written to {out}")
    return 0


COMMANDS = {"translocation": (_translocation_args, cmd_translocation),
            "porosity": (_porosity_args, cmd_porosity)}
# module commands that are not analyses of images
EXTRA = {"bio": {"combine": ("one experiment from result folders analysed separately (e.g. one per repetition)",
                             _combine_args, cmd_combine)},
         "materials": {"combine": ("one SEM experiment from result folders analysed separately (e.g. one per repetition)",
                                   _materials_combine_args, cmd_materials_combine)}}


def cmd_run(a) -> int:
    from .modules import load_settings

    try:
        an, _, _ = load_settings(a.settings)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    from .core.report import read_settings_yaml

    doc = read_settings_yaml(a.settings)
    out = ["--out", a.out] if a.out else []
    if doc.get("combined_from"):  # a combine.yaml: combine the same result folders again
        _help, add_args, run = EXTRA[an.module]["combine"]
        reps = ["--repetitions", *[str(r) for r in doc["repetitions"]]] if doc.get("repetitions") else []
        args = [*[str(f) for f in doc["combined_from"]], "--config", a.settings, *reps, *out]
    else:
        add_args, run = COMMANDS[an.key]
        args = ["--config", a.settings, *out]
    p = argparse.ArgumentParser(add_help=False)
    add_args(p)
    return run(p.parse_args(args))


def cmd_modules(_a) -> int:
    for m in MODULES:
        print(f"{m.title}  ({m.key}): {m.summary}")
        for an in m.analyses:
            print(f"    microscount {m.key} {an.key:<14} {an.title}: {an.summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="microscount", description=f"{APP_NAME} {__version__}: microscopy image analysis. "
                                "Run without arguments to open the graphical interface.")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.add_parser("gui", help="open the graphical interface (default)")
    for m in MODULES:
        mp = sub.add_parser(m.key, help=f"{m.title}: {m.summary.lower()}")
        msub = mp.add_subparsers(dest="analysis", metavar="ANALYSIS", required=True)
        for an in m.analyses:
            ap = msub.add_parser(an.key, help=an.summary, description=f"{m.title} ▸ {an.title}: {an.summary}")
            COMMANDS[an.key][0](ap)
        for key, (help_, add_args, _run) in EXTRA.get(m.key, {}).items():
            ap = msub.add_parser(key, help=help_, description=f"{m.title} ▸ {help_}")
            add_args(ap)
    for an_key, (add_args, _run) in COMMANDS.items():  # shortcuts kept from version 0.1
        ap = sub.add_parser(an_key, help=f"shortcut for '{_module_of(an_key)} {an_key}'")
        add_args(ap)
    r = sub.add_parser("run", help="repeat an analysis from a saved settings.yaml")
    r.add_argument("settings")
    r.add_argument("--out", help="output folder")
    sub.add_parser("modules", help="list the modules and their analyses")
    st = sub.add_parser("selftest", help="run the built-in self-test")
    st.add_argument("--report", help="also write the report to this file")
    return p


def _module_of(analysis_key: str) -> str:
    from .modules import analysis

    return analysis(analysis_key).module


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # macOS passes -psn_* when launched from Finder on old systems
    argv = [x for x in argv if not x.startswith("-psn_")]
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.command in (None, "gui"):
        from .gui.app import run_gui

        return run_gui()
    if a.command == "selftest":
        from .selftest import run

        return 0 if run(a.report) else 1
    if a.command == "modules":
        return cmd_modules(a)
    if a.command == "run":
        return cmd_run(a)
    key = getattr(a, "analysis", None) or a.command
    if key in EXTRA.get(a.command, {}):
        return EXTRA[a.command][key][2](a)
    if key in COMMANDS:
        return COMMANDS[key][1](a)
    parser.print_help()
    return 2
