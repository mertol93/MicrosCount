"""Command line: ``microscount`` (GUI), ``microscount translocation|porosity|selftest``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import APP_NAME, __version__


def _expand(inputs: list[str]) -> list[Path]:
    from .imageio import is_supported, list_images

    out: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            out += list_images(p)
        elif p.is_file() and is_supported(p):
            out.append(p)
        else:
            print(f"skipping {s}: not a TIFF/PNG/JPEG file or folder", file=sys.stderr)
    return out


def _progress(i: int, n: int, msg: str) -> None:
    print(f"[{i}/{n}] {msg}", flush=True)


def cmd_translocation(a) -> int:
    from .pairing import pair_files, scan_files
    from .reporting import default_output_dir, load_settings_yaml, run_translocation
    from .translocation import FieldSpec, TranslocationSettings

    s = TranslocationSettings.paper() if a.method == "paper" else TranslocationSettings()
    specs: list[FieldSpec] = []
    if a.config:
        module, s, inputs = load_settings_yaml(a.config)
        if module != "translocation":
            print("settings file is not for nuclear translocation", file=sys.stderr)
            return 2
        if a.method:
            s.method = a.method
        specs = [FieldSpec(i["nuclear"], i["target"], i.get("condition", ""), i.get("field", "")) for i in inputs]
    if a.inputs:
        files = _expand(a.inputs)
        pairs, unpaired = pair_files(scan_files(files))
        for u in unpaired:
            print(f"unpaired: {u.path} {u.error}", file=sys.stderr)
        specs = [FieldSpec(p.nuclear, p.target, a.condition or p.condition, p.field_id) for p in pairs]
        for p in pairs:
            print(f"{p.field_id}: {Path(p.nuclear).name} + {Path(p.target).name} [{p.confidence}] {p.note}")
    if not specs:
        print("no fields to analyse", file=sys.stderr)
        return 2
    if a.pixel_size:
        s.pixel_size_um = a.pixel_size
    out = Path(a.out) if a.out else default_output_dir([specs[0].nuclear_path])
    res = run_translocation(specs, s, out, progress=_progress)
    for row in res["conditions"]:
        print(row)
    for e in res["errors"]:
        print("error:", e, file=sys.stderr)
    print(f"results written to {out}")
    return 0 if res["results"] else 1


def cmd_porosity(a) -> int:
    from .porosity import PorositySettings
    from .reporting import default_output_dir, load_settings_yaml, run_porosity

    s = PorositySettings()
    items: list[tuple[str, str]] = []
    if a.config:
        module, s, inputs = load_settings_yaml(a.config)
        if module != "porosity":
            print("settings file is not for SEM porosity", file=sys.stderr)
            return 2
        items = [(i["image"], i.get("condition", "")) for i in inputs]
    if a.inputs:
        items = [(str(p), a.condition or p.parent.name) for p in _expand(a.inputs)]
    if a.pixel_size:
        s.pixel_size_um = a.pixel_size
    if a.n_thresholds:
        s.n_thresholds = a.n_thresholds
    if a.crop_bottom is not None:
        s.crop_bottom_px = a.crop_bottom
    if not items:
        print("no images to analyse", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else default_output_dir([items[0][0]])
    res = run_porosity(items, s, out, progress=_progress)
    for r in res["results"]:
        sm = r.summary
        print(f"{sm['image']}: porosity {sm['porosity']:.4f}, {sm['n_pores']} pores, mean radius {sm['mean_pore_radius_um']:.3f} µm")
    for e in res["errors"]:
        print("error:", e, file=sys.stderr)
    print(f"results written to {out}")
    return 0 if res["results"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="microscount", description=f"{APP_NAME} {__version__}: microscopy image analysis. "
                                "Run without arguments to open the graphical interface.")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("gui", help="open the graphical interface (default)")
    t = sub.add_parser("translocation", help="nuclear/cytoplasmic ratio of fluorescence images")
    t.add_argument("inputs", nargs="*", help="image files or folders (TIFF/PNG/JPEG)")
    t.add_argument("--method", choices=["per_cell", "paper"], default=None)
    t.add_argument("--config", help="settings.yaml saved by MicrosCount")
    t.add_argument("--condition", help="condition label for all inputs")
    t.add_argument("--pixel-size", type=float, help="micrometres per pixel")
    t.add_argument("--out", help="output folder")
    s = sub.add_parser("porosity", help="porosity and pore sizes of SEM images")
    s.add_argument("inputs", nargs="*")
    s.add_argument("--config")
    s.add_argument("--condition")
    s.add_argument("--pixel-size", type=float, help="micrometres per pixel (MATLAB 'Resolution')")
    s.add_argument("--n-thresholds", type=int, help="MATLAB 'N' (default 4)")
    s.add_argument("--crop-bottom", type=int, help="pixels to remove at the bottom (SEM data bar)")
    s.add_argument("--out")
    st = sub.add_parser("selftest", help="run the built-in self-test")
    st.add_argument("--report", help="also write the report to this file")
    return p


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
    if a.command == "translocation":
        if a.method is None and not a.config:
            a.method = "per_cell"
        return cmd_translocation(a)
    if a.command == "porosity":
        return cmd_porosity(a)
    parser.print_help()
    return 2
