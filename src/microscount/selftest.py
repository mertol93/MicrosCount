"""End-to-end self-test (run by ``microscount selftest`` and by the installer CI)."""

from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


def run(report: str | Path | None = None) -> bool:
    lines: list[str] = []
    ok = True

    def check(name: str, cond: bool, detail: str = ""):
        nonlocal ok
        ok &= bool(cond)
        lines.append(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    t0 = time.time()
    try:
        from PIL import Image
        import tifffile

        from . import __version__
        from .imageio import load_image
        from .porosity import PorositySettings, analyse_sem
        from .reporting import run_porosity, run_translocation
        from .synthetic import sem_image, translocation_field
        from .thresholds import ij_default
        from .translocation import FieldSpec, TranslocationSettings, analyse_field

        lines.append(f"MicrosCount {__version__} self-test")
        # ImageJ Default threshold: reference value computed with ImageJ 1.54
        hist = np.bincount(np.r_[np.full(900, 10), np.arange(5, 60), np.full(300, 200), np.arange(150, 250)], minlength=256)
        check("ImageJ Default threshold", ij_default(hist) == 106, f"level {ij_default(hist)} (ImageJ 1.54: 106)")

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            n8, t8, truth = translocation_field(ratio=2.0, seed=1)
            files = {
                "tif": (d / "f_dapi.tif", d / "f_gfp.tif"),
                "png": (d / "f_dapi.png", d / "f_gfp.png"),
                "jpg": (d / "f_dapi.jpg", d / "f_gfp.jpg"),
            }
            tifffile.imwrite(files["tif"][0], n8)
            tifffile.imwrite(files["tif"][1], t8)
            Image.fromarray(n8).save(files["png"][0])
            Image.fromarray(t8).save(files["png"][1])
            Image.fromarray(n8).save(files["jpg"][0], quality=95)
            Image.fromarray(t8).save(files["jpg"][1], quality=95)
            for fmt, (a, b) in files.items():
                na, tb = load_image(a), load_image(b)
                check(f"read {fmt.upper()}", na.data.shape == (1,) + n8.shape and tb.data.shape == (1,) + t8.shape)
            # 16-bit TIFF round trip
            tifffile.imwrite(d / "s16.tif", (t8.astype(np.uint16) * 16))
            check("read 16-bit TIFF", load_image(d / "s16.tif").data.dtype == np.uint16)

            nuc, tgt = load_image(files["tif"][0]), load_image(files["tif"][1])
            s = TranslocationSettings()
            s.exclude_border_cells = True
            r = analyse_field(nuc, tgt, s, "synthetic", "test")
            med = r.summary["median_ratio"]
            check("per-cell N/C ratio on synthetic cells", abs(med - truth["ratio"]) / truth["ratio"] < 0.10,
                  f"median {med:.3f}, truth {truth['ratio']:.3f}, cells {r.summary['n_cells_analysed']}")
            p = analyse_field(nuc, tgt, TranslocationSettings.paper(), "synthetic", "test")
            check("paper method runs", np.isfinite(p.summary["paper_ratio"]), f"R = {p.summary['paper_ratio']:.3f}")

            out = run_translocation([FieldSpec(str(files["png"][0]), str(files["png"][1]), "test", "field_001")],
                                    TranslocationSettings(), d / "out_t")
            check("translocation report files", all((d / "out_t" / f).exists() for f in ("per_field.csv", "per_cell.csv", "CITATION.txt", "summary.png")))

            sem, true_por = sem_image(seed=2)
            Image.fromarray(np.stack([sem] * 3, -1)).save(d / "sem.png")
            res = analyse_sem(load_image(d / "sem.png"), PorositySettings(n_thresholds=2))
            por = res.summary["porosity"]
            check("SEM porosity on synthetic pores", abs(por - true_por) < 0.05, f"{por:.3f} vs true {true_por:.3f}")
            out2 = run_porosity([(str(d / "sem.png"), "test")], PorositySettings(n_thresholds=2), d / "out_p")
            check("porosity report files", all((d / "out_p" / f).exists() for f in ("porosity_summary.csv", "pores.csv", "CITATION.txt")))
    except Exception:  # noqa: BLE001
        ok = False
        lines.append("[FAIL] exception:\n" + traceback.format_exc())
    lines.append(f"{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'} in {time.time() - t0:.1f} s")
    text = "\n".join(lines)
    if report:
        Path(report).write_text(text + "\n", encoding="utf-8")
    try:
        print(text)
    except Exception:  # noqa: BLE001 - windowed builds have no console
        pass
    return ok
