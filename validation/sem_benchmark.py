"""Benchmark of the SEM porosity analysis on synthetic micrographs with known pores.

    python validation/sem_benchmark.py [--images 6] [--n-thresholds 4] [--out scores.csv]

Each scenario is a set of SEM-like membrane surfaces (``microscount.synthetic.sem_surface``) with
the true pore mask saved next to every image, scored with ``microscount materials check``: the
porosity found against the true porosity, the overlap of the pore pixels, the share of pores
found, and the ratio of mean pore diameters. Hand-traced images can be scored the same way with
``microscount materials check <folder>``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from microscount.materials.check import check_images
from microscount.materials.porosity import PorositySettings
from microscount.synthetic import sem_surface

SCENARIOS = {
    "flat pores": dict(style="flat", porosity=0.05),
    "pores with grey walls": dict(porosity=0.05),
    "low porosity": dict(porosity=0.015, radius=(2.5, 30)),
    "high porosity": dict(porosity=0.15),
    "uneven brightness": dict(porosity=0.05, shading=0.3),
    "charging patches": dict(porosity=0.05, blotches=0.45),
    "published figure": dict(porosity=0.05, shading=0.15, figure=True),
}
EXPOSURES = [(1.0, 0.0), (0.6, -10.0), (1.3, 10.0), (0.8, 30.0), (1.1, -25.0), (0.7, 20.0)]


def run(n_images: int = 6, settings: PorositySettings | None = None) -> list[dict]:
    s = settings or PorositySettings()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, kw in list(SCENARIOS.items()) + [("different exposures", dict(porosity=0.05))]:
            folder = Path(tmp) / name.replace(" ", "_")
            folder.mkdir()
            paths = []
            for i in range(n_images):
                extra = {}
                if name == "different exposures":
                    extra = dict(zip(("gain", "offset"), EXPOSURES[i % len(EXPOSURES)]))
                g, truth = sem_surface(seed=100 * i + 7, **kw, **extra)
                p = folder / f"image_{i:02d}.png"
                Image.fromarray(g).save(p)
                Image.fromarray(truth.astype(np.uint8) * 255).save(folder / f"image_{i:02d}_pores.png")
                paths.append(p)
            _, mean = check_images(paths, s)
            rows.append({"scenario": name, **{k: v for k, v in mean.items() if k not in ("image", "mask")}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--images", type=int, default=6, help="images per scenario (default 6)")
    ap.add_argument("--n-thresholds", type=int, default=4, help="MATLAB 'N' (default 4)")
    ap.add_argument("--out", help="also write the table to this CSV file")
    a = ap.parse_args()
    rows = run(a.images, PorositySettings(n_thresholds=a.n_thresholds))
    print(f"{'scenario':22s} {'true %':>7s} {'found %':>8s} {'overlap':>8s} {'pores found':>12s} {'size ratio':>11s}")
    for r in rows:
        print(f"{r['scenario']:22s} {r['porosity_traced_percent']:7.2f} {r['porosity_found_percent']:8.2f} "
              f"{r['overlap']:8.2f} {r['pores_found']:12.2f} {r['size_ratio']:11.2f}")
    if a.out:
        from microscount.core.report import write_csv

        write_csv(Path(a.out), rows)


if __name__ == "__main__":
    main()
