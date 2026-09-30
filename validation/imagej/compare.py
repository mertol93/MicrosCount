"""Compare MicrosCount's paper method with ImageJ on one field.

Usage: python compare.py path/to/ij.jar nuclear.tif target.tif
(single-channel 8-bit TIFFs; compile the Java files first, see README.md)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from microscount.imageio import load_image
from microscount.translocation import TranslocationSettings, analyse_field, median3x3


def main(ij_jar: str, nuclear: str, target: str) -> int:
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as d:
        prefix = str(Path(d) / "ij")
        cp = f"{ij_jar}{';' if sys.platform == 'win32' else ':'}{here}"
        out = subprocess.run(["java", "-Djava.awt.headless=true", "-cp", cp, "PaperPipeline", nuclear, target, prefix],
                             capture_output=True, text=True, timeout=600)
        line = next((ln for ln in (out.stdout + out.stderr).splitlines() if ln.startswith("RESULT")), None)
        if line is None:
            print(out.stdout, out.stderr)
            return 1
        nl, nu, tl, tu, na, nm, ca, cm, ratio = map(float, line.split(",")[1:])
        s = TranslocationSettings.paper()
        s.exclude_annotations = False
        nuc, tgt = load_image(nuclear), load_image(target)
        r = analyse_field(nuc, tgt, s)
        S = r.summary
        same_median = np.array_equal(median3x3(nuc.channel()), tifffile.imread(prefix + "_Nf.tif")) and np.array_equal(
            median3x3(tgt.channel()), tifffile.imread(prefix + "_Tf.tif"))
        same_n = np.array_equal(tifffile.imread(prefix + "_Nmask.tif") > 0, r.layers["paper_nuclear"])
        same_c = np.array_equal(tifffile.imread(prefix + "_Cmask.tif") > 0, r.layers["paper_cytoplasm"])
    print(f"median filter identical : {same_median}")
    print(f"nuclear ROI identical   : {same_n}")
    print(f"cytoplasmic ROI identical: {same_c}")
    print(f"ImageJ     : N >= {nl:g}, T >= {tl:g}; N mean {nm:.6f} (n={int(na)}), C mean {cm:.6f} (n={int(ca)}), R = {ratio:.6f}")
    print(f"MicrosCount: N > {S['nuclear_threshold']:g}, T > {S['target_threshold']:g}; N mean {S['paper_nuclear_mean']:.6f} "
          f"(n={S['paper_nuclear_area_px']}), C mean {S['paper_cytoplasm_mean']:.6f} (n={S['paper_cytoplasm_area_px']}), "
          f"R = {S['paper_ratio']:.6f}")
    return 0 if (same_median and same_n and same_c and abs(ratio - S["paper_ratio"]) < 1e-9) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
