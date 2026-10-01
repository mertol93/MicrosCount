"""Compare MicrosCount's rolling ball and particle measurements with ImageJ's own classes.

    javac -cp ij.jar RollingBallTest.java ParticleTest.java
    python compare_protocol.py ij.jar target.tif mask.tif

``target.tif``: an 8- or 16-bit single-channel image (the target channel of a field);
``mask.tif``: a 0/255 binary image (e.g. nuclei). Needs ImageJ 1.54 (``ij.jar``) and Java.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from microscount.core.imagej import particle_shape, subtract_background

HERE = Path(__file__).parent


def java(jar: str, *args: str) -> str:
    cp = f"{jar}{';' if sys.platform.startswith('win') else ':'}{HERE}"
    return subprocess.run(["java", "-Djava.awt.headless=true", "-cp", cp, *args], capture_output=True, text=True,
                          check=True).stdout


def main(jar: str, target: str, mask: str) -> None:
    img = tifffile.imread(target)
    with tempfile.TemporaryDirectory() as tmp:
        for radius in (30, 40, 50):
            out = str(Path(tmp) / f"rb{radius}.tif")
            java(jar, "RollingBallTest", target, str(radius), out)
            ij = tifffile.imread(out)
            mc = subtract_background(img, radius)
            print(f"rolling ball {radius} px: {'identical' if np.array_equal(ij, mc) else 'DIFFERENT'} "
                  f"({int((ij != mc).sum())} of {ij.size} pixels differ)")
    rows = [line.split(",") for line in java(jar, "ParticleTest", mask).splitlines() if line.startswith("P,")]
    lab, n = ndi.label(tifffile.imread(mask) > 0, structure=np.ones((3, 3)))
    shapes = particle_shape(lab, n)
    same = 0
    for _, xs, ys, area, perim, circ in rows:
        s = shapes[lab[int(ys), int(xs)]]
        same += (s["area"] == float(area) and abs(s["perimeter"] - float(perim)) < 1e-9
                 and abs(s["circularity"] - float(circ)) < 1e-9)
    print(f"particles: ImageJ {len(rows)}, MicrosCount {n}; area, perimeter and circularity identical for {same}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
