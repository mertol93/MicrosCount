# ImageJ 1.39u cross-validation

Noursadeghi et al. (2008) used ImageJ 1.39. These programs run that version's own classes.

| File | What it does |
|---|---|
| `Thr139.java` | Reads 256-bin histograms (one per line, comma-separated) and prints the two automatic levels of ImageJ 1.39u: *Process ▸ Binary ▸ Convert to Mask* (`Thresholder`, raw histogram) and *Image ▸ Adjust ▸ Threshold ▸ Auto* (`ThresholdAdjuster`, modal bin clipped first) |
| `Paper139.java` | The paper method with either route: 3 × 3 median, automatic levels (pixels ≥ level), nuclear mask, target mask minus nuclear mask, both applied to the original target image, means without the zero bin, and their ratio |
| `stubs/` | Empty stand-ins for `com.sun.image.codec.jpeg`, which ImageJ 1.39 imports and current Java no longer has |

Build ImageJ 1.39u from the ImageJA history and compile the programs against it (javac compiles only the ImageJ
classes they need):

```bash
git clone https://github.com/imagej/ImageJA && git -C ImageJA checkout v1.39u
mkdir -p build
javac -nowarn -encoding ISO-8859-1 -d build -sourcepath ImageJA:stubs Paper139.java Thr139.java
java -cp build Thr139 histograms.csv
java -cp build Paper139 nuclear.tif target.tif out auto      # or: mask
```

With MicrosCount's default settings (`ij139_auto`) the paper method gives the same levels, ROI areas, means and
ratio as `Paper139 … auto` (checked on a synthetic field pair, stored in `tests/data/imagej139_reference.json`,
and on a 2048 × 2048 confocal field), and `ij139_auto` / `ij139_mask` give the same levels as `Thr139` on the 80
stored test histograms.
