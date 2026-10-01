# ImageJ cross-validation (ImageJ 1.54)

These Java programs run ImageJ's own classes so that MicrosCount's re-implementations can be compared with
ImageJ directly.

| File | What it does |
|---|---|
| `ThrTest.java` | Reads 256-bin histograms (one per line, comma-separated) and prints ImageJ's *Default* and *IJ_IsoData* thresholds |
| `PaperPipeline.java` | Noursadeghi et al. (2008) with the current *Default* threshold: median filter (RankFilters, radius 1), `setAutoThreshold("Default dark")` on both channels, nuclear mask, target mask minus nuclear mask, then the mean target intensity in both ROIs and their ratio. It also writes the filtered images and masks as TIFFs. |
| `compare.py` | Runs `PaperPipeline` and MicrosCount on the same pair of single-channel 8-bit TIFFs and reports the differences |
| `RollingBallTest.java` | *Process ▸ Subtract Background* (rolling ball, dialog defaults) with ImageJ's `BackgroundSubtracter` |
| `ParticleTest.java` | *Analyze ▸ Analyze Particles* (size 0–Infinity, circularity 0–1) on a 0/255 mask: area, perimeter and circularity of every particle |
| `compare_protocol.py` | Runs the two programs above and MicrosCount on the same image and mask |

```bash
# ij.jar: download ImageJ 1.x or build it from https://github.com/imagej/ImageJ
javac -cp ij.jar ThrTest.java PaperPipeline.java RollingBallTest.java ParticleTest.java
python compare.py ij.jar nuclear.tif target.tif
python compare_protocol.py ij.jar target.tif nuclei_mask.tif
```

Results obtained with ImageJ 1.54 built from source:

- *Default* threshold identical on 2,000 random histograms;
- on two 2048 × 2048 confocal fields (DAPI + target), identical median-filtered images, pixel-identical ROIs, and identical means and ratios (R = 1.162687 and 1.154091);
- rolling ball at radius 30, 40 and 50 px pixel-identical on both fields;
- area, perimeter and circularity identical for all 655 and 782 particles of the two fields' nuclear masks.

The ImageJ 1.39u comparison of the paper method as published is in [`../imagej139`](../imagej139).
The unit tests (`tests/test_reference.py`) check a stored subset of these reference values.
