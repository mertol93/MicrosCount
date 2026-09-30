# ImageJ cross-validation

These Java programs run ImageJ's own classes so that MicrosCount's re-implementation of the
Noursadeghi et al. (2008) pipeline can be compared with ImageJ directly.

| File | What it does |
|---|---|
| `ThrTest.java` | Reads 256-bin histograms (one per line, comma-separated) and prints ImageJ's *Default* and *IJ_IsoData* thresholds |
| `PaperPipeline.java` | Median filter (RankFilters, radius 1), `setAutoThreshold("Default dark")` on both channels, nuclear mask, target mask minus nuclear mask, then mean target intensity in both ROIs and their ratio. It also writes the filtered images and masks as TIFFs. |
| `compare.py` | Runs `PaperPipeline` and MicrosCount on the same pair of single-channel 8-bit TIFFs and reports the differences |

```bash
# ij.jar: download ImageJ 1.x or build it from https://github.com/imagej/ImageJ
javac -cp ij.jar ThrTest.java PaperPipeline.java
python compare.py ij.jar nuclear.tif target.tif
```

Results obtained with ImageJ 1.54 built from source:

- *Default* threshold identical on 2,000 random histograms;
- on two 2048 × 2048 confocal fields (DAPI + target), identical median-filtered images, pixel-identical ROIs, and identical means and ratios (R = 1.162687 and 1.154091).

The unit tests (`tests/test_reference.py`) check a stored subset of these reference values.
