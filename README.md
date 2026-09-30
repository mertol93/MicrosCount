# MicrosCount

**Open-source microscopy image analysis with a point-and-click interface.** No programming needed.

MicrosCount has two tools:

| Tool | What it measures | Typical images |
|---|---|---|
| **Nuclear translocation** | Nuclear/cytoplasmic (N/C) intensity ratio of a protein (e.g. NF-κB p65/RelA), per cell and per field | Fluorescence/confocal: a nuclear stain (DAPI, Hoechst) + a target channel |
| **SEM porosity** | Porosity and pore-size distribution | Scanning electron micrographs of membranes, rocks, foams… |

It reads **TIFF, PNG and JPEG** (8/16-bit, greyscale, RGB exports, multi-channel TIFF) and installs on **Windows, macOS and Linux**.

> ### 📄 Please cite
> If you use MicrosCount in published work, please cite:
>
> Acarer-Arat, S., Pir, İ., Tüfekci, M., Güneş-Durak, S., Akman, A., & Tüfekci, N. (2024). Heavy Metal Rejection Performance and Mechanical Performance of Cellulose-Nanofibril-Reinforced Cellulose Acetate Membranes. *ACS Omega*, 9(41), 42159–42171. https://doi.org/10.1021/acsomega.4c03038
>
> GitHub's **"Cite this repository"** button (right-hand panel) gives this reference in APA and BibTeX. Every results folder also contains a `CITATION.txt`, which additionally lists the method papers for the tool you used (see [Method references](#method-references)).

![Nuclear translocation: per-cell segmentation preview](docs/images/translocation_preview.png)

---

## Download and install

Download the installer for your system from the **[Releases page](https://github.com/mertol93/microscount/releases/latest)**:

| System | File | How to install |
|---|---|---|
| Windows 10/11 (64-bit) | `MicrosCount-<version>-Windows-x64-Setup.exe` | Double-click and follow the wizard. No administrator rights needed. If SmartScreen says *"Windows protected your PC"*, click **More info → Run anyway** (the installer is not code-signed). |
| macOS, Apple silicon (M1–M4) | `MicrosCount-<version>-macOS-arm64.dmg` | Open the DMG and drag **MicrosCount** to **Applications**. On first launch, right-click the app and choose **Open**. If macOS still refuses, go to **System Settings → Privacy & Security** and click **Open Anyway**. The app is not notarised by Apple. |
| macOS, Intel | `MicrosCount-<version>-macOS-x86_64.dmg` | As above. |
| Ubuntu / Debian | `microscount_<version>_amd64.deb` | `sudo apt install ./microscount_<version>_amd64.deb`, then start *MicrosCount* from the applications menu. |
| Any Linux | `MicrosCount-<version>-x86_64.AppImage` | `chmod +x MicrosCount-*.AppImage` and double-click it. |
| Any Linux | `MicrosCount-<version>-Linux-x86_64.tar.gz` | Extract, then run `./install.sh` to add a menu entry (or run `./MicrosCount` directly). |

The Linux builds need glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 or later).

Python users can instead run `pip install git+https://github.com/mertol93/microscount` and then `microscount`.

---

## Quick start: nuclear translocation

1. **Add images.** Use *Add files…* or *Add folder…*, or drag files onto the table. MicrosCount pairs each nuclear-stain image with its target image automatically:
   - blue single-colour exports are treated as the nuclear stain, and green/red ones as the target;
   - files are paired by name (`img_ch00`/`img_ch01`, `C1-x`/`C2-x`, `…DAPI`/`…AF488`);
   - if names don't match, files are paired by image content (the same nuclei appear in both channels).
   
   Multi-channel TIFFs and merged RGB images are used as they are. Right-click a row to swap or change a pairing. Type a **Condition** name for each field, e.g. *control*, *LPS 30 min*.
2. **Choose the method.**
   - **Per-cell (recommended):** every nucleus gets its own ratio.
   - **Paper method:** the published whole-field procedure of Noursadeghi et al. (2008).
3. **Check the settings.** The defaults work for most images. *Pixel size → From scale bar…* reads a burned-in scale bar.
4. **Preview** a field. Measured nuclei are outlined in cyan, their cytoplasm is shaded magenta, and excluded nuclei are outlined in red. Hover over a cell to see its ratio.
5. **Analyse all fields.** Results open in the *Results* tab and are saved to a new folder next to your images.

### What you get

| File | Contents |
|---|---|
| `per_cell.csv` | One row per nucleus: position, area, nuclear and cytoplasmic mean, N/C, log2 N/C, and whether and why it was excluded |
| `per_field.csv` | One row per field: median/mean/geometric-mean N/C, IQR, fraction of cells above the cut-off, paper-method ratio, thresholds, background, QC flags |
| `per_condition.csv` | Mean ± SD over fields (as in the paper) and pooled per-cell statistics |
| `results.xlsx` | All of the above plus the settings, in one workbook |
| `overlays/*.png` | Segmentation overlays for checking every field |
| `summary.png` | Per-cell distributions and per-field ratios by condition |
| `settings.yaml` | Every setting and input file. Reopen it with *File → Open settings / previous analysis* to reproduce the run. |
| `CITATION.txt` | How to cite |

## Quick start: SEM porosity

1. **Add SEM images.** Pores should be the darkest regions; tick *Pores are bright* otherwise.
2. **Set the pixel size** in µm/px from the micrograph's scale bar. If the image has a microscope data bar, remove it with *Crop at bottom*.
3. **Preview** an image and check the overlay, the binary segmentation, the MATLAB-style depth map and the pore segmentation.
4. **Analyse all images.** You get:
   - porosity, pore count, and mean/SD/median equivalent pore radius per image and per condition;
   - a `pores.csv` table of every pore;
   - the MATLAB-style images for each micrograph: binary segmentation, depth map, pore-space segmentation, and the pore-size histogram.

![SEM porosity preview](docs/images/porosity_preview.png)

---

## Methods

### Nuclear translocation: paper method

This re-implements Noursadeghi et al. (2008), *J. Immunol. Methods* 329:194–200, exactly as published:

1. a 3 × 3 median filter on both channels (ImageJ *Median…* radius 1);
2. ImageJ's *Default* (modified IsoData) auto-threshold of each filtered channel;
3. nuclear ROI = the nuclear-stain mask, cytoplasmic ROI = target mask minus nuclear mask;
4. N/C = mean *unfiltered* target intensity in the nuclear ROI ÷ that in the cytoplasmic ROI, pooled over the whole field.

The threshold is a line-by-line port of ImageJ 1.54 (`AutoThresholder.defaultIsoData`), including the modal-bin clipping and the 8-bit scaling of 16-bit data.

### Nuclear translocation: per-cell method

The paper's cytoplasmic ROI is thresholded on the same channel that is being measured. When the target protein leaves the cytoplasm, dim cytoplasm drops below the threshold and the cytoplasmic mean is overestimated, so the ratio is compressed towards 1 exactly when translocation is strongest. The per-cell method avoids this:

- **Nuclei** are detected on the nuclear stain after light Gaussian smoothing, thresholded (ImageJ Default by default), then:
  - holes are filled;
  - touching nuclei are split by a distance-transform watershed, which only splits objects larger than one typical nucleus with a real "neck";
  - debris, clumps, irregular (merged) nuclei and nuclei touching the image edge are excluded.
- **Cytoplasm** is a ring around each nucleus, 0.3 × the nucleus diameter wide by default, 1 px away from it. Pixels closer to a neighbouring nucleus are assigned to that nucleus. The ring does not depend on the target intensity; it only drops pixels at background level.
- **Background** (the histogram mode of the target channel away from nuclei) is subtracted before the ratio. You can also set it manually or switch it off.
- **Saturated pixels** are ignored, and cells with more than 5% saturated pixels are excluded.
- The ratio is computed for each cell. The field and condition are summarised by the median, the IQR and the fraction of cells above a cut-off (default N/C > 1). The paper-method ratio is reported alongside for comparison.

### SEM porosity

This is a Python port of `SEM_Porosity.m` by Arash Rabbani (BSD-3-Clause), the script used for the membrane SEM analysis in Acarer-Arat et al. (2024). Each MATLAB step keeps its MATLAB semantics:

1. `rgb2gray`;
2. `multithresh(I, N)`, including MATLAB's `fminsearch` search for N ≥ 3 (an *exhaustive* global search is available as an option);
3. pores are the darkest class;
4. `bwmorph(…,'majority')`;
5. a city-block distance map, `medfilt2`, and an 8-connected watershed to separate touching pores;
6. `bwareaopen(…, 9)`;
7. equivalent pore radius = pixel size × √(area/π);
8. porosity = pore fraction before the watershed split.

Optional additions, all off by default:
- cropping an SEM data bar;
- counting more than one dark class as pore;
- leaving edge-cut pores out of the size statistics;
- bright pores.

### Input handling

- **TIFF:** 8/16/32-bit, multi-page, multi-channel (ImageJ, OME), RGB and palette. Pixel size is read from OME or ImageJ metadata or the resolution tags. Z/T stacks are max-projected.
- **PNG / JPEG:**
  - greyscale, RGB, palette (decoded through the palette) and 16-bit PNG, including 16-bit RGB;
  - single-colour exports (e.g. a blue DAPI snapshot) are recognised and the populated channel is used;
  - burned-in annotations such as scale bars and text are detected from colour and excluded from every calculation.
- **Use raw exports for quantification.** Microscope "snapshots" are display-scaled 8-bit images and JPEG is lossy, so prefer raw TIFF (or 16-bit) exports.

---

## Validation

| Check | Result |
|---|---|
| ImageJ *Default* threshold vs ImageJ 1.54 (built from source) on 2,000 random histograms | **2,000/2,000 identical** |
| Paper method vs ImageJ 1.54 on two 2048 × 2048 confocal fields (DAPI + target) | Identical median-filtered images and pixel-identical ROIs; means and ratios equal to 6 decimals (R = 1.162687 and 1.154091) |
| SEM porosity vs the original MATLAB outputs (Rabbani's sample images) | Binary segmentation **pixel-identical**; porosity identical (0.09945, 0.19567); pore count 330/330 and 454/453; mean pore radius within 0.2% |
| Per-cell method on synthetic cells with a known N/C (0.6, 1.0, 2.0, 2.5) | Recovered within 8%. On a synthetic field with N/C = 2.5, the paper method is biased towards 1. |

The ImageJ comparisons can be re-run with the Java harness in [`validation/imagej`](validation/imagej). The unit tests (`pytest`) check against stored ImageJ and MATLAB reference values.

## Command line

The installers include `microscount-cli`, and a pip install provides `microscount`:

```bash
microscount translocation path/to/images --out results          # per-cell method
microscount translocation path/to/images --method paper         # Noursadeghi et al. 2008
microscount translocation --config results/settings.yaml        # re-run a saved analysis
microscount porosity path/to/sem --pixel-size 0.459 --n-thresholds 4 --crop-bottom 60
microscount selftest
```

## Building from source

```bash
git clone https://github.com/mertol93/microscount && cd microscount
pip install -e .[dev]
pytest
microscount            # opens the GUI
```

Installers are built with PyInstaller by [`.github/workflows/build.yml`](.github/workflows/build.yml) on GitHub's Windows, macOS and Linux runners. Each frozen app runs its self-test before it is packaged. To build locally, run `packaging/build_windows.ps1`, `packaging/build_macos.sh` or `packaging/build_linux.sh`.

## Method references

Please also cite the method papers for the tool you used. These references are in `CITATION.txt` in every results folder.

- **Nuclear translocation (paper method):** Noursadeghi, M., Tsang, J., Haustein, T., Miller, R. F., Chain, B. M., & Katz, D. R. (2008). Quantitative imaging assay for NF-κB nuclear translocation in primary human macrophages. *Journal of Immunological Methods*, 329(1–2), 194–200. https://doi.org/10.1016/j.jim.2007.10.015
- **SEM porosity (original MATLAB code):**
  - Rabbani, A., & Salehi, S. (2017). Dynamic modeling of the formation damage and mud cake deposition using filtration theories coupled with SEM image processing. *Journal of Natural Gas Science and Engineering*, 42, 157–168. https://doi.org/10.1016/j.jngse.2017.02.047
  - Ezeakacha, C. P., Rabbani, A., Salehi, S., & Ghalambor, A. (2018). Integrated image processing and computational techniques to characterize formation damage. *SPE International Conference and Exhibition on Formation Damage Control*.

## Licence

MicrosCount is free software under the **GNU General Public License v3.0 or later** (see [`LICENSE`](LICENSE)). It comes with no warranty.

It bundles third-party components under their own licences, including Qt for Python (LGPL-3.0) and A. Rabbani's SEM_Porosity (BSD-3-Clause); see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Bug reports and suggestions are welcome on the [issue tracker](https://github.com/mertol93/microscount/issues).
