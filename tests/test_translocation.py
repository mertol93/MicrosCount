import numpy as np
import pytest
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from microscount.core.imageio import load_image
from microscount.bio.pairing import pair_files, scan_files
from microscount.synthetic import translocation_field
from microscount.bio.translocation import TranslocationSettings, analyse_field, median3x3
from microscount.core.thresholds import compute_threshold


def _pair(tmp_path, **kw):
    n, t, truth = translocation_field(**kw)
    tifffile.imwrite(tmp_path / "a_dapi.tif", n)
    tifffile.imwrite(tmp_path / "a_gfp.tif", t)
    return load_image(tmp_path / "a_dapi.tif"), load_image(tmp_path / "a_gfp.tif"), truth


@pytest.mark.parametrize("ratio", [0.6, 1.0, 2.5])
def test_per_cell_recovers_true_ratio(tmp_path, ratio):
    nuc, tgt, truth = _pair(tmp_path, ratio=ratio, seed=3)
    s = TranslocationSettings.per_cell()
    r = analyse_field(nuc, tgt, s)
    assert r.summary["n_cells_analysed"] >= 0.8 * truth["n_cells"] - 12  # edge cells excluded
    # the lab protocol measures whole particles and a ring touching them: blur mixes a little
    assert r.summary["median_nc"] == pytest.approx(ratio, rel=0.12)
    assert r.summary["median_cn"] == pytest.approx(1 / ratio, rel=0.12)
    # trimming the nucleus edge and leaving a gap removes the mixing
    s.nucleus_erode_px, s.ring_gap_px = 1, 1.0
    assert analyse_field(nuc, tgt, s).summary["median_nc"] == pytest.approx(ratio, rel=0.08)


def test_lab_protocol_values_per_cell(tmp_path):
    nuc, tgt, _ = _pair(tmp_path, ratio=2.0, seed=12)
    r = analyse_field(nuc, tgt, TranslocationSettings())  # defaults: paper + per-cell lab protocol
    assert r.method == "per_cell" and np.isfinite(r.summary["paper_ratio"])
    c = next(c for c in r.cells if c["included"])
    bg = r.summary["background_mean"]
    assert c["background_mean"] == bg
    assert c["nuc_corr"] == pytest.approx(c["nuc_mean"] - bg)
    assert c["cyto_corr"] == pytest.approx(c["cyto_mean"] - bg)
    assert c["nc_ratio"] == pytest.approx(c["nuc_corr"] / c["cyto_corr"])
    assert c["cn_ratio"] == pytest.approx(c["cyto_corr"] / c["nuc_corr"])
    assert 0.2 <= c["circularity"] <= 1.0
    # particle filters as ImageJ Analyze Particles
    s = TranslocationSettings(size_min_px2=10_000)
    r2 = analyse_field(nuc, tgt, s)
    assert r2.summary["n_cells_analysed"] == 0
    assert all(c["exclusion_reason"] == "size outside the particle filter" for c in r2.cells)
    s = TranslocationSettings(circularity_min=0.99)
    reasons = {c["exclusion_reason"] for c in analyse_field(nuc, tgt, s).cells}
    assert "circularity outside the particle filter" in reasons


def test_rolling_ball_is_applied_to_the_target(tmp_path):
    nuc, tgt, _ = _pair(tmp_path, ratio=2.0, seed=13)
    tgt.data = tgt.data.astype(np.uint16) + np.linspace(0, 400, tgt.data.shape[-1], dtype=np.uint16)[None, None, :]
    on = analyse_field(nuc, tgt, TranslocationSettings()).summary["median_nc"]
    off = analyse_field(nuc, tgt, TranslocationSettings(rolling_ball_radius=0, background="none",
                                                        restrict_to_cells=False)).summary["median_nc"]
    assert on == pytest.approx(2.0, rel=0.15)  # the ramp is removed
    assert off < 0.8 * on  # without it the offset compresses the ratio


def test_paper_method_is_compressed_towards_one(tmp_path):
    nuc, tgt, truth = _pair(tmp_path, ratio=2.5, seed=4)
    paper = analyse_field(nuc, tgt, TranslocationSettings.paper()).summary["paper_ratio"]
    per_cell = analyse_field(nuc, tgt, TranslocationSettings.per_cell()).summary["median_nc"]
    assert 1.0 < paper < per_cell


def test_median3x3_equals_scipy():
    a = np.random.default_rng(5).integers(0, 4096, (57, 43)).astype(np.uint16)
    assert np.array_equal(median3x3(a), ndi.median_filter(a, 3, mode="nearest"))


def test_manual_threshold_and_background(tmp_path):
    nuc, tgt, _ = _pair(tmp_path, ratio=2.0, seed=6)
    s = TranslocationSettings.per_cell()
    s.nuclear_threshold, s.nuclear_threshold_value = "manual", 40
    s.background, s.background_value = "manual", 1.0
    r = analyse_field(nuc, tgt, s)
    assert r.summary["background_mean"] == 1.0 and r.summary["background_source"] == "manual"
    assert r.summary["median_nc"] == pytest.approx(2.0, rel=0.12)


def test_paper_outputs_histograms_and_counts(tmp_path):
    nuc, tgt, truth = _pair(tmp_path, ratio=2.0, seed=8)
    r = analyse_field(nuc, tgt, TranslocationSettings.paper())
    h = r.histograms
    assert h is not None and abs(h["nuclear_pct"].sum() - 100) < 1e-6 and h["nuclear_pct"][0] == 0  # zero bin excluded
    assert abs(r.summary["nuclei_count"] - truth["n_cells"]) <= 0.1 * truth["n_cells"]


def test_threshold_16bit_matches_8bit_scaling():
    a = np.random.default_rng(0).integers(0, 4096, (100, 100)).astype(np.uint16)
    t = compute_threshold(a, "ij_default")
    assert t.level8 is not None and 0 <= t.level8 <= 255


def test_pairing_by_name_and_by_image(tmp_path):
    n1, t1, _ = translocation_field(seed=10, shape=(256, 256))
    n2, t2, _ = translocation_field(seed=11, shape=(256, 256))
    # by name (Leica style)
    tifffile.imwrite(tmp_path / "field1_ch00.tif", n1)
    tifffile.imwrite(tmp_path / "field1_ch01.tif", t1)
    # by image content: single-colour RGB exports with meaningless names
    def rgb(a, c):
        x = np.zeros(a.shape + (3,), np.uint8)
        x[..., c] = a
        return x
    Image.fromarray(rgb(n2, 2)).save(tmp_path / "x9.png")
    Image.fromarray(rgb(t1, 1)).save(tmp_path / "q1.png")  # decoy: other field
    Image.fromarray(rgb(t2, 1)).save(tmp_path / "k3.png")
    Image.fromarray(rgb(n1, 2)).save(tmp_path / "z0.png")
    files = sorted(str(p) for p in tmp_path.iterdir())
    pairs, unpaired = pair_files(scan_files(files))
    got = {(p.nuclear.split("/")[-1].split("\\")[-1], p.target.split("/")[-1].split("\\")[-1]) for p in pairs}
    assert ("field1_ch00.tif", "field1_ch01.tif") in got
    assert ("x9.png", "k3.png") in got and ("z0.png", "q1.png") in got
    assert not unpaired


def test_leica_export_folder_with_third_channel_and_metadata(tmp_path):
    """A LAS X export: _ch00 (blue), _ch01 (green), _ch02 (transmitted light) and a MetaData folder."""
    from microscount.core.imageio import list_images

    def rgb(a, c):
        x = np.zeros(a.shape + (3,), np.uint8)
        x[..., c] = a
        return x

    for k, seed in ((1, 20), (2, 21)):
        n, t, _ = translocation_field(seed=seed, shape=(256, 256))
        Image.fromarray(rgb(n, 2)).save(tmp_path / f"exp_vehicle-{k}_ch00.tif")
        Image.fromarray(rgb(t, 1)).save(tmp_path / f"exp_vehicle-{k}_ch01.tif")
        bright = (128 + 20 * np.random.default_rng(seed).standard_normal((256, 256))).clip(0, 255).astype(np.uint8)
        Image.fromarray(bright).save(tmp_path / f"exp_vehicle-{k}_ch02.tif")
    meta = tmp_path / "MetaData"
    meta.mkdir()
    ramp = np.tile(np.arange(200, dtype=np.uint8), (500, 1))
    for k in (1, 2):
        for c, ch in ((2, 0), (1, 1)):
            Image.fromarray(rgb(ramp, c)).save(meta / f"exp_vehicle-{k}ch{ch}LUT.png")
    (meta / "exp_vehicle-1.xml").write_text("<xml/>")
    # a copied subfolder where the nuclear file is missing and _ch02 is blank (an unused detector)
    sub = tmp_path / "copies"
    sub.mkdir()
    Image.fromarray(rgb(translocation_field(seed=22, shape=(256, 256))[1], 1)).save(sub / "exp_vehicle-3_ch01.tif")
    Image.fromarray(np.ones((256, 256), np.uint8)).save(sub / "exp_vehicle-3_ch02.tif")
    Image.fromarray(rgb(ramp, 2)).save(tmp_path / "exp_vehicle-1ch0LUT.png")  # a legend next to the images

    files = list_images(tmp_path)
    expected = [f"exp_vehicle-{k}_ch0{c}.tif" for k in (1, 2) for c in (0, 1, 2)] + ["exp_vehicle-3_ch01.tif", "exp_vehicle-3_ch02.tif"]
    assert sorted(p.name for p in files) == sorted(expected)
    pairs, unpaired = pair_files(scan_files(files))
    got = sorted((p.nuclear.split("/")[-1], p.target.split("/")[-1], p.confidence) for p in pairs)
    assert got == [(f"exp_vehicle-{k}_ch00.tif", f"exp_vehicle-{k}_ch01.tif", "name") for k in (1, 2)]
    left = {u.path.split("/")[-1]: u.note for u in unpaired}
    assert sorted(left) == ["exp_vehicle-1_ch02.tif", "exp_vehicle-2_ch02.tif", "exp_vehicle-3_ch01.tif", "exp_vehicle-3_ch02.tif"]
    assert left["exp_vehicle-1_ch02.tif"].startswith("extra channel") and left["exp_vehicle-3_ch02.tif"].startswith("blank")
    assert left["exp_vehicle-3_ch01.tif"] == ""  # a target without its nuclear file stays unpaired
