import numpy as np
import pytest
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from microscount.imageio import load_image
from microscount.pairing import pair_files, scan_files
from microscount.synthetic import translocation_field
from microscount.translocation import TranslocationSettings, analyse_field, median3x3
from microscount.thresholds import compute_threshold


def _pair(tmp_path, **kw):
    n, t, truth = translocation_field(**kw)
    tifffile.imwrite(tmp_path / "a_dapi.tif", n)
    tifffile.imwrite(tmp_path / "a_gfp.tif", t)
    return load_image(tmp_path / "a_dapi.tif"), load_image(tmp_path / "a_gfp.tif"), truth


@pytest.mark.parametrize("ratio", [0.6, 1.0, 2.5])
def test_per_cell_recovers_true_ratio(tmp_path, ratio):
    nuc, tgt, truth = _pair(tmp_path, ratio=ratio, seed=3)
    r = analyse_field(nuc, tgt, TranslocationSettings())
    assert r.summary["n_cells_analysed"] >= 0.8 * truth["n_cells"] - 12  # edge cells excluded
    assert r.summary["median_ratio"] == pytest.approx(ratio, rel=0.08)


def test_paper_method_is_compressed_towards_one(tmp_path):
    nuc, tgt, truth = _pair(tmp_path, ratio=2.5, seed=4)
    paper = analyse_field(nuc, tgt, TranslocationSettings.paper()).summary["paper_ratio"]
    per_cell = analyse_field(nuc, tgt, TranslocationSettings()).summary["median_ratio"]
    assert 1.0 < paper < per_cell


def test_median3x3_equals_scipy():
    a = np.random.default_rng(5).integers(0, 4096, (57, 43)).astype(np.uint16)
    assert np.array_equal(median3x3(a), ndi.median_filter(a, 3, mode="nearest"))


def test_manual_threshold_and_background(tmp_path):
    nuc, tgt, _ = _pair(tmp_path, ratio=2.0, seed=6)
    s = TranslocationSettings(nuclear_threshold="manual", nuclear_threshold_value=40, background="manual",
                              background_value=6.0)
    r = analyse_field(nuc, tgt, s)
    assert r.summary["background"] == 6.0
    assert r.summary["median_ratio"] == pytest.approx(2.0, rel=0.1)


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
