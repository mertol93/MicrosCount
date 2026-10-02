"""Agreement with the reference implementations (ImageJ 1.54 and the original MATLAB script)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage as ndi

from microscount.core.imageio import load_image
from microscount.materials.porosity import PorositySettings, analyse_sem, bwmorph_majority, matlab_multithresh
from microscount.core.thresholds import ij139_auto, ij139_mask, ij_default, ij_isodata, _bilevel
from microscount.bio.translocation import TranslocationSettings, analyse_field

DATA = Path(__file__).parent / "data"
REF = json.loads((DATA / "imagej_reference.json").read_text())  # ImageJ 1.54
REF139 = json.loads((DATA / "imagej139_reference.json").read_text())  # ImageJ 1.39u


@pytest.mark.parametrize("case", REF139["thresholds"])
def test_imagej139_threshold_routes(case):
    h = np.array(case["histogram"])
    assert ij139_mask(h) == case["mask"]  # Process > Binary > Convert to Mask
    assert ij139_auto(h) == case["auto"]  # Image > Adjust > Threshold > Auto


@pytest.mark.parametrize("route,method", [("auto", "ij139_auto"), ("mask", "ij139_mask")])
def test_paper_method_matches_imagej139(route, method):
    """The default paper method reproduces the published ImageJ 1.39 workflow exactly."""
    ref = REF139[route]
    s = TranslocationSettings()  # the defaults are the paper method
    s.nuclear_threshold = s.target_threshold = method
    r = analyse_field(load_image(DATA / "synthetic_nuclear.png"), load_image(DATA / "synthetic_target.png"), s)
    S = r.summary
    assert S["nuclear_threshold_ij8"] == ref["level_nuclear"] and S["target_threshold_ij8"] == ref["level_target"]
    assert S["paper_nuclear_area_px"] == ref["nuclear_px"]
    assert S["paper_cytoplasm_area_px"] == ref["cytoplasm_px"]
    assert S["paper_nuclear_mean"] == pytest.approx(ref["nuclear_mean"], rel=1e-12)
    assert S["paper_cytoplasm_mean"] == pytest.approx(ref["cytoplasm_mean"], rel=1e-12)
    assert S["paper_ratio"] == pytest.approx(ref["ratio"], rel=1e-12)


def test_paper_defaults():
    s = TranslocationSettings()  # paper method + per-cell lab protocol
    assert s.method == "per_cell" and TranslocationSettings.paper().method == "paper"
    assert s.median_size == 3 and s.nuclear_threshold == s.target_threshold == "ij139_auto" and s.exclude_zero_pixels
    # the lab protocol
    assert (s.size_min_px2, s.size_max_px2, s.circularity_min, s.circularity_max) == (0.0, None, 0.2, 1.0)
    assert s.rolling_ball_radius == 50 and s.background == "auto" and not s.exclude_saturated


def test_paper_ratio_is_the_same_with_the_per_cell_step():
    nuc, tgt = load_image(DATA / "synthetic_nuclear.png"), load_image(DATA / "synthetic_target.png")
    a = analyse_field(nuc, tgt, TranslocationSettings.paper()).summary
    b = analyse_field(nuc, tgt, TranslocationSettings()).summary
    for k in ("paper_ratio", "paper_nuclear_mean", "paper_cytoplasm_mean", "paper_nuclear_area_px"):
        assert a[k] == b[k]


@pytest.mark.parametrize("case", REF["thresholds"])
def test_imagej_default_threshold(case):
    h = np.array(case["histogram"])
    assert ij_default(h) == case["default"]
    b = _bilevel(h)
    assert (b if b >= 0 else ij_isodata(h)) == case["ij_isodata"]


def test_paper_pipeline_matches_current_imagej():
    """With the ImageJ 1.42+ threshold variant (and zero-valued pixels kept, as that macro did)."""
    ref = REF["paper_pipeline_synthetic"]
    s = TranslocationSettings.paper()
    s.nuclear_threshold = s.target_threshold = "ij_default"
    s.exclude_zero_pixels = False
    r = analyse_field(load_image(DATA / "synthetic_nuclear.png"), load_image(DATA / "synthetic_target.png"), s)
    S = r.summary
    # ImageJ "Default dark" keeps pixels >= lower; MicrosCount reports the exclusive level (> t)
    assert S["nuclear_threshold"] == ref["nuclear_lower"] - 0.5
    assert S["target_threshold"] == ref["target_lower"] - 0.5
    assert S["paper_nuclear_area_px"] == ref["nuclear_px"]
    assert S["paper_cytoplasm_area_px"] == ref["cytoplasm_px"]
    assert S["paper_nuclear_mean"] == pytest.approx(ref["nuclear_mean"], rel=1e-12)
    assert S["paper_cytoplasm_mean"] == pytest.approx(ref["cytoplasm_mean"], rel=1e-12)
    assert S["paper_ratio"] == pytest.approx(ref["ratio"], rel=1e-12)


@pytest.mark.parametrize("name,matlab_first_level", [("SEM1", 83), ("SEM2", 92)])
def test_multithresh_matches_matlab(name, matlab_first_level):
    grey = load_image(DATA / f"{name}.jpg").data[0]
    assert matlab_multithresh(grey, 4, "matlab")[0] == matlab_first_level


@pytest.mark.parametrize("name", ["SEM1", "SEM2"])
def test_sem_porosity_matches_matlab(name):
    res = analyse_sem(load_image(DATA / f"{name}.jpg"), PorositySettings(pixel_size_um=0.459))  # MATLAB "Resolution"
    binary_matlab = np.asarray(Image.open(DATA / f"{name}_Binary Segmentation.png")).astype(bool)
    assert np.array_equal(res.layers["solid"], binary_matlab)  # pixel-identical segmentation
    assert res.summary["porosity"] == pytest.approx(1 - binary_matlab.mean(), abs=1e-12)
    seg = np.asarray(Image.open(DATA / f"{name}_Pore Space Segmentation.png").convert("RGB"))
    pores_matlab = ~np.all(seg == 255, axis=-1)
    assert np.array_equal(res.layers["pores"], pores_matlab)  # pixel-identical watershed split
    lab, n = ndi.label(pores_matlab, structure=np.ones((3, 3)))
    radii = 0.459 * np.sqrt(np.bincount(lab.ravel())[1:] / np.pi)
    assert res.summary["n_pores"] == n
    assert res.summary["mean_pore_radius_um"] == pytest.approx(radii.mean(), rel=1e-12)


def test_bwmorph_majority_border_zero_padding():
    m = np.ones((5, 5), bool)
    out = bwmorph_majority(m, 1)
    assert out[2, 2] and out[0, 1]  # an edge pixel sees 6 ones of 9
    assert not out[0, 0]  # a corner sees only 4 ones (zero padding, as MATLAB)
