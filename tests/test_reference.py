"""Agreement with the reference implementations (ImageJ 1.54 and the original MATLAB script)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage as ndi

from microscount.imageio import load_image
from microscount.porosity import PorositySettings, analyse_sem, bwmorph_majority, matlab_multithresh
from microscount.thresholds import ij_default, ij_isodata, _bilevel
from microscount.translocation import TranslocationSettings, analyse_field

DATA = Path(__file__).parent / "data"
REF = json.loads((DATA / "imagej_reference.json").read_text())


@pytest.mark.parametrize("case", REF["thresholds"])
def test_imagej_default_threshold(case):
    h = np.array(case["histogram"])
    assert ij_default(h) == case["default"]
    b = _bilevel(h)
    assert (b if b >= 0 else ij_isodata(h)) == case["ij_isodata"]


def test_paper_pipeline_matches_imagej():
    ref = REF["paper_pipeline_synthetic"]
    s = TranslocationSettings.paper()
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
    res = analyse_sem(load_image(DATA / f"{name}.jpg"), PorositySettings())
    binary_matlab = np.asarray(Image.open(DATA / f"{name}_Binary Segmentation.png")).astype(bool)
    assert np.array_equal(res.layers["solid"], binary_matlab)  # pixel-identical segmentation
    assert res.summary["porosity"] == pytest.approx(1 - binary_matlab.mean(), abs=1e-12)
    seg = np.asarray(Image.open(DATA / f"{name}_Pore Space Segmentation.png").convert("RGB"))
    pores_matlab = ~np.all(seg == 255, axis=-1)
    lab, n = ndi.label(pores_matlab, structure=np.ones((3, 3)))
    radii = 0.459 * np.sqrt(np.bincount(lab.ravel())[1:] / np.pi)
    # watershed lines are placed slightly differently from MATLAB's implementation
    assert abs(res.summary["n_pores"] - n) <= 0.01 * n
    assert res.summary["mean_pore_radius_um"] == pytest.approx(radii.mean(), rel=0.01)
    iou = (res.layers["pores"] & pores_matlab).sum() / (res.layers["pores"] | pores_matlab).sum()
    assert iou > 0.98


def test_bwmorph_majority_border_zero_padding():
    m = np.ones((5, 5), bool)
    out = bwmorph_majority(m, 1)
    assert out[2, 2] and out[0, 1]  # an edge pixel sees 6 ones of 9
    assert not out[0, 0]  # a corner sees only 4 ones (zero padding, as MATLAB)
