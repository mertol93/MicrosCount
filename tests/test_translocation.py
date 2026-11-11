import numpy as np
import pytest
import tifffile
import yaml
from PIL import Image
from scipy import ndimage as ndi

from microscount.core.imageio import load_image
from microscount.bio.pairing import pair_files, scan_files
from microscount.synthetic import hazy_translocation_field, translocation_field
from microscount.bio.translocation import TranslocationSettings, analyse_field, median3x3, nuclear_threshold_check
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
    tgt.saturation_value = None
    assert analyse_field(nuc, tgt, TranslocationSettings()).summary["median_nc"] == pytest.approx(2.0, rel=0.15)
    ring = dict(cytoplasm="ring", restrict_to_cells=False)
    on = analyse_field(nuc, tgt, TranslocationSettings(**ring)).summary["median_nc"]
    off = analyse_field(nuc, tgt, TranslocationSettings(rolling_ball_radius=0, background="none", **ring))
    assert on == pytest.approx(2.0, rel=0.15)  # the ramp is removed
    assert off.summary["median_nc"] < 0.8 * on  # without it the offset compresses the ratio
    # nor can the cells be outlined: the bright end of the ramp counts as cell
    off = analyse_field(nuc, tgt, TranslocationSettings(rolling_ball_radius=0, background="none"))
    assert off.summary["median_nc"] < 0.8 * 2.0


def _jaccard(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


@pytest.mark.parametrize("spacing", [64, 46])  # separate cells, and cells that touch
def test_whole_cell_boundaries(tmp_path, spacing):
    """Boundary 1 = the nucleus in the nuclear stain, boundary 2 = the cell outlined on the target;
    the cytoplasm is everything between them, and only the target is measured."""
    nuc, tgt, truth = _pair(tmp_path, ratio=2.5, seed=21, spacing=spacing, cell_radius=26.0)
    r = analyse_field(nuc, tgt, TranslocationSettings())
    L, S = r.layers, r.summary
    assert S["cytoplasm_region"] == "cell" and S["median_nc"] == pytest.approx(2.5, rel=0.12)
    nuclei, cells, cyto = L["nuclear_labels"], L["cell_labels"], L["cytoplasm_labels"]
    assert np.array_equal(cells[nuclei > 0], nuclei[nuclei > 0])  # each cell holds its own nucleus
    assert not (cyto & (nuclei > 0)).any() and np.array_equal(cyto > 0, (cells > 0) & (nuclei == 0))
    # every true cell is found as one cell, with its true outline
    t_cells, t_nuc = truth["cells"], truth["nuclei"]
    scores = []
    for c in r.cells:
        if not c["included"]:
            continue
        k = np.bincount(t_nuc[nuclei == c["cell"]]).argmax()
        assert k > 0
        scores.append(_jaccard(cells == c["cell"], t_cells == k))
        assert c["cell_area_px"] == int((cells == c["cell"]).sum())
        assert c["cytoplasm_pixels"] == int((cyto == c["cell"]).sum())
    assert len(scores) >= 0.6 * truth["n_cells"] and np.median(scores) > 0.85 and min(scores) > 0.6
    # cells cut by the image edge are not measured
    edge = {int(v) for v in np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]]) if v}
    assert edge and all(not c["included"] for c in r.cells if c["cell"] in edge)
    assert any(c["exclusion_reason"] == "cell touches image edge" for c in r.cells)


def test_whole_cell_options(tmp_path):
    nuc, tgt, truth = _pair(tmp_path, ratio=2.0, seed=22)
    y, x = (int(round(v)) for v in ndi.center_of_mass(truth["nuclei"] == 20))
    x += 17  # a speck of DAPI-positive debris in the cytoplasm of cell 20: no cell of its own
    nuc.data = nuc.data.copy()
    nuc.data[0, y - 1:y + 2, x - 1:x + 2] = 160
    r = analyse_field(nuc, tgt, TranslocationSettings())
    speck = int(r.layers["nuclear_labels"][y, x])
    host = int(r.layers["nuclear_labels"][truth["nuclei"] == 20].max())
    assert speck and speck != host and r.layers["cell_labels"][y, x] == host
    assert next(c for c in r.cells if c["cell"] == speck)["exclusion_reason"] == \
        "too small for a nucleus (no cell outline)"
    # an ImageJ or manual threshold draws the outline where the threshold lies
    lo = analyse_field(nuc, tgt, TranslocationSettings(cell_threshold="manual", cell_threshold_value=8))
    hi = analyse_field(nuc, tgt, TranslocationSettings(cell_threshold="manual", cell_threshold_value=25))
    assert lo.summary["cell_outline_threshold"] == 8 and hi.summary["median_cell_area_px"] < \
        lo.summary["median_cell_area_px"]
    otsu = analyse_field(nuc, tgt, TranslocationSettings(cell_threshold="otsu"))
    assert otsu.summary["median_nc"] == pytest.approx(2.0, rel=0.15)
    # the ring and the territory are still there
    for region in ("ring", "territory"):
        res = analyse_field(nuc, tgt, TranslocationSettings(cytoplasm=region))
        assert res.layers["cell_labels"] is None and res.summary["median_nc"] == pytest.approx(2.0, rel=0.15)
    assert "cell_area_px" not in res.cells[0]


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


# ---------------------------------------------------------------------- nuclear threshold check and fallback


def _hazy_pair(tmp_path, seed=4, ratio=2.0, name="hazy"):
    """A dim, hazy nuclear stain with bright debris: ImageJ 1.39 Auto settles in the background peak."""
    n, t, truth = hazy_translocation_field(ratio=ratio, seed=seed)
    tifffile.imwrite(tmp_path / f"{name}_dapi.tif", n)
    tifffile.imwrite(tmp_path / f"{name}_gfp.tif", t)
    return load_image(tmp_path / f"{name}_dapi.tif"), load_image(tmp_path / f"{name}_gfp.tif"), truth


def test_nuclear_threshold_check():
    rng = np.random.default_rng(1)
    img = 10 + rng.normal(0, 1.0, (200, 200))
    img[50:90, 50:90] += 20  # one bright square: 4% of the field
    region = np.ones(img.shape, bool)
    good = nuclear_threshold_check(img, 20.0, region)
    assert not good["failed"] and good["mask_fraction"] == pytest.approx(0.04, abs=0.002)
    assert good["background"] == pytest.approx(10, abs=0.3) and good["noise_sd"] == pytest.approx(1.0, rel=0.2)
    assert good["snr"] == pytest.approx(10, rel=0.2)
    noise = nuclear_threshold_check(img, 11.0, region)
    assert noise["failed"] and "background noise" in noise["reason"]
    assert nuclear_threshold_check(img, 40.0, region)["reason"] == "finds almost no nuclei"
    img[:120] += 20  # bright over more than half of the field
    assert "% of the field in the nuclei" in nuclear_threshold_check(img, 25.0, region)["reason"]


@pytest.mark.parametrize("seed", [4, 7])
def test_failed_nuclear_threshold_falls_back_to_isodata(tmp_path, seed):
    nuc, tgt, truth = _hazy_pair(tmp_path, seed=seed)
    off = analyse_field(nuc, tgt, TranslocationSettings(nuclear_threshold_fallback="none"))
    assert off.summary["nuclear_threshold_failed"] and not off.summary["segmentation_fallback"]
    assert off.summary["segmentation_snr"] < 1  # ImageJ Auto lies in the background peak
    assert any(w.startswith("nuclei could not be found reliably") for w in off.warnings)
    assert abs(off.summary["median_nc"] - 2.0) > 0.5  # and N/C is wrong
    r = analyse_field(nuc, tgt, TranslocationSettings())  # default: fallback "isodata"
    assert r.summary["nuclear_threshold_failed"] and r.summary["segmentation_fallback"]
    assert r.summary["segmentation_method"] == "isodata, smoothing 4 px" and r.summary["segmentation_snr"] > 2.5
    assert any("nuclei found with 'isodata' on the stain smoothed with 4 px instead" in w for w in r.warnings)
    assert r.summary["n_cells_analysed"] >= 0.7 * truth["n_cells"]
    assert r.summary["median_nc"] == pytest.approx(2.0, rel=0.05)


def test_fallback_leaves_a_good_threshold_alone(tmp_path):
    nuc, tgt, _ = _pair(tmp_path, ratio=2.0, seed=5, shape=(512, 512), spacing=56)
    r = analyse_field(nuc, tgt, TranslocationSettings())
    off = analyse_field(nuc, tgt, TranslocationSettings(nuclear_threshold_fallback="none"))
    assert r.summary["nuclear_threshold_failed"] is False and r.summary["segmentation_fallback"] is False
    assert r.summary["segmentation_method"] == "ij139_auto, smoothing 2 px"
    assert r.summary["median_nc"] == off.summary["median_nc"] and r.cells == off.cells
    # as in a run where another field needed the fallback: tried first, and used where it works
    p = analyse_field(nuc, tgt, TranslocationSettings(), prefer_fallback=True)
    assert p.summary["segmentation_fallback"] and p.summary["nuclear_threshold_failed"] is None
    assert p.summary["median_nc"] == pytest.approx(2.0, rel=0.08)
    # a manual threshold has no fallback; an unknown fallback method is an error
    s = TranslocationSettings(nuclear_threshold="manual", nuclear_threshold_value=40)
    assert not analyse_field(nuc, tgt, s, prefer_fallback=True).summary["segmentation_fallback"]
    with pytest.raises(ValueError, match="nuclear_threshold_fallback"):
        analyse_field(nuc, tgt, TranslocationSettings(nuclear_threshold_fallback="isodta"))


def test_run_measures_every_field_alike_once_one_needs_the_fallback(tmp_path):
    from microscount.bio.experiment import combine_result_folders, segmentation_methods
    from microscount.bio.report import run_translocation
    from microscount.bio.translocation import FieldSpec

    nuc, tgt, _ = _hazy_pair(tmp_path, seed=4)
    specs = [FieldSpec(str(tmp_path / "hazy_dapi.tif"), str(tmp_path / "hazy_gfp.tif"), "stimulus", "1", "1")]
    for k in (1, 2):
        n, t, _ = translocation_field(shape=(512, 512), ratio=1.0, seed=10 + k, spacing=56)
        tifffile.imwrite(tmp_path / f"c{k}_dapi.tif", n)
        tifffile.imwrite(tmp_path / f"c{k}_gfp.tif", t)
        specs.append(FieldSpec(str(tmp_path / f"c{k}_dapi.tif"), str(tmp_path / f"c{k}_gfp.tif"), "vehicle", str(k),
                               "1"))
    s = TranslocationSettings(control_condition="vehicle")
    # the vehicle fields alone keep ImageJ Auto
    alone = run_translocation(specs[1:], s, tmp_path / "vehicle", save_overlays=False)
    assert {r.summary["segmentation_method"] for r in alone["results"]} == {"ij139_auto, smoothing 2 px"}
    assert not alone["experiment"].notes
    # with the hazy field, every field is found with IsoData
    res = run_translocation(specs, s, tmp_path / "joint", save_overlays=False)
    assert not res["errors"]
    rows = {r.field_id + r.condition: r.summary for r in res["results"]}
    assert {v["segmentation_method"] for v in rows.values()} == {"isodata, smoothing 4 px"}
    assert rows["1stimulus"]["nuclear_threshold_failed"] is True
    assert rows["1vehicle"]["nuclear_threshold_failed"] is False  # its own threshold worked
    v1 = analyse_field(load_image(specs[1].nuclear_path), load_image(specs[1].target_path), s, prefer_fallback=True)
    assert rows["1vehicle"]["median_nc"] == v1.summary["median_nc"]
    note = next(n for n in res["experiment"].notes if n.startswith("Nuclei (boundary 1)"))
    assert "failed in 1 of 3 fields" in note and "the nuclei of every field were found with 'isodata'" in note
    doc = yaml.safe_load((tmp_path / "joint" / "settings.yaml").read_text(encoding="utf-8"))
    assert doc["notes"] == res["experiment"].notes and doc["settings"]["nuclear_threshold_fallback"] == "isodata"
    assert "segmentation_method" in (tmp_path / "joint" / "per_field.csv").read_text(encoding="utf-8").splitlines()[0]
    # result folders found with different thresholds are flagged when combined, also folders from before 0.5.2
    s2 = TranslocationSettings(control_condition="vehicle", nuclear_threshold_fallback="none")
    run_translocation(specs[:1], s2, tmp_path / "hazy", save_overlays=False)
    exp, _, _ = combine_result_folders([tmp_path / "vehicle", tmp_path / "joint"], repetitions=["1", "2"])
    assert any("not all found alike" in n and "isodata, smoothing 4 px (3 fields)" in n for n in exp.notes)
    f = tmp_path / "hazy" / "per_field.csv"
    lines = f.read_text(encoding="utf-8").splitlines()
    k = lines[0].split(",").index("segmentation_method")
    f.write_text("\n".join(",".join(c for j, c in enumerate(x.split(",")) if j != k) for x in lines), encoding="utf-8")
    assert segmentation_methods(tmp_path / "hazy", []) == {}
    assert segmentation_methods(tmp_path / "hazy", [{"method": "per_cell"}]) == {"ij139_auto, smoothing 2 px": 1}
    exp, _, _ = combine_result_folders([tmp_path / "vehicle", tmp_path / "hazy"], repetitions=["1", "2"])
    assert not any("not all found alike" in n for n in exp.notes)


def test_small_nuclei_are_flagged():
    from microscount.bio.report import _flag_small_nuclei
    from microscount.bio.translocation import FieldResult

    def field(i, d):
        return FieldResult(str(i), "c", "", "", "per_cell", {"nucleus_diameter_px": d}, [], [], repetition="1")

    rs = [field(1, 20.0), field(2, 22.0), field(3, 21.0), field(4, 5.1)]
    notes = _flag_small_nuclei(rs)
    assert [bool(r.warnings) for r in rs] == [False, False, False, True]
    assert "diameter 5.1 px, against a median of 20.5 px" in rs[3].warnings[0]
    assert len(notes) == 1 and "field 4 of c" in notes[0]
    assert _flag_small_nuclei([field(1, 20.0), field(2, 5.0)]) == []  # too few fields to tell


def test_only_a_field_with_cells_switches_the_run():
    from microscount.bio.report import _same_nuclei_everywhere
    from microscount.bio.translocation import FieldResult

    def field(i, cells, fallback, failed):
        return FieldResult(str(i), "c", "", "", "per_cell", {"n_cells_analysed": cells, "segmentation_method": "x",
                           "segmentation_fallback": fallback, "nuclear_threshold_failed": failed}, [], [],
                           repetition="1")

    def never(*a, **k):
        raise AssertionError("no field should be analysed again")

    # an empty field with a speck of debris, rescued: the other fields keep their own threshold
    rs = [field(1, 2, True, True), field(2, 700, False, False), field(3, 650, False, False)]
    notes = _same_nuclei_everywhere(rs, [(None, r.field_id) for r in rs], never, TranslocationSettings(), [])
    assert len(notes) == 1 and "too few cells" in notes[0] and "field 1 of c" in notes[0]
    # a field of cells, rescued: the others are analysed again with the fallback first
    calls = []

    def again(spec, fid, prefer_fallback=False):
        calls.append((fid, prefer_fallback))
        return field(fid, 690, True, None)

    rs = [field(1, 400, True, True), field(2, 700, False, False), field(3, 650, False, False)]
    notes = _same_nuclei_everywhere(rs, [(None, r.field_id) for r in rs], again, TranslocationSettings(), [])
    assert calls == [("2", True), ("3", True)]
    assert rs[1].summary["segmentation_fallback"] and rs[1].summary["nuclear_threshold_failed"] is False
    assert len(notes) == 1 and "failed in 1 of 3 fields" in notes[0] and "the nuclei of every field" in notes[0]
