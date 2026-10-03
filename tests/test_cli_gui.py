
import numpy as np
import pytest
import yaml
from PIL import Image

from microscount.cli import main
from microscount.synthetic import sem_image, translocation_field


def test_cli_translocation_and_porosity(tmp_path):
    n, t, _ = translocation_field(seed=21)
    Image.fromarray(n).save(tmp_path / "w1_dapi.png")
    Image.fromarray(t).save(tmp_path / "w1_gfp.png")
    # default: paper method + per-cell lab protocol
    assert main(["bio", "translocation", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    for f in ("per_field.csv", "per_condition.csv", "per_cell.csv", "histograms.csv", "histograms.png"):
        assert (tmp_path / "out" / f).exists()
    assert "please cite" in (tmp_path / "out" / "CITATION.txt").read_text().lower()
    doc = yaml.safe_load((tmp_path / "out" / "settings.yaml").read_text(encoding="utf-8"))
    assert (doc["module"], doc["analysis"]) == ("bio", "translocation")
    # the 0.1 shortcut, paper method only, protocol options
    assert main(["translocation", str(tmp_path), "--method", "paper", "--out", str(tmp_path / "out2")]) == 0
    assert not (tmp_path / "out2" / "per_cell.csv").exists()
    assert main(["translocation", str(tmp_path), "--size", "20-inf", "--circularity", "0.3-1", "--rolling-ball", "30",
                 "--background", "2.5", "--out", str(tmp_path / "out3")]) == 0
    s = yaml.safe_load((tmp_path / "out3" / "settings.yaml").read_text(encoding="utf-8"))["settings"]
    assert (s["size_min_px2"], s["size_max_px2"], s["circularity_min"], s["rolling_ball_radius"]) == (20, None, 0.3, 30)
    assert (s["background"], s["background_value"]) == ("manual", 2.5)
    assert main(["run", str(tmp_path / "out3" / "settings.yaml"), "--out", str(tmp_path / "out4")]) == 0
    assert (tmp_path / "out4" / "per_cell.csv").exists()
    sem, _ = sem_image(seed=3)
    Image.fromarray(sem).save(tmp_path / "sem.jpg", quality=95)
    assert main(["materials", "porosity", str(tmp_path / "sem.jpg"), "--n-thresholds", "2", "--out",
                 str(tmp_path / "po")]) == 0
    assert (tmp_path / "po" / "per_image.csv").exists() and (tmp_path / "po" / "summary.csv").exists()
    assert main(["porosity", str(tmp_path / "sem.jpg"), "--n-thresholds", "2", "--out", str(tmp_path / "po2")]) == 0


def test_modules_registry(capsys):
    from microscount.modules import MODULES, analysis, settings_class

    assert [m.title for m in MODULES] == ["Bio & Cells", "Materials & Mechanics"]
    assert analysis("translocation").module == "bio" and analysis("porosity").module == "materials"
    assert settings_class("porosity")().n_thresholds == 4
    assert main(["modules"]) == 0
    assert "Materials & Mechanics" in capsys.readouterr().out


def test_selftest():
    assert main(["selftest"]) == 0


def test_gui_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    import time

    from PySide6.QtWidgets import QApplication

    import microscount.gui.bio_translocation as tp
    from microscount.gui.app import MainWindow

    dialogs = []
    monkeypatch.setattr(tp, "message", lambda *a, **k: dialogs.append(a))
    n, t, _ = translocation_field(seed=22)
    Image.fromarray(n).save(tmp_path / "f_dapi.png")
    Image.fromarray(t).save(tmp_path / "f_gfp.png")
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()

    def wait(page):
        t0 = time.time()
        while page.task is not None and time.time() - t0 < 60:
            app.processEvents()
            time.sleep(0.01)
        for _ in range(10):
            app.processEvents()

    assert [w.tabs.tabText(i).replace("&&", "&") for i in range(w.tabs.count())] == ["Bio & Cells", "Materials & Mechanics"]
    p = w.trans
    assert w.page() is p and w.show_analysis("porosity") is w.poro
    w.show_analysis("translocation")
    p.add_paths([str(tmp_path)])
    wait(p)
    assert len(p.pairs) == 1
    p.table.selectRow(0)
    assert p.cb_cell.isChecked()  # paper ratio + per-cell lab protocol by default
    p.run_preview()
    wait(p)
    assert np.isfinite(p.preview.summary["paper_ratio"]) and p.preview.summary["n_cells_analysed"] > 20
    p.view_combo.setCurrentIndex(p.view_combo.findData("target_corrected"))
    p._hover(10, 10)
    p.cb_cell.setChecked(False)
    assert p.get_settings().method == "paper"
    p.run_preview()
    wait(p)
    assert p.preview.method == "paper" and not p.preview.cells
    p.cb_cell.setChecked(True)
    p.out_edit.setText(str(tmp_path / "gui_out"))
    p.run_all()
    wait(p)
    assert (tmp_path / "gui_out" / "per_field.csv").exists()
    assert not dialogs
    w.close()


def test_gui_experiment(tmp_path, monkeypatch):
    """Repetitions from folders, control, comparisons, typed edits kept, run, combine saved results."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    import time

    from PySide6.QtWidgets import QApplication

    import microscount.gui.bio_translocation as tp
    from microscount.gui.app import MainWindow

    dialogs = []
    monkeypatch.setattr(tp, "message", lambda *a, **k: dialogs.append(a))
    root = tmp_path / "data"
    seed = 30
    for rep in ("3", "4"):
        d = root / f"Experiment-{rep}"
        d.mkdir(parents=True)
        for cond, r in (("vehicle", 1.0), ("stimulus 30 min", 2.5)):
            for k in (1, 2):
                seed += 1
                n, t, _ = translocation_field(shape=(160, 160), ratio=r, seed=seed)
                rgb = np.zeros(n.shape + (3,), np.uint8)
                rgb[..., 2] = n
                Image.fromarray(rgb).save(d / f"Exp-{rep}_{cond}-{k}_ch00.tif")
                rgb = np.zeros(n.shape + (3,), np.uint8)
                rgb[..., 1] = t
                Image.fromarray(rgb).save(d / f"Exp-{rep}_{cond}-{k}_ch01.tif")
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    p = w.trans

    def wait():
        t0 = time.time()
        while p.task is not None and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.01)
        for _ in range(10):
            app.processEvents()

    p.add_paths([str(root)])
    wait()
    assert len(p.pairs) == 8 and {q.repetition for q in p.pairs} == {"3", "4"}
    assert {q.condition for q in p.pairs} == {"vehicle", "stimulus 30 min"}
    assert sorted(p.control.itemData(i) for i in range(p.control.count())) == ["", "stimulus 30 min", "vehicle"]
    p.control.setCurrentIndex(p.control.findData("vehicle"))
    monkeypatch.setattr(p, "_choose_conditions", lambda *a: ["stimulus 30 min", "vehicle"])
    p.add_comparison()
    s = p.get_settings()
    assert s.control_condition == "vehicle" and s.comparisons == [["vehicle", "stimulus 30 min"]]
    # a typed repetition survives reading the names again
    r0 = next(r for r, (kind, _i) in enumerate(p.rows) if kind == "pair")
    p.table.item(r0, 2).setText("X")
    p.cond_from.setCurrentIndex(p.cond_from.findData("name"))
    assert "X" in {q.repetition for q in p.pairs}
    p.table.item(r0, 2).setText("3")  # back to its folder's repetition
    # settings round trip
    p.apply_settings(p.get_settings())
    assert p.get_settings().comparisons == s.comparisons and p.get_settings().control_condition == "vehicle"
    out = tmp_path / "outs" / "run"
    p.out_edit.setText(str(out))
    p.run_all()
    wait()
    assert (out / "comparisons.csv").exists() and (out / "summary.csv").exists()
    assert p.comp_table.rowCount() == 3  # two repetitions and all repetitions
    assert p.cond_table.rowCount() == 2  # two conditions across repetitions
    monkeypatch.setattr(tp.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path / "outs"))
    p.combine_saved()
    wait()
    assert list((tmp_path / "outs").glob("microscount_combined_*/summary.csv"))
    assert not dialogs
    w.close()


def test_gui_materials_experiment(tmp_path, monkeypatch):
    """Samples and repetitions from names, pixel size per image, reference, comparison, typed edits, run, combine."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    import time

    import tifffile
    from PySide6.QtWidgets import QApplication

    import microscount.gui.materials_porosity as mp
    from microscount.gui.app import MainWindow

    dialogs = []
    monkeypatch.setattr(mp, "message", lambda *a, **k: dialogs.append(a))
    root = tmp_path / "data"
    seed = 0
    for rep in (1, 2):
        d = root / f"batch-{rep}"
        d.mkdir(parents=True)
        for name, por in (("CA", 0.10), ("CA-CNF1", 0.18)):
            for k in (1, 2):
                seed += 1
                Image.fromarray(sem_image(porosity=por, seed=seed)[0]).save(d / f"Membrane_{name}_5kx_{k:02d}.png")
    fei = "[Scan]\r\nPixelWidth=5e-008\r\nPixelHeight=5e-008\r\n"  # an FEI TIFF records its pixel size
    tifffile.imwrite(root / "batch-1" / "Membrane_CA_5kx_03.tif", sem_image(porosity=0.1, seed=99)[0],
                     extratags=[(34682, "s", 0, fei, True)])
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    p = w.poro

    def wait():
        t0 = time.time()
        while p.task is not None and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.01)
        for _ in range(10):
            app.processEvents()

    p.path_box.edit.setText(f'"{root}"')  # a pasted path, quotes and all: the folder and its subfolders
    p.path_box.submit()
    wait()
    assert len(p.specs) == 9 and {s.repetition for s in p.specs} == {"1", "2"}
    assert {s.sample for s in p.specs} == {"CA", "CA-CNF1"}
    tif = next(r for r, s in enumerate(p.specs) if s.path.endswith(".tif"))
    assert p.table.item(tif, 3).text() == "0.05" and "FEI" in p.table.item(tif, 4).text()
    png = next(r for r, s in enumerate(p.specs) if s.path.endswith(".png"))
    assert p.table.item(png, 4).text() == "unknown"
    p.pixel.setValue(0.02)  # default pixel size for files without one
    assert p.table.item(png, 3).text() == "0.02" and p.table.item(png, 4).text() == "default"
    p.table.item(png, 3).setText("0,03")  # typed pixel size, comma decimal
    assert p.specs[png].pixel_size_um == 0.03 and p.table.item(png, 4).text() == "typed"
    p.table.item(png, 3).setText("")
    assert p.specs[png].pixel_size_um is None
    assert sorted(p.reference.itemData(i) for i in range(p.reference.count())) == ["", "CA", "CA-CNF1"]
    p.reference.setCurrentIndex(p.reference.findData("CA"))
    monkeypatch.setattr(p, "_choose_samples", lambda *a: ["CA-CNF1", "CA"])
    p.add_comparison()
    s = p.get_settings()
    assert s.reference_sample == "CA" and s.comparisons == [["CA", "CA-CNF1"]] and s.pixel_size_um == 0.02
    # a typed sample survives reading the names again
    sample = p.specs[png].sample
    p.table.item(png, 1).setText("X")
    p.samples_from.setCurrentIndex(p.samples_from.findData("folder"))
    assert p.specs[png].sample == "X" and {s.sample for s in p.specs} - {"X"} == {"data"}  # batch-n: repetitions
    assert {s.repetition for s in p.specs} == {"1", "2"}
    p.samples_from.setCurrentIndex(p.samples_from.findData("auto"))
    p.table.item(png, 1).setText(sample)
    assert {s.sample for s in p.specs} == {"CA", "CA-CNF1"}
    p.apply_settings(p.get_settings())  # settings round trip
    assert p.get_settings().comparisons == s.comparisons and p.get_settings().reference_sample == "CA"
    from microscount.materials.porosity import PorositySettings

    p.apply_settings(PorositySettings(samples_from="folder"))  # settings opened without their images
    assert {q.sample for i, q in enumerate(p.specs) if i != png} == {"data"}
    p.apply_settings(s)
    assert {q.sample for q in p.specs} == {"CA", "CA-CNF1"} and p.get_settings().reference_sample == "CA"
    out = tmp_path / "outs" / "run"
    p.out_edit.setText(str(out))
    p.run_all()
    wait()
    for f in ("summary.csv", "per_image.csv", "comparisons.csv", "pore_size_distribution.csv", "results.xlsx"):
        assert (out / f).exists(), f
    assert p.sum_table.rowCount() == 2 and p.img_table.rowCount() == 9 and p.comp_table.rowCount() > 0
    doc = yaml.safe_load((out / "settings.yaml").read_text(encoding="utf-8"))
    assert {d["pixel_size_um"] for d in doc["inputs"]} == {None}  # nothing typed: file or default
    monkeypatch.setattr(mp.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path / "outs"))
    p.combine_saved()
    wait()
    assert list((tmp_path / "outs").glob("microscount_combined_*/summary.csv"))
    assert not dialogs
    w.close()


def test_gui_check_against_traced_pores(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import time

    from PySide6.QtWidgets import QApplication

    import microscount.gui.materials_porosity as mp
    from microscount.gui.app import MainWindow
    from microscount.synthetic import sem_surface

    g, truth = sem_surface(seed=2, porosity=0.05, style="flat")
    Image.fromarray(g).save(tmp_path / "flat.png")
    Image.fromarray(truth.astype(np.uint8) * 255).save(tmp_path / "flat_pores.png")
    shown = []
    monkeypatch.setattr(mp.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(mp.QDialog, "exec", lambda self: shown.append(self) or 0)
    app = QApplication.instance() or QApplication([])
    p = MainWindow().poro
    p.nthr.setValue(2)  # the check uses the settings in the window
    p.check_traced()
    t0 = time.time()
    while p.task is not None and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(10):
        app.processEvents()
    saved = list(tmp_path.glob("microscount_check_*.csv"))
    assert shown and saved
    mean = saved[0].read_text(encoding="utf-8-sig").splitlines()[-1].split(",")
    assert mean[0] == "mean" and float(mean[5]) > 0.9  # overlap with two thresholds on flat pores
