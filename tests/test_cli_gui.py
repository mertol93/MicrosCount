import os

import numpy as np
import pytest
from PIL import Image

from microscount.cli import main
from microscount.synthetic import sem_image, translocation_field


def test_cli_translocation_and_porosity(tmp_path):
    n, t, _ = translocation_field(seed=21)
    Image.fromarray(n).save(tmp_path / "w1_dapi.png")
    Image.fromarray(t).save(tmp_path / "w1_gfp.png")
    assert main(["translocation", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "per_cell.csv").exists()
    assert "please cite" in (tmp_path / "out" / "CITATION.txt").read_text().lower()
    sem, _ = sem_image(seed=3)
    Image.fromarray(sem).save(tmp_path / "sem.jpg", quality=95)
    assert main(["porosity", str(tmp_path / "sem.jpg"), "--n-thresholds", "2", "--out", str(tmp_path / "po")]) == 0
    assert (tmp_path / "po" / "porosity_summary.csv").exists()


def test_selftest():
    assert main(["selftest"]) == 0


def test_gui_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    import time

    from PySide6.QtWidgets import QApplication

    import microscount.gui.translocation_page as tp
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

    p = w.trans
    p.add_paths([str(tmp_path)])
    wait(p)
    assert len(p.pairs) == 1
    p.table.selectRow(0)
    p.run_preview()
    wait(p)
    assert p.preview is not None and p.preview.summary["n_cells_analysed"] > 20
    p.out_edit.setText(str(tmp_path / "gui_out"))
    p.run_all()
    wait(p)
    assert (tmp_path / "gui_out" / "per_field.csv").exists()
    assert not dialogs
    w.close()
