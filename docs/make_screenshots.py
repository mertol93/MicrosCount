"""Screenshots for the README, made from synthetic images (python docs/make_screenshots.py)."""

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from microscount.synthetic import translocation_field  # noqa: E402

OUT = Path(__file__).parent / "images"
CONDITIONS = {"vehicle": 1.0, "stimulus 30 min": 2.6, "stimulus 60 min": 2.1}


def experiment(root: Path) -> Path:
    """Two repetitions of three conditions, three fields each, named like a microscope export."""
    rng = np.random.default_rng(7)
    seed = 100
    for rep in (1, 2):
        d = root / f"Experiment-{rep}"
        d.mkdir(parents=True)
        for cond, ratio in CONDITIONS.items():
            for k in (1, 2, 3):
                seed += 1
                r = ratio * rng.uniform(0.9, 1.1) * (1.0 if rep == 1 else 0.93)
                n, t, _ = translocation_field(shape=(640, 640), ratio=r, seed=seed, noise=3.0)
                for img, c, ch in ((n, 2, "ch00"), (t, 1, "ch01")):
                    rgb = np.zeros(img.shape + (3,), np.uint8)
                    rgb[..., c] = img
                    Image.fromarray(rgb).save(d / f"Exp {rep}_{cond}-{k}_{ch}.tif")
    return root


def main():
    from PySide6.QtWidgets import QApplication

    import microscount.gui.bio_translocation as tp
    import microscount.gui.materials_porosity as pp
    from microscount.gui.app import MainWindow, _light_palette

    tp.message = pp.message = lambda *a, **k: print("dialog:", a[1:])
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    w = MainWindow()
    w.resize(1400, 880)
    w.show()

    def settle(page=None, timeout=300):
        t0 = time.time()
        while page is not None and page.task is not None and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.02)
        for _ in range(30):
            app.processEvents()

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = experiment(Path(tmp) / "data")
        p = w.trans
        p.add_paths([str(root)])
        settle(p)
        p.control.setCurrentIndex(p.control.findData("vehicle"))
        row = next(r for r, (kind, i) in enumerate(p.rows) if kind == "pair" and p.pairs[i].condition == "stimulus 30 min")
        p.table.selectRow(row)
        p.run_preview()
        settle(p)
        p.view_combo.setCurrentIndex(p.view_combo.findData("composite"))
        settle()
        p.viewer.fit()
        settle()
        w.grab().save(str(OUT / "translocation_preview.png"))
        p.out_edit.setText(str(Path(tmp) / "results"))
        p.run_all()
        settle(p)
        from PySide6.QtWidgets import QScrollArea

        area = p.findChild(QScrollArea)
        area.ensureWidgetVisible(p.cb_paper_qc)  # show the Experiment panel
        area.horizontalScrollBar().setValue(0)
        settle()
        w.grab().save(str(OUT / "translocation_results.png"))
        w.show_analysis("porosity")
        q = w.poro
        q.add_paths([str(Path(__file__).parents[1] / "tests" / "data" / "SEM1.jpg")])
        settle(q)
        q.table.selectRow(0)
        q.run_preview()
        settle(q)
        w.grab().save(str(OUT / "porosity_preview.png"))
    w.close()
    print("screenshots written to", OUT)


if __name__ == "__main__":
    main()
