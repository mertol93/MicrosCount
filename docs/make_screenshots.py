"""Screenshots for the README, made from synthetic images and the SEM_Porosity.m example images
(python docs/make_screenshots.py)."""

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from microscount.synthetic import translocation_field  # noqa: E402

OUT = Path(__file__).parent / "images"
DATA = Path(__file__).parents[1] / "tests" / "data"
CONDITIONS = {"vehicle": 1.0, "stimulus 30 min": 2.6, "stimulus 60 min": 2.1}
PIXEL_UM = 0.459  # the 'Resolution' of the SEM_Porosity.m example images


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


def _font(size):
    import matplotlib

    try:
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), size)
    except OSError:
        return ImageFont.load_default()


def with_data_bar(img: np.ndarray) -> np.ndarray:
    """Append a 40-row SEM data bar: acquisition details and a 50 µm scale bar."""
    h, w = img.shape
    bar = Image.new("L", (w, 40), 0)
    d = ImageDraw.Draw(bar)
    font = _font(11)
    d.text((8, 5), f"HV 5.00 kV   WD 10.0 mm   HFW {w * PIXEL_UM:.0f} µm   ETD", fill=235, font=font)
    n = round(50 / PIXEL_UM)
    d.rectangle([w - 12 - n, 28, w - 12, 32], fill=255)
    d.text((w - 12 - n, 13), "50 µm", fill=235, font=font)
    return np.vstack([img, np.asarray(bar)])


def sem_experiment(root: Path) -> Path:
    """Two membranes (samples) from two batches (repetitions), three images each, cut from the example
    images; batch 1 as SEM TIFFs with a data bar and the pixel size in their metadata."""
    import tifffile

    from microscount.core.imageio import load_image

    src = {"neat": load_image(DATA / "SEM1.jpg").data[0], "filled": load_image(DATA / "SEM2.jpg").data[0]}
    rng = np.random.default_rng(5)
    for rep in (1, 2):
        d = root / f"batch-{rep}"
        d.mkdir(parents=True)
        for name, g in src.items():
            for k in (1, 2, 3):
                t = g[::-1] if (k + rep) % 2 else g
                t = t[:, ::-1] if k == 2 else t
                y, x = int(rng.integers(0, t.shape[0] - 300)), int(rng.integers(0, t.shape[1] - 400))
                img = np.ascontiguousarray(t[y:y + 300, x:x + 400])
                stem = f"Membrane_{name}_{k:02d}"
                if rep == 1:
                    meta = (f"[Beam]\r\nHV=5000\r\n[Scan]\r\nPixelWidth={PIXEL_UM * 1e-6:.4e}\r\n"
                            f"PixelHeight={PIXEL_UM * 1e-6:.4e}\r\n[Stage]\r\nWorkingDistance=0.0100\r\n"
                            "[Detectors]\r\nName=ETD\r\nMode=SE\r\n[Image]\r\nResolutionX=400\r\nResolutionY=300\r\n")
                    tifffile.imwrite(d / f"{stem}.tif", with_data_bar(img), extratags=[(34682, "s", 0, meta, True)])
                else:
                    Image.fromarray(img).save(d / f"{stem}.png")
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
        q.add_paths([str(sem_experiment(Path(tmp) / "sem"))])
        settle(q)
        q.pixel.setValue(PIXEL_UM)  # default for the PNGs, which record no pixel size
        q.reference.setCurrentIndex(q.reference.findData("neat"))
        q.table.selectRow(next(r for r, s in enumerate(q.specs) if s.path.endswith("Membrane_filled_02.tif")))
        q.run_preview()
        settle(q)
        q.viewer.fit()
        settle()
        w.grab().save(str(OUT / "porosity_preview.png"))
        q.out_edit.setText(str(Path(tmp) / "sem_results"))
        q.run_all()
        settle(q)
        area = q.findChild(QScrollArea)
        area.ensureWidgetVisible(q.btn_combine)  # show the Samples and Run panels
        area.horizontalScrollBar().setValue(0)
        settle()
        w.grab().save(str(OUT / "porosity_results.png"))
    w.close()
    print("screenshots written to", OUT)


if __name__ == "__main__":
    main()
