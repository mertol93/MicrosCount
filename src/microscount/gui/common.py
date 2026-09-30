"""Shared GUI pieces: image viewer, background tasks, small dialogs and helpers."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

from .. import APP_NAME, __version__
from ..imageio import SUPPORTED_EXTENSIONS

IMAGE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg *.TIF *.TIFF *.PNG *.JPG *.JPEG);;All files (*)"
REPO_URL = "https://github.com/mertol93/microscount"


def resource(name: str) -> Path:
    return Path(__file__).resolve().parent / "resources" / name


# ---------------------------------------------------------------------- background tasks


class TaskSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    """Run ``fn(*args, progress=..., cancel=..., **kw)`` on the thread pool."""

    def __init__(self, fn, *args, pass_progress: bool = True, **kwargs):
        super().__init__()
        self.setAutoDelete(False)
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.pass_progress = pass_progress
        self.signals = TaskSignals()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        try:
            kw = dict(self.kwargs)
            if self.pass_progress:
                kw["progress"] = lambda i, n, m: self.signals.progress.emit(int(i), int(n), str(m))
                kw["cancel"] = self.cancelled
            res = self.fn(*self.args, **kw)
        except Exception:  # noqa: BLE001
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.finished.emit(res)


def start_task(task: Task) -> Task:
    QThreadPool.globalInstance().start(task)
    return task


# ---------------------------------------------------------------------- image viewer


class ImageViewer(QGraphicsView):
    """Zoom (wheel), pan (drag), fit (double-click); nearest-neighbour when zoomed in."""

    hovered = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.FastTransformation)
        self._scene.addItem(self._item)
        self._buf = None
        self._fitted = True
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#1f1f1f"))
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self._placeholder = QLabel("Select a field and press Preview", self)
        self._placeholder.setStyleSheet("color: #bdbcb6; font-size: 13px;")
        self._placeholder.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._placeholder.setGeometry(0, 0, self.width(), self.height())
        if self._fitted and self._buf is not None:
            self.fit()

    def set_placeholder(self, text: str):
        self._placeholder.setText(text)

    def set_image(self, rgb: np.ndarray | None, keep_view: bool = False):
        if rgb is None:
            self._item.setPixmap(QPixmap())
            self._buf = None
            self._placeholder.show()
            return
        self._placeholder.hide()
        rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
        h, w = rgb.shape[:2]
        self._buf = rgb
        img = QImage(self._buf.data, w, h, 3 * w, QImage.Format_RGB888)
        self._item.setPixmap(QPixmap.fromImage(img.copy()))
        self._scene.setSceneRect(0, 0, w, h)
        if not keep_view:
            self.fit()

    def fit(self):
        if self._buf is None:
            return
        self.fitInView(self._item, Qt.KeepAspectRatio)
        self._fitted = True

    def actual_size(self):
        self.resetTransform()
        self._fitted = False

    def wheelEvent(self, e):
        if self._buf is None:
            return
        f = 1.25 ** (e.angleDelta().y() / 120.0)
        cur = self.transform().m11()
        if (cur * f > 40 and f > 1) or (cur * f < 0.02 and f < 1):
            return
        self.scale(f, f)
        self._fitted = False

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        if self._buf is not None:
            p = self.mapToScene(e.position().toPoint())
            x, y = int(math.floor(p.x())), int(math.floor(p.y()))
            if 0 <= x < self._buf.shape[1] and 0 <= y < self._buf.shape[0]:
                self.hovered.emit(x, y)


# ---------------------------------------------------------------------- helpers


def open_folder(path: str | Path):
    path = str(path)
    if not os.path.exists(path):
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def message(parent, title: str, text: str, icon=QMessageBox.Information):
    box = QMessageBox(icon, title, text, QMessageBox.Ok, parent)
    box.exec()


def fill_table(table: QTableWidget, rows: list[dict], columns: list[tuple[str, str]], digits: int = 4):
    table.clear()
    table.setColumnCount(len(columns))
    table.setRowCount(len(rows))
    table.setHorizontalHeaderLabels([c[1] for c in columns])
    for i, r in enumerate(rows):
        for j, (key, _) in enumerate(columns):
            v = r.get(key, "")
            if isinstance(v, float):
                text = "" if not math.isfinite(v) else (f"{v:.{digits}g}" if abs(v) < 1e5 else f"{v:.4e}")
            else:
                text = str(v)
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(i, j, item)
    table.resizeColumnsToContents()


def expand_paths(paths: list[str]) -> list[str]:
    from ..imageio import list_images

    out = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out += [str(x) for x in list_images(pp)]
        elif pp.suffix.lower() in SUPPORTED_EXTENSIONS and pp.is_file():
            out.append(str(pp))
    seen = set()
    return [p for p in out if not (p in seen or seen.add(p))]


class DropTable(QTableWidget):
    """Table that accepts dropped files and folders."""

    dropped = Signal(list)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QTableWidget.DropOnly)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
            self.dropped.emit(paths)
            e.acceptProposedAction()
        else:
            super().dropEvent(e)


class ScaleBarDialog(QDialog):
    """Pixel size from a burned-in scale bar of known length."""

    def __init__(self, parent, length_px: float | None):
        super().__init__(parent)
        self.setWindowTitle("Pixel size from scale bar")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.px = QDoubleSpinBox()
        self.px.setRange(0, 100000)
        self.px.setDecimals(1)
        self.px.setValue(length_px or 0)
        self.um = QDoubleSpinBox()
        self.um.setRange(0, 1e6)
        self.um.setDecimals(3)
        self.um.setValue(100.0)
        self.result_label = QLabel()
        form.addRow("Scale bar length (pixels):", self.px)
        form.addRow("Scale bar length (µm):", self.um)
        form.addRow("Pixel size:", self.result_label)
        if length_px:
            note = QLabel(f"Detected a burned-in scale bar of {length_px:g} px in the selected image.")
        else:
            note = QLabel("No burned-in scale bar was detected; enter its length in pixels.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.px.valueChanged.connect(self._update)
        self.um.valueChanged.connect(self._update)
        self._update()

    def value(self) -> float:
        return self.um.value() / self.px.value() if self.px.value() > 0 else 0.0

    def _update(self):
        v = self.value()
        self.result_label.setText(f"{v:.5g} µm per pixel" if v else "–")


def about_html() -> str:
    from ..citation import METHOD_POROSITY, METHOD_TRANSLOCATION, PREFERRED, SOFTWARE

    por = "<br>".join(METHOD_POROSITY.split("\n"))
    return f"""
    <h2 style='margin-bottom:0'>{APP_NAME} {__version__}</h2>
    <p style='margin-top:4px'>Open-source microscopy image analysis: nuclear translocation (N/C ratio) in
    fluorescence images and porosity / pore-size distribution in SEM images.</p>
    <p><b>Please cite</b> when you publish results obtained with {APP_NAME}:<br>{PREFERRED}</p>
    <p>and the software: {SOFTWARE}</p>
    <p><b>Method references</b><br>Nuclear translocation (paper method): {METHOD_TRANSLOCATION}<br><br>
    SEM porosity (port of A. Rabbani's SEM_Porosity, BSD-3-Clause): <br>{por}</p>
    <p><b>Licence</b>: GNU General Public License v3.0 or later. {APP_NAME} comes with ABSOLUTELY NO WARRANTY.<br>
    Source code: <a href='{REPO_URL}'>{REPO_URL}</a></p>
    <p style='color:#52514e'>Built with Python, NumPy, SciPy, scikit-image, tifffile, Pillow, Matplotlib and
    Qt for Python (PySide6, LGPL-3.0). SEM_Porosity © 2020 Arash Rabbani, BSD-3-Clause.</p>
    """


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from ..citation import BIBTEX, PREFERRED

        self.setWindowTitle(f"About {APP_NAME}")
        self.resize(640, 560)
        lay = QVBoxLayout(self)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        tb.setHtml(about_html())
        lay.addWidget(tb)
        row = QHBoxLayout()
        b1 = QPushButton("Copy citation")
        b1.clicked.connect(lambda: QGuiApplication.clipboard().setText(PREFERRED))
        b2 = QPushButton("Copy BibTeX")
        b2.clicked.connect(lambda: QGuiApplication.clipboard().setText(BIBTEX))
        b3 = QPushButton("Close")
        b3.clicked.connect(self.accept)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch(1)
        row.addWidget(b3)
        lay.addLayout(row)


class FitImageLabel(QLabel):
    """Shows a picture scaled to the available width, keeping its aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pm: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_file(self, path):
        self._pm = QPixmap(str(path))
        self._rescale()

    def clear_picture(self):
        self._pm = None
        self.clear()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pm is None or self._pm.isNull():
            return
        w, h = max(50, self.width() - 8), max(50, self.height() - 8)
        self.setPixmap(self._pm.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class LogView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)

    def log(self, text: str):
        self.appendPlainText(text)


class MplCanvas(QWidget):
    """Small matplotlib canvas (pyplot-free)."""

    def __init__(self, parent=None, height: int = 170):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 1.8), dpi=100)
        self.figure.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def histogram(self, values: np.ndarray, xlabel: str, marker: float | None = None, marker_label: str = "",
                  ref: float | None = None, ref_label: str = "", bins: int = 40):
        from ..reporting import AXIS, GRID, INK, INK2, SERIES, SURFACE

        f = self.figure
        f.clear()
        ax = f.add_subplot(111)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if v.size:
            hi = np.percentile(v, 99.5)
            ax.hist(v[v <= hi] if hi > v.min() else v, bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=0.8)
            if ref is not None:
                ax.axvline(ref, color=INK2, linewidth=0.8)
                ax.annotate(ref_label, xy=(ref, 1), xycoords=("data", "axes fraction"), xytext=(3, -10),
                            textcoords="offset points", fontsize=8, color=INK2)
            if marker is not None and math.isfinite(marker):
                ax.axvline(marker, color=INK, linewidth=1.2)
                ax.annotate(marker_label, xy=(marker, 1), xycoords=("data", "axes fraction"), xytext=(3, -22),
                            textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
        f.tight_layout(pad=0.4)
        self.canvas.draw_idle()

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()
