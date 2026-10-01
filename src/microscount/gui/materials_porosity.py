"""SEM porosity page."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QMenu, QProgressBar, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..materials.porosity import PorositySettings, analyse_sem
from ..core.render import jet_map, label_colours, porosity_overlay
from .common import (
    IMAGE_FILTER, DropTable, FitImageLabel, ImageViewer, LogView, MplCanvas, Task, expand_paths, fill_table, message, open_folder,
    start_task,
)

IMAGE_COLS = [("image", "Image"), ("condition", "Condition"), ("porosity_percent", "Porosity (%)"),
              ("n_pores", "Pores"), ("mean_pore_radius_um", "Mean radius (µm)"), ("sd_pore_radius_um", "SD (µm)"),
              ("median_pore_radius_um", "Median radius (µm)"), ("pore_threshold", "Pore threshold")]
COND_COLS = [("condition", "Condition"), ("n_images", "Images"), ("porosity_mean", "Porosity (mean)"),
             ("porosity_sd", "SD"), ("mean_pore_radius_um_mean", "Mean radius (µm)"),
             ("mean_pore_radius_um_sd", "SD (µm)"), ("pooled_n_pores", "Pores (all images)")]
VIEWS = [("Overlay", "overlay"), ("Original", "grey"), ("Depth map (MATLAB style)", "depth"),
         ("Binary segmentation", "binary"), ("Pore segmentation", "pores")]


class PorosityPage(QWidget):
    status = Signal(str)
    analysis = "porosity"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[list[str]] = []  # [path, condition]
        self.preview = None
        self.task: Task | None = None
        self.last_out: Path | None = None
        self._build()
        self.apply_settings(PorositySettings())

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)

        g1 = QGroupBox("1   SEM images")
        v1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        for text, slot in (("Add files…", self.add_files), ("Add folder…", self.add_folder),
                           ("Remove", self.remove_selected), ("Clear", self.clear_all)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        v1.addLayout(row)
        self.table = DropTable(0, 2)
        self.table.setHorizontalHeaderLabels(["Image", "Condition"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(150)
        self.table.dropped.connect(self.add_paths)
        self.table.itemChanged.connect(self._item_changed)
        self.table.cellDoubleClicked.connect(lambda r, c: c == 0 and self.run_preview())
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        v1.addWidget(self.table)
        row = QHBoxLayout()
        b = QPushButton("Set condition…")
        b.clicked.connect(self.set_condition)
        row.addWidget(b)
        row.addStretch(1)
        v1.addLayout(row)
        hint = QLabel("Greyscale SEM images in which pores are the darkest regions (tick 'Pores are bright' otherwise). "
                      "Crop away the microscope's data bar if the image has one.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#52514e")
        v1.addWidget(hint)
        lv.addWidget(g1)

        g2 = QGroupBox("2   Settings")
        f = QFormLayout(g2)
        self.pixel = QDoubleSpinBox()
        self.pixel.setRange(0.00001, 100000)
        self.pixel.setDecimals(5)
        self.pixel.setSuffix(" µm/px")
        self.pixel.setToolTip("Size of one pixel (MATLAB script: 'Resolution'). Read it from the SEM scale bar.")
        f.addRow("Pixel size:", self.pixel)
        self.nthr = QSpinBox()
        self.nthr.setRange(1, 8)
        self.nthr.setToolTip("Number of multithresh levels (MATLAB script: 'N'). If porosity looks overestimated, "
                             "increase it, and vice versa.")
        f.addRow("Number of thresholds (N):", self.nthr)
        self.search = QComboBox()
        self.search.addItem("MATLAB-compatible (as published)", "matlab")
        self.search.addItem("Exhaustive (global optimum)", "exhaustive")
        self.search.setToolTip("For N >= 3 MATLAB's multithresh uses fminsearch, which can stop at a local optimum. "
                               "'MATLAB-compatible' reproduces the published numbers exactly.")
        f.addRow("Threshold search:", self.search)
        self.pclass = QSpinBox()
        self.pclass.setRange(1, 8)
        self.pclass.setToolTip("How many of the darkest intensity classes count as pore (MATLAB script: 1).")
        f.addRow("Dark classes counted as pore:", self.pclass)
        self.crop = QSpinBox()
        self.crop.setRange(0, 10000)
        self.crop.setSuffix(" px")
        self.crop.setToolTip("Rows removed at the bottom of every image (SEM data bar).")
        f.addRow("Crop at bottom:", self.crop)
        self.minarea = QSpinBox()
        self.minarea.setRange(1, 100000)
        self.minarea.setSuffix(" px")
        self.minarea.setToolTip("Smaller pores are ignored (MATLAB: bwareaopen 9).")
        f.addRow("Minimum pore area:", self.minarea)
        self.cb_split = QCheckBox("Separate touching pores (watershed)")
        self.cb_edge = QCheckBox("Leave pores cut by the image edge out of size statistics")
        self.cb_bright = QCheckBox("Pores are bright (invert)")
        f.addRow(self.cb_split)
        f.addRow(self.cb_edge)
        f.addRow(self.cb_bright)
        self.bins = QSpinBox()
        self.bins.setRange(5, 200)
        f.addRow("Histogram bins:", self.bins)
        lv.addWidget(g2)

        g3 = QGroupBox("3   Run")
        v3 = QVBoxLayout(g3)
        orow = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("automatic: a new folder next to the images")
        orow.addWidget(QLabel("Output folder:"))
        orow.addWidget(self.out_edit, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self.browse_out)
        orow.addWidget(b)
        v3.addLayout(orow)
        row = QHBoxLayout()
        self.btn_preview = QPushButton("Preview selected image")
        self.btn_preview.clicked.connect(self.run_preview)
        self.btn_run = QPushButton("Analyse all images")
        self.btn_run.setStyleSheet("font-weight:600")
        self.btn_run.clicked.connect(self.run_all)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        v3.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        v3.addWidget(self.progress)
        lv.addWidget(g3)
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(430)
        split.addWidget(scroll)

        self.tabs = QTabWidget()
        pv = QWidget()
        pvl = QVBoxLayout(pv)
        bar = QHBoxLayout()
        self.view_combo = QComboBox()
        for label, key in VIEWS:
            self.view_combo.addItem(label, key)
        self.view_combo.currentIndexChanged.connect(self.render_preview)
        bar.addWidget(QLabel("Show:"))
        bar.addWidget(self.view_combo)
        bar.addStretch(1)
        b = QPushButton("Fit")
        b.clicked.connect(lambda: self.viewer.fit())
        bar.addWidget(b)
        b = QPushButton("100%")
        b.clicked.connect(lambda: self.viewer.actual_size())
        bar.addWidget(b)
        pvl.addLayout(bar)
        self.viewer = ImageViewer()
        self.viewer.set_placeholder("Select an image and press Preview")
        pvl.addWidget(self.viewer, 1)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        pvl.addWidget(self.info)
        self.hist = MplCanvas()
        pvl.addWidget(self.hist)
        self.tabs.addTab(pv, "Preview")
        rs = QWidget()
        rsl = QVBoxLayout(rs)
        top = QHBoxLayout()
        self.res_label = QLabel("No results yet.")
        self.res_label.setWordWrap(True)
        top.addWidget(self.res_label, 1)
        self.btn_open = QPushButton("Open results folder")
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(lambda: self.last_out and open_folder(self.last_out))
        top.addWidget(self.btn_open)
        rsl.addLayout(top)
        rsl.addWidget(QLabel("<b>Per condition</b>"))
        self.cond_table = QTableWidget()
        self.cond_table.setMaximumHeight(110)
        rsl.addWidget(self.cond_table)
        rsl.addWidget(QLabel("<b>Per image</b>"))
        self.img_table = QTableWidget()
        rsl.addWidget(self.img_table, 1)
        self.fig_label = FitImageLabel()
        rsl.addWidget(self.fig_label, 2)
        cite = QLabel("Publishing these results? Please cite MicrosCount and the SEM method papers "
                      "(Help ▸ How to cite; a CITATION.txt is saved with every result).")
        cite.setStyleSheet("color:#52514e")
        rsl.addWidget(cite)
        self.tabs.addTab(rs, "Results")
        self.logview = LogView()
        self.tabs.addTab(self.logview, "Log")
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([480, 900])

    # ------------------------------------------------------------------ settings
    def get_settings(self) -> PorositySettings:
        return PorositySettings(
            pixel_size_um=self.pixel.value(), n_thresholds=self.nthr.value(), threshold_search=self.search.currentData(),
            pore_classes=self.pclass.value(), crop_bottom_px=self.crop.value(), min_pore_area_px=self.minarea.value(),
            split_pores=self.cb_split.isChecked(), exclude_edge_pores=self.cb_edge.isChecked(),
            pores_are_bright=self.cb_bright.isChecked(), histogram_bins=self.bins.value(),
        )

    def apply_settings(self, s: PorositySettings):
        self.pixel.setValue(s.pixel_size_um)
        self.nthr.setValue(int(s.n_thresholds))
        i = self.search.findData(s.threshold_search)
        self.search.setCurrentIndex(max(0, i))
        self.pclass.setValue(int(s.pore_classes))
        self.crop.setValue(int(s.crop_bottom_px))
        self.minarea.setValue(int(s.min_pore_area_px))
        self.cb_split.setChecked(s.split_pores)
        self.cb_edge.setChecked(s.exclude_edge_pores)
        self.cb_bright.setChecked(s.pores_are_bright)
        self.bins.setValue(int(s.histogram_bins))

    def reset_defaults(self):
        self.apply_settings(PorositySettings())

    # ------------------------------------------------------------------ files
    def log(self, text):
        self.logview.log(text)
        self.status.emit(text)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add SEM images", "", IMAGE_FILTER)
        if paths:
            self.add_paths(paths)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add a folder of SEM images")
        if d:
            self.add_paths([d])

    def add_paths(self, paths):
        have = {p for p, _ in self.items}
        new = [p for p in expand_paths(paths) if p not in have]
        for p in new:
            self.items.append([p, Path(p).parent.name])
        if new:
            self.log(f"Added {len(new)} image(s).")
        self._refresh()

    def load_inputs(self, inputs: list[dict]):
        self.items = [[d["image"], d.get("condition", "")] for d in inputs if Path(d.get("image", "")).exists()]
        self._refresh()

    def _refresh(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.items))
        for r, (p, c) in enumerate(self.items):
            it = QTableWidgetItem(Path(p).name)
            it.setToolTip(p)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(c))
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self.btn_run.setText(f"Analyse all images ({len(self.items)})" if self.items else "Analyse all images")
        if self.items and not self.table.selectedItems():
            self.table.selectRow(0)

    def _item_changed(self, item):
        if item.column() == 1 and item.row() < len(self.items):
            self.items[item.row()][1] = item.text().strip()

    def _rows(self):
        return sorted({i.row() for i in self.table.selectedItems()})

    def remove_selected(self):
        rows = set(self._rows())
        self.items = [x for i, x in enumerate(self.items) if i not in rows]
        self._refresh()

    def clear_all(self):
        self.items = []
        self._refresh()
        self.preview = None
        self.viewer.set_image(None)
        self.info.setText("")
        self.hist.clear()

    def set_condition(self):
        rows = self._rows()
        if not rows:
            message(self, "Set condition", "Select one or more images first.")
            return
        text, ok = QInputDialog.getText(self, "Set condition", "Condition label for the selected images:")
        if ok:
            for r in rows:
                self.items[r][1] = text.strip()
            self._refresh()

    def _menu(self, pos):
        r = self.table.rowAt(pos.y())
        if r < 0:
            return
        self.table.selectRow(r)
        m = QMenu(self)
        m.addAction("Preview", self.run_preview)
        m.addAction("Set condition…", self.set_condition)
        m.addAction("Remove", self.remove_selected)
        m.exec(self.table.viewport().mapToGlobal(pos))

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.out_edit.setText(d)

    # ------------------------------------------------------------------ tasks
    def _busy(self):
        if self.task is not None:
            message(self, "Busy", "Please wait for the current task to finish (or press Cancel).")
            return True
        return False

    def _start(self, task, on_done, label):
        self.task = task
        self.btn_cancel.setEnabled(True)
        self.btn_run.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat(label + "…")
        task.signals.progress.connect(self._on_progress)
        task.signals.finished.connect(lambda res: self._done(res, on_done))
        task.signals.failed.connect(self._failed)
        start_task(task)

    def _on_progress(self, i, n, msg):
        if n > 0:
            self.progress.setRange(0, n)
            self.progress.setValue(i)
            self.progress.setFormat(f"{i}/{n}  {msg}")
        self.status.emit(msg)

    def _finish(self):
        self.task = None
        self.btn_cancel.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("Ready")

    def _done(self, res, on_done):
        self._finish()
        on_done(res)

    def _failed(self, tb):
        self._finish()
        self.log(tb)
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        message(self, "Something went wrong", last + "\n\nDetails are in the Log tab.")

    def cancel(self):
        if self.task:
            self.task.cancel()
            self.log("Cancelling after the current image…")

    def run_preview(self):
        rows = self._rows()
        if not rows:
            message(self, "Preview", "Add SEM images and select one first.")
            return
        if self._busy():
            return
        path, cond = self.items[rows[0]]
        s = self.get_settings()

        def job():
            from ..core.imageio import load_image

            return analyse_sem(load_image(path), s, cond, keep_layers=True)

        self.viewer.set_placeholder("Analysing…")
        self._start(Task(job, pass_progress=False), self._preview_done, f"Analysing {Path(path).name}")

    def _preview_done(self, res):
        self.preview = res
        self.tabs.setCurrentIndex(0)
        self.render_preview(keep=False)
        S = res.summary
        text = (f"<b>{html.escape(res.name)}</b> &nbsp; porosity <b>{100 * S['porosity']:.2f}%</b>; {S['n_pores']} pores; "
                f"mean equivalent radius <b>{S['mean_pore_radius_um']:.3f} µm</b> (SD {S['sd_pore_radius_um']:.3f}); "
                f"thresholds {S['thresholds']} (pores ≤ {S['pore_threshold']:g})")
        for w in res.warnings:
            text += f"<br><span style='color:#c05800'>⚠ {html.escape(w)}</span>"
        self.info.setText(text)
        r = np.array([p["equivalent_radius_um"] for p in res.pores if p["included"]])
        self.hist.histogram(r, "Equivalent pore radius (µm)", S["mean_pore_radius_um"],
                            f"mean {S['mean_pore_radius_um']:.2f} µm", bins=self.bins.value())
        self.log(f"Preview {res.name}: porosity {S['porosity']:.4f}, {S['n_pores']} pores")

    def render_preview(self, *_, keep=True):
        if self.preview is None or self.preview.layers is None:
            return
        L = self.preview.layers
        key = self.view_combo.currentData()
        if key == "grey":
            g = L["grey"].astype(np.uint8) if L["grey"].dtype == np.uint8 else (255 * (L["grey"] / max(1, L["grey"].max()))).astype(np.uint8)
            rgb = np.repeat(g[..., None], 3, axis=2)
        elif key == "depth":
            rgb = jet_map(L["grey"])
        elif key == "binary":
            rgb = np.repeat((L["solid"].astype(np.uint8) * 255)[..., None], 3, axis=2)
        elif key == "pores":
            rgb = label_colours(L["pore_labels"])
        else:
            rgb = porosity_overlay(L["grey"], L["pore_labels"], L["valid"])
        self.viewer.set_image(rgb, keep_view=keep)

    def run_all(self):
        if not self.items:
            message(self, "Nothing to analyse", "Add SEM images first.")
            return
        if self._busy():
            return
        from ..materials.report import default_output_dir, run_porosity

        out = Path(self.out_edit.text().strip()) if self.out_edit.text().strip() else default_output_dir([self.items[0][0]])
        s = self.get_settings()
        self.log(f"Analysing {len(self.items)} image(s) → {out}")
        self._start(Task(run_porosity, [tuple(x) for x in self.items], s, out), self._run_done, "Analysing")

    def _run_done(self, res):
        out = res["out_dir"]
        self.last_out = out
        self.btn_open.setEnabled(True)
        fill_table(self.img_table, [r.summary for r in res["results"]], IMAGE_COLS)
        fill_table(self.cond_table, res["conditions"], COND_COLS)
        fig = res["files"].get("figure")
        if fig and Path(fig).exists():
            self.fig_label.set_file(fig)
        n_err = len([e for e in res["errors"] if e.get("image")])
        self.res_label.setText(f"<b>{len(res['results'])}</b> image(s) analysed" + (f", <span style='color:#c05800'>"
                               f"{n_err} failed (see Log)</span>" if n_err else "") + f".<br>Saved to {out}")
        for e in res["errors"]:
            self.log(f"Error: {e}")
        self.log(f"Done. Results in {out}")
        self.tabs.setCurrentIndex(1)
