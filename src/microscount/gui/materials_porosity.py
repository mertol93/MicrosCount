"""SEM porosity page (Materials & Mechanics)."""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMenu, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..core.experiment import GroupDesign, order_groups
from ..core.naming import group_key
from ..core.render import EXCLUDED_TINT, _blend, jet_map, label_colours, porosity_overlay
from ..materials.experiment import ImageSpec, assign_samples, design_from_settings, label_images, scan_sem_files
from ..materials.porosity import PorositySettings, analyse_sem
from .common import (
    IMAGE_FILTER, DropTable, FitImageLabel, ImageViewer, LogView, MplCanvas, PathBox, ScaleBarDialog, Task,
    expand_paths, fill_table, message, open_folder, start_task,
)

VIEWS = [("Overlay", "overlay"), ("Original", "grey"), ("Depth map (MATLAB style)", "depth"),
         ("Binary segmentation", "binary"), ("Pore segmentation", "pores")]
SAMPLE_SOURCE_ITEMS = [("Automatic", "auto"), ("File names", "name"), ("Folder names", "folder")]
DATA_BAR_ITEMS = [("Automatic (file or detected)", "auto"), ("None", "none"), ("Manual: rows below", "manual")]
COMP_COLS = [("measure", "Measure"), ("repetition", "Repetition"), ("sample_b", "Sample"), ("sample_a", "compared with"),
             ("mean_b", "Value"), ("mean_a", "Reference value"), ("difference_pct", "Change (%)"),
             ("welch_p", "Welch p (images)"), ("per_repetition", "Per repetition"), ("paired_p", "Paired p (repetitions)")]
GREY = "color:#52514e"


def _dspin(lo, hi, val, dec=2, step=1.0, special="", suffix="", tip=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if special:
        s.setSpecialValueText(special)
    if suffix:
        s.setSuffix(suffix)
    if tip:
        s.setToolTip(tip)
    return s


def _combo(items, tip=""):
    c = QComboBox()
    for label, data in items:
        c.addItem(label, data)
    if tip:
        c.setToolTip(tip)
    return c


def _set_combo(c: QComboBox, data):
    i = c.findData(data)
    if i >= 0:
        c.setCurrentIndex(i)


def _short_source(src: str) -> str:
    """``"FEI / Thermo Fisher metadata"`` -> ``"file (FEI)"``, for the narrow column of the image table."""
    if src in ("typed", "default", "unknown"):
        return src
    name = src.replace(" metadata", "").replace(" calibration", "").replace(" resolution", "")
    return f"file ({name.split(' /')[0]})"


class PorosityPage(QWidget):
    status = Signal(str)
    analysis = "porosity"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.specs: list[ImageSpec] = []
        self.preview = None
        self.task: Task | None = None
        self.last_out: Path | None = None
        self._base = PorositySettings()
        self._typed_sample: dict[str, str] = {}  # path -> sample typed by the user
        self._typed_rep: dict[str, str] = {}
        self._reference = ""
        self._comparisons: list[list[str]] = []
        self._references: dict[str, str] = {}
        self._order: list[str] = []
        self._build()
        self.apply_settings(PorositySettings())

    @property
    def items(self) -> list[list[str]]:  # [path, sample] (version 0.2)
        return [[s.path, s.sample] for s in self.specs]

    # ------------------------------------------------------------------ layout
    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        g1 = QGroupBox("1   SEM images")
        v1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        for text, slot in (("Add files…", self.add_files), ("Add folder…", self.add_folder),
                           ("Remove", self.remove_selected), ("Clear", self.clear_all)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            if text == "Add folder…":
                b.setToolTip("Every image in the folder and in all its subfolders, at any depth")
            row.addWidget(b)
        v1.addLayout(row)
        self.path_box = PathBox()
        self.path_box.paths.connect(self.add_paths)
        v1.addWidget(self.path_box)
        self.table = DropTable(0, 5)
        self.table.setHorizontalHeaderLabels(["Image", "Sample", "Repetition", "Pixel size (µm)", "from"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(170)
        self.table.dropped.connect(self.add_paths)
        self.table.itemChanged.connect(self._item_changed)
        self.table.cellDoubleClicked.connect(lambda r, c: c == 0 and self.run_preview())
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        v1.addWidget(self.table)
        row = QHBoxLayout()
        for text, slot, tip in (("Set sample…", self.set_sample, "Sample of the selected images"),
                                ("Set repetition…", self.set_repetition, "Repetition (independent membrane or batch) of "
                                                                         "the selected images"),
                                ("Set pixel size…", self.set_pixel_size, "Pixel size of the selected images (empty: "
                                                                         "from the file or the default)"),
                                ("From scale bar…", self.pixel_from_scale_bar, "Measure the scale bar in the data bar "
                                                                               "of the selected image")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        v1.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Samples from:"))
        self.samples_from = _combo(SAMPLE_SOURCE_ITEMS,
                                   "Automatic: from the file names when a folder holds several samples (CA_01, "
                                   "CA-CNF1_01 …) and each folder is a repetition; otherwise each folder is a sample and "
                                   "its parent folder a repetition.")
        self.samples_from.currentIndexChanged.connect(self._reassign)
        row.addWidget(self.samples_from)
        row.addStretch(1)
        v1.addLayout(row)
        hint = QLabel("SEM images in which pores are the darkest regions (tick 'Pores are bright' otherwise). A folder "
                      "brings the images of all its subfolders. Samples and repetitions are read from the names; the "
                      "pixel size from the SEM file when it records one. All can be edited in the table.")
        hint.setWordWrap(True)
        hint.setStyleSheet(GREY)
        v1.addWidget(hint)
        lv.addWidget(g1)

        g2 = QGroupBox("2   Settings")
        f = QFormLayout(g2)
        self.pixel = _dspin(0, 100000, 0, 5, 0.01, "unknown", " µm/px",
                            "Pixel size of images whose file does not record one (MATLAB script: 'Resolution'). "
                            "Unknown: sizes in pixels.")
        f.addRow("Default pixel size:", self.pixel)
        self.cb_meta = QCheckBox("Use the pixel size recorded in the file (SEM metadata, calibration)")
        self.cb_meta.setToolTip("FEI / Thermo Fisher and Zeiss SEM TIFFs record the pixel size of every image.")
        self.cb_meta.toggled.connect(self._refresh)
        self.pixel.valueChanged.connect(self._refresh)
        f.addRow(self.cb_meta)
        barrow = QHBoxLayout()
        self.bar = _combo(DATA_BAR_ITEMS, "The SEM data bar at the bottom of the image is left out. Automatic: its "
                                          "height from the file (FEI / Thermo Fisher) or found as a block of flat "
                                          "graphic rows.")
        self.crop = QSpinBox()
        self.crop.setRange(0, 10000)
        self.crop.setSuffix(" px")
        self.bar.currentIndexChanged.connect(self._sync_enabled)
        barrow.addWidget(self.bar, 2)
        barrow.addWidget(self.crop, 1)
        f.addRow("Data bar:", barrow)
        self.nthr = QSpinBox()
        self.nthr.setRange(1, 8)
        self.nthr.setToolTip("Number of multithresh levels (MATLAB script: 'N'). If porosity looks overestimated, "
                             "increase it, and vice versa.")
        f.addRow("Number of thresholds (N):", self.nthr)
        self.search = _combo([("MATLAB-compatible (as published)", "matlab"), ("Exhaustive (global optimum)", "exhaustive")],
                             "For N >= 3 MATLAB's multithresh uses fminsearch, which can stop at a local optimum. "
                             "'MATLAB-compatible' reproduces the published numbers exactly.")
        f.addRow("Threshold search:", self.search)
        throw = QHBoxLayout()
        self.thr_mode = _combo([("Automatic (multithresh)", "auto"), ("Fixed grey level", "fixed")],
                               "A fixed grey level treats every image alike; automatic adapts to each image.")
        self.thr_val = _dspin(0, 1e9, 0, 1, 1)
        self.thr_mode.currentIndexChanged.connect(self._sync_enabled)
        throw.addWidget(self.thr_mode, 2)
        throw.addWidget(self.thr_val, 1)
        f.addRow("Pore threshold:", throw)
        self.pclass = QSpinBox()
        self.pclass.setRange(1, 8)
        self.pclass.setToolTip("How many of the darkest intensity classes count as pore (MATLAB script: 1).")
        f.addRow("Dark classes counted as pore:", self.pclass)
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
        self.bins.setToolTip("Bins of the pore-size distributions.")
        f.addRow("Distribution bins:", self.bins)
        lv.addWidget(g2)

        g3 = QGroupBox("3   Samples")
        f3 = QFormLayout(g3)
        self.reference = QComboBox()
        self.reference.setToolTip("Reference sample (e.g. the unmodified membrane): percent changes and the default "
                                  "comparisons.")
        self.reference.currentIndexChanged.connect(self._reference_changed)
        f3.addRow("Reference sample:", self.reference)
        cw = QWidget()
        cl = QHBoxLayout(cw)
        cl.setContentsMargins(0, 0, 0, 0)
        self.comp_list = QListWidget()
        self.comp_list.setMaximumHeight(86)
        self.comp_list.setToolTip("Each line compares two samples for every measure, in each repetition (Welch t-test "
                                  "on the images, exploratory) and across repetitions (paired t-test with at least 3). "
                                  "Empty: every sample is compared with the reference.")
        cl.addWidget(self.comp_list, 1)
        cb = QVBoxLayout()
        b = QPushButton("Add…")
        b.clicked.connect(self.add_comparison)
        cb.addWidget(b)
        b = QPushButton("Remove")
        b.clicked.connect(self.remove_comparison)
        cb.addWidget(b)
        cb.addStretch(1)
        cl.addLayout(cb)
        f3.addRow("Comparisons:", cw)
        b = QPushButton("Order and references…")
        b.setToolTip("Order of the samples in tables and figures, and the reference of each sample's changes.")
        b.clicked.connect(self.edit_order)
        f3.addRow(b)
        lv.addWidget(g3)

        g4 = QGroupBox("4   Run")
        v4 = QVBoxLayout(g4)
        orow = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("automatic: a new folder next to the images")
        orow.addWidget(QLabel("Output folder:"))
        orow.addWidget(self.out_edit, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self.browse_out)
        orow.addWidget(b)
        v4.addLayout(orow)
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
        v4.addLayout(row)
        row = QHBoxLayout()
        self.btn_combine = QPushButton("Combine saved results…")
        self.btn_combine.setToolTip("One experiment from SEM result folders analysed separately (e.g. one per "
                                    "repetition): choose the folder that holds them.")
        self.btn_combine.clicked.connect(self.combine_saved)
        row.addWidget(self.btn_combine)
        row.addStretch(1)
        v4.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        v4.addWidget(self.progress)
        lv.addWidget(g4)
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(460)
        split.addWidget(scroll)

        self.tabs = QTabWidget()
        pv = QWidget()
        pvl = QVBoxLayout(pv)
        bar = QHBoxLayout()
        self.view_combo = _combo(VIEWS)
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
        legend = QLabel("<span style='color:#ff3c00'>■</span> pores &nbsp; <span style='color:#c8a800'>■</span> pore "
                        "outlines &nbsp; <span style='color:#2828a0'>■</span> left out (data bar, annotations)")
        pvl.addWidget(legend)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
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
        self.sum_title = QLabel("<b>Per sample</b>")
        rsl.addWidget(self.sum_title)
        self.sum_table = QTableWidget()
        self.sum_table.setMaximumHeight(150)
        rsl.addWidget(self.sum_table)
        self.comp_title = QLabel("<b>Comparisons</b>")
        rsl.addWidget(self.comp_title)
        self.comp_table = QTableWidget()
        self.comp_table.setMaximumHeight(130)
        rsl.addWidget(self.comp_table)
        rsl.addWidget(QLabel("<b>Per image</b>"))
        self.img_table = QTableWidget()
        rsl.addWidget(self.img_table, 1)
        self.fig_label = FitImageLabel()
        rsl.addWidget(self.fig_label, 2)
        cite = QLabel("Publishing these results? Please cite MicrosCount and the SEM method papers "
                      "(Help ▸ How to cite; a CITATION.txt is saved with every result).")
        cite.setStyleSheet(GREY)
        rsl.addWidget(cite)
        self.tabs.addTab(rs, "Results")
        self.logview = LogView()
        self.tabs.addTab(self.logview, "Log")
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([500, 900])

    # ------------------------------------------------------------------ settings
    def _sync_enabled(self, *_):
        self.crop.setEnabled(self.bar.currentData() == "manual")
        self.thr_val.setEnabled(self.thr_mode.currentData() == "fixed")

    def get_settings(self) -> PorositySettings:
        import copy

        s = copy.deepcopy(self._base)  # keeps settings that have no widget (e.g. from a loaded file)
        s.pixel_size_um = self.pixel.value() or None
        s.use_metadata_pixel_size = self.cb_meta.isChecked()
        s.n_thresholds = self.nthr.value()
        s.threshold_search = self.search.currentData()
        s.pore_threshold_value = self.thr_val.value() if self.thr_mode.currentData() == "fixed" else None
        s.pore_classes = self.pclass.value()
        s.data_bar = self.bar.currentData()
        s.crop_bottom_px = self.crop.value()
        s.min_pore_area_px = self.minarea.value()
        s.split_pores = self.cb_split.isChecked()
        s.exclude_edge_pores = self.cb_edge.isChecked()
        s.pores_are_bright = self.cb_bright.isChecked()
        s.histogram_bins = self.bins.value()
        s.samples_from = self.samples_from.currentData()
        s.reference_sample = self._reference
        s.references = dict(self._references)
        s.comparisons = [list(c) for c in self._comparisons]
        s.sample_order = list(self._order)
        return s

    def apply_settings(self, s: PorositySettings):
        import copy

        self._base = copy.deepcopy(s)
        source = self.samples_from.currentData()
        for w in (self.pixel, self.cb_meta, self.samples_from):
            w.blockSignals(True)
        self.pixel.setValue(s.pixel_size_um or 0)
        self.cb_meta.setChecked(bool(s.use_metadata_pixel_size))
        _set_combo(self.samples_from, s.samples_from or "auto")
        for w in (self.pixel, self.cb_meta, self.samples_from):
            w.blockSignals(False)
        if self.specs and self.samples_from.currentData() != source:
            self._assign()  # the images already listed, read with the new source
        self.nthr.setValue(int(s.n_thresholds))
        _set_combo(self.search, s.threshold_search)
        _set_combo(self.thr_mode, "fixed" if s.pore_threshold_value is not None else "auto")
        self.thr_val.setValue(s.pore_threshold_value or 0)
        self.pclass.setValue(int(s.pore_classes))
        _set_combo(self.bar, s.data_bar)
        self.crop.setValue(int(s.crop_bottom_px or 0))
        self.minarea.setValue(int(s.min_pore_area_px))
        self.cb_split.setChecked(s.split_pores)
        self.cb_edge.setChecked(s.exclude_edge_pores)
        self.cb_bright.setChecked(s.pores_are_bright)
        self.bins.setValue(int(s.histogram_bins))
        self._reference = s.reference_sample or ""
        self._references = dict(s.references or {})
        self._comparisons = [list(c) for c in (s.comparisons or [])]
        self._order = list(s.sample_order or [])
        self._sync_enabled()
        self._sync_design()
        self._refresh()

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
        have = {s.path for s in self.specs}
        new = [p for p in expand_paths(paths) if p not in have]
        if not new:
            if paths:
                self.log("No new TIFF/PNG/JPEG images found.")
            return
        if self._busy():
            return
        self.log(f"Reading {len(new)} image(s)…")
        self._start(Task(scan_sem_files, new), self._scanned, "Reading images")

    def _scanned(self, specs: list[ImageSpec]):
        self.specs += specs
        for s in specs:
            if s.file_pixel_size_um:
                self.log(f"{Path(s.path).name}: {s.file_pixel_size_um:.5g} µm/px ({s.file_pixel_size_source})")
        self._assign()
        self._refresh()

    def load_inputs(self, inputs: list[dict]):
        """Restore images saved in a settings.yaml."""
        self.clear_all()
        specs = []
        for d in inputs:
            p = d.get("image", "")
            if not Path(p).exists():
                self.log(f"missing file: {p}")
                continue
            sp = scan_sem_files([p])[0]
            sp.sample = d.get("sample") or d.get("condition", "") or ""
            sp.repetition = str(d.get("repetition") or "1")
            sp.pixel_size_um = d.get("pixel_size_um")
            self._typed_sample[p] = sp.sample
            self._typed_rep[p] = sp.repetition
            specs.append(sp)
        self.specs = specs
        label_images(self.specs)
        self._sync_design()
        self._refresh()

    def _assign(self):
        if not self.specs:
            self._sync_design()
            return
        used = assign_samples(self.specs, self.samples_from.currentData() or "auto")
        for s in self.specs:
            s.sample = self._typed_sample.get(s.path, s.sample)
            s.repetition = self._typed_rep.get(s.path, s.repetition)
        label_images(self.specs)
        reps = {s.repetition for s in self.specs}
        self.log(f"Samples from {'file' if used == 'name' else 'folder'} names: "
                 f"{len({group_key(s.sample) for s in self.specs})} sample(s) in {len(reps)} repetition(s).")
        self._sync_design()

    def _reassign(self, *_):
        self._assign()
        self._refresh()

    def samples(self) -> list[str]:
        return order_groups([s.sample for s in self.specs], self._order or ([self._reference] if self._reference else []))

    def _pixel(self, s: ImageSpec) -> tuple[float | None, str]:
        """As ``analyse_sem``: typed, the file's calibration, the default, a bare TIFF resolution tag."""
        from ..core.imageio import TIFF_RESOLUTION

        file_px = s.file_pixel_size_um if self.cb_meta.isChecked() else None
        weak = s.file_pixel_size_source == TIFF_RESOLUTION
        if s.pixel_size_um:
            return s.pixel_size_um, "typed"
        if file_px and not (weak and self.pixel.value()):
            return file_px, s.file_pixel_size_source
        if self.pixel.value():
            return self.pixel.value(), "default"
        return None, "unknown"

    def _refresh(self, *_):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.specs))
        for r, s in enumerate(self.specs):
            px, src = self._pixel(s)
            vals = [Path(s.path).name, s.sample, s.repetition, f"{px:.5g}" if px else "", _short_source(src)]
            tips = [s.path, "Double-click to edit", "Double-click to edit",
                    "Double-click to type the pixel size of this image (empty: from the file or the default)",
                    f"Pixel size: {src}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setToolTip(tips[c])
                if c not in (1, 2, 3):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if c == 4 and v == "unknown":
                    it.setForeground(QColor("#c05800"))
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self.btn_run.setText(f"Analyse all images ({len(self.specs)})" if self.specs else "Analyse all images")
        if self.specs and not self.table.selectedItems():
            self.table.selectRow(0)

    def _item_changed(self, item):
        r, c = item.row(), item.column()
        if r >= len(self.specs) or c not in (1, 2, 3):
            return
        s = self.specs[r]
        text = item.text().strip()
        if c == 1:
            s.sample = self._typed_sample[s.path] = text
        elif c == 2:
            s.repetition = self._typed_rep[s.path] = text or "1"
        else:
            try:
                v = float(text.replace(",", ".")) if text else None
            except ValueError:
                v = s.pixel_size_um
            s.pixel_size_um = v if v and v > 0 else None
        label_images(self.specs)
        self._sync_design()
        self._refresh()

    def _rows(self):
        return sorted({i.row() for i in self.table.selectedItems()})

    def remove_selected(self):
        rows = set(self._rows())
        self.specs = [x for i, x in enumerate(self.specs) if i not in rows]
        label_images(self.specs)
        self._sync_design()
        self._refresh()

    def clear_all(self):
        self.specs = []
        self._typed_sample.clear()
        self._typed_rep.clear()
        self._refresh()
        self._sync_design()
        self.preview = None
        self.viewer.set_image(None)
        self.info.setText("")
        self.hist.clear()

    def _ask_selected(self, title, label) -> tuple[list[int], str | None]:
        rows = self._rows()
        if not rows:
            message(self, title, "Select one or more images first.")
            return [], None
        text, ok = QInputDialog.getText(self, title, label)
        return rows, (text.strip() if ok else None)

    def set_sample(self):
        rows, text = self._ask_selected("Set sample", "Sample of the selected images:")
        if text is not None:
            for r in rows:
                s = self.specs[r]
                s.sample = self._typed_sample[s.path] = text
            label_images(self.specs)
            self._sync_design()
            self._refresh()

    def set_repetition(self):
        rows, text = self._ask_selected("Set repetition", "Repetition (independent membrane or batch) of the selected "
                                                          "images:")
        if text:
            for r in rows:
                s = self.specs[r]
                s.repetition = self._typed_rep[s.path] = text
            label_images(self.specs)
            self._refresh()

    def set_pixel_size(self):
        rows, text = self._ask_selected("Set pixel size", "Pixel size of the selected images in µm (empty: from the file "
                                                          "or the default):")
        if text is None:
            return
        try:
            v = float(text.replace(",", ".")) if text else None
        except ValueError:
            message(self, "Set pixel size", f"'{text}' is not a number.")
            return
        for r in rows:
            self.specs[r].pixel_size_um = v if v and v > 0 else None
        self._refresh()

    def pixel_from_scale_bar(self):
        from ..core.imageio import load_image
        from ..materials.porosity import detect_data_bar, grey_from_image, measure_data_bar_scale

        rows = self._rows()
        if not rows:
            message(self, "Pixel size from scale bar", "Select one or more images taken at the same magnification.")
            return
        s = self.specs[rows[0]]
        length = None
        try:
            img = load_image(s.path)
            grey, _ = grey_from_image(img)
            bar = (img.metadata or {}).get("data_bar_px") or detect_data_bar(grey)
            length = measure_data_bar_scale(grey, bar) if bar else None
        except Exception:  # noqa: BLE001
            length = None
        dlg = ScaleBarDialog(self, length)
        dlg.um.setValue(1.0 if length else dlg.um.value())
        if dlg.exec() and dlg.value() > 0:
            for r in rows:
                self.specs[r].pixel_size_um = dlg.value()
            self.log(f"Pixel size {dlg.value():.5g} µm/px for {len(rows)} image(s)")
            self._refresh()

    def _menu(self, pos):
        r = self.table.rowAt(pos.y())
        if r < 0:
            return
        if r not in self._rows():
            self.table.selectRow(r)
        m = QMenu(self)
        m.addAction("Preview", self.run_preview)
        m.addAction("Set sample…", self.set_sample)
        m.addAction("Set repetition…", self.set_repetition)
        m.addAction("Set pixel size…", self.set_pixel_size)
        m.addAction("Pixel size from scale bar…", self.pixel_from_scale_bar)
        m.addSeparator()
        m.addAction("Remove", self.remove_selected)
        m.exec(self.table.viewport().mapToGlobal(pos))

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.out_edit.setText(d)

    # ------------------------------------------------------------------ samples
    def _sync_design(self):
        samples = self.samples()
        self.reference.blockSignals(True)
        self.reference.clear()
        self.reference.addItem("(none)", "")
        for g in samples:
            self.reference.addItem(g, g)
        if self._reference and group_key(self._reference) not in {group_key(g) for g in samples}:
            self.reference.addItem(f"{self._reference} (not among the images)", self._reference)
        idx = next((i for i in range(self.reference.count())
                    if group_key(self.reference.itemData(i) or "") == group_key(self._reference)), 0)
        self.reference.setCurrentIndex(idx if self._reference else 0)
        self.reference.blockSignals(False)
        self.comp_list.clear()
        for a, b in self._comparisons:
            self.comp_list.addItem(f"{b}   vs   {a}")
        if not self._comparisons:
            self.comp_list.addItem("(each sample vs the reference)" if self._reference else "(none: set a reference or add)")

    def _reference_changed(self, *_):
        self._reference = self.reference.currentData() or ""
        self._sync_design()

    def _choose_samples(self, title: str, labels: tuple[str, str]) -> list[str] | None:
        samples = self.samples()
        if len(samples) < 2:
            message(self, title, "Add images of at least two samples first.")
            return None
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        form = QFormLayout(dlg)
        boxes = []
        for i, lab in enumerate(labels):
            c = QComboBox()
            c.addItems(samples)
            c.setCurrentIndex(1 if i == 0 else 0)
            form.addRow(lab, c)
            boxes.append(c)
        if self._reference in samples:
            boxes[1].setCurrentIndex(samples.index(self._reference))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if not dlg.exec():
            return None
        return [b.currentText() for b in boxes]

    def add_comparison(self):
        got = self._choose_samples("Add a comparison", ("Sample:", "compared with:"))
        if got and got[0] != got[1]:
            self._comparisons.append([got[1], got[0]])
            self._sync_design()

    def remove_comparison(self):
        for r in sorted({self.comp_list.row(i) for i in self.comp_list.selectedItems()}, reverse=True):
            if r < len(self._comparisons):
                self._comparisons.pop(r)
        self._sync_design()

    def edit_order(self):
        samples = self.samples()
        if not samples:
            message(self, "Samples", "Add images first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Order and references")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Order of the samples, and what each sample's changes are relative to."))
        table = QTableWidget(len(samples), 2)
        table.setHorizontalHeaderLabels(["Sample", "Changes relative to"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        refs = {group_key(k): val for k, val in self._references.items()}

        def fill(order):
            for r, g in enumerate(order):
                it = QTableWidgetItem(g)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                table.setItem(r, 0, it)
                box = QComboBox()
                box.addItem("the reference sample", "")
                for o in samples:
                    box.addItem(o, o)
                box.setCurrentIndex(max(0, box.findData(refs.get(group_key(g), ""))))
                box.currentIndexChanged.connect(lambda _i, g=g, box=box: refs.__setitem__(group_key(g), box.currentData()))
                table.setCellWidget(r, 1, box)
            table.resizeColumnsToContents()

        order = list(samples)
        fill(order)
        v.addWidget(table)
        row = QHBoxLayout()

        def move(d):
            r = table.currentRow()
            if r >= 0 and 0 <= r + d < len(order):
                order[r], order[r + d] = order[r + d], order[r]
                fill(order)
                table.selectRow(r + d)

        for text, d in (("Move up", -1), ("Move down", 1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, d=d: move(d))
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.resize(560, 360)
        if dlg.exec():
            self._order = list(order)
            by_key = {group_key(g): g for g in samples}
            self._references = {by_key[k]: r for k, r in refs.items() if r and k in by_key}
            self._sync_design()

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
        sp = self.specs[rows[0]]
        s = self.get_settings()

        def job():
            from ..core.imageio import load_image

            return analyse_sem(load_image(sp.path), s, sp.sample, keep_layers=True, pixel_size_um=sp.pixel_size_um,
                               repetition=sp.repetition, image_id=sp.image_id)

        self.viewer.set_placeholder("Analysing…")
        self._start(Task(job, pass_progress=False), self._preview_done, f"Analysing {Path(sp.path).name}")

    def _preview_done(self, res):
        self.preview = res
        self.tabs.setCurrentIndex(0)
        self.render_preview(keep=False)
        S = res.summary
        known = math.isfinite(S["pixel_size_um"])
        unit, sfx = ("µm", "um") if known else ("px", "px")
        key = f"equivalent_diameter_{sfx}"
        md = S[f"mean_pore_diameter_{sfx}"]

        def f(v, d=3):
            return "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}g}"

        parts = [f"<b>{html.escape(res.image_id or res.name)}</b> &nbsp; {html.escape(res.name)}",
                 f"Porosity <b>{f(S['porosity_percent'], 4)}%</b>; {S['n_pores']} pores; mean equivalent diameter "
                 f"<b>{f(md)} {unit}</b> (median {f(S[f'median_pore_diameter_{sfx}'])}, "
                 f"d10–d90 {f(S[f'd10_pore_diameter_{sfx}'])}–{f(S[f'd90_pore_diameter_{sfx}'])} {unit}); "
                 f"threshold {S['pore_threshold']:g} ({html.escape(S['threshold_source'])}; levels {S['thresholds']})",
                 f"Pixel size {f(S['pixel_size_um'], 5)} µm ({html.escape(S['pixel_size_source'])}); data bar "
                 f"{S['data_bar_px']} px ({html.escape(S['data_bar_source'])})"]
        if S.get("instrument"):
            extra = [f"{S['voltage_kv']:g} kV" if "voltage_kv" in S else "",
                     f"WD {S['working_distance_mm']:.3g} mm" if "working_distance_mm" in S else "",
                     S.get("detector", ""), S.get("magnification", "")]
            parts.append(html.escape(S["instrument"] + ": " + ", ".join(x for x in extra if x)))
        for w in res.warnings:
            parts.append(f"<span style='color:#c05800'>⚠ {html.escape(w)}</span>")
        self.info.setText("<br>".join(parts))
        d = np.array([p[key] for p in res.pores if p["included"]])
        self.hist.histogram(d, f"Equivalent pore diameter ({unit})", md, f"mean {f(md)} {unit}", bins=self.bins.value())
        self.log(f"Preview {res.name}: porosity {S['porosity']:.4f}, {S['n_pores']} pores")

    def render_preview(self, *_, keep=True):
        if self.preview is None or self.preview.layers is None:
            return
        L = self.preview.layers
        key = self.view_combo.currentData()
        g = L["grey"]
        if key == "grey":
            g8 = g.astype(np.uint8) if g.dtype == np.uint8 else (255 * (g / max(1, g.max()))).astype(np.uint8)
            rgb = np.repeat(g8[..., None], 3, axis=2)
        elif key == "depth":
            rgb = jet_map(g)
        elif key == "binary":
            rgb = np.repeat((L["solid"].astype(np.uint8) * 255)[..., None], 3, axis=2)
        elif key == "pores":
            rgb = label_colours(L["pore_labels"])
        else:
            rgb = porosity_overlay(g, L["pore_labels"], L["valid"])
        crop = int(L.get("crop") or 0)
        if crop:  # show the data bar that was left out
            full = L["grey_full"]
            bar = full[-crop:]
            b8 = bar.astype(np.uint8) if bar.dtype == np.uint8 else (255 * (bar / max(1, full.max()))).astype(np.uint8)
            brgb = np.repeat(b8[..., None], 3, axis=2)
            _blend(brgb, np.ones(bar.shape, bool), EXCLUDED_TINT, 0.55)
            rgb = np.vstack([rgb, brgb])
        self.viewer.set_image(rgb, keep_view=keep)

    def _specs_for_run(self) -> list[ImageSpec]:
        return [ImageSpec(s.path, s.sample, s.repetition or "1", s.image_id, s.field_no, s.pixel_size_um)
                for s in self.specs]

    def run_all(self):
        if not self.specs:
            message(self, "Nothing to analyse", "Add SEM images first.")
            return
        if self._busy():
            return
        from ..materials.report import default_output_dir, run_porosity

        out = Path(self.out_edit.text().strip()) if self.out_edit.text().strip() else default_output_dir([self.specs[0].path])
        s = self.get_settings()
        self.log(f"Analysing {len(self.specs)} image(s) → {out}")
        self._start(Task(run_porosity, self._specs_for_run(), s, out), self._run_done, "Analysing")

    def _run_done(self, res):
        n_err = len([e for e in res["errors"] if e.get("image")])
        head = f"<b>{len(res['results'])}</b> image(s) analysed" + (
            f", <span style='color:#c05800'>{n_err} failed (see Log)</span>" if n_err else "")
        self._show_results(res, head)
        for e in res["errors"]:
            self.log(f"Error: {e}")
        for r in res["results"]:
            for w in r.warnings:
                self.log(f"{r.image_id}: {w}")
        self.log(f"Done. Results in {res['out_dir']}")

    def _show_results(self, res, head: str):
        out = res["out_dir"]
        self.last_out = out
        self.btn_open.setEnabled(True)
        ms = res["experiment"]
        several = len(ms.repetitions) > 1
        over = f"mean ± SD of {len(ms.repetitions)} repetitions" if several else "mean ± SD of images"
        self.sum_title.setText(f"<b>Per sample</b> ({over}; change relative to the reference)")
        cols = [("sample", "Sample"), ("n_images", "Images")] + ([("n_repetitions", "Repetitions")] if several else [])
        for m in ms.measures:
            short = {"Porosity": "Porosity", "Mean pore diameter": "Mean d", "Median pore diameter": "Median d",
                     "Pore density": "Density"}.get(m.label, m.label)
            cols += [(f"{m.key}_mean", f"{short} ({m.unit})"), (f"{m.key}_sd", "SD")]
        cols += [(f"{ms.measures[0].key}_change_pct", "Porosity change (%)"),
                 (f"{ms.measures[1].key}_change_pct", "Mean d change (%)"), ("reference", "relative to")]
        fill_table(self.sum_table, ms.summary, cols)
        comps = [c for c in ms.comparisons if c["key"] in (ms.measures[0].key, ms.measures[1].key)]
        self.comp_title.setVisible(bool(comps))
        self.comp_table.setVisible(bool(comps))
        if comps:
            fill_table(self.comp_table, comps, COMP_COLS)
        unit = ms.size_unit
        sfx = "um" if unit == "µm" else "px"
        img_cols = [("image_id", "Image"), ("sample", "Sample"), ("repetition", "Repetition"),
                    ("porosity_percent", "Porosity (%)"), ("n_pores", "Pores"),
                    (f"mean_pore_diameter_{sfx}", f"Mean d ({unit})"), (f"median_pore_diameter_{sfx}", f"Median d ({unit})"),
                    ("pixel_size_um", "Pixel size (µm)"), ("pixel_size_source", "from"), ("data_bar_px", "Data bar (px)"),
                    ("warnings", "Warnings"), ("image", "File")]
        fill_table(self.img_table, res.get("image_rows", []), img_cols)
        fig = res["files"].get("figure")
        if fig and Path(fig).exists():
            self.fig_label.set_file(fig)
        notes = "".join(f"<br><span style='{GREY}'>{html.escape(n)}</span>" for n in ms.notes)
        self.res_label.setText(f"{head}.<br>Saved to {html.escape(str(out))}{notes}")
        self.tabs.setCurrentIndex(1)

    def combine_saved(self):
        if self._busy():
            return
        d = QFileDialog.getExistingDirectory(self, "Folder holding the SEM result folders to combine")
        if not d:
            return
        from ..core.report import find_result_folders

        root = Path(d)
        found = find_result_folders(root, "per_image.csv")
        if not found:
            message(self, "Combine saved results", "No MicrosCount SEM result folders (with per_image.csv) were found there.")
            return
        import datetime as _dt

        from ..materials.report import combine_results

        s = self.get_settings()
        design: GroupDesign = design_from_settings(s)
        out = root / ("microscount_combined_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.log(f"Combining {len(found)} result folder(s): " + ", ".join(f.name for f in found))
        self._start(Task(combine_results, [str(f) for f in found], design, out, None, s.histogram_bins,
                         pass_progress=False),
                    lambda res: self._show_results(res, f"<b>{len(found)}</b> result folder(s) combined"), "Combining")
