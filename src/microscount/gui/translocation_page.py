"""Nuclear translocation page."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMenu, QProgressBar, QPushButton, QRadioButton,
    QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from ..pairing import FileEntry, Pairing, pair_files, scan_files
from ..render import translocation_overlay
from ..thresholds import METHOD_LABELS
from ..translocation import FieldSpec, TranslocationSettings, analyse_field
from .common import (
    IMAGE_FILTER, DropTable, FitImageLabel, ImageViewer, LogView, MplCanvas, ScaleBarDialog, Task, expand_paths, fill_table,
    message, open_folder, start_task,
)

CHANNELS = [("Automatic", "auto"), ("Blue", "blue"), ("Green", "green"), ("Red", "red"),
            ("Channel 1 (index 0)", "ch0"), ("Channel 2 (index 1)", "ch1"), ("Channel 3 (index 2)", "ch2"),
            ("Channel 4 (index 3)", "ch3")]
BACKGROUND = [("Automatic (histogram mode)", "auto"), ("None", "none"), ("Manual value", "manual")]
CYTOPLASM = [("Perinuclear ring", "ring"), ("Cell territory", "territory")]

FIELD_COLS = [("field", "Field"), ("condition", "Condition"), ("n_cells_analysed", "Cells"),
              ("median_ratio", "Median N/C"), ("q1_ratio", "Q1"), ("q3_ratio", "Q3"),
              ("responder_fraction", "Fraction N/C > cut-off"), ("paper_ratio", "Paper-method ratio"),
              ("background", "Background"), ("nuclear_file", "Nuclear image"), ("target_file", "Target image")]
PAPER_FIELD_COLS = [("field", "Field"), ("condition", "Condition"), ("paper_ratio", "N/C ratio"),
                    ("paper_nuclear_mean", "Nuclear mean"), ("paper_cytoplasm_mean", "Cytoplasmic mean"),
                    ("nuclear_threshold", "Nuclear threshold"), ("target_threshold", "Target threshold"),
                    ("nuclear_file", "Nuclear image"), ("target_file", "Target image")]
COND_COLS = [("condition", "Condition"), ("n_fields", "Fields"), ("n_cells", "Cells"),
             ("pooled_median_ratio", "Median N/C (all cells)"), ("field_median_ratio_mean", "Mean of field medians"),
             ("field_median_ratio_sd", "SD"), ("responder_fraction", "Fraction N/C > cut-off"),
             ("paper_ratio_mean", "Paper-method ratio (mean)"), ("paper_ratio_sd", "SD ")]


def _combo(items, tooltip=""):
    c = QComboBox()
    for label, data in items:
        c.addItem(label, data)
    if tooltip:
        c.setToolTip(tooltip)
    return c


def _set_combo(c: QComboBox, data):
    i = c.findData(data)
    if i >= 0:
        c.setCurrentIndex(i)


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


class TranslocationPage(QWidget):
    status = Signal(str)
    module = "translocation"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries: dict[str, FileEntry] = {}
        self.pairs: list[Pairing] = []
        self.unpaired: list[FileEntry] = []
        self.rows: list[tuple[str, int]] = []
        self.preview = None
        self.task: Task | None = None
        self.last_out: Path | None = None
        self._build()
        self.apply_settings(TranslocationSettings())

    # ------------------------------------------------------------------ layout
    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        g1 = QGroupBox("1   Images")
        v1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        for text, slot in (("Add files…", self.add_files), ("Add folder…", self.add_folder),
                           ("Remove", self.remove_selected), ("Clear", self.clear_all)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        v1.addLayout(row)
        self.table = DropTable(0, 5)
        self.table.setHorizontalHeaderLabels(["Field", "Condition", "Nuclear stain image", "Target image", "Pairing"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(170)
        self.table.dropped.connect(self.add_paths)
        self.table.itemChanged.connect(self._item_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda r, c: self.run_preview())
        v1.addWidget(self.table)
        row = QHBoxLayout()
        b = QPushButton("Set condition…")
        b.clicked.connect(self.set_condition)
        row.addWidget(b)
        b = QPushButton("Swap channels")
        b.setToolTip("Swap nuclear-stain and target images of the selected fields")
        b.clicked.connect(self.swap_selected)
        row.addWidget(b)
        row.addStretch(1)
        v1.addLayout(row)
        hint = QLabel("Drop TIFF, PNG or JPEG files or folders here. Blue exports are used as the nuclear stain and "
                      "green/red as the target; matching files are paired by name or, failing that, by image content. "
                      "Right-click a row to change a pairing.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#52514e")
        v1.addWidget(hint)
        lv.addWidget(g1)

        g2 = QGroupBox("2   Method")
        v2 = QVBoxLayout(g2)
        self.rb_cell = QRadioButton("Per-cell N/C ratio (recommended)")
        self.rb_cell.setToolTip("Each nucleus is segmented; its cytoplasm is a ring around it, independent of the target "
                                "intensity. Background is subtracted. Reports the distribution of per-cell ratios.")
        self.rb_paper = QRadioButton("Paper method: Noursadeghi et al. (2008), whole field")
        self.rb_paper.setToolTip("Exact re-implementation of the published ImageJ procedure (3x3 median, ImageJ Default "
                                 "threshold of both channels, cytoplasm = target mask minus nuclear mask), one ratio per "
                                 "field. Validated pixel-for-pixel against ImageJ.")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_cell)
        self.mode_group.addButton(self.rb_paper)
        self.rb_cell.toggled.connect(self._mode_changed)
        v2.addWidget(self.rb_cell)
        v2.addWidget(self.rb_paper)
        lv.addWidget(g2)

        g3 = QGroupBox("3   Settings")
        f3 = QFormLayout(g3)
        self.nuc_ch = _combo(CHANNELS, "Channel holding the nuclear stain (DAPI/Hoechst). Automatic uses the populated "
                                       "colour of single-colour exports, or blue in merged images.")
        self.tgt_ch = _combo(CHANNELS, "Channel holding the protein whose translocation is measured.")
        f3.addRow("Nuclear stain channel:", self.nuc_ch)
        f3.addRow("Target channel:", self.tgt_ch)
        pxrow = QHBoxLayout()
        self.pixel = _dspin(0, 1000, 0, 5, 0.01, "unknown", " µm/px", "Only used to report sizes in µm.")
        pxrow.addWidget(self.pixel, 1)
        b = QPushButton("From scale bar…")
        b.clicked.connect(self.pixel_from_scale_bar)
        pxrow.addWidget(b)
        f3.addRow("Pixel size:", pxrow)
        self.nuc_thr = _combo([(v, k) for k, v in METHOD_LABELS.items()], "Automatic threshold for the nuclear stain.")
        self.nuc_thr_val = _dspin(0, 1e9, 0, 2, 1, tip="Used when the threshold method is 'Manual value'.")
        f3.addRow("Nucleus threshold:", self.nuc_thr)
        f3.addRow("   manual value:", self.nuc_thr_val)
        bgrow = QHBoxLayout()
        self.bg = _combo(BACKGROUND, "Constant background subtracted from the target channel before the ratio.")
        self.bg_val = _dspin(0, 1e9, 0, 2, 1)
        bgrow.addWidget(self.bg, 2)
        bgrow.addWidget(self.bg_val, 1)
        f3.addRow("Target background:", bgrow)
        self.ring = _dspin(0, 500, 0, 1, 1, "automatic", " px", "Width of the perinuclear cytoplasmic ring (automatic = "
                           "0.3 x nucleus diameter).")
        f3.addRow("Cytoplasm ring width:", self.ring)
        self.cb_border = QCheckBox("Exclude cells touching the image edge")
        self.cb_sat = QCheckBox("Ignore saturated pixels (and cells with >5% saturation)")
        f3.addRow(self.cb_border)
        f3.addRow(self.cb_sat)
        self.adv_btn = QToolButton()
        self.adv_btn.setText("Advanced settings")
        self.adv_btn.setCheckable(True)
        self.adv_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.adv_btn.setArrowType(Qt.RightArrow)
        self.adv_btn.toggled.connect(self._toggle_advanced)
        f3.addRow(self.adv_btn)
        self.adv = QWidget()
        fa = QFormLayout(self.adv)
        fa.setContentsMargins(12, 0, 0, 0)
        self.median = QSpinBox()
        self.median.setRange(1, 15)
        self.median.setSingleStep(2)
        self.median.setToolTip("Median filter kernel (3 = ImageJ radius 1, as in the paper). 1 = off.")
        fa.addRow("Median filter (px):", self.median)
        self.tgt_thr = _combo([(v, k) for k, v in METHOD_LABELS.items()], "Target-channel threshold (paper method).")
        self.tgt_thr_val = _dspin(0, 1e9, 0, 2, 1)
        fa.addRow("Target threshold (paper):", self.tgt_thr)
        fa.addRow("   manual value:", self.tgt_thr_val)
        self.smooth = _dspin(0, 20, 2, 1, 0.5, "off", " px", "Gaussian smoothing of the nuclear stain before "
                             "detecting nuclei (per-cell method).")
        fa.addRow("Nucleus detection smoothing:", self.smooth)
        self.diam = _dspin(0, 1000, 0, 1, 1, "automatic", " px", "Typical nucleus diameter; automatic estimates it.")
        fa.addRow("Nucleus diameter:", self.diam)
        self.cb_split = QCheckBox("Split touching nuclei (watershed)")
        fa.addRow(self.cb_split)
        self.split_sens = _dspin(0.02, 0.5, 0.10, 2, 0.01, tip="Neck depth needed to split, as a fraction of the "
                                 "nucleus diameter. Lower = more splitting.")
        fa.addRow("Split sensitivity:", self.split_sens)
        self.min_area = _dspin(0.05, 2, 0.3, 2, 0.05, tip="Objects smaller than this x typical nucleus area are "
                               "discarded as debris.")
        self.max_area = _dspin(1, 20, 4, 1, 0.5, tip="Objects larger than this x typical area are excluded as clumps.")
        self.min_sol = _dspin(0, 1, 0.8, 2, 0.05, tip="Nuclei less convex than this are excluded (merged nuclei).")
        fa.addRow("Min nucleus size (x typical):", self.min_area)
        fa.addRow("Max nucleus size (x typical):", self.max_area)
        fa.addRow("Min solidity:", self.min_sol)
        self.erode = QSpinBox()
        self.erode.setRange(0, 10)
        self.erode.setToolTip("Pixels trimmed from each nucleus edge before measuring (limits blur mixing).")
        fa.addRow("Nucleus edge trim (px):", self.erode)
        self.gap = _dspin(0, 50, 1, 1, 1, suffix=" px", tip="Gap between nucleus and cytoplasm ring.")
        fa.addRow("Ring gap:", self.gap)
        self.cyto = _combo(CYTOPLASM)
        fa.addRow("Cytoplasm region:", self.cyto)
        self.territory = _dspin(0, 1000, 0, 1, 1, "automatic", " px", "Reach of the cell territory (automatic = "
                                "1 x nucleus diameter).")
        fa.addRow("Territory reach:", self.territory)
        self.cb_restrict = QCheckBox("Drop cytoplasm pixels at background level")
        fa.addRow(self.cb_restrict)
        self.cell_sig = _dspin(0, 20, 3, 1, 0.5, tip="Cell pixels must exceed background + this x noise SD.")
        fa.addRow("Cell detection (x noise SD):", self.cell_sig)
        self.resp = _dspin(0, 100, 1.0, 2, 0.1, "none", tip="Report the fraction of cells with N/C above this value.")
        fa.addRow("Responder cut-off (N/C):", self.resp)
        self.cb_legacy = QCheckBox("ImageJ <= 1.41 threshold convention (keep pixels equal to the level)")
        self.cb_zero = QCheckBox("Ignore zero-valued pixels (paper histogram convention)")
        fa.addRow(self.cb_legacy)
        fa.addRow(self.cb_zero)
        self.adv.setVisible(False)
        f3.addRow(self.adv)
        self.nuc_thr.currentIndexChanged.connect(self._sync_enabled)
        self.tgt_thr.currentIndexChanged.connect(self._sync_enabled)
        self.bg.currentIndexChanged.connect(self._sync_enabled)
        self.cyto.currentIndexChanged.connect(self._sync_enabled)
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
        self.btn_preview = QPushButton("Preview selected field")
        self.btn_preview.clicked.connect(self.run_preview)
        self.btn_run = QPushButton("Analyse all fields")
        self.btn_run.setDefault(True)
        self.btn_run.setStyleSheet("font-weight:600")
        self.btn_run.clicked.connect(self.run_all)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        v4.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setValue(0)
        v4.addWidget(self.progress)
        lv.addWidget(g4)
        lv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(460)
        split.addWidget(scroll)

        # right side
        self.tabs = QTabWidget()
        pv = QWidget()
        pvl = QVBoxLayout(pv)
        bar = QHBoxLayout()
        self.view_combo = _combo([("Target channel", "target"), ("Nuclear stain", "nuclear"), ("Composite", "composite")])
        self.view_combo.currentIndexChanged.connect(self.render_preview)
        self.cb_show_nuc = QCheckBox("Nuclei")
        self.cb_show_cyto = QCheckBox("Cytoplasm")
        self.cb_show_exc = QCheckBox("Excluded")
        for cb in (self.cb_show_nuc, self.cb_show_cyto, self.cb_show_exc):
            cb.setChecked(True)
            cb.toggled.connect(self.render_preview)
        bar.addWidget(QLabel("Show:"))
        bar.addWidget(self.view_combo)
        bar.addWidget(self.cb_show_nuc)
        bar.addWidget(self.cb_show_cyto)
        bar.addWidget(self.cb_show_exc)
        bar.addStretch(1)
        b = QPushButton("Fit")
        b.clicked.connect(lambda: self.viewer.fit())
        bar.addWidget(b)
        b = QPushButton("100%")
        b.clicked.connect(lambda: self.viewer.actual_size())
        bar.addWidget(b)
        pvl.addLayout(bar)
        self.viewer = ImageViewer()
        self.viewer.hovered.connect(self._hover)
        pvl.addWidget(self.viewer, 1)
        self.hover_label = QLabel(" ")
        self.hover_label.setStyleSheet("color:#52514e")
        pvl.addWidget(self.hover_label)
        legend = QLabel("<span style='color:#00b8d4'>■</span> nucleus (measured) &nbsp; "
                        "<span style='color:#ff00c8'>■</span> cytoplasm (measured) &nbsp; "
                        "<span style='color:#ff4646'>■</span> excluded nucleus &nbsp; "
                        "<span style='color:#2828a0'>■</span> ignored area (annotation / exclusion)")
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
        rsl.addWidget(QLabel("<b>Per condition</b>"))
        self.cond_table = QTableWidget()
        self.cond_table.setMaximumHeight(110)
        rsl.addWidget(self.cond_table)
        rsl.addWidget(QLabel("<b>Per field</b>"))
        self.field_table = QTableWidget()
        rsl.addWidget(self.field_table, 1)
        self.fig_label = FitImageLabel()
        rsl.addWidget(self.fig_label, 2)
        cite = QLabel("Publishing these results? Please cite MicrosCount (Help ▸ How to cite; a CITATION.txt is saved "
                      "with every result).")
        cite.setStyleSheet("color:#52514e")
        rsl.addWidget(cite)
        self.tabs.addTab(rs, "Results")

        self.logview = LogView()
        self.tabs.addTab(self.logview, "Log")
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([520, 900])

    # ------------------------------------------------------------------ settings
    def _toggle_advanced(self, on):
        self.adv.setVisible(on)
        self.adv_btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def _mode_changed(self, *_):
        paper = self.rb_paper.isChecked()
        if paper:
            _set_combo(self.bg, "none")
            self.cb_sat.setChecked(False)
        else:
            _set_combo(self.bg, "auto")
            self.cb_sat.setChecked(True)
        self._sync_enabled()

    def _sync_enabled(self, *_):
        per_cell = self.rb_cell.isChecked()
        for w in (self.ring, self.cb_border, self.smooth, self.diam, self.cb_split, self.split_sens, self.min_area,
                  self.max_area, self.min_sol, self.erode, self.gap, self.cyto, self.cb_restrict, self.cell_sig,
                  self.resp):
            w.setEnabled(per_cell)
        self.territory.setEnabled(per_cell and self.cyto.currentData() == "territory")
        self.ring.setEnabled(per_cell and self.cyto.currentData() == "ring")
        self.nuc_thr_val.setEnabled(self.nuc_thr.currentData() == "manual")
        self.tgt_thr_val.setEnabled(self.tgt_thr.currentData() == "manual")
        self.bg_val.setEnabled(self.bg.currentData() == "manual")

    def get_settings(self) -> TranslocationSettings:
        s = TranslocationSettings()
        s.method = "per_cell" if self.rb_cell.isChecked() else "paper"
        s.nuclear_channel = self.nuc_ch.currentData()
        s.target_channel = self.tgt_ch.currentData()
        s.pixel_size_um = self.pixel.value() or None
        s.nuclear_threshold = self.nuc_thr.currentData()
        s.nuclear_threshold_value = self.nuc_thr_val.value() if s.nuclear_threshold == "manual" else None
        s.target_threshold = self.tgt_thr.currentData()
        s.target_threshold_value = self.tgt_thr_val.value() if s.target_threshold == "manual" else None
        s.background = self.bg.currentData()
        s.background_value = self.bg_val.value()
        s.ring_width_px = self.ring.value() or None
        s.exclude_border_cells = self.cb_border.isChecked()
        s.exclude_saturated = self.cb_sat.isChecked()
        s.median_size = self.median.value()
        s.segmentation_smoothing_px = self.smooth.value()
        s.nucleus_diameter_px = self.diam.value() or None
        s.split_touching = self.cb_split.isChecked()
        s.split_sensitivity = self.split_sens.value()
        s.min_area_fraction = self.min_area.value()
        s.max_area_fraction = self.max_area.value()
        s.min_solidity = self.min_sol.value()
        s.nucleus_erode_px = self.erode.value()
        s.ring_gap_px = self.gap.value()
        s.cytoplasm = self.cyto.currentData()
        s.territory_px = self.territory.value() or None
        s.restrict_to_cells = self.cb_restrict.isChecked()
        s.cell_detection_sigmas = self.cell_sig.value()
        s.responder_ratio = self.resp.value() or None
        s.legacy_inclusive_threshold = self.cb_legacy.isChecked()
        s.exclude_zero_pixels = self.cb_zero.isChecked()
        return s

    def apply_settings(self, s: TranslocationSettings):
        widgets = [self.rb_cell, self.rb_paper]
        for w in widgets:
            w.blockSignals(True)
        (self.rb_paper if s.method == "paper" else self.rb_cell).setChecked(True)
        for w in widgets:
            w.blockSignals(False)
        _set_combo(self.nuc_ch, s.nuclear_channel)
        _set_combo(self.tgt_ch, s.target_channel)
        self.pixel.setValue(s.pixel_size_um or 0)
        _set_combo(self.nuc_thr, s.nuclear_threshold)
        self.nuc_thr_val.setValue(s.nuclear_threshold_value or 0)
        _set_combo(self.tgt_thr, s.target_threshold)
        self.tgt_thr_val.setValue(s.target_threshold_value or 0)
        _set_combo(self.bg, s.background)
        self.bg_val.setValue(s.background_value or 0)
        self.ring.setValue(s.ring_width_px or 0)
        self.cb_border.setChecked(s.exclude_border_cells)
        self.cb_sat.setChecked(s.exclude_saturated)
        self.median.setValue(int(s.median_size or 1))
        self.smooth.setValue(s.segmentation_smoothing_px or 0)
        self.diam.setValue(s.nucleus_diameter_px or 0)
        self.cb_split.setChecked(s.split_touching)
        self.split_sens.setValue(s.split_sensitivity)
        self.min_area.setValue(s.min_area_fraction)
        self.max_area.setValue(s.max_area_fraction)
        self.min_sol.setValue(s.min_solidity)
        self.erode.setValue(int(s.nucleus_erode_px))
        self.gap.setValue(s.ring_gap_px if s.ring_gap_px is not None else 1.0)
        _set_combo(self.cyto, s.cytoplasm)
        self.territory.setValue(s.territory_px or 0)
        self.cb_restrict.setChecked(s.restrict_to_cells)
        self.cell_sig.setValue(s.cell_detection_sigmas)
        self.resp.setValue(s.responder_ratio or 0)
        self.cb_legacy.setChecked(s.legacy_inclusive_threshold)
        self.cb_zero.setChecked(s.exclude_zero_pixels)
        self._sync_enabled()

    def reset_defaults(self):
        self.apply_settings(TranslocationSettings.paper() if self.rb_paper.isChecked() else TranslocationSettings())

    # ------------------------------------------------------------------ files
    def log(self, text: str):
        self.logview.log(text)
        self.status.emit(text)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add images", "", IMAGE_FILTER)
        if paths:
            self.add_paths(paths)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add a folder of images")
        if d:
            self.add_paths([d])

    def add_paths(self, paths: list[str]):
        files = [p for p in expand_paths(paths) if p not in self.entries]
        if not files:
            if paths:
                self.log("No new TIFF/PNG/JPEG images found.")
            return
        if self._busy():
            return
        self.log(f"Reading {len(files)} image(s)…")
        self._start(Task(scan_files, files), self._scanned, "Reading images")

    def _scanned(self, entries: list[FileEntry]):
        for e in entries:
            self.entries[e.path] = e
            if e.error:
                self.log(f"Could not read {Path(e.path).name}: {e.error}")
            else:
                self.log(f"{Path(e.path).name}: {e.description} → {e.role}")
        self._repair()

    def _repair(self):
        """Re-pair all files, keeping manual pairs and typed conditions."""
        keep = [p for p in self.pairs if p.confidence == "manual"]
        conditions = {p.nuclear: p.condition for p in self.pairs}
        used = {p.nuclear for p in keep} | {p.target for p in keep}
        auto, unpaired = pair_files([e for e in self.entries.values() if e.path not in used])
        pairs = keep + auto
        pairs.sort(key=lambda q: (str(Path(q.nuclear).parent), Path(q.nuclear).name))
        for i, p in enumerate(pairs):
            p.field_id = f"field_{i + 1:03d}"
            if p.nuclear in conditions and conditions[p.nuclear]:
                p.condition = conditions[p.nuclear]
        self.pairs, self.unpaired = pairs, unpaired
        for p in pairs:
            if p.confidence in ("image", "weak") and p.note:
                self.log(f"{p.field_id}: {Path(p.nuclear).name} + {Path(p.target).name} ({p.note})")
        self._refresh_table()

    def _refresh_table(self):
        self.table.blockSignals(True)
        self.rows = [("pair", i) for i in range(len(self.pairs))] + [("unpaired", i) for i in range(len(self.unpaired))]
        self.table.setRowCount(len(self.rows))
        for r, (kind, i) in enumerate(self.rows):
            if kind == "pair":
                p = self.pairs[i]
                same = p.nuclear == p.target
                vals = [p.field_id, p.condition, Path(p.nuclear).name,
                        "(same file)" if same else Path(p.target).name,
                        {"self": "one file, two channels", "name": "by file name", "image": "by image content",
                         "manual": "set by you", "weak": "uncertain – check"}.get(p.confidence, p.confidence)]
                tips = ["", "Double-click to edit", p.nuclear, p.target, p.note]
            else:
                e = self.unpaired[i]
                vals = ["–", "", Path(e.path).name if e.role != "target" else "",
                        Path(e.path).name if e.role == "target" else "", "unpaired" if not e.error else "unreadable"]
                tips = ["", "", e.path, e.path, e.error or "Right-click ▸ Pair with… to choose its partner"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setToolTip(tips[c])
                if not (kind == "pair" and c == 1):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if c == 4 and (v.startswith("uncertain") or v in ("unpaired", "unreadable")):
                    it.setForeground(QColor("#c05800"))
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        n_ok = len(self.pairs)
        self.btn_run.setText(f"Analyse all fields ({n_ok})" if n_ok else "Analyse all fields")
        if n_ok and not self.table.selectedItems():
            self.table.selectRow(0)

    def _item_changed(self, item: QTableWidgetItem):
        r = item.row()
        if r < len(self.rows) and self.rows[r][0] == "pair" and item.column() == 1:
            self.pairs[self.rows[r][1]].condition = item.text().strip()

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.table.selectedItems()})

    def selected_pair(self) -> Pairing | None:
        rows = self._selected_rows()
        if not rows:
            return None
        kind, i = self.rows[rows[0]]
        return self.pairs[i] if kind == "pair" else None

    def remove_selected(self):
        rows = self._selected_rows()
        drop = set()
        for r in rows:
            kind, i = self.rows[r]
            if kind == "pair":
                drop |= {self.pairs[i].nuclear, self.pairs[i].target}
            else:
                drop.add(self.unpaired[i].path)
        for p in drop:
            self.entries.pop(p, None)
        self.pairs = [p for p in self.pairs if p.nuclear not in drop and p.target not in drop]
        self._repair()

    def clear_all(self):
        self.entries.clear()
        self.pairs.clear()
        self.unpaired.clear()
        self._refresh_table()
        self.preview = None
        self.viewer.set_image(None)
        self.info.setText("")
        self.hist.clear()

    def set_condition(self):
        rows = [r for r in self._selected_rows() if self.rows[r][0] == "pair"]
        if not rows:
            message(self, "Set condition", "Select one or more fields first.")
            return
        text, ok = QInputDialog.getText(self, "Set condition", "Condition label for the selected fields:")
        if ok:
            for r in rows:
                self.pairs[self.rows[r][1]].condition = text.strip()
            self._refresh_table()

    def swap_selected(self):
        for r in self._selected_rows():
            kind, i = self.rows[r]
            if kind == "pair" and self.pairs[i].nuclear != self.pairs[i].target:
                p = self.pairs[i]
                p.nuclear, p.target = p.target, p.nuclear
                p.confidence = "manual"
        self._refresh_table()

    def _choose(self, title) -> str | None:
        path, _ = QFileDialog.getOpenFileName(self, title, "", IMAGE_FILTER)
        return path or None

    def _context_menu(self, pos):
        r = self.table.rowAt(pos.y())
        if r < 0:
            return
        self.table.selectRow(r)
        kind, i = self.rows[r]
        menu = QMenu(self)
        if kind == "pair":
            menu.addAction("Preview this field", self.run_preview)
            menu.addAction("Swap nuclear / target", self.swap_selected)
            menu.addAction("Choose nuclear-stain image…", lambda: self._replace(i, "nuclear"))
            menu.addAction("Choose target image…", lambda: self._replace(i, "target"))
            menu.addAction("Set condition…", self.set_condition)
        else:
            menu.addAction("Pair with…", lambda: self._pair_unpaired(i))
        menu.addSeparator()
        menu.addAction("Remove", self.remove_selected)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _replace(self, i: int, side: str):
        path = self._choose(f"Choose the {side} image")
        if not path:
            return
        p = self.pairs[i]
        setattr(p, side, path)
        p.confidence = "manual"
        if path not in self.entries:
            self.entries[path] = FileEntry(path, side, "", "", "")
        self._refresh_table()

    def _pair_unpaired(self, i: int):
        e = self.unpaired[i]
        other = self._choose("Choose the partner image")
        if not other:
            return
        nuc, tgt = (other, e.path) if e.role == "target" else (e.path, other)
        self.pairs.append(Pairing(nuc, tgt, Path(nuc).parent.name, confidence="manual"))
        if other not in self.entries:
            self.entries[other] = FileEntry(other, "unknown", "", "", "")
        self._repair()

    def load_inputs(self, inputs: list[dict]):
        """Restore fields saved in a settings.yaml."""
        self.clear_all()
        for k, d in enumerate(inputs):
            n, t = d.get("nuclear"), d.get("target")
            if n and t and Path(n).exists() and Path(t).exists():
                self.entries[n] = FileEntry(n, "nuclear", "", "", "")
                self.entries[t] = FileEntry(t, "target", "", "", "")
                self.pairs.append(Pairing(n, t, d.get("condition", ""), d.get("field") or f"field_{k + 1:03d}", "manual"))
            else:
                self.log(f"missing files for {d.get('field', '?')}: {n}, {t}")
        self._refresh_table()

    # ------------------------------------------------------------------ run
    def _busy(self) -> bool:
        if self.task is not None:
            message(self, "Busy", "Please wait for the current task to finish (or press Cancel).")
            return True
        return False

    def _start(self, task: Task, on_done, label: str):
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

    def _finish_task(self):
        self.task = None
        self.btn_cancel.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("Ready")

    def _done(self, res, on_done):
        self._finish_task()
        on_done(res)

    def _failed(self, tb: str):
        self._finish_task()
        self.log(tb)
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        message(self, "Something went wrong", last + "\n\nDetails are in the Log tab.")

    def cancel(self):
        if self.task:
            self.task.cancel()
            self.log("Cancelling after the current field…")

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.out_edit.setText(d)

    def pixel_from_scale_bar(self):
        from ..imageio import load_image, measure_scale_bar

        p = self.selected_pair()
        length = None
        if p is not None:
            try:
                img = load_image(p.nuclear)
                sb = measure_scale_bar(img.annotation_mask)
                if sb is None and p.target != p.nuclear:
                    sb = measure_scale_bar(load_image(p.target).annotation_mask)
                length = sb["length_px"] if sb else None
            except Exception:  # noqa: BLE001
                length = None
        dlg = ScaleBarDialog(self, length)
        if dlg.exec() and dlg.value() > 0:
            self.pixel.setValue(dlg.value())
            self.log(f"Pixel size set to {dlg.value():.5g} µm/px")

    def run_preview(self):
        p = self.selected_pair()
        if p is None:
            message(self, "Preview", "Add images and select a paired field first.")
            return
        if self._busy():
            return
        s = self.get_settings()

        def job():
            from ..imageio import load_image

            nuc = load_image(p.nuclear)
            tgt = nuc if p.target == p.nuclear else load_image(p.target)
            return analyse_field(nuc, tgt, s, p.field_id, p.condition, keep_layers=True)

        self.viewer.set_placeholder("Analysing…")
        self._start(Task(job, pass_progress=False), self._preview_done, f"Analysing {p.field_id}")

    def _preview_done(self, res):
        self.preview = res
        self.tabs.setCurrentIndex(0)
        self.render_preview(keep=False)
        S = res.summary
        parts = [f"<b>{res.field_id}</b> &nbsp; {res.nuclear_file} + {res.target_file}"]
        if res.method == "per_cell":
            parts.append(
                f"<b>{S.get('n_cells_analysed', 0)}</b> cells measured of {S.get('n_nuclei_detected', 0)} nuclei detected; "
                f"median N/C <b>{S.get('median_ratio', math.nan):.3f}</b> "
                f"(IQR {S.get('q1_ratio', math.nan):.3f}–{S.get('q3_ratio', math.nan):.3f}); "
                f"paper-method ratio {S['paper_ratio']:.3f}")
            if S.get("excluded_cells"):
                parts.append(f"Excluded: {S['excluded_cells']}")
        else:
            parts.append(f"N/C ratio (paper method) <b>{S['paper_ratio']:.4f}</b> "
                         f"(nuclear mean {S['paper_nuclear_mean']:.2f}, cytoplasmic mean {S['paper_cytoplasm_mean']:.2f})")
        parts.append(f"Thresholds: nuclear > {S['nuclear_threshold']:g}, target > {S['target_threshold']:g}; "
                     f"background {S['background']:g} ({S['background_source']}); nucleus Ø {S['nucleus_diameter_px']:.1f} px")
        for w in res.warnings:
            parts.append(f"<span style='color:#c05800'>⚠ {w}</span>")
        self.info.setText("<br>".join(parts))
        if res.method == "per_cell":
            r = np.array([c["ratio"] for c in res.cells if c["included"]])
            self.hist.show()
            self.hist.histogram(r, "N/C ratio per cell", S.get("median_ratio"), f"median {S.get('median_ratio', 0):.2f}",
                                ref=1.0, ref_label="N = C")
        else:
            self.hist.hide()
        self.log(f"Preview {res.field_id}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                         for k, v in S.items() if k in ("paper_ratio", "median_ratio",
                                                                                         "n_cells_analysed")))

    def render_preview(self, *_, keep=True):
        if self.preview is None or self.preview.layers is None:
            return
        rgb = translocation_overlay(self.preview.layers, base=self.view_combo.currentData(),
                                    show_cytoplasm=self.cb_show_cyto.isChecked(),
                                    show_nuclei=self.cb_show_nuc.isChecked(),
                                    show_excluded=self.cb_show_exc.isChecked())
        self.viewer.set_image(rgb, keep_view=keep)

    def _hover(self, x, y):
        if self.preview is None or self.preview.layers is None:
            return
        L = self.preview.layers
        text = f"x {x}, y {y}   nuclear {L['nuclear'][y, x]}   target {L['target'][y, x]}"
        lab = L.get("nuclear_labels")
        if lab is not None:
            cid = int(lab[y, x]) or int(L["cytoplasm_labels"][y, x])
            if cid:
                cell = next((c for c in self.preview.cells if c["cell"] == cid), None)
                if cell:
                    text += f"   cell {cid}: N/C {cell['ratio']:.3f}" if math.isfinite(cell["ratio"]) else f"   cell {cid}"
                    if not cell["included"]:
                        text += f" (excluded: {cell['exclusion_reason']})"
        self.hover_label.setText(text)

    def _selection_changed(self):
        pass

    def run_all(self):
        if not self.pairs:
            message(self, "Nothing to analyse", "Add images first (each field needs a nuclear-stain and a target image).")
            return
        if self._busy():
            return
        from ..reporting import default_output_dir, run_translocation

        specs = [FieldSpec(p.nuclear, p.target, p.condition, p.field_id) for p in self.pairs]
        out = Path(self.out_edit.text().strip()) if self.out_edit.text().strip() else default_output_dir([self.pairs[0].nuclear])
        s = self.get_settings()
        if self.unpaired:
            self.log(f"{len(self.unpaired)} unpaired file(s) will be skipped.")
        self.log(f"Analysing {len(specs)} field(s) → {out}")
        self._start(Task(run_translocation, specs, s, out), self._run_done, "Analysing")

    def _run_done(self, res):
        out = res["out_dir"]
        self.last_out = out
        self.btn_open.setEnabled(True)
        results = res["results"]
        cols = FIELD_COLS if any(r.method == "per_cell" for r in results) else PAPER_FIELD_COLS
        fill_table(self.field_table, [r.summary for r in results], cols)
        fill_table(self.cond_table, res["conditions"], COND_COLS)
        fig = res["files"].get("figure")
        if fig and Path(fig).exists():
            self.fig_label.set_file(fig)
        n_err = len([e for e in res["errors"] if e.get("field")])
        self.res_label.setText(f"<b>{len(results)}</b> field(s) analysed" + (f", <span style='color:#c05800'>{n_err} "
                               f"failed (see Log)</span>" if n_err else "") + f".<br>Saved to {out}")
        for e in res["errors"]:
            self.log(f"Error: {e}")
        for r in results:
            for w in r.warnings:
                self.log(f"{r.field_id}: {w}")
        self.log(f"Done. Results in {out}")
        self.tabs.setCurrentIndex(1)
