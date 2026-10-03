"""Nuclear translocation page."""

from __future__ import annotations

import copy
import html
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMenu, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)

from ..bio.experiment import ExperimentDesign, order_conditions
from ..bio.pairing import FileEntry, Pairing, assign_conditions, condition_key, label_fields, pair_files, scan_files
from ..core.render import translocation_overlay
from ..core.thresholds import METHOD_LABELS
from ..bio.translocation import FieldSpec, TranslocationSettings, analyse_field
from .common import (
    IMAGE_FILTER, DropTable, FitImageLabel, ImageViewer, LogView, MplCanvas, PathBox, ScaleBarDialog, Task, expand_paths,
    fill_table, message, open_folder, start_task,
)

CHANNELS = [("Automatic", "auto"), ("Blue", "blue"), ("Green", "green"), ("Red", "red"),
            ("Channel 1 (index 0)", "ch0"), ("Channel 2 (index 1)", "ch1"), ("Channel 3 (index 2)", "ch2"),
            ("Channel 4 (index 3)", "ch3")]
BACKGROUND = [("Cell-free area (automatic)", "auto"), ("Manual value", "manual"), ("None", "none")]
CYTOPLASM = [("Perinuclear ring", "ring"), ("Cell territory", "territory")]

FIELD_COLS = [("field", "Field"), ("condition", "Condition"), ("repetition", "Repetition"), ("n_cells_analysed", "Cells"),
              ("median_nc", "Median N/C"), ("q1_nc", "Q1"), ("q3_nc", "Q3"), ("median_cn", "Median C/N"),
              ("background_mean", "Background_Mean"), ("responder_fraction", "Fraction N/C > cut-off"),
              ("paper_ratio", "Paper N/C (field)"), ("paper_ok", "Paper threshold OK"), ("warnings", "Warnings"),
              ("nuclear_file", "Nuclear image"), ("target_file", "Target image")]
PAPER_FIELD_COLS = [("field", "Field"), ("condition", "Condition"), ("repetition", "Repetition"), ("paper_ratio", "N/C ratio"),
                    ("paper_ok", "Threshold OK"),
                    ("paper_nuclear_mean", "Nuclear mean"), ("paper_cytoplasm_mean", "Cytoplasmic mean"),
                    ("nuclei_count", "Nuclei (approx.)"), ("nuclear_threshold", "DAPI threshold (>)"),
                    ("target_threshold", "Target threshold (>)"), ("nuclear_file", "Nuclear image"),
                    ("target_file", "Target image")]
COND_COLS = [("repetition", "Repetition"), ("condition", "Condition"), ("n_fields", "Fields"), ("n_cells", "Cells"),
             ("field_median_nc_mean", "N/C (mean of fields)"), ("field_median_nc_sd", "SD"),
             ("fold_change", "Fold change"), ("fold_reference", "relative to"), ("responder_fraction", "Responders (fraction)"),
             ("median_nc", "Median N/C (all cells)"), ("median_cn", "Median C/N"), ("paper_ratio_mean", "Paper N/C"),
             ("paper_ratio_sd", "SD"), ("paper_fields_excluded", "Paper fields left out"), ("criteria_notes", "Paper criteria")]
PAPER_COND_COLS = [("repetition", "Repetition"), ("condition", "Condition"), ("n_fields", "Fields"),
                   ("paper_ratio_mean", "Paper N/C (mean)"), ("paper_ratio_sd", "SD"), ("fold_change", "Fold change"),
                   ("fold_reference", "relative to"), ("paper_fields_excluded", "Fields left out"), ("nuclei_count", "Nuclei"),
                   ("criteria_notes", "Paper criteria")]
COMP_COLS = [("repetition", "Repetition"), ("condition_b", "Condition"), ("condition_a", "compared with"),
             ("mean_b", "Value"), ("mean_a", "Reference value"), ("difference_pct", "Difference (%)"),
             ("welch_p", "Welch p (fields)"), ("per_repetition", "Per repetition"), ("paired_p", "Paired p (repetitions)")]
CONDITION_SOURCE_ITEMS = [("Automatic", "auto"), ("File names", "name"), ("Folder names", "folder")]
GREY = "color:#52514e"


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
    analysis = "translocation"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries: dict[str, FileEntry] = {}
        self.pairs: list[Pairing] = []
        self.unpaired: list[FileEntry] = []
        self.rows: list[tuple[str, int]] = []
        self.preview = None
        self.task: Task | None = None
        self.last_out: Path | None = None
        self._base = TranslocationSettings()
        self._cell_index: dict[int, dict] = {}
        self._typed_cond: dict[str, str] = {}  # nuclear path -> condition typed by the user
        self._typed_rep: dict[str, str] = {}
        self._comparisons: list[list[str]] = []
        self._references: dict[str, str] = {}
        self._order: list[str] = []
        self._control = ""
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
            if text == "Add folder…":
                b.setToolTip("Every image in the folder and in all its subfolders, at any depth")
            row.addWidget(b)
        v1.addLayout(row)
        self.path_box = PathBox()
        self.path_box.paths.connect(self.add_paths)
        v1.addWidget(self.path_box)
        self.table = DropTable(0, 6)
        self.table.setHorizontalHeaderLabels(["Field", "Condition", "Repetition", "Nuclear stain image", "Target image",
                                              "Pairing"])
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
        b = QPushButton("Set repetition…")
        b.setToolTip("Repetition (independent experiment) of the selected fields")
        b.clicked.connect(self.set_repetition)
        row.addWidget(b)
        b = QPushButton("Swap channels")
        b.setToolTip("Swap nuclear-stain and target images of the selected fields")
        b.clicked.connect(self.swap_selected)
        row.addWidget(b)
        row.addStretch(1)
        v1.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Conditions from:"))
        self.cond_from = _combo(CONDITION_SOURCE_ITEMS,
                                "Automatic: from the file names when a folder holds several conditions (e.g. a "
                                "microscope export named Experiment_vehicle-1, Experiment_stimulus-1 …) and each folder is a "
                                "repetition; otherwise each folder is a condition and its parent folder a repetition.")
        self.cond_from.currentIndexChanged.connect(self._reassign)
        row.addWidget(self.cond_from)
        row.addStretch(1)
        v1.addLayout(row)
        hint = QLabel("Drop TIFF, PNG or JPEG files or folders here (a folder brings the images of all its "
                      "subfolders). Blue exports are used as the nuclear stain and "
                      "green/red as the target; matching files are paired by name or, failing that, by image content. "
                      "Conditions and repetitions are read from the names and can be edited in the table. "
                      "Right-click a row to change a pairing.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#52514e")
        v1.addWidget(hint)
        lv.addWidget(g1)

        g2 = QGroupBox("2   Method")
        v2 = QVBoxLayout(g2)
        paper = QLabel("<b>Noursadeghi et al. (2008)</b>: one N/C ratio per field and mean ± SD per condition, "
                       "always computed exactly as published (it reproduces ImageJ 1.39).")
        paper.setWordWrap(True)
        paper.setToolTip("3x3 median filter, ImageJ 1.39 automatic IsoData threshold of each stain, nuclear ROI = DAPI "
                         "mask, cytoplasmic ROI = target mask minus DAPI mask, masks applied to the original target "
                         "image, normalised histograms (zero bin excluded) compared as N/C.")
        v2.addWidget(paper)
        self.cb_cell = QCheckBox("Also measure every cell (lab protocol)")
        self.cb_cell.setToolTip("Nuclei are found in the nuclear stain as by ImageJ Analyze Particles (size and "
                                "circularity filters), the target is background-subtracted with the rolling ball, and "
                                "for every cell: Nuc_corr = Nuc_Mean - Background_Mean, Cyto_corr = Cyto_Mean - "
                                "Background_Mean, N/C = Nuc_corr / Cyto_corr (C/N is reported too).")
        self.cb_cell.toggled.connect(self._sync_enabled)
        v2.addWidget(self.cb_cell)
        self.method_note = QLabel("")
        self.method_note.setWordWrap(True)
        self.method_note.setStyleSheet(GREY)
        v2.addWidget(self.method_note)
        lv.addWidget(g2)

        g3 = QGroupBox("3   Settings")
        f3 = QFormLayout(g3)
        self.nuc_ch = _combo(CHANNELS, "Channel holding the nuclear stain (DAPI/Hoechst). Automatic uses the populated "
                                       "colour of single-colour exports, or blue in merged images.")
        self.tgt_ch = _combo(CHANNELS, "Channel holding the protein whose translocation is measured (e.g. p65).")
        f3.addRow("Nuclear stain channel:", self.nuc_ch)
        f3.addRow("Target channel:", self.tgt_ch)
        pxrow = QHBoxLayout()
        self.pixel = _dspin(0, 1000, 0, 5, 0.01, "unknown", " µm/px", "Only used to report sizes in µm.")
        pxrow.addWidget(self.pixel, 1)
        b = QPushButton("From scale bar…")
        b.clicked.connect(self.pixel_from_scale_bar)
        pxrow.addWidget(b)
        f3.addRow("Pixel size:", pxrow)
        self.nuc_thr = _combo([(v, k) for k, v in METHOD_LABELS.items()],
                              "Automatic threshold for the DAPI and target masks (paper method) and for finding the "
                              "nuclei (per cell). The paper used ImageJ's IsoData auto-threshold (ImageJ 1.39). A "
                              "different target threshold can be set under Advanced settings.")
        self.nuc_thr_val = _dspin(0, 1e9, 0, 2, 1, tip="Used when the threshold method is 'Manual value'.")
        f3.addRow("Threshold:", self.nuc_thr)
        f3.addRow("   manual value:", self.nuc_thr_val)

        self.cell_hdr = QLabel("<b>Per-cell lab protocol</b>")
        f3.addRow(self.cell_hdr)
        size_row = QHBoxLayout()
        self.size_min = _dspin(0, 1e8, 0, 0, 10, suffix=" px²", tip="Analyze Particles 'Size' (pixel units): "
                               "smallest nucleus kept. The protocol uses 0.")
        self.size_max = _dspin(0, 1e8, 0, 0, 100, "Infinity", " px²", "Largest nucleus kept (Infinity = no limit).")
        size_row.addWidget(self.size_min, 1)
        size_row.addWidget(QLabel("–"))
        size_row.addWidget(self.size_max, 1)
        f3.addRow("Nucleus size:", size_row)
        circ_row = QHBoxLayout()
        self.circ_min = _dspin(0, 1, 0.2, 2, 0.05, tip="Analyze Particles 'Circularity' = 4π·area/perimeter² "
                               "(ImageJ's traced perimeter). The protocol uses 0.20–1.00.")
        self.circ_max = _dspin(0, 1, 1.0, 2, 0.05)
        circ_row.addWidget(self.circ_min, 1)
        circ_row.addWidget(QLabel("–"))
        circ_row.addWidget(self.circ_max, 1)
        f3.addRow("Circularity:", circ_row)
        self.cb_border = QCheckBox("Exclude nuclei touching the image edge")
        f3.addRow(self.cb_border)
        self.rb_radius = _dspin(0, 5000, 50, 1, 5, "off", " px", "Process ▸ Subtract Background (rolling ball) on the "
                                "target channel before measuring. The protocol uses 30–50 px.")
        f3.addRow("Rolling ball radius:", self.rb_radius)
        bgrow = QHBoxLayout()
        self.bg = _combo(BACKGROUND, "Background_Mean, subtracted from Nuc_Mean and Cyto_Mean: the mean of the "
                                     "background-subtracted target over the cell-free area of the field, a value "
                                     "you enter, or none.")
        self.bg_val = _dspin(0, 1e9, 0, 2, 1)
        bgrow.addWidget(self.bg, 2)
        bgrow.addWidget(self.bg_val, 1)
        f3.addRow("Background_Mean:", bgrow)
        self.ring = _dspin(0, 500, 0, 1, 1, "automatic", " px", "Width of the cytoplasmic ring around each nucleus "
                           "where Cyto_Mean is measured (automatic = 0.3 x nucleus diameter).")
        f3.addRow("Cytoplasm ring width:", self.ring)
        self.cb_sat = QCheckBox("Ignore saturated pixels (and cells with >5% saturation)")
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
        fa.addRow(QLabel("<i>Paper method</i>"))
        self.median = QSpinBox()
        self.median.setRange(1, 15)
        self.median.setSingleStep(2)
        self.median.setToolTip("Median filter kernel (3 = ImageJ radius 1, as in the paper). 1 = off.")
        fa.addRow("Median filter (px):", self.median)
        self.tgt_thr = _combo([(v, k) for k, v in METHOD_LABELS.items()], "Target-channel threshold (paper method).")
        self.tgt_thr_val = _dspin(0, 1e9, 0, 2, 1)
        fa.addRow("Target threshold:", self.tgt_thr)
        fa.addRow("   manual value:", self.tgt_thr_val)
        self.cb_legacy = QCheckBox("ImageJ 1.42+ variant: keep pixels equal to the level")
        self.cb_zero = QCheckBox("Ignore zero-valued pixels (the paper's histograms drop the zero bin)")
        fa.addRow(self.cb_legacy)
        fa.addRow(self.cb_zero)
        self.adv_cell_hdr = QLabel("<i>Per-cell lab protocol</i>")
        fa.addRow(self.adv_cell_hdr)
        self.smooth = _dspin(0, 20, 2, 1, 0.5, "off", " px", "Gaussian smoothing of the nuclear stain before it is "
                             "thresholded, so that dim, noisy nuclei come out whole.")
        fa.addRow("Nucleus detection smoothing:", self.smooth)
        self.cb_split = QCheckBox("Split touching nuclei (watershed)")
        fa.addRow(self.cb_split)
        self.split_sens = _dspin(0.02, 0.5, 0.10, 2, 0.01, tip="Neck depth needed to split, as a fraction of the "
                                 "nucleus diameter. Lower = more splitting.")
        fa.addRow("Split sensitivity:", self.split_sens)
        self.diam = _dspin(0, 1000, 0, 1, 1, "automatic", " px", "Typical nucleus diameter; automatic estimates it.")
        fa.addRow("Nucleus diameter:", self.diam)
        self.erode = QSpinBox()
        self.erode.setRange(0, 10)
        self.erode.setToolTip("Pixels trimmed from each nucleus edge before measuring Nuc_Mean (0 = the whole "
                              "particle, as ImageJ measures it).")
        fa.addRow("Nucleus edge trim (px):", self.erode)
        self.gap = _dspin(0, 50, 0, 1, 1, suffix=" px", tip="Gap between the nucleus and the cytoplasmic ring.")
        fa.addRow("Ring gap:", self.gap)
        self.cyto = _combo(CYTOPLASM)
        fa.addRow("Cytoplasm region:", self.cyto)
        self.territory = _dspin(0, 1000, 0, 1, 1, "automatic", " px", "Reach of the cell territory (automatic = "
                                "1 x nucleus diameter).")
        fa.addRow("Territory reach:", self.territory)
        self.cb_restrict = QCheckBox("Leave out ring pixels at background level (off the cell)")
        fa.addRow(self.cb_restrict)
        self.cell_sig = _dspin(0, 20, 3, 1, 0.5, tip="A pixel belongs to a cell when the background-subtracted target "
                               "exceeds the background by this many noise SDs (also defines the cell-free area).")
        fa.addRow("Cell detection (x noise SD):", self.cell_sig)
        self.resp = _dspin(0, 100, 1.0, 2, 0.1, "none", tip="Report the fraction of cells with N/C above this value.")
        fa.addRow("Responder cut-off (N/C):", self.resp)
        self.adv.setVisible(False)
        f3.addRow(self.adv)
        self._cell_widgets = [self.cell_hdr, self.size_min, self.size_max, self.circ_min, self.circ_max, self.cb_border,
                              self.rb_radius, self.bg, self.ring, self.cb_sat, self.adv_cell_hdr, self.smooth,
                              self.cb_split, self.split_sens, self.diam, self.erode, self.gap, self.cyto,
                              self.cb_restrict, self.cell_sig, self.resp]
        self.nuc_thr.currentIndexChanged.connect(self._sync_target_threshold)
        self.nuc_thr.currentIndexChanged.connect(self._sync_enabled)
        self.tgt_thr.currentIndexChanged.connect(self._sync_enabled)
        self.bg.currentIndexChanged.connect(self._sync_enabled)
        self.cyto.currentIndexChanged.connect(self._sync_enabled)
        lv.addWidget(g3)

        g5 = QGroupBox("4   Experiment")
        f5 = QFormLayout(g5)
        self.control = QComboBox()
        self.control.setToolTip("Control condition: the reference for fold changes, the responder cut-off and the "
                                "default comparisons.")
        self.control.currentIndexChanged.connect(self._control_changed)
        f5.addRow("Control condition:", self.control)
        cw = QWidget()
        cl = QHBoxLayout(cw)
        cl.setContentsMargins(0, 0, 0, 0)
        self.comp_list = QListWidget()
        self.comp_list.setMaximumHeight(86)
        self.comp_list.setToolTip("Each line compares a condition with another, in every repetition (Welch t-test on "
                                  "the fields, exploratory) and across repetitions (paired t-test when there are at "
                                  "least 3). Empty: every condition is compared with the control.")
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
        f5.addRow("Comparisons:", cw)
        self.btn_order = QPushButton("Order and fold-change references…")
        self.btn_order.setToolTip("Order of the conditions in tables and figures, and for each condition the "
                                  "condition its fold change is relative to (the control unless set).")
        self.btn_order.clicked.connect(self.edit_order)
        f5.addRow(self.btn_order)
        self.resp_pct = _dspin(0, 100, 95, 0, 5, "fixed N/C cut-off", "th percentile of the control",
                               "Responders: cells whose N/C is above this percentile of the control's cells in the "
                               "same repetition. Off: the fixed cut-off under Advanced settings.")
        f5.addRow("Responders above:", self.resp_pct)
        self.cb_paper_qc = QCheckBox("Leave fields with a failed automatic threshold out of paper-method means")
        self.cb_paper_qc.setToolTip("A field fails when its nuclear mask covers more than 60% or less than 0.2% of the "
                                    "field, or its target mask more than 97% or less than 0.5%.")
        f5.addRow(self.cb_paper_qc)
        lv.addWidget(g5)

        g4 = QGroupBox("5   Run")
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
        row = QHBoxLayout()
        self.btn_combine = QPushButton("Combine saved results…")
        self.btn_combine.setToolTip("One experiment from result folders analysed separately (e.g. one per repetition): "
                                    "choose the folder that holds them.")
        self.btn_combine.clicked.connect(self.combine_saved)
        row.addWidget(self.btn_combine)
        row.addStretch(1)
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
        self.view_combo = _combo([("Target channel", "target"), ("Target, rolling ball applied", "target_corrected"),
                                  ("Nuclear stain", "nuclear"), ("Composite", "composite")])
        self.view_combo.currentIndexChanged.connect(self.render_preview)
        self.cb_show_nuc = QCheckBox("Nuclei")
        self.cb_show_cyto = QCheckBox("Cytoplasm")
        self.cb_show_bg = QCheckBox("Background area")
        self.cb_show_exc = QCheckBox("Excluded")
        for cb in (self.cb_show_nuc, self.cb_show_cyto, self.cb_show_bg, self.cb_show_exc):
            cb.setChecked(True)
            cb.toggled.connect(self.render_preview)
        bar.addWidget(QLabel("Show:"))
        bar.addWidget(self.view_combo)
        bar.addWidget(self.cb_show_nuc)
        bar.addWidget(self.cb_show_cyto)
        bar.addWidget(self.cb_show_bg)
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
                        "<span style='color:#ff4646'>■</span> not measured &nbsp; "
                        "<span style='color:#c8a800'>■</span> cell-free area (Background_Mean) &nbsp; "
                        "<span style='color:#2828a0'>■</span> ignored (annotation / exclusion)")
        legend.setWordWrap(True)
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
        self.cond_title = QLabel("<b>Per condition</b>")
        rsl.addWidget(self.cond_title)
        self.cond_table = QTableWidget()
        self.cond_table.setMaximumHeight(150)
        rsl.addWidget(self.cond_table)
        self.comp_title = QLabel("<b>Comparisons</b>")
        rsl.addWidget(self.comp_title)
        self.comp_table = QTableWidget()
        self.comp_table.setMaximumHeight(120)
        rsl.addWidget(self.comp_table)
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

    def _sync_target_threshold(self, *_):
        _set_combo(self.tgt_thr, self.nuc_thr.currentData())

    def _sync_enabled(self, *_):
        per_cell = self.cb_cell.isChecked()
        self.method_note.setText(
            "Per cell: nuclei found in the nuclear stain (size and circularity filters as ImageJ Analyze Particles), "
            "target background removed with the rolling ball, then N/C = Nuc_corr / Cyto_corr for every cell."
            if per_cell else
            "Paper method only: one N/C ratio per field, mean ± SD per condition (the paper used 5 fields and at "
            "least 500 cells per condition).")
        for w in self._cell_widgets:
            w.setEnabled(per_cell)
        self.territory.setEnabled(per_cell and self.cyto.currentData() == "territory")
        self.ring.setEnabled(per_cell and self.cyto.currentData() == "ring")
        self.nuc_thr_val.setEnabled(self.nuc_thr.currentData() == "manual")
        self.tgt_thr_val.setEnabled(self.tgt_thr.currentData() == "manual")
        self.bg_val.setEnabled(per_cell and self.bg.currentData() == "manual")

    def get_settings(self) -> TranslocationSettings:
        s = copy.deepcopy(self._base)  # keeps settings that have no widget (e.g. from a loaded file)
        s.method = "per_cell" if self.cb_cell.isChecked() else "paper"
        s.nuclear_channel = self.nuc_ch.currentData()
        s.target_channel = self.tgt_ch.currentData()
        s.pixel_size_um = self.pixel.value() or None
        s.nuclear_threshold = self.nuc_thr.currentData()
        s.nuclear_threshold_value = self.nuc_thr_val.value() if s.nuclear_threshold == "manual" else None
        s.target_threshold = self.tgt_thr.currentData()
        s.target_threshold_value = self.tgt_thr_val.value() if s.target_threshold == "manual" else None
        s.median_size = self.median.value()
        s.legacy_inclusive_threshold = self.cb_legacy.isChecked()
        s.exclude_zero_pixels = self.cb_zero.isChecked()
        s.size_min_px2 = self.size_min.value()
        s.size_max_px2 = self.size_max.value() or None
        s.circularity_min = self.circ_min.value()
        s.circularity_max = self.circ_max.value()
        s.exclude_border_cells = self.cb_border.isChecked()
        s.rolling_ball_radius = self.rb_radius.value()
        s.background = self.bg.currentData()
        s.background_value = self.bg_val.value()
        s.ring_width_px = self.ring.value() or None
        s.exclude_saturated = self.cb_sat.isChecked()
        s.segmentation_smoothing_px = self.smooth.value()
        s.split_touching = self.cb_split.isChecked()
        s.split_sensitivity = self.split_sens.value()
        s.nucleus_diameter_px = self.diam.value() or None
        s.nucleus_erode_px = self.erode.value()
        s.ring_gap_px = self.gap.value()
        s.cytoplasm = self.cyto.currentData()
        s.territory_px = self.territory.value() or None
        s.restrict_to_cells = self.cb_restrict.isChecked()
        s.cell_detection_sigmas = self.cell_sig.value()
        s.responder_ratio = self.resp.value() or None
        s.conditions_from = self.cond_from.currentData()
        s.control_condition = self._control
        s.comparisons = [list(c) for c in self._comparisons]
        s.fold_references = dict(self._references)
        s.condition_order = list(self._order)
        s.responder_percentile = self.resp_pct.value() or None
        s.exclude_failed_paper_fields = self.cb_paper_qc.isChecked()
        return s

    def apply_settings(self, s: TranslocationSettings):
        self._base = copy.deepcopy(s)
        self.cb_cell.blockSignals(True)
        self.cb_cell.setChecked(s.method != "paper")
        self.cb_cell.blockSignals(False)
        _set_combo(self.nuc_ch, s.nuclear_channel)
        _set_combo(self.tgt_ch, s.target_channel)
        self.pixel.setValue(s.pixel_size_um or 0)
        self.nuc_thr.blockSignals(True)
        _set_combo(self.nuc_thr, s.nuclear_threshold)
        self.nuc_thr.blockSignals(False)
        self.nuc_thr_val.setValue(s.nuclear_threshold_value or 0)
        _set_combo(self.tgt_thr, s.target_threshold)
        self.tgt_thr_val.setValue(s.target_threshold_value or 0)
        self.median.setValue(int(s.median_size or 1))
        self.cb_legacy.setChecked(s.legacy_inclusive_threshold)
        self.cb_zero.setChecked(s.exclude_zero_pixels)
        self.size_min.setValue(s.size_min_px2 or 0)
        self.size_max.setValue(s.size_max_px2 or 0)
        self.circ_min.setValue(s.circularity_min)
        self.circ_max.setValue(s.circularity_max)
        self.cb_border.setChecked(s.exclude_border_cells)
        self.rb_radius.setValue(s.rolling_ball_radius or 0)
        _set_combo(self.bg, s.background)
        self.bg_val.setValue(s.background_value or 0)
        self.ring.setValue(s.ring_width_px or 0)
        self.cb_sat.setChecked(s.exclude_saturated)
        self.smooth.setValue(s.segmentation_smoothing_px or 0)
        self.cb_split.setChecked(s.split_touching)
        self.split_sens.setValue(s.split_sensitivity)
        self.diam.setValue(s.nucleus_diameter_px or 0)
        self.erode.setValue(int(s.nucleus_erode_px or 0))
        self.gap.setValue(s.ring_gap_px or 0)
        _set_combo(self.cyto, s.cytoplasm)
        self.territory.setValue(s.territory_px or 0)
        self.cb_restrict.setChecked(s.restrict_to_cells)
        self.cell_sig.setValue(s.cell_detection_sigmas)
        self.resp.setValue(s.responder_ratio or 0)
        source = self.cond_from.currentData()
        self.cond_from.blockSignals(True)
        _set_combo(self.cond_from, s.conditions_from or "auto")
        self.cond_from.blockSignals(False)
        self._control = s.control_condition or ""
        self._comparisons = [list(c) for c in (s.comparisons or [])]
        self._references = dict(s.fold_references or {})
        self._order = list(s.condition_order or [])
        self.resp_pct.setValue(s.responder_percentile or 0)
        self.cb_paper_qc.setChecked(bool(s.exclude_failed_paper_fields))
        if getattr(self, "pairs", None) and self.cond_from.currentData() != source:
            self._reassign()  # the fields already listed, read with the new source
        self._sync_design()
        self._sync_enabled()

    def reset_defaults(self):
        self.apply_settings(TranslocationSettings())

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
        """Re-pair all files, keeping manual pairs, then read conditions and repetitions (typed ones are kept)."""
        keep = [p for p in self.pairs if p.confidence == "manual"]
        used = {p.nuclear for p in keep} | {p.target for p in keep}
        auto, unpaired = pair_files([e for e in self.entries.values() if e.path not in used], condition_from_folder=False)
        pairs = keep + auto
        pairs.sort(key=lambda q: (str(Path(q.nuclear).parent), Path(q.nuclear).name))
        self.pairs, self.unpaired = pairs, unpaired
        self._assign()
        for p in pairs:
            if p.confidence in ("image", "weak") and p.note:
                self.log(f"{p.field_id}: {Path(p.nuclear).name} + {Path(p.target).name} ({p.note})")
        self._refresh_table()

    def _assign(self):
        if not self.pairs:
            self._sync_design()
            return
        used = assign_conditions(self.pairs, self.cond_from.currentData() or "auto")
        for p in self.pairs:
            p.condition = self._typed_cond.get(p.nuclear, p.condition)
            p.repetition = self._typed_rep.get(p.nuclear, p.repetition)
        label_fields(self.pairs)
        reps = sorted({p.repetition for p in self.pairs})
        self.log(f"Conditions from {'file' if used == 'name' else 'folder'} names: "
                 f"{len({condition_key(p.condition) for p in self.pairs})} condition(s) in {len(reps)} repetition(s).")
        self._sync_design()

    def _reassign(self, *_):
        self._assign()
        self._refresh_table()

    def conditions(self) -> list[str]:
        return order_conditions([p.condition for p in self.pairs], self._order or ([self._control] if self._control else []))

    def _refresh_table(self):
        self.table.blockSignals(True)
        self.rows = [("pair", i) for i in range(len(self.pairs))] + [("unpaired", i) for i in range(len(self.unpaired))]
        self.table.setRowCount(len(self.rows))
        for r, (kind, i) in enumerate(self.rows):
            if kind == "pair":
                p = self.pairs[i]
                same = p.nuclear == p.target
                vals = [p.field_id, p.condition, p.repetition, Path(p.nuclear).name,
                        "(same file)" if same else Path(p.target).name,
                        {"self": "one file, two channels", "name": "by file name", "image": "by image content",
                         "manual": "set by you", "weak": "uncertain – check"}.get(p.confidence, p.confidence)]
                tips = ["", "Double-click to edit", "Double-click to edit", p.nuclear, p.target, p.note]
            else:
                e = self.unpaired[i]
                note = getattr(e, "note", "")
                status = "unreadable" if e.error else ("not used" if note else "unpaired")
                vals = ["–", "", "", Path(e.path).name if e.role != "target" else "",
                        Path(e.path).name if e.role == "target" else "", status]
                tips = ["", "", "", e.path, e.path, e.error or note or "Right-click ▸ Pair with… to choose its partner"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setToolTip(tips[c])
                if not (kind == "pair" and c in (1, 2)):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if c == 5 and (v.startswith("uncertain") or v in ("unpaired", "unreadable")):
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
        if r < len(self.rows) and self.rows[r][0] == "pair" and item.column() in (1, 2):
            p = self.pairs[self.rows[r][1]]
            text = item.text().strip()
            if item.column() == 1:
                p.condition = text
                self._typed_cond[p.nuclear] = text
            else:
                p.repetition = text
                self._typed_rep[p.nuclear] = text
            label_fields(self.pairs)
            self._sync_design()
            self._refresh_table()

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
        self._typed_cond.clear()
        self._typed_rep.clear()
        self._refresh_table()
        self._sync_design()
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
                p = self.pairs[self.rows[r][1]]
                p.condition = self._typed_cond[p.nuclear] = text.strip()
            label_fields(self.pairs)
            self._sync_design()
            self._refresh_table()

    def set_repetition(self):
        rows = [r for r in self._selected_rows() if self.rows[r][0] == "pair"]
        if not rows:
            message(self, "Set repetition", "Select one or more fields first.")
            return
        text, ok = QInputDialog.getText(self, "Set repetition", "Repetition (independent experiment) of the selected "
                                        "fields:")
        if ok and text.strip():
            for r in rows:
                p = self.pairs[self.rows[r][1]]
                p.repetition = self._typed_rep[p.nuclear] = text.strip()
            label_fields(self.pairs)
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
        self.pairs.append(Pairing(nuc, tgt, confidence="manual"))
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
                rep_ = str(d.get("repetition") or "1")
                self.pairs.append(Pairing(n, t, d.get("condition", ""), d.get("field") or f"field_{k + 1:03d}", "manual",
                                          repetition=rep_))
                self._typed_cond[n] = d.get("condition", "")
                self._typed_rep[n] = rep_
            else:
                self.log(f"missing files for {d.get('field', '?')}: {n}, {t}")
        self._sync_design()
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
        from ..core.imageio import load_image, measure_scale_bar

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
            from ..core.imageio import load_image

            nuc = load_image(p.nuclear)
            tgt = nuc if p.target == p.nuclear else load_image(p.target)
            return analyse_field(nuc, tgt, s, p.field_id, p.condition, keep_layers=True, repetition=p.repetition)

        self.viewer.set_placeholder("Analysing…")
        self._start(Task(job, pass_progress=False), self._preview_done, f"Analysing {p.field_id}")

    def _preview_done(self, res):
        self.preview = res
        self._cell_index = {c["cell"]: c for c in res.cells}
        self.tabs.setCurrentIndex(0)
        self.render_preview(keep=False)
        S = res.summary

        def f(v, d=3):
            return "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"

        parts = [f"<b>{html.escape(res.field_id)}</b> &nbsp; {html.escape(res.nuclear_file)} + {html.escape(res.target_file)}"]
        parts.append(f"Paper method: N/C <b>{f(S['paper_ratio'], 4)}</b> (nuclear mean {f(S['paper_nuclear_mean'], 2)} "
                     f"over {S['paper_nuclear_area_px']} px, cytoplasmic mean {f(S['paper_cytoplasm_mean'], 2)} over "
                     f"{S['paper_cytoplasm_area_px']} px; thresholds: nuclear > {S['nuclear_threshold']:g}, "
                     f"target > {S['target_threshold']:g})")
        if res.method != "paper":
            parts.append(
                f"Per cell: <b>{S.get('n_cells_analysed', 0)}</b> cells measured of {S.get('n_nuclei', 0)} nuclei; "
                f"median N/C <b>{f(S.get('median_nc'))}</b> (IQR {f(S.get('q1_nc'))}–{f(S.get('q3_nc'))}), "
                f"median C/N {f(S.get('median_cn'))}; Background_Mean {f(S.get('background_mean'), 2)} "
                f"({S.get('background_source', '')}), rolling ball {S.get('rolling_ball_radius', 0):g} px, "
                f"nucleus Ø {S['nucleus_diameter_px']:.1f} px, ring {f(S.get('ring_width_px'), 0)} px")
            if S.get("excluded_cells"):
                parts.append(f"Not measured: {html.escape(S['excluded_cells'])}")
        else:
            parts.append(f"About {S.get('nuclei_count', 0)} nuclei in this field")
        for w in res.warnings:
            parts.append(f"<span style='color:#c05800'>⚠ {html.escape(w)}</span>")
        self.info.setText("<br>".join(parts))
        self.hist.show()
        if res.method != "paper":
            r = np.array([c["nc_ratio"] for c in res.cells if c["included"]])
            self.hist.histogram(r, "N/C = Nuc_corr / Cyto_corr, per cell", S.get("median_nc"),
                                f"median {f(S.get('median_nc'), 2)}", ref=1.0, ref_label="N = C")
        elif res.histograms:
            self.hist.roi_histograms(res.histograms)
        self.log(f"Preview {res.field_id}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                         for k, v in S.items() if k in ("paper_ratio", "median_nc",
                                                                                         "n_cells_analysed")))

    def render_preview(self, *_, keep=True):
        if self.preview is None or self.preview.layers is None:
            return
        rgb = translocation_overlay(self.preview.layers, base=self.view_combo.currentData(),
                                    show_cytoplasm=self.cb_show_cyto.isChecked(),
                                    show_nuclei=self.cb_show_nuc.isChecked(),
                                    show_excluded=self.cb_show_exc.isChecked(),
                                    show_background=self.cb_show_bg.isChecked())
        self.viewer.set_image(rgb, keep_view=keep)

    def _hover(self, x, y):
        if self.preview is None or self.preview.layers is None:
            return
        L = self.preview.layers
        text = f"x {x}, y {y}   nuclear {L['nuclear'][y, x]}   target {L['target'][y, x]}"
        if L.get("target_corrected") is not None:
            text += f" (after rolling ball {L['target_corrected'][y, x]})"
        lab = L.get("nuclear_labels")
        if lab is not None:
            cid = int(lab[y, x]) or int(L["cytoplasm_labels"][y, x])
            cell = self._cell_index.get(cid) if cid else None
            if cell:
                text += f"   cell {cid}: Nuc_corr {cell['nuc_corr']:.1f}, Cyto_corr {cell['cyto_corr']:.1f}"
                if math.isfinite(cell["nc_ratio"]):
                    text += f", N/C {cell['nc_ratio']:.3f}"
                if not cell["included"]:
                    text += f" (not measured: {cell['exclusion_reason']})"
        self.hover_label.setText(text)

    def _selection_changed(self):
        pass

    def run_all(self):
        if not self.pairs:
            message(self, "Nothing to analyse", "Add images first (each field needs a nuclear-stain and a target image).")
            return
        if self._busy():
            return
        from ..bio.report import default_output_dir, run_translocation

        specs = [FieldSpec(p.nuclear, p.target, p.condition, p.field_id, p.repetition) for p in self.pairs]
        out = Path(self.out_edit.text().strip()) if self.out_edit.text().strip() else default_output_dir([self.pairs[0].nuclear])
        s = self.get_settings()
        if self.unpaired:
            extra = sum(1 for e in self.unpaired if getattr(e, "note", ""))
            if len(self.unpaired) - extra:
                self.log(f"{len(self.unpaired) - extra} unpaired file(s) will be skipped.")
            if extra:
                self.log(f"{extra} extra or blank channel file(s) are not analysed.")
        self.log(f"Analysing {len(specs)} field(s) → {out}")
        self._start(Task(run_translocation, specs, s, out), self._run_done, "Analysing")

    def _run_done(self, res):
        results = res["results"]
        n_err = len([e for e in res["errors"] if e.get("field")])
        head = f"<b>{len(results)}</b> field(s) analysed" + (f", <span style='color:#c05800'>{n_err} failed (see Log)</span>"
                                                              if n_err else "")
        self._show_results(res, head)
        for e in res["errors"]:
            self.log(f"Error: {e}")
        for r in results:
            for w in r.warnings:
                self.log(f"{r.field_id}: {w}")
        self.log(f"Done. Results in {res['out_dir']}")

    def _show_results(self, res, head: str):
        out = res["out_dir"]
        self.last_out = out
        self.btn_open.setEnabled(True)
        exp = res["experiment"]
        per_cell = exp.measure.startswith("median")
        if len(exp.repetitions) > 1:
            self.cond_title.setText(f"<b>Per condition, {len(exp.repetitions)} repetitions</b> (mean ± SD of the "
                                    "repetition values; fold change and responders averaged over repetitions)")
            cols = [("condition", "Condition"), ("n_repetitions", "Repetitions"), ("n_fields", "Fields")]
            cols += [("n_cells", "Cells")] if per_cell else []
            cols += [("mean", "Mean"), ("sd", "SD")] + [(f"repetition {r}", f"Rep. {r}") for r in exp.repetitions]
            cols += [("fold_change_mean", "Fold change"), ("fold_reference", "relative to")]
            cols += [("responder_fraction_mean", "Responders (fraction)")] if per_cell else [("paper_ratio_mean", "Paper N/C")]
            fill_table(self.cond_table, exp.overall, cols)
        else:
            self.cond_title.setText("<b>Per condition</b> (" + ("median N/C per field, mean ± SD" if per_cell else
                                                                 "paper-method N/C per field, mean ± SD") + ")")
            fill_table(self.cond_table, exp.conditions, COND_COLS if per_cell else PAPER_COND_COLS)
        self.comp_title.setVisible(bool(exp.comparisons))
        self.comp_table.setVisible(bool(exp.comparisons))
        if exp.comparisons:
            fill_table(self.comp_table, exp.comparisons, COMP_COLS)
        fill_table(self.field_table, res.get("field_rows", []), FIELD_COLS if per_cell else PAPER_FIELD_COLS)
        fig = res["files"].get("figure")
        if fig and Path(fig).exists():
            self.fig_label.set_file(fig)
        notes = "".join(f"<br><span style='color:#52514e'>{html.escape(n)}</span>" for n in exp.notes)
        self.res_label.setText(f"{head}.<br>Saved to {html.escape(str(out))}{notes}")
        self.tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------ experiment design
    def _sync_design(self):
        """Refresh the control list and the comparison list from the current conditions."""
        conds = self.conditions()
        self.control.blockSignals(True)
        self.control.clear()
        self.control.addItem("(none)", "")
        for c in conds:
            self.control.addItem(c, c)
        if self._control and condition_key(self._control) not in {condition_key(c) for c in conds}:
            self.control.addItem(f"{self._control} (not among the images)", self._control)
        idx = next((i for i in range(self.control.count())
                    if condition_key(self.control.itemData(i) or "") == condition_key(self._control)), 0)
        self.control.setCurrentIndex(idx if self._control else 0)
        self.control.blockSignals(False)
        self.comp_list.clear()
        for a, b in self._comparisons:
            self.comp_list.addItem(f"{b}   vs   {a}")
        if not self._comparisons:
            self.comp_list.addItem("(each condition vs the control)" if self._control else "(none: set a control or add)")

    def _control_changed(self, *_):
        self._control = self.control.currentData() or ""
        self._sync_design()

    def _choose_conditions(self, title: str, labels: tuple[str, str]) -> list[str] | None:
        conds = self.conditions()
        if len(conds) < 2:
            message(self, title, "Add images of at least two conditions first.")
            return None
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        form = QFormLayout(dlg)
        boxes = []
        for i, lab in enumerate(labels):
            c = QComboBox()
            c.addItems(conds)
            c.setCurrentIndex(min(i, len(conds) - 1) if i == 0 else 1)
            form.addRow(lab, c)
            boxes.append(c)
        if self._control and self._control in conds:
            boxes[1].setCurrentIndex(conds.index(self._control))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if not dlg.exec():
            return None
        return [b.currentText() for b in boxes]

    def add_comparison(self):
        got = self._choose_conditions("Add a comparison", ("Condition:", "compared with:"))
        if got and got[0] != got[1]:
            self._comparisons.append([got[1], got[0]])
            self._sync_design()

    def remove_comparison(self):
        rows = sorted({self.comp_list.row(i) for i in self.comp_list.selectedItems()}, reverse=True)
        for r in rows:
            if r < len(self._comparisons):
                self._comparisons.pop(r)
        self._sync_design()

    def edit_order(self):
        conds = self.conditions()
        if not conds:
            message(self, "Conditions", "Add images first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Order and fold-change references")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Order of the conditions, and what each fold change is relative to."))
        table = QTableWidget(len(conds), 2)
        table.setHorizontalHeaderLabels(["Condition", "Fold change relative to"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        refs = {condition_key(k): val for k, val in self._references.items()}

        def fill(order):
            for r, c in enumerate(order):
                it = QTableWidgetItem(c)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                table.setItem(r, 0, it)
                box = QComboBox()
                box.addItem("the control", "")
                for o in conds:
                    box.addItem(o, o)
                i = box.findData(refs.get(condition_key(c), ""))
                box.setCurrentIndex(max(0, i))
                box.currentIndexChanged.connect(lambda _i, c=c, box=box: refs.__setitem__(condition_key(c), box.currentData()))
                table.setCellWidget(r, 1, box)
            table.resizeColumnsToContents()

        order = list(conds)
        fill(order)
        v.addWidget(table)
        row = QHBoxLayout()

        def move(d):
            r = table.currentRow()
            if 0 <= r + d < len(order) and r >= 0:
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
            by_key = {condition_key(c): c for c in conds}
            self._references = {by_key[k]: r for k, r in refs.items() if r and k in by_key}
            self._sync_design()

    def combine_saved(self):
        if self._busy():
            return
        d = QFileDialog.getExistingDirectory(self, "Folder holding the result folders to combine")
        if not d:
            return
        from ..core.report import find_result_folders

        root = Path(d)
        found = find_result_folders(root, "per_field.csv")
        if not found:
            message(self, "Combine saved results", "No MicrosCount result folders (with per_field.csv) were found there.")
            return
        import datetime as _dt

        from ..bio.report import combine_results

        design = ExperimentDesign.from_settings(self.get_settings())
        out = root / ("microscount_combined_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.log(f"Combining {len(found)} result folder(s): " + ", ".join(f.name for f in found))
        self._start(Task(combine_results, [str(f) for f in found], design, out, pass_progress=False),
                    lambda res: self._show_results(res, f"<b>{len(found)}</b> result folder(s) combined"), "Combining")
