"""Main window and application entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QKeySequence, QPalette
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QTabWidget

from .. import APP_NAME, __version__
from ..modules import MODULES
from .bio_translocation import TranslocationPage
from .common import REPO_URL, AboutDialog, message, resource
from .materials_porosity import PorosityPage

# analysis key -> page class; every analysis in modules.MODULES needs one
PAGES = {"translocation": TranslocationPage, "porosity": PorosityPage}


def _light_palette() -> QPalette:
    p = QPalette()
    base, text, mid = QColor("#ffffff"), QColor("#0b0b0b"), QColor("#f4f4f2")
    p.setColor(QPalette.Window, mid)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, QColor("#f7f7f5"))
    p.setColor(QPalette.ToolTipBase, base)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, QColor("#fbfbfa"))
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.BrightText, QColor("#e34948"))
    p.setColor(QPalette.Highlight, QColor("#2a78d6"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.PlaceholderText, QColor("#8a8984"))
    p.setColor(QPalette.Link, QColor("#256abf"))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        p.setColor(QPalette.Disabled, role, QColor("#a3a29c"))
    return p


class MainWindow(QMainWindow):
    """Two modules (Bio & Cells, Materials & Mechanics), each a tab holding its analyses."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        icon = resource("icon.png")
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        self.tabs = QTabWidget()  # one tab per module
        self.tabs.setDocumentMode(True)
        self.module_tabs: dict[str, QTabWidget] = {}
        self.pages: dict[str, object] = {}
        self.status_label = QLabel("Ready")
        for m in MODULES:
            inner = QTabWidget()
            inner.setDocumentMode(True)
            for an in m.analyses:
                page = PAGES[an.key]()
                page.status.connect(self.status_label.setText)
                inner.addTab(page, an.title)
                inner.setTabToolTip(inner.count() - 1, an.summary)
                self.pages[an.key] = page
            self.tabs.addTab(inner, m.title.replace("&", "&&"))  # a single & would become a keyboard shortcut
            self.tabs.setTabToolTip(self.tabs.count() - 1, m.summary)
            self.module_tabs[m.key] = inner
        self.trans = self.pages["translocation"]
        self.poro = self.pages["porosity"]
        self.setCentralWidget(self.tabs)
        self.statusBar().addWidget(self.status_label, 1)
        self._menus()
        st = QSettings(APP_NAME, APP_NAME)
        geo = st.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1480, 920)

    def page(self):
        return self.tabs.currentWidget().currentWidget()

    def show_analysis(self, key: str):
        from ..modules import analysis

        inner = self.module_tabs[analysis(key).module]
        self.tabs.setCurrentWidget(inner)
        inner.setCurrentWidget(self.pages[key])
        return self.pages[key]

    def _menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        a = QAction("Add images…", self)
        a.setShortcut(QKeySequence.Open)
        a.triggered.connect(lambda: self.page().add_files())
        m.addAction(a)
        a = QAction("Add folder…", self)
        a.triggered.connect(lambda: self.page().add_folder())
        m.addAction(a)
        m.addSeparator()
        a = QAction("Open settings / previous analysis…", self)
        a.triggered.connect(self.load_settings)
        m.addAction(a)
        a = QAction("Save settings…", self)
        a.setShortcut(QKeySequence.Save)
        a.triggered.connect(self.save_settings)
        m.addAction(a)
        a = QAction("Reset settings to defaults", self)
        a.triggered.connect(lambda: self.page().reset_defaults())
        m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.setShortcut(QKeySequence.Quit)
        a.triggered.connect(self.close)
        m.addAction(a)
        h = mb.addMenu("&Help")
        a = QAction("User guide (online)", self)
        a.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(REPO_URL + "#readme")))
        h.addAction(a)
        a = QAction("How to cite…", self)
        a.triggered.connect(lambda: AboutDialog(self).exec())
        h.addAction(a)
        a = QAction("Report a problem (GitHub)", self)
        a.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(REPO_URL + "/issues")))
        h.addAction(a)
        h.addSeparator()
        a = QAction(f"About {APP_NAME}", self)
        a.triggered.connect(lambda: AboutDialog(self).exec())
        h.addAction(a)

    def save_settings(self):
        from ..core.report import write_settings_yaml

        page = self.page()
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", f"microscount_{page.analysis}_settings.yaml",
                                              "Settings (*.yaml *.yml)")
        if path:
            write_settings_yaml(Path(path), page.analysis, page.get_settings())
            self.status_label.setText(f"Settings saved to {path}")

    def load_settings(self):
        from ..modules import load_settings

        path, _ = QFileDialog.getOpenFileName(self, "Open settings or a previous analysis", "",
                                              "Settings (*.yaml *.yml)")
        if not path:
            return
        try:
            an, s, inputs = load_settings(path)
        except Exception as exc:  # noqa: BLE001
            message(self, "Could not read settings", str(exc), QMessageBox.Warning)
            return
        page = self.show_analysis(an.key)
        page.apply_settings(s)
        if inputs:
            r = QMessageBox.question(self, "Load images too?",
                                     f"This file lists {len(inputs)} input(s). Load them as well?")
            if r == QMessageBox.Yes:
                page.load_inputs(inputs)
        self.status_label.setText(f"Settings loaded from {path}")

    def closeEvent(self, e):
        QSettings(APP_NAME, APP_NAME).setValue("geometry", self.saveGeometry())
        super().closeEvent(e)


def run_gui(argv: list[str] | None = None) -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    icon = resource("icon.png")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    w = MainWindow()
    w.show()
    quit_after = os.environ.get("MICROSCOUNT_QUIT_AFTER_MS")  # used by the installer smoke tests
    if quit_after:
        QTimer.singleShot(int(quit_after), app.quit)
    return app.exec()
