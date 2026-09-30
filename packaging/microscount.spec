# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for MicrosCount: one folder with a windowed app ("MicrosCount")
# and a console twin ("microscount-cli"); a .app bundle on macOS.
# Build from the repository root:  pyinstaller --noconfirm packaging/microscount.spec

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from microscount import __version__  # noqa: E402

if sys.platform == "win32":
    ICON = ROOT / "packaging" / "icons" / "microscount.ico"
elif sys.platform == "darwin":
    ICON = ROOT / "packaging" / "icons" / "microscount.icns"
else:
    ICON = ROOT / "packaging" / "icons" / "microscount.png"

hidden = ["png", "yaml", "openpyxl", "tifffile", "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_agg"]
for pkg in ("skimage.filters", "skimage.morphology", "skimage.segmentation", "skimage.measure",
            "skimage.registration", "skimage.feature", "skimage.exposure", "skimage.util", "skimage._shared"):
    hidden += collect_submodules(pkg)
hidden += collect_submodules("microscount")

datas = [(str(ROOT / "src" / "microscount" / "gui" / "resources"), "microscount/gui/resources")]
datas += collect_data_files("skimage", includes=["**/*.pyi"])  # lazy-loader stubs
datas += [(str(ROOT / "LICENSE"), "."), (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."), (str(ROOT / "CITATION.cff"), ".")]

a = Analysis(
    [str(ROOT / "packaging" / "launch_microscount.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "IPython", "pandas", "pytest", "sphinx", "notebook",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick3D",
              "cv2", "imageio", "imageio_ffmpeg", "sklearn", "lxml", "gi", "gtk", "torch", "tensorflow", "dask",
              "numba", "pyarrow", "zarr", "h5py", "sympy", "pooch", "networkx", "jupyter", "jedi", "cffconvert"],
    noarchive=False,
)
# Qt's optional GTK platform theme would drag the build machine's GTK stack into the bundle
_drop = ("libqgtk3", "libgtk-3", "libgdk-3", "libatk-1", "libatk-bridge", "libgdk_pixbuf", "libcairo-gobject",
         "libepoxy", "libatspi")
a.binaries = [b for b in a.binaries if not any(x in Path(b[0]).name for x in _drop)]
pyz = PYZ(a.pure)

exe_gui = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="MicrosCount",
    console=False,
    icon=str(ICON),
    upx=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
exe_cli = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="microscount-cli",
    console=True,
    icon=str(ICON),
    upx=False,
)
coll = COLLECT(exe_gui, exe_cli, a.binaries, a.datas, strip=False, upx=False, name="MicrosCount")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="MicrosCount.app",
        icon=str(ICON),
        bundle_identifier="io.github.mertol93.microscount",
        version=__version__,
        info_plist={
            "CFBundleName": "MicrosCount",
            "CFBundleDisplayName": "MicrosCount",
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "GPL-3.0-or-later",
            "LSMinimumSystemVersion": "11.0",
        },
    )
