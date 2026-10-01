"""The two MicrosCount modules and the analyses they contain.

* **Bio & Cells** (``bio``): nuclear translocation of a fluorescent protein (N/C ratio).
* **Materials & Mechanics** (``materials``): porosity and pore sizes of SEM images.

The GUI, the command line and saved settings files all use this registry, so a new
analysis only has to be added here (plus its page and its ``run_*`` function).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Analysis:
    key: str  # command-line name and settings-file tag, e.g. "translocation"
    module: str  # key of the module it belongs to
    title: str
    summary: str
    settings: str  # "package.module:Class" of its settings dataclass


@dataclass(frozen=True)
class Module:
    key: str
    title: str
    summary: str
    analyses: tuple[Analysis, ...]


MODULES: tuple[Module, ...] = (
    Module(
        "bio",
        "Bio & Cells",
        "Fluorescence microscopy of cells",
        (
            Analysis(
                "translocation", "bio", "Nuclear translocation",
                "Nuclear/cytoplasmic ratio of a fluorescent protein (e.g. NF-κB p65): Noursadeghi et al. (2008) "
                "per field, and the per-cell lab protocol",
                "microscount.bio.translocation:TranslocationSettings",
            ),
        ),
    ),
    Module(
        "materials",
        "Materials & Mechanics",
        "Electron microscopy of materials",
        (
            Analysis(
                "porosity", "materials", "SEM porosity",
                "Porosity and pore-size distribution of membranes and porous materials (port of A. Rabbani's "
                "SEM_Porosity)",
                "microscount.materials.porosity:PorositySettings",
            ),
        ),
    ),
)

ANALYSES: dict[str, Analysis] = {a.key: a for m in MODULES for a in m.analyses}


def module(key: str) -> Module:
    for m in MODULES:
        if m.key == key:
            return m
    raise KeyError(f"unknown module {key!r}; choose from {', '.join(m.key for m in MODULES)}")


def analysis(key: str) -> Analysis:
    try:
        return ANALYSES[key]
    except KeyError:
        raise KeyError(f"unknown analysis {key!r}; choose from {', '.join(ANALYSES)}") from None


def settings_class(key: str):
    mod, _, cls = analysis(key).settings.partition(":")
    return getattr(importlib.import_module(mod), cls)


def load_settings(path: str | Path):
    """Read a settings.yaml written by MicrosCount: ``(analysis, settings, inputs)``.

    Files from version 0.1 stored the analysis under ``module``; both forms are read.
    """
    from .core.report import read_settings_yaml

    doc = read_settings_yaml(path)
    key = doc.get("analysis") or doc.get("module")
    if key not in ANALYSES:
        raise ValueError(f"{Path(path).name} is not a MicrosCount settings file (no known analysis)")
    a = analysis(key)
    return a, settings_class(key).from_dict(doc.get("settings") or {}), doc.get("inputs") or []
