"""Group images into fields of view (nuclear stain + target channel).

Rules, in order:

1. A multi-channel TIFF or merged RGB image is a field on its own.
2. A single-colour RGB export is nuclear if blue/cyan, target otherwise.
3. A greyscale file gets its role from its name (DAPI, Hoechst, GFP, AF488 ...).
4. Nuclear and target files are paired when their names match after removing
   channel words (``img01_ch00`` <-> ``img01_ch01``, ``C1-x`` <-> ``C2-x``).
5. Whatever is left is paired by image content: nuclei appear in both channels
   (bright or dark), so the band-passed images of a true pair correlate; the
   assignment maximising total |correlation| is used and weak pairs are flagged.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .imageio import LoadedImage, load_image

NUCLEAR_WORDS = ("dapi", "hoechst", "nuclear", "nuclei", "nucleus", "nuc", "draq5", "h33342", "blue")
TARGET_WORDS = (
    "green", "red", "gfp", "fitc", "tritc", "cy3", "cy5", "af488", "af555", "af568", "af594", "af633", "af647",
    "alexa488", "alexa555", "alexa568", "alexa594", "alexa633", "alexa647", "488", "555", "568", "594", "633", "647",
    "texasred", "rhodamine", "p65", "rela", "target",
)
CHANNEL_INDEX = r"(?:ch\d+|c\d+|w\d+|channel\d+)"
_SPLIT = re.compile(r"[\s_\-\.\(\)\[\]]+")


@dataclass
class FileEntry:
    path: str
    role: str  # "nuclear" | "target" | "both" | "unknown"
    kind: str
    description: str
    key: str
    thumb: np.ndarray | None = field(default=None, repr=False)
    error: str = ""


@dataclass
class Pairing:
    nuclear: str
    target: str
    condition: str = ""
    field_id: str = ""
    confidence: str = "name"  # "self" | "name" | "image" | "manual" | "weak"
    note: str = ""


def _tokens(stem: str) -> list[str]:
    return [t for t in _SPLIT.split(stem.lower()) if t]


def field_key(path: str | Path) -> str:
    """Filename stem with channel words removed (``img_ch00`` -> ``img``)."""
    stem = Path(path).stem
    stem = re.sub(r"(?i)^c\d+-", "", stem)  # Fiji "C1-name"
    toks = _tokens(stem)
    keep = [
        t for t in toks
        if not re.fullmatch(CHANNEL_INDEX, t) and t not in NUCLEAR_WORDS and t not in TARGET_WORDS
    ]
    return "_".join(keep) if keep else "_".join(toks)


def role_from_name(path: str | Path) -> str:
    toks = set(_tokens(Path(path).stem))
    if toks & set(NUCLEAR_WORDS):
        return "nuclear"
    if toks & set(TARGET_WORDS):
        return "target"
    return "unknown"


def role_from_image(img: LoadedImage, path: str) -> str:
    if img.kind in ("multichannel", "merged_rgb"):
        return "both"
    if img.kind == "single_colour":
        return "nuclear" if img.colour in ("blue", "cyan") else "target"
    return role_from_name(path)


def thumbnail(img: LoadedImage, max_side: int = 384) -> np.ndarray:
    """Band-passed, standardised thumbnail of the signal channel for matching."""
    ch = img.resolve_channel("auto") if img.kind in ("grey", "single_colour") else 0
    a = img.data[ch].astype(np.float32)
    if img.annotation_mask is not None:
        a = np.where(img.annotation_mask, np.median(a), a)
    f = max(1, int(math.ceil(max(a.shape) / max_side)))
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    a = a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    b = ndi.gaussian_filter(a, 1.0) - ndi.gaussian_filter(a, 6.0)
    b -= b.mean()
    sd = b.std()
    return b / sd if sd > 0 else b


def scan_files(paths: list[str | Path], progress=None, cancel=None) -> list[FileEntry]:
    entries = []
    for i, p in enumerate(paths):
        if cancel and cancel():
            break
        p = str(p)
        if progress:
            progress(i, len(paths), Path(p).name)
        try:
            img = load_image(p)
            role = role_from_image(img, p)
            thumb = thumbnail(img) if role in ("nuclear", "target", "unknown") else None
            entries.append(FileEntry(p, role, img.kind, img.describe(), field_key(p), thumb))
        except Exception as exc:  # noqa: BLE001
            entries.append(FileEntry(p, "unknown", "error", "", field_key(p), None, str(exc)))
    return entries


def _similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(abs(np.mean(a * b)))


def pair_files(entries: list[FileEntry], condition_from_folder: bool = True) -> tuple[list[Pairing], list[FileEntry]]:
    """Return (pairings, unpaired entries)."""
    from scipy.optimize import linear_sum_assignment

    pairs: list[Pairing] = []
    good = [e for e in entries if not e.error]
    unpaired = [e for e in entries if e.error]
    for e in good:
        if e.role == "both":
            pairs.append(Pairing(e.path, e.path, confidence="self"))
    rest = [e for e in good if e.role != "both"]

    # 1) name-based groups
    groups: dict[tuple[str, str], list[FileEntry]] = {}
    for e in rest:
        groups.setdefault((str(Path(e.path).parent), e.key), []).append(e)
    leftovers: list[FileEntry] = []
    for (_, _key), g in groups.items():
        nucs = [e for e in g if e.role == "nuclear"]
        tgts = [e for e in g if e.role == "target"]
        unk = [e for e in g if e.role == "unknown"]
        if len(g) == 2 and len(nucs) == 1 and len(tgts) == 1:
            pairs.append(Pairing(nucs[0].path, tgts[0].path, confidence="name"))
        elif len(g) == 2 and len(unk) == 1 and (len(nucs) == 1 or len(tgts) == 1):
            n = nucs[0] if nucs else unk[0]
            t = tgts[0] if tgts else unk[0]
            pairs.append(Pairing(n.path, t.path, confidence="name"))
        elif len(g) == 2 and len(unk) == 2:
            a, b = sorted(g, key=lambda e: e.path)
            pairs.append(Pairing(a.path, b.path, confidence="weak",
                                 note="roles guessed from file order; swap if wrong"))
        else:
            leftovers.extend(g)

    # 2) content-based assignment for the rest
    nucs = [e for e in leftovers if e.role == "nuclear"]
    tgts = [e for e in leftovers if e.role == "target"]
    others = [e for e in leftovers if e.role == "unknown"]
    if nucs and tgts:
        sim = np.array([[_similarity(n.thumb, t.thumb) for t in tgts] for n in nucs])
        rows, cols = linear_sum_assignment(-sim)
        used_n, used_t = set(), set()
        for r, c in zip(rows, cols):
            s = sim[r, c]
            conf = "image" if s >= 0.3 else "weak"
            note = f"matched by image content (r = {s:.2f})"
            if s < 0.3:
                note += "; check this pair"
            pairs.append(Pairing(nucs[r].path, tgts[c].path, confidence=conf, note=note))
            used_n.add(r)
            used_t.add(c)
        unpaired += [n for i, n in enumerate(nucs) if i not in used_n]
        unpaired += [t for i, t in enumerate(tgts) if i not in used_t]
    else:
        unpaired += nucs + tgts
    unpaired += others

    # field ids and default conditions
    pairs.sort(key=lambda q: (str(Path(q.nuclear).parent), Path(q.nuclear).name))
    for i, p in enumerate(pairs):
        p.field_id = f"field_{i + 1:03d}"
        if condition_from_folder and not p.condition:
            p.condition = Path(p.nuclear).parent.name
    return pairs, unpaired
