"""Group images into fields of view (nuclear stain + target channel).

Rules, in order:

1. A multi-channel TIFF or merged RGB image is a field on its own.
2. A single-colour RGB export is nuclear if blue/cyan, target otherwise.
3. A greyscale file gets its role from its name (DAPI, Hoechst, GFP, AF488 ...).
4. Nuclear and target files are paired when their names match after removing
   channel words (``img01_ch00`` <-> ``img01_ch01``, ``C1-x`` <-> ``C2-x``); other
   files of the same field with no stain role (e.g. a transmitted-light ``_ch02``)
   are set aside as extra channels. Blank images (no signal, e.g. an unused
   detector) are never paired.
5. Whatever is left is paired by image content: nuclei appear in both channels
   (bright or dark), so the band-passed images of a true pair correlate; the
   assignment maximising total |correlation| is used and weak pairs are flagged.

Conditions and repetitions (``assign_conditions``) come from the names:

* when the file names in a folder name several conditions (a microscope export such
  as ``Experiment_vehicle-1_ch00.tif``, ``Experiment_stimulus 30 min-2_ch00.tif``), the
  condition is the file name without the shared experiment prefix and the trailing
  field number, and each folder is one repetition;
* otherwise the folder name is the condition and its parent folder the repetition.

Spellings that differ only in case, spaces or punctuation (``stimulus 30 min`` and
``Stimulus-30min``) are merged, so repetitions line up.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..core.imageio import LoadedImage, load_image

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
    role: str  # "nuclear" | "target" | "both" | "unknown" | "blank"
    kind: str
    description: str
    key: str
    thumb: np.ndarray | None = field(default=None, repr=False)
    error: str = ""
    note: str = ""  # why a readable file was left out (e.g. an extra channel of a paired field)


@dataclass
class Pairing:
    nuclear: str
    target: str
    condition: str = ""
    field_id: str = ""
    confidence: str = "name"  # "self" | "name" | "image" | "manual" | "weak"
    note: str = ""
    repetition: str = ""
    field_no: int | None = None  # field number within its condition (from the file name when it has one)


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


_CHANNEL_WORD = re.compile(
    r"(?i)^(?:ch\d+|c\d+|w\d+|channel\d+|" + "|".join(re.escape(w) for w in NUCLEAR_WORDS + TARGET_WORDS) + r")$")
_FIELD_WORDS = r"(?:field|pos|position|img|image|series|xy|view|fov|site|point)"
_FIELD_NO = (
    re.compile(r"^(?P<base>.*?\S)\s*[-_#]\s*(?:" + _FIELD_WORDS + r"[\s_\-]*)?(?P<n>\d+)\s*$", re.I),  # "vehicle-1", "x_002"
    re.compile(r"^(?P<base>.*?\S)\s*\(\s*(?P<n>\d+)\s*\)\s*$"),  # "vehicle (2)"
    re.compile(r"^(?P<base>.*?\S)[\s_\-]*" + _FIELD_WORDS + r"[\s_\-]*(?P<n>\d+)\s*$", re.I),  # "vehicle field 3"
)


def image_name(path: str | Path) -> str:
    """File name without extension and channel words, spelling kept (``Exp_vehicle-1_ch00.tif`` -> ``Exp_vehicle-1``)."""
    stem = re.sub(r"(?i)^c\d+-", "", Path(path).stem)  # Fiji "C1-name"
    parts = re.split(r"([\s_\-\.]+)", stem)  # words at even, separators at odd positions
    while len(parts) >= 3 and _CHANNEL_WORD.match(parts[-1]):
        parts = parts[:-2]
    while len(parts) >= 3 and _CHANNEL_WORD.match(parts[0]):
        parts = parts[2:]
    return "".join(parts).strip(" _-.") or stem


def split_field_number(name: str) -> tuple[str, int | None]:
    """``"stimulus 30 min-2"`` -> ``("stimulus 30 min", 2)``.

    A number counts as a field number when it ends the name after ``-``, ``_`` or ``#``,
    in brackets, or after a word such as *field* or *pos*; a number after a plain space
    (``"dose 10"``) is kept as part of the condition.
    """
    for rx in _FIELD_NO:
        m = rx.match(name)
        if m:
            return m.group("base").rstrip(" _-.#"), int(m.group("n"))
    return name.strip(), None


def condition_key(name: str) -> str:
    """Spelling-insensitive identity of a condition: ``"stimulus 30 min"`` == ``"Stimulus-30min"``."""
    return re.sub(r"[^0-9a-zà-ÿα-ω]+", "", (name or "").lower())


def _strip_shared(names: list[str]) -> list[str]:
    """Remove the part every name shares up to an underscore: the experiment or project name."""
    if len(set(names)) < 2:
        return names
    cut = os.path.commonprefix(names).rfind("_") + 1
    if cut and all(len(n) > cut for n in names):
        names = [n[cut:] for n in names]
    suffix = os.path.commonprefix([n[::-1] for n in names])[::-1]
    i = suffix.find("_")
    if i >= 0:
        cut = len(suffix) - i
        if all(len(n) > cut for n in names):
            names = [n[:-cut] for n in names]
    return names


def short_labels(labels: list[str]) -> dict[str, str]:
    """Shorten folder names to what tells them apart: ``exp-3``, ``exp-4`` -> ``3``, ``4``."""
    uniq = list(dict.fromkeys(labels))
    out = {u: u for u in uniq}
    if len(uniq) < 2:
        return out
    pre = re.sub(r"\d+$", "", os.path.commonprefix(uniq))  # never cut through a number
    if pre and all(len(u) > len(pre) for u in uniq) and (not pre[-1].isalnum() or all(u[len(pre)].isdigit() for u in uniq)):
        out = {u: u[len(pre):] for u in uniq}
    rest = list(out.values())
    suf = os.path.commonprefix([r[::-1] for r in rest])[::-1]
    if suf and not suf[0].isalnum() and all(len(r) > len(suf) for r in rest):
        out = {u: v[: -len(suf)] for u, v in out.items()}
    out = {u: v.strip(" _-.") or u for u, v in out.items()}
    return out if len(set(out.values())) == len(uniq) else {u: u for u in uniq}


CONDITION_SOURCES = ("auto", "name", "folder")


def assign_conditions(pairs: list[Pairing], source: str = "auto") -> str:
    """Set the condition, repetition and field number of every pair from its names.

    ``source``: ``"name"`` (file names), ``"folder"`` (folder names) or ``"auto"``: file names when
    those of any one folder name more than one condition, folder names otherwise. Returns the
    source used. Manual edits are the caller's to restore.
    """
    if source not in CONDITION_SOURCES:
        raise ValueError(f"conditions from {source!r}: use one of {', '.join(CONDITION_SOURCES)}")
    folders: dict[str, list[Pairing]] = {}
    for p in pairs:
        folders.setdefault(str(Path(p.nuclear).parent), []).append(p)
    parsed: dict[int, tuple[str, int | None]] = {}
    several = False
    for ps in folders.values():
        names = _strip_shared([image_name(p.nuclear) for p in ps])
        split = [split_field_number(n) for n in names]
        for p, (base, n) in zip(ps, split):
            parsed[id(p)] = (base, n)
        several |= len({condition_key(b) for b, _ in split}) > 1
    used = "name" if source == "name" or (source == "auto" and several) else "folder"
    if used == "name":
        reps = short_labels([Path(f).name for f in folders]) if len(folders) > 1 else {}
        for f, ps in folders.items():
            for p in ps:
                p.condition, p.field_no = parsed[id(p)]
                p.repetition = reps.get(Path(f).name, "1")
    else:
        parents = [Path(f).parent.name for f in folders]
        reps = short_labels(parents) if len(set(parents)) > 1 else {}
        for f, ps in folders.items():
            for p in ps:
                p.condition = Path(f).name
                p.field_no = parsed[id(p)][1]
                p.repetition = reps.get(Path(f).parent.name, "1")
    unify_conditions(pairs)
    return used


def unify_conditions(pairs: list[Pairing]) -> None:
    """One spelling per condition: the first one met (repetitions often differ by a space)."""
    first: dict[str, str] = {}
    for p in pairs:
        k = condition_key(p.condition)
        if k:
            p.condition = first.setdefault(k, p.condition)


def label_fields(pairs: list[Pairing]) -> None:
    """Readable, unique field ids: ``"vehicle #1"``, or ``"3 · vehicle #1"`` when there are several repetitions."""
    several = len({p.repetition for p in pairs}) > 1
    groups: dict[tuple[str, str], list[Pairing]] = {}
    for p in pairs:
        groups.setdefault((p.repetition, condition_key(p.condition)), []).append(p)
    seen: set[str] = set()
    for (rep, _k), ps in groups.items():
        nums = [p.field_no for p in ps]
        use = all(n is not None for n in nums) and len(set(nums)) == len(nums)
        for i, p in enumerate(sorted(ps, key=lambda q: (q.field_no if use else 0, Path(q.nuclear).name)), start=1):
            n = p.field_no if use else i
            base = f"{p.condition or '(no condition)'} #{n}"
            fid = f"{rep} · {base}" if several and rep else base
            while fid in seen:  # same names in two folders of one repetition
                fid += "'"
            seen.add(fid)
            p.field_id = fid


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


def is_blank(img: LoadedImage) -> bool:
    """True for a channel without signal, e.g. an unused detector saved as ``_ch02``."""
    if img.kind not in ("grey", "single_colour"):
        return False
    a = img.data[img.resolve_channel("auto")]
    step = max(1, int(math.sqrt(a.size / 250_000)))
    lo, hi = np.percentile(a[::step, ::step], [1, 99.9])
    return float(hi - lo) <= 2


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
            role = "blank" if is_blank(img) else role_from_image(img, p)
            thumb = thumbnail(img) if role in ("nuclear", "target", "unknown") else None
            entries.append(FileEntry(p, role, img.kind, img.describe(), field_key(p), thumb))
        except Exception as exc:  # noqa: BLE001
            entries.append(FileEntry(p, "unknown", "error", "", field_key(p), None, str(exc)))
    return entries


def _similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(abs(np.mean(a * b)))


def pair_files(entries: list[FileEntry], condition_from_folder: bool = True,
               conditions_from: str = "auto") -> tuple[list[Pairing], list[FileEntry]]:
    """Return (pairings, unpaired entries).

    Conditions, repetitions and field ids are set by ``assign_conditions`` (``conditions_from``);
    ``condition_from_folder=False`` leaves the conditions empty.
    """
    from scipy.optimize import linear_sum_assignment

    pairs: list[Pairing] = []
    for e in entries:
        e.note = ""
    good = [e for e in entries if not e.error]
    unpaired = [e for e in entries if e.error]
    for e in good:
        if e.role == "both":
            pairs.append(Pairing(e.path, e.path, confidence="self"))
        elif e.role == "blank":
            e.note = "blank image (no signal), not used"
            unpaired.append(e)
    rest = [e for e in good if e.role not in ("both", "blank")]

    # 1) name-based groups
    groups: dict[tuple[str, str], list[FileEntry]] = {}
    for e in rest:
        groups.setdefault((str(Path(e.path).parent), e.key), []).append(e)
    leftovers: list[FileEntry] = []
    for (_, _key), g in groups.items():
        nucs = [e for e in g if e.role == "nuclear"]
        tgts = [e for e in g if e.role == "target"]
        unk = [e for e in g if e.role == "unknown"]
        if len(nucs) == 1 and len(tgts) == 1:
            # one nuclear and one target file; any others of the same field are extra channels
            # (e.g. the transmitted-light _ch02 of a Leica export) and are not analysed
            pairs.append(Pairing(nucs[0].path, tgts[0].path, confidence="name"))
            for e in unk:
                e.note = "extra channel of a paired field, not used"
                unpaired.append(e)
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

    # conditions, repetitions and field ids
    pairs.sort(key=lambda q: (str(Path(q.nuclear).parent), Path(q.nuclear).name))
    if condition_from_folder:
        assign_conditions(pairs, conditions_from)
    label_fields(pairs)
    return pairs, unpaired
