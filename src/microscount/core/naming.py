"""Groups (conditions or samples) and repetitions from file and folder names.

* When the file names in a folder name several groups (a microscope export such as
  ``Experiment_vehicle-1.tif``, ``Experiment_stimulus 30 min-2.tif``), the group is the
  file name without the parts every name shares (the experiment name before the first
  ``_``, a shared ``_suffix``) and without the trailing field number, and each folder is
  one repetition.
* Otherwise each folder is a group and its parent folder the repetition.

Spellings that differ only in case, spaces or separators (``stimulus 30 min`` and
``Stimulus-30min``) are one group, so repetitions line up. Folders with the same name are
told apart by the folders above them, and a folder that holds one group only is read the
way the other folders are. Repetitions of one experiment hold mostly the same groups:
folders that share few of their groups (two studies that both have a ``neat`` sample)
are separate experiments, and a group name found in several of them takes the
experiment's folder name (``StudyA/neat``). The functions work on any objects that carry
the group attribute, ``repetition`` and ``field_no``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

GROUP_SOURCES = ("auto", "name", "folder")
_FIELD_WORDS = r"(?:field|pos|position|img|image|series|xy|view|fov|site|point)"
_FIELD_NO = (
    re.compile(r"^(?P<base>.*?\S)\s*[-_#]\s*(?:" + _FIELD_WORDS + r"[\s_\-]*)?(?P<n>\d+)\s*$", re.I),  # "vehicle-1", "x_002"
    re.compile(r"^(?P<base>.*?\S)\s*\(\s*(?P<n>\d+)\s*\)\s*$"),  # "vehicle (2)"
    re.compile(r"^(?P<base>.*?\S)[\s_\-]*" + _FIELD_WORDS + r"[\s_\-]*(?P<n>\d+)\s*$", re.I),  # "vehicle field 3"
)


def split_field_number(name: str) -> tuple[str, int | None]:
    """``"stimulus 30 min-2"`` -> ``("stimulus 30 min", 2)``.

    A number counts as a field number when it ends the name after ``-``, ``_`` or ``#``,
    in brackets, or after a word such as *field* or *pos*; a number after a plain space
    (``"dose 10"``) is kept as part of the group.
    """
    for rx in _FIELD_NO:
        m = rx.match(name)
        if m:
            return m.group("base").rstrip(" _.#"), int(m.group("n"))  # a final minus sign stays
    return name.strip(), None


_SEPARATORS = re.compile(r"[\s\-_.,;:/\\()\[\]{}'\"`|]+")


def group_key(name: str) -> str:
    """Spelling-insensitive identity of a group: ``"stimulus 30 min"`` == ``"Stimulus-30min"``.

    Case, spaces and separators (``- _ . , ; : / \\ ( ) [ ]`` …) do not count; letters of any
    script, digits and other symbols do, and so does a final minus sign: ``"PES+"``, ``"PES-"``
    and ``"PES"`` are three groups.
    """
    s = (name or "").casefold().strip()
    key = _SEPARATORS.sub("", s)
    if s.endswith(("-", "−", "–")):
        key += "−"
    return key or s


condition_key = group_key


def strip_shared(names: list[str]) -> list[str]:
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


def path_labels(paths: list[str]) -> dict[str, str]:
    """Short labels that tell folders apart: their names (``batch-1``, ``batch-2`` -> ``1``, ``2``), or
    as many of their last path parts as needed (``day1/SEM``, ``day2/SEM`` -> ``1``, ``2``)."""
    uniq = list(dict.fromkeys(str(p) for p in paths))
    if len(uniq) < 2:
        return {u: "1" for u in uniq}
    parts = {u: Path(u).parts or (u,) for u in uniq}
    k = 1
    while True:
        labels = {u: "/".join(parts[u][-k:]) for u in uniq}
        if len(set(labels.values())) == len(uniq) or all(len(parts[u]) <= k for u in uniq):
            break
        k += 1
    if len(set(labels.values())) < len(uniq):  # cannot happen for distinct paths; keep them apart anyway
        labels = {u: u for u in uniq}
    short = short_labels(list(labels.values()))
    return {u: short[labels[u]] for u in uniq}


def _tokens_match(tokens: list[str], group: str) -> bool:
    gt = [group_key(t) for t in group.split("_")]
    keys = [group_key(t) for t in tokens]
    return any(keys[i:i + len(gt)] == gt for i in range(len(keys) - len(gt) + 1))


def _single_group(names: list[str], base: str, others: list[list[str]], known: list[str]) -> str:
    """Group of a folder whose files all show one group, read the way the other folders' names are read.

    A folder of one repetition may hold only one sample (``Membrane_CA_5kx_01``): its names
    share everything, so the group is found by reading them together with the names of a
    folder that holds several groups, or as a group of that folder within the name.
    """
    for o in others:
        own = [group_key(b) for b in parse_names(o)[0]]
        joint = parse_names(o + names)[0]
        if [group_key(b) for b in joint[:len(o)]] == own:  # the other folder reads the same: same scheme
            cand = {group_key(b): b for b in joint[len(o):]}
            if len(cand) == 1 and next(iter(cand)):
                return next(iter(cand.values()))
    tokens = base.split("_")
    hits = [g for g in known if _tokens_match(tokens, g)]
    return max(hits, key=lambda g: len(group_key(g))) if hits else base


def parse_names(names: list[str]) -> tuple[list[str], list[int | None]]:
    """Group part and field number of the names of one folder.

    The field number is split off first (``img_1`` -> ``img``, 1), then the parts all names
    share are removed (``Exp_vehicle``, ``Exp_drug`` -> ``vehicle``, ``drug``). A number
    hidden before a shared suffix (``vehicle-1_20x``) is found after the suffix is removed.
    """
    split = [split_field_number(n) for n in names]
    bases, nums = strip_shared([b for b, _ in split]), [k for _, k in split]
    if any(k is None for k in nums):
        split2 = [split_field_number(n) for n in strip_shared(names)]
        if sum(k is not None for _, k in split2) > sum(k is not None for k in nums):
            bases, nums = strip_shared([b for b, _ in split2]), [k for _, k in split2]
    return bases, nums


def assign_groups(items: list, path_of: Callable, name_of: Callable | None = None, source: str = "auto",
                  attr: str = "condition") -> str:
    """Set ``attr``, ``repetition`` and ``field_no`` of every item from its file and folder names.

    ``source``: ``"name"`` (file names), ``"folder"`` (folder names) or ``"auto"``: file names when
    those of any one folder name more than one group, folder names otherwise. Returns the source used.
    """
    if source not in GROUP_SOURCES:
        raise ValueError(f"groups from {source!r}: use one of {', '.join(GROUP_SOURCES)}")
    name_of = name_of or (lambda p: Path(p).stem)
    folders: dict[str, list] = {}
    for it in items:
        folders.setdefault(str(Path(path_of(it)).parent), []).append(it)
    names = {f: [name_of(path_of(it)) for it in its] for f, its in folders.items()}
    parsed = {f: parse_names(names[f]) for f in folders}
    multi = [f for f in folders if len({group_key(b) for b in parsed[f][0]}) > 1]
    used = "name" if source == "name" or (source == "auto" and multi) else "folder"
    if used == "name":
        singles = [f for f in folders if f not in multi and parsed[f][0]]
        if not multi and len(singles) > 1:  # every folder one group: what their names do not share
            single = dict(zip(singles, strip_shared([parsed[f][0][0] for f in singles])))
        else:
            known = list(dict.fromkeys(b for f in multi for b in parsed[f][0]))
            single = {f: _single_group(names[f], parsed[f][0][0], [names[m] for m in multi], known) for f in singles}
        sets = {f: {group_key(single.get(f, b)) for b in parsed[f][0]} for f in folders}
        rep, exp = _split_experiments(sets)
        shared = _shared_groups(sets, exp)
        for f, its in folders.items():
            bases, nums = parsed[f]
            for it, base, n in zip(its, bases, nums):
                g = single.get(f, base)
                setattr(it, attr, f"{exp[f]}/{g}" if group_key(g) in shared else g)
                it.field_no = n
                it.repetition = rep[f]
    else:
        group, rep = _folder_groups(list(folders))
        for f, its in folders.items():
            for it, n in zip(its, parsed[f][1]):
                setattr(it, attr, group[f])
                it.field_no = n
                it.repetition = rep[f]
    unify_groups(items, attr)
    return used


_REPETITION_NAME = re.compile(
    r"^(?:(?:rep|repeat|repetition|replicate|batch|run|day|exp|experiment|series|trial|set|r)[\s_\-#.]*\d+"
    r"|\d{1,4}[.\-_]\d{1,2}[.\-_]\d{1,4})$", re.I)


def looks_like_repetition(name: str) -> bool:
    """``rep1``, ``Batch 2``, ``day-3``, ``R4``, a date: a folder named after a repetition, not a group."""
    return bool(_REPETITION_NAME.match(name.strip()))


def _folder_groups(folders: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Group and repetition of each image folder when folders name the groups.

    Usually a folder is a group and its parent folder the repetition (``rep1/vehicle``). When
    every image folder is named like a repetition and the parents are not (``CA/batch-1``,
    ``CA/batch-2``), the parent is the group and the folder the repetition. A group name found
    in more than one place for the same repetition (``PES/neat/batch-1``, ``PSf/neat/batch-1``)
    takes as much of its path as tells the places apart (``PES/neat``, ``PSf/neat``).
    """
    names = {f: Path(f).name for f in folders}
    parent = {f: str(Path(f).parent) for f in folders}
    swap = all(looks_like_repetition(n) for n in names.values()) and not all(
        looks_like_repetition(Path(p).name) for p in parent.values())
    if not swap:
        sets: dict[str, set[str]] = {}
        for f in folders:
            sets.setdefault(parent[f], set()).add(group_key(names[f]))
        rep, exp = _split_experiments(sets)
        shared = _shared_groups(sets, exp)
        group = {f: f"{exp[parent[f]]}/{n}" if group_key(n) in shared else n for f, n in names.items()}
        return group, {f: rep[parent[f]] for f in folders}
    short = short_labels(list(names.values())) if len(set(names.values())) > 1 else {}
    rep = {f: short.get(names[f], names[f]) if short else "1" for f in folders}
    group = {f: Path(parent[f]).name for f in folders}
    places: dict[str, dict[str, set[str]]] = {}  # group -> repetition -> the folders it is in
    for f in folders:
        places.setdefault(group_key(group[f]), {}).setdefault(rep[f], set()).add(parent[f])
    for by_rep in places.values():
        if all(len(dirs) < 2 for dirs in by_rep.values()):
            continue  # one place per repetition: one group, even if its batches sit in different folders
        dirs = set().union(*by_rep.values())
        parts = {d: Path(d).parts for d in dirs}
        k = 2
        while len({"/".join(p[-k:]) for p in parts.values()}) < len(dirs) and any(len(p) > k for p in parts.values()):
            k += 1
        for f in folders:
            if parent[f] in dirs:
                group[f] = "/".join(parts[parent[f]][-k:])
    return group, rep


def _split_experiments(sets: dict[str, set[str]]) -> tuple[dict[str, str], dict[str, str]]:
    """Repetition and experiment label of each folder, from the groups each folder holds.

    Folders are repetitions of one experiment when they share at least half of their groups
    (Jaccard index), directly or through other folders, or when both are named like
    repetitions (``rep1``, a date). Repetitions are labelled within their experiment and
    experiments by the first part of their path that tells them apart; with one experiment
    the experiment label is empty.
    """
    dirs = list(sets)
    root = {d: d for d in dirs}

    def find(d: str) -> str:
        while root[d] != d:
            root[d] = root[root[d]]
            d = root[d]
        return d

    for i, a in enumerate(dirs):
        for b in dirs[i + 1:]:
            sa, sb = sets[a], sets[b]
            linked = bool(sa and sb) and len(sa & sb) >= 0.5 * len(sa | sb)
            if linked or (looks_like_repetition(Path(a).name) and looks_like_repetition(Path(b).name)):
                root[find(b)] = find(a)
    experiments: dict[str, list[str]] = {}
    for d in dirs:
        experiments.setdefault(find(d), []).append(d)
    names = _experiment_names(list(experiments.values())) if len(experiments) > 1 else {}
    rep, exp = {}, {}
    for key, members in experiments.items():
        reps = path_labels(members) if len(members) > 1 else {}
        for d in members:
            rep[d], exp[d] = reps.get(d, "1"), names.get(key, "")
    return rep, exp


def _experiment_names(experiments: list[list[str]]) -> dict[str, str]:
    """Experiment label of each folder: the first parts of its experiment's shared path, after
    the path all experiments share, that tell the experiments apart."""
    try:
        bases = [os.path.commonpath(m) if len(m) > 1 else m[0] for m in experiments]
        top = os.path.commonpath(bases) if len(bases) > 1 else ""
        rel = [Path(os.path.relpath(b, top)).parts if top else Path(b).parts for b in bases]
    except ValueError:  # different drives
        rel = [Path(m[0]).parts for m in experiments]
    k = 1
    while len({"/".join(r[:k]) for r in rel}) < len(rel) and any(len(r) > k for r in rel):
        k += 1
    out = {}
    for m, r in zip(experiments, rel):
        label = "/".join(r[:k]) if r and r != (".",) else Path(m[0]).name
        for d in m:
            out[d] = label
    return out


def _shared_groups(sets: dict[str, set[str]], exp: dict[str, str]) -> set[str]:
    """Group keys found in more than one experiment."""
    where: dict[str, set[str]] = {}
    for d, keys in sets.items():
        for k in keys:
            where.setdefault(k, set()).add(exp[d])
    return {k for k, e in where.items() if len(e) > 1}


def unify_groups(items: list, attr: str = "condition") -> None:
    """One spelling per group: the first one met (repetitions often differ by a space)."""
    first: dict[str, str] = {}
    for it in items:
        k = group_key(getattr(it, attr))
        if k:
            setattr(it, attr, first.setdefault(k, getattr(it, attr)))


def number_from_id(item_id: str) -> int | None:
    """The number of an id made by ``label_items``: ``"3 · vehicle #2"`` -> 2."""
    m = re.search(r"#(\d+)'*$", item_id or "")
    return int(m.group(1)) if m else None


def label_items(items: list, path_of: Callable, attr: str = "condition", id_attr: str = "field_id",
                empty: str = "(no condition)") -> None:
    """Readable, unique ids: ``"vehicle #1"``, or ``"3 · vehicle #1"`` when there are several repetitions."""
    several = len({it.repetition for it in items}) > 1
    groups: dict[tuple[str, str], list] = {}
    for it in items:
        groups.setdefault((it.repetition, group_key(getattr(it, attr))), []).append(it)
    seen: set[str] = set()
    for (rep, _k), its in groups.items():
        nums = [it.field_no for it in its]
        use = all(n is not None for n in nums) and len(set(nums)) == len(nums)
        for i, it in enumerate(sorted(its, key=lambda q: (q.field_no if use else 0, Path(path_of(q)).name)), start=1):
            n = it.field_no if use else i
            base = f"{getattr(it, attr) or empty} #{n}"
            fid = f"{rep} · {base}" if several and rep else base
            while fid in seen:  # same names in two folders of one repetition
                fid += "'"
            seen.add(fid)
            setattr(it, id_attr, fid)
