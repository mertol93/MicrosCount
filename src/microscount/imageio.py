"""Loading of TIFF, PNG and JPEG microscopy images.

Every file is returned as a :class:`LoadedImage` holding a ``(C, Y, X)`` array in
its native dtype, plus what the analyses need to know about it:

* what kind of image it is (greyscale, a single-colour RGB export such as a
  blue DAPI snapshot, a merged RGB composite, or a multi-channel TIFF);
* which colour carries the signal in a single-colour export;
* a mask of burned-in annotations (scale bars, text) detected from colour;
* the saturation value, and the pixel size when the file records one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
LOSSY_EXTENSIONS = (".jpg", ".jpeg")
RGB_NAMES = ("red", "green", "blue")
_COLOUR_BY_SET = {
    frozenset({0}): "red",
    frozenset({1}): "green",
    frozenset({2}): "blue",
    frozenset({0, 1}): "yellow",
    frozenset({0, 2}): "magenta",
    frozenset({1, 2}): "cyan",
    frozenset({0, 1, 2}): "grey",
}


class ImageLoadError(RuntimeError):
    """Raised when a file cannot be read as an image."""


class ChannelError(ValueError):
    """Raised when a requested channel does not exist or is ambiguous."""


@dataclass
class LoadedImage:
    path: Path
    data: np.ndarray  # (C, Y, X), native dtype
    channel_names: list[str]
    kind: str  # "grey" | "single_colour" | "merged_rgb" | "multichannel"
    saturation_value: float | None
    lossy: bool = False
    pixel_size_um: float | None = None
    annotation_mask: np.ndarray | None = None  # True = burned-in overlay pixel
    signal_channel: int | None = None  # populated channel of a single-colour export
    colour: str | None = None  # display colour of a single-colour export
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.data.shape[1:])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def name(self) -> str:
        return self.path.name

    def describe(self) -> str:
        y, x = self.shape
        dt = str(self.data.dtype)
        if self.kind == "single_colour":
            what = f"single-colour RGB export ({self.colour})"
        elif self.kind == "merged_rgb":
            what = "merged RGB composite"
        elif self.kind == "multichannel":
            what = f"{self.n_channels}-channel image ({', '.join(self.channel_names)})"
        else:
            what = "greyscale"
        return f"{x}×{y} {dt} {what}"

    def resolve_channel(self, spec: str | int | None = "auto") -> int:
        """Return the channel index for ``spec``.

        ``spec`` may be ``"auto"``, an integer index, a channel name
        (``"blue"``, ``"DAPI"``, ...) or ``"chN"`` (0-based).
        """
        names = [n.lower() for n in self.channel_names]
        if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "auto")):
            if self.kind == "grey" or self.n_channels == 1:
                return 0
            if self.kind == "single_colour" and self.signal_channel is not None:
                return self.signal_channel
            raise ChannelError(
                f"{self.name} has {self.n_channels} channels ({', '.join(self.channel_names)}); "
                "choose which one to use"
            )
        if isinstance(spec, (int, np.integer)):
            idx = int(spec)
        else:
            s = str(spec).strip().lower()
            if s in names:
                return names.index(s)
            m = re.fullmatch(r"(?:ch|channel)\s*0*(\d+)", s)
            if s.isdigit():
                idx = int(s)
            elif m:
                idx = int(m.group(1))
            elif s in RGB_NAMES and self.n_channels == 1:
                # a greyscale file asked for by colour: there is only one channel
                return 0
            else:
                raise ChannelError(
                    f"{self.name}: no channel '{spec}' (available: {', '.join(self.channel_names)})"
                )
        if not 0 <= idx < self.n_channels:
            raise ChannelError(f"{self.name}: channel index {idx} out of range 0..{self.n_channels - 1}")
        return idx

    def channel(self, spec: str | int | None = "auto") -> np.ndarray:
        return self.data[self.resolve_channel(spec)]


# --------------------------------------------------------------------------- loading


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def list_images(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Supported image files under ``folder``, skipping hidden and result folders."""
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    out = []
    for p in sorted(folder.glob(pattern)):
        if not p.is_file() or not is_supported(p):
            continue
        rel = p.relative_to(folder).parts
        if any(part.startswith(".") for part in rel):
            continue
        if any(part.lower().startswith("microscount_results") for part in rel[:-1]):
            continue
        out.append(p)
    return out


def load_image(path: str | Path) -> LoadedImage:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"{path.name}: unsupported file type '{ext}' (use TIFF, PNG or JPEG)")
    if not path.exists():
        raise ImageLoadError(f"{path}: file not found")
    notes: list[str] = []
    try:
        if ext in (".tif", ".tiff"):
            arr, is_rgb, names, px = _read_tiff(path, notes)
        else:
            arr, is_rgb, names, px = _read_pillow(path, notes)
    except ImageLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - report any reader failure uniformly
        raise ImageLoadError(f"{path.name}: could not be read ({exc})") from exc
    return _build(path, arr, is_rgb, names, px, ext in LOSSY_EXTENSIONS, notes)


def _read_tiff(path: Path, notes: list[str]):
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        arr = np.asarray(series.asarray())
        axes = series.axes.upper()
        page = tif.pages[0]
        photometric = int(getattr(page, "photometric", 1))
        colormap = getattr(page, "colormap", None)
        px = _tiff_pixel_size_um(tif)
        names = _tiff_channel_names(tif)

    if photometric == 3 and colormap is not None and np.issubdtype(arr.dtype, np.integer):
        # palette TIFF: map indices through the colour table to RGB
        cmap = np.asarray(colormap)
        if cmap.max() > 255:
            cmap = (cmap / 257.0).round().astype(np.uint8)
        arr = np.moveaxis(cmap[:, arr], 0, -1)
        axes = axes + "S"
        notes.append("palette TIFF converted to RGB")

    arr, axes = _squeeze_axes(arr, axes)
    if "C" not in axes and "S" not in axes:
        for a in ("I", "Q"):
            if a in axes and arr.shape[axes.index(a)] <= 8:
                axes = axes.replace(a, "C")
                notes.append(f"{arr.shape[axes.index('C')]} pages treated as channels")
                break
    for a in list(axes):
        if a in "YXCS":
            continue
        i = axes.index(a)
        notes.append(f"{a} axis with {arr.shape[i]} planes: maximum-intensity projection used")
        arr = arr.max(axis=i)
        axes = axes[:i] + axes[i + 1 :]
    if "Y" not in axes or "X" not in axes:
        raise ImageLoadError(f"{path.name}: no 2-D image plane found (axes {axes})")
    samples = arr.shape[axes.index("S")] if "S" in axes else 1
    is_rgb = "S" in axes and samples in (3, 4) and "C" not in axes
    chan_axes = [a for a in axes if a in "CS"]
    order = [axes.index(a) for a in chan_axes] + [axes.index("Y"), axes.index("X")]
    arr = np.transpose(arr, order).reshape((-1,) + tuple(arr.shape[axes.index(k)] for k in "YX"))
    if is_rgb:
        arr = arr[:3]
        names = None
    elif names is not None and len(names) != arr.shape[0]:
        names = None
    return arr, is_rgb, names, px


def _squeeze_axes(arr: np.ndarray, axes: str):
    keep = [i for i, n in enumerate(arr.shape) if n > 1 or axes[i] in "YX"]
    return arr.reshape([arr.shape[i] for i in keep]), "".join(axes[i] for i in keep)


_UNIT_TO_UM = {
    "m": 1e6, "mm": 1e3, "cm": 1e4, "um": 1.0, "µm": 1.0, "μm": 1.0, "micron": 1.0,
    "microns": 1.0, "micrometer": 1.0, "micrometre": 1.0, "nm": 1e-3, "\\u00b5m": 1.0,
}


def _unit_factor(unit: str | None) -> float | None:
    if not unit:
        return None
    return _UNIT_TO_UM.get(unit.strip().lower())


def _tiff_pixel_size_um(tif) -> float | None:
    try:
        if getattr(tif, "is_ome", False) and tif.ome_metadata:
            m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata)
            if m:
                u = re.search(r'PhysicalSizeXUnit="([^"]+)"', tif.ome_metadata)
                f = _unit_factor(u.group(1) if u else "µm") or 1.0
                val = float(m.group(1)) * f
                if 1e-4 < val < 1e3:
                    return val
    except Exception:  # noqa: BLE001
        pass
    try:
        page = tif.pages[0]
        xres = page.tags.get("XResolution")
        if xres is None:
            return None
        num, den = xres.value
        ppu = num / den if den else 0
        if ppu <= 0:
            return None
        ij = tif.imagej_metadata or {}
        unit = ij.get("unit")
        resunit = page.tags.get("ResolutionUnit")
        resunit = int(resunit.value) if resunit is not None else 2
        if unit and _unit_factor(unit):
            val = _unit_factor(unit) / ppu
        elif resunit == 3:  # centimetre
            val = 1e4 / ppu
        elif resunit == 2:  # inch: only believable at microscope-like densities
            val = 25400.0 / ppu
            if val > 50:
                return None
        else:
            return None
        return val if 1e-4 < val < 1e3 else None
    except Exception:  # noqa: BLE001
        return None


def _tiff_channel_names(tif) -> list[str] | None:
    try:
        if getattr(tif, "is_ome", False) and tif.ome_metadata:
            names = re.findall(r'<Channel[^>]*?Name="([^"]*)"', tif.ome_metadata)
            if names:
                return names
    except Exception:  # noqa: BLE001
        pass
    return None


def _read_pillow(path: Path, notes: list[str]):
    from PIL import Image

    with Image.open(path) as im:
        rawmode = im.tile[0][3] if im.tile and len(im.tile[0]) > 3 else ""
        rawmode = rawmode if isinstance(rawmode, str) else str(rawmode)
        if im.format == "PNG" and im.mode in ("RGB", "RGBA") and ";16" in rawmode:
            arr = _read_png16_rgb(path, notes)
            if arr is not None:
                return np.moveaxis(arr[..., :3], -1, 0), True, None, None
        im.load()
        mode = im.mode
        if mode == "P":
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        elif mode == "PA":
            im = im.convert("RGBA")
        elif mode == "1":
            im = im.convert("L")
        elif mode == "LA":
            im = im.convert("L")
        elif mode in ("CMYK", "YCbCr", "LAB", "HSV"):
            im = im.convert("RGB")
        arr = np.asarray(im)
    if arr.dtype == np.int32 and arr.size and arr.min() >= 0 and arr.max() <= 65535:
        arr = arr.astype(np.uint16)
    if arr.ndim == 3:
        return np.moveaxis(arr[..., :3], -1, 0), True, None, None
    return arr[None], False, None, None


def _read_png16_rgb(path: Path, notes: list[str]) -> np.ndarray | None:
    """16-bit-per-channel RGB PNG (Pillow would truncate these to 8 bit)."""
    try:
        import png  # pypng

        w, h, rows, info = png.Reader(filename=str(path)).asDirect()
        planes = info["planes"]
        arr = np.vstack([np.asarray(r, dtype=np.uint16) for r in rows]).reshape(h, w, planes)
        return arr
    except Exception:  # noqa: BLE001
        notes.append("16-bit RGB PNG read at 8-bit precision (pypng unavailable)")
        return None


# ------------------------------------------------------------------ classification


def _build(path, arr, is_rgb, names, px, lossy, notes) -> LoadedImage:
    annotation = None
    signal = None
    colour = None
    if is_rgb:
        kind, signal, colour, empty = _classify_rgb(arr, lossy)
        if kind == "grey":
            annotation = _colour_on_grey_mask(arr, lossy)
            data = arr[1:2].copy()
            names = ["grey"]
            notes.append("RGB file with identical channels: treated as greyscale")
        else:
            data = arr
            names = list(RGB_NAMES)
            annotation = _annotation_from_empty(arr, empty, lossy)
    else:
        data = arr
        if data.shape[0] == 1:
            kind, names = "grey", names or ["grey"]
        else:
            kind = "multichannel"
            names = names or [f"ch{i}" for i in range(data.shape[0])]
    if annotation is not None and annotation.any():
        frac = annotation.mean()
        notes.append(f"burned-in annotation detected ({100 * frac:.2f}% of pixels) and excluded")
    else:
        annotation = None
    if lossy:
        notes.append("JPEG is lossy: compression alters intensities; prefer TIFF/PNG for quantification")
    return LoadedImage(
        path=path,
        data=data,
        channel_names=list(names),
        kind=kind,
        saturation_value=_saturation_value(data),
        lossy=lossy,
        pixel_size_um=px,
        annotation_mask=annotation,
        signal_channel=signal,
        colour=colour,
        notes=notes,
    )


def _classify_rgb(rgb: np.ndarray, lossy: bool):
    """Return (kind, signal_channel, colour_name, empty_channels)."""
    step = max(1, int(np.sqrt(rgb.shape[1] * rgb.shape[2] / 250_000)))
    sub = rgb[:, ::step, ::step].astype(np.int32)
    tol_eq = 6 if lossy else 0
    tol_empty = 12 if lossy else 0
    hi = np.percentile(sub.reshape(3, -1), 99.5, axis=1)
    empty = [c for c in range(3) if hi[c] <= tol_empty]
    populated = [c for c in range(3) if c not in empty]
    if not populated:
        return "grey", None, None, empty
    # judge channel identity only on informative (non-dark) pixels
    maxc = sub.max(axis=0)
    level = max(tol_empty + 10, np.percentile(maxc, 90))
    informative = maxc >= level
    if informative.sum() < 100:
        informative = np.ones_like(maxc, dtype=bool)

    def same(a, b):
        return np.mean(np.abs(sub[a][informative] - sub[b][informative]) <= tol_eq) >= 0.98

    if len(populated) == 3 and same(0, 1) and same(1, 2):
        return "grey", None, None, empty
    if len(populated) == 1 or all(same(populated[0], c) for c in populated[1:]):
        colour = _COLOUR_BY_SET[frozenset(populated)]
        return "single_colour", populated[0], colour, empty
    return "merged_rgb", None, None, empty


def _annotation_from_empty(rgb: np.ndarray, empty: list[int], lossy: bool) -> np.ndarray | None:
    """Pixels lit in channels that the image otherwise never uses = overlays."""
    if not empty:
        return None
    thr = 40 if lossy else 0
    mask = np.zeros(rgb.shape[1:], dtype=bool)
    for c in empty:
        mask |= rgb[c] > thr
    if not mask.any():
        return None
    return ndi.binary_dilation(mask, iterations=3 if lossy else 1)


def _colour_on_grey_mask(rgb: np.ndarray, lossy: bool) -> np.ndarray | None:
    """Coloured overlay (e.g. a yellow scale bar) on an otherwise grey image."""
    spread = rgb.max(axis=0).astype(np.int32) - rgb.min(axis=0).astype(np.int32)
    mask = spread > (40 if lossy else 0)
    if not mask.any():
        return None
    return ndi.binary_dilation(mask, iterations=3 if lossy else 1)


def _saturation_value(data: np.ndarray) -> float | None:
    if data.dtype == np.uint8:
        return 255.0
    if data.dtype == np.uint16:
        m = int(data.max()) if data.size else 0
        for bits in (10, 12, 14):
            if m == 2**bits - 1:
                return float(m)
        return 65535.0
    if np.issubdtype(data.dtype, np.integer):
        return float(np.iinfo(data.dtype).max)
    return None


# ------------------------------------------------------------------ scale bars


def measure_scale_bar(annotation_mask: np.ndarray | None) -> dict | None:
    """Length in pixels of a burned-in horizontal scale bar.

    Uses the centres of the end ticks when present (Leica style), otherwise the
    full length of the longest horizontal run of annotation pixels.
    """
    if annotation_mask is None or not annotation_mask.any():
        return None
    m = annotation_mask.astype(np.int8)
    d = np.diff(np.pad(m, ((0, 0), (1, 1))), axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    if len(starts) == 0:
        return None
    lengths = ends[:, 1] - starts[:, 1]
    k = int(np.argmax(lengths))
    length = int(lengths[k])
    if length < 20:
        return None
    y, x0 = int(starts[k, 0]), int(starts[k, 1])
    x1 = x0 + length  # exclusive
    result = {"length_px": float(length), "row": y, "x0": x0, "x1": x1, "ticks": 0, "method": "line"}
    # vertical ticks rising from (or hanging below) the line
    for direction in (-1, 1):
        rows = [y + direction * k for k in range(2, 6)]
        rows = [r for r in rows if 0 <= r < annotation_mask.shape[0]]
        if len(rows) < 3:
            continue
        cols = annotation_mask[rows, x0:x1].sum(axis=0) >= len(rows) - 1
        if not cols.any():
            continue
        lab, n = ndi.label(cols)
        if n < 2:
            continue
        centres = ndi.center_of_mass(cols, lab, range(1, n + 1))
        centres = sorted(c[0] for c in centres)
        if centres[0] <= 6 and centres[-1] >= length - 7:
            result.update(length_px=float(centres[-1] - centres[0]), ticks=n, method="ticks")
            break
    return result
