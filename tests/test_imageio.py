import numpy as np
import pytest
import tifffile
from PIL import Image, ImageDraw

from microscount.core.imageio import ChannelError, ImageLoadError, load_image, measure_scale_bar


def _blue_export(tmp_path, with_bar=True, palette=True):
    rng = np.random.default_rng(0)
    b = rng.integers(1, 60, (200, 300)).astype(np.uint8)
    rgb = np.zeros((200, 300, 3), np.uint8)
    rgb[..., 2] = b
    if with_bar:  # Leica-style white scale bar with end ticks
        rgb[190:192, 200:278] = 255
        rgb[184:190, 200:203] = 255
        rgb[184:190, 275:278] = 255
    im = Image.fromarray(rgb)
    if palette:
        im = im.convert("P", palette=Image.ADAPTIVE, colors=256) if len(np.unique(rgb.reshape(-1, 3), axis=0)) > 256 else im.quantize(256)
    p = tmp_path / "blue.png"
    Image.fromarray(rgb).save(p)
    return p, rgb


def test_single_colour_export_and_annotation(tmp_path):
    p, rgb = _blue_export(tmp_path)
    img = load_image(p)
    assert img.kind == "single_colour" and img.colour == "blue"
    assert np.array_equal(img.channel("auto"), rgb[..., 2])
    assert img.annotation_mask is not None
    assert img.annotation_mask[191, 240] and not img.annotation_mask[50, 50]
    sb = measure_scale_bar(img.annotation_mask)
    assert sb["method"] == "ticks" and sb["length_px"] == pytest.approx(75.0)


def test_palette_png_is_decoded_through_the_palette(tmp_path):
    rgb = np.zeros((50, 60, 3), np.uint8)
    rgb[..., 1] = np.arange(60, dtype=np.uint8)[None, :] * 3
    im = Image.fromarray(rgb).quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
    im.save(tmp_path / "pal.png")
    img = load_image(tmp_path / "pal.png")
    assert img.kind == "single_colour" and img.colour == "green"
    ref = np.asarray(im.convert("RGB"))[..., 1]
    assert np.array_equal(img.channel(), ref)


def test_grey_tiff_16bit_and_pixel_size(tmp_path):
    a = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 13) % 4096
    tifffile.imwrite(tmp_path / "g.tif", a, resolution=(1 / 0.25, 1 / 0.25), imagej=True, metadata={"unit": "um"})
    img = load_image(tmp_path / "g.tif")
    assert img.kind == "grey" and img.data.dtype == np.uint16
    assert img.pixel_size_um == pytest.approx(0.25)
    assert np.array_equal(img.channel(), a)


def test_multichannel_tiff(tmp_path):
    a = np.stack([np.full((20, 30), 10, np.uint16), np.full((20, 30), 20, np.uint16)])
    tifffile.imwrite(tmp_path / "mc.tif", a, imagej=True, metadata={"axes": "CYX"})
    img = load_image(tmp_path / "mc.tif")
    assert img.kind == "multichannel" and img.n_channels == 2
    assert img.channel("ch1")[0, 0] == 20
    with pytest.raises(ChannelError):
        img.channel("auto")


def test_merged_rgb_and_jpeg(tmp_path):
    rng = np.random.default_rng(1)
    rgb = np.zeros((80, 80, 3), np.uint8)
    rgb[..., 1] = rng.integers(0, 200, (80, 80))
    rgb[..., 2] = rng.integers(0, 200, (80, 80))
    Image.fromarray(rgb).save(tmp_path / "m.png")
    assert load_image(tmp_path / "m.png").kind == "merged_rgb"
    Image.fromarray(rgb[..., 1]).save(tmp_path / "g.jpg", quality=95)
    j = load_image(tmp_path / "g.jpg")
    assert j.lossy and j.kind == "grey"


def test_grey_as_rgb_is_grey(tmp_path):
    g = np.random.default_rng(2).integers(0, 255, (40, 40)).astype(np.uint8)
    Image.fromarray(np.stack([g] * 3, -1)).save(tmp_path / "g3.png")
    img = load_image(tmp_path / "g3.png")
    assert img.kind == "grey" and np.array_equal(img.channel(), g)


def test_png16_rgb_full_precision(tmp_path):
    png = pytest.importorskip("png")
    a = (np.arange(10 * 12 * 3, dtype=np.uint16).reshape(10, 12, 3) * 97) % 65535
    a[..., 0] = 0
    with open(tmp_path / "rgb16.png", "wb") as f:
        png.Writer(12, 10, greyscale=False, bitdepth=16).write(f, a.reshape(10, -1).tolist())
    img = load_image(tmp_path / "rgb16.png")
    assert img.data.dtype == np.uint16
    assert np.array_equal(img.data[1], a[..., 1])


def test_unsupported(tmp_path):
    (tmp_path / "x.bmp").write_bytes(b"BM")
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "x.bmp")
