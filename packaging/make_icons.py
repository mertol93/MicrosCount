"""Generate the MicrosCount icon (PNG, ICO, ICNS) with Pillow. Run from the repository root."""

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
S = 1024


def cell_icon() -> Image.Image:
    bg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(bg)
    d.rounded_rectangle((40, 40, S - 40, S - 40), radius=210, fill=(16, 20, 28, 255))
    # cell body: soft green glow
    cell = Image.new("L", (S, S), 0)
    dc = ImageDraw.Draw(cell)
    dc.ellipse((190, 230, 840, 800), fill=255)
    dc.ellipse((520, 170, 880, 520), fill=255)
    dc.ellipse((150, 420, 470, 860), fill=255)
    cell = cell.filter(ImageFilter.GaussianBlur(38))
    green = Image.new("RGBA", (S, S), (60, 205, 110, 255))
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    layer.paste(green, (0, 0), cell.point(lambda v: int(v * 0.72)))
    bg = Image.alpha_composite(bg, layer)
    # bright nucleus (nuclear translocation: nucleus brighter than cytoplasm)
    nuc = Image.new("L", (S, S), 0)
    ImageDraw.Draw(nuc).ellipse((370, 390, 650, 640), fill=255)
    nuc = nuc.filter(ImageFilter.GaussianBlur(10))
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    layer.paste(Image.new("RGBA", (S, S), (150, 245, 170, 255)), (0, 0), nuc)
    bg = Image.alpha_composite(bg, layer)
    # blue nuclear stain core
    core = Image.new("L", (S, S), 0)
    ImageDraw.Draw(core).ellipse((420, 430, 600, 600), fill=255)
    core = core.filter(ImageFilter.GaussianBlur(26))
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    layer.paste(Image.new("RGBA", (S, S), (42, 120, 214, 255)), (0, 0), core.point(lambda v: int(v * 0.55)))
    bg = Image.alpha_composite(bg, layer)
    # segmentation outline (cyan) and cytoplasm ring (magenta), as drawn by the app
    d = ImageDraw.Draw(bg)
    d.ellipse((370, 390, 650, 640), outline=(0, 230, 255, 255), width=18)
    d.ellipse((318, 340, 702, 690), outline=(255, 0, 200, 200), width=14)
    # clip to the rounded square
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((40, 40, S - 40, S - 40), radius=210, fill=255)
    bg.putalpha(ImageChops.multiply(bg.getchannel("A"), mask))
    return bg


def main():
    icon = cell_icon()
    out = ROOT / "packaging" / "icons"
    out.mkdir(parents=True, exist_ok=True)
    icon.save(out / "microscount.png")
    icon.resize((512, 512), Image.LANCZOS).save(ROOT / "src" / "microscount" / "gui" / "resources" / "icon.png")
    icon.save(out / "microscount.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    icon.save(out / "microscount.icns")
    print("icons written to", out)


if __name__ == "__main__":
    main()
