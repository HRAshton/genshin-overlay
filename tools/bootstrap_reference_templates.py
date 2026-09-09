from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genshin_overlay.vision.digits import normalize_glyph, white_text_mask


GLYPHS = {
    "2": (3592, 1313, 13, 19),
    "5": (3615, 1313, 13, 19),
    "8": (3744, 1282, 20, 27),
    "7": (3778, 1282, 19, 27),
}


def main() -> int:
    source = ROOT / "Screenshot 2026-09-09 192650.png"
    frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(source)
    output = ROOT / "src" / "genshin_overlay" / "assets" / "digits"
    output.mkdir(parents=True, exist_ok=True)
    font_path = Path("C:/Windows/Fonts/calibrib.ttf")
    if not font_path.exists():
        raise FileNotFoundError("Windows Calibri Bold font is required for fallback templates")
    font = ImageFont.truetype(str(font_path), 64)
    for digit in "0123456789":
        destination = output / f"{digit}.png"
        if digit == "1" and destination.exists():
            continue
        canvas = Image.new("L", (100, 100))
        ImageDraw.Draw(canvas).text((10, -4), digit, font=font, fill=255)
        cv2.imwrite(
            str(destination),
            normalize_glyph(np.asarray(canvas)),
        )
    for digit, (x, y, width, height) in GLYPHS.items():
        glyph = white_text_mask(frame[y : y + height, x : x + width])
        cv2.imwrite(str(output / f"{digit}.png"), normalize_glyph(glyph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
