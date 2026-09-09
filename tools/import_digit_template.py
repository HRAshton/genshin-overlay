from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genshin_overlay.vision.digits import normalize_glyph, white_text_mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import one verified game-font digit")
    parser.add_argument("image", type=Path)
    parser.add_argument("digit", choices=tuple("0123456789"))
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    args = parser.parse_args(argv)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    if (
        args.x < 0
        or args.y < 0
        or args.x + args.width > image.shape[1]
        or args.y + args.height > image.shape[0]
    ):
        parser.error("glyph rectangle is outside image")

    crop = image[
        args.y : args.y + args.height,
        args.x : args.x + args.width,
    ]
    glyph = normalize_glyph(white_text_mask(crop))
    destination = (
        ROOT / "src" / "genshin_overlay" / "assets" / "digits" / f"{args.digit}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), glyph):
        raise OSError(f"failed to write template: {destination}")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
