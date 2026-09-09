from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genshin_overlay import ScreenAnalyzer
from genshin_overlay.debug import save_debug_images


def _reading_text(reading) -> str:
    value = "" if reading.seconds is None else f" {reading.seconds:g}"
    return f"{reading.state.value}{value} confidence={reading.confidence:.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect static Genshin screenshots")
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "debug_output")
    args = parser.parse_args(argv)
    images = args.images or sorted(ROOT.glob("Screenshot *.png"))
    if not images:
        parser.error("no screenshots found")

    analyzer = ScreenAnalyzer()
    for path in images:
        frame, observation = analyzer.analyze_file(str(path))
        viewport = observation.viewport
        party = "unknown" if observation.party is None else str(observation.party.active_slot)
        print(path.name)
        print(f"  viewport: {viewport.x},{viewport.y} {viewport.width}x{viewport.height}")
        print(f"  active slot: {party}")
        print(f"  E: {_reading_text(observation.e)}")
        print(f"  Q: {_reading_text(observation.q)}")
        save_debug_images(frame, observation, args.output, path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

