from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genshin_overlay.analyzer import ScreenAnalyzer
from genshin_overlay.models import Rect, SkillHud


class AnalyzerCacheTests(unittest.TestCase):
    def test_retries_hud_every_frame_while_missing(self) -> None:
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        viewport = Rect(0, 0, 160, 100)
        hud = SkillHud(
            Rect(90, 80, 10, 8),
            Rect(120, 80, 10, 8),
            Rect(85, 55, 25, 20),
            Rect(110, 50, 35, 25),
            0.25,
            0.9,
        )
        analyzer = ScreenAnalyzer()
        with (
            patch(
                "genshin_overlay.analyzer.locate_viewport", return_value=viewport
            ) as locate_viewport,
            patch(
                "genshin_overlay.analyzer.locate_skill_hud", side_effect=(None, hud)
            ) as locate_hud,
            patch("genshin_overlay.analyzer.locate_active_party", return_value=None),
        ):
            missing = analyzer.analyze(frame, refresh_geometry=True)
            recovered = analyzer.analyze(frame, refresh_geometry=False)

        self.assertIsNone(missing.hud)
        self.assertIs(hud, recovered.hud)
        self.assertEqual(1, locate_viewport.call_count)
        self.assertEqual(2, locate_hud.call_count)


if __name__ == "__main__":
    unittest.main()
