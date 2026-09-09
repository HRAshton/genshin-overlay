from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genshin_overlay import ScreenAnalyzer
from genshin_overlay.models import CooldownState


class StaticAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = ScreenAnalyzer()

    def analyze_named(self, suffix: str):
        path = next(ROOT.glob(f"*{suffix}.png"))
        return self.analyzer.analyze_file(str(path))[1]

    def test_ready_reference(self) -> None:
        result = self.analyze_named("192540")
        actual = (
            result.viewport.x,
            result.viewport.y,
            result.viewport.width,
            result.viewport.height,
        )
        self.assertEqual((1120, 0, 2880, 1440), actual)
        self.assertIsNotNone(result.hud)
        self.assertIsNotNone(result.party)
        self.assertEqual(1, result.party.active_slot)
        self.assertEqual(
            (3675, 301, 125, 110),
            (
                result.party.portrait_rect.x,
                result.party.portrait_rect.y,
                result.party.portrait_rect.width,
                result.party.portrait_rect.height,
            ),
        )
        self.assertEqual(CooldownState.READY, result.e.state)
        self.assertEqual(CooldownState.READY, result.q.state)

    def test_cooldown_reference(self) -> None:
        result = self.analyze_named("192650")
        self.assertIsNotNone(result.hud)
        self.assertIsNotNone(result.party)
        self.assertEqual(2, result.party.active_slot)
        self.assertEqual(CooldownState.COOLDOWN, result.e.state)
        self.assertAlmostEqual(2.5, result.e.seconds)
        self.assertEqual(CooldownState.COOLDOWN, result.q.state)
        self.assertAlmostEqual(8.7, result.q.seconds)

    def test_half_scale_cooldown_reference(self) -> None:
        path = next(ROOT.glob("*192650.png"))
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        result = self.analyzer.analyze(frame)
        self.assertEqual((560, 0, 1440, 720), (
            result.viewport.x,
            result.viewport.y,
            result.viewport.width,
            result.viewport.height,
        ))
        self.assertEqual(2, result.party.active_slot)
        self.assertAlmostEqual(2.5, result.e.seconds)
        self.assertAlmostEqual(8.7, result.q.seconds)


if __name__ == "__main__":
    unittest.main()
