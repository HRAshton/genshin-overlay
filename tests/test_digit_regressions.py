from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genshin_overlay.models import CooldownState
from genshin_overlay.vision.digits import DigitRecognizer


class DigitRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recognizer = DigitRecognizer()

    def read(self, name: str):
        image = cv2.imread(str(ROOT / "tests" / "fixtures" / name))
        self.assertIsNotNone(image)
        return self.recognizer.read(image)

    def test_furina_keeps_leading_one(self) -> None:
        first = self.read("furina_e_18_9.png")
        second = self.read("furina_e_18_8.png")
        self.assertEqual(CooldownState.COOLDOWN, first.state)
        self.assertAlmostEqual(18.9, first.seconds)
        self.assertEqual(CooldownState.COOLDOWN, second.state)
        self.assertAlmostEqual(18.8, second.seconds)

    def test_ready_icon_fragment_is_not_cooldown(self) -> None:
        reading = self.read("ready_icon_false_4.png")
        self.assertNotEqual(CooldownState.COOLDOWN, reading.state)


if __name__ == "__main__":
    unittest.main()
