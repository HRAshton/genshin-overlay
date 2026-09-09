from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genshin_overlay.models import (
    CooldownReading,
    CooldownState,
    PartyObservation,
    Rect,
    ScreenObservation,
)
from genshin_overlay.tracker import CooldownTracker


def observation(slot: int, seconds: float | None) -> ScreenObservation:
    reading = (
        CooldownReading(CooldownState.COOLDOWN, seconds, 0.95, str(seconds))
        if seconds is not None
        else CooldownReading(CooldownState.UNKNOWN, None, 0.0)
    )
    return ScreenObservation(
        viewport=Rect(0, 0, 100, 100),
        hud=None,
        party=PartyObservation(slot, Rect(10, 10, 20, 20), 0.9),
        e=reading,
        q=CooldownReading(CooldownState.UNKNOWN, None, 0.0),
    )


class TrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_confirms_and_keeps_cooldown_after_character_switch(self) -> None:
        tracker = CooldownTracker()
        self.assertEqual((), tracker.update(self.frame, observation(2, 8.7), 100.0))
        result = tracker.update(self.frame, observation(2, 8.6), 100.1)
        self.assertEqual(1, len(result))
        self.assertEqual((2, "E"), (result[0].party_slot, result[0].skill))
        self.assertEqual("confirmed", tracker.last_decisions[0].action)

        switched = tracker.update(self.frame, observation(1, None), 101.1)
        self.assertEqual(1, len(switched))
        self.assertAlmostEqual(7.6, switched[0].remaining_seconds, places=1)

    def test_expires_at_zero(self) -> None:
        tracker = CooldownTracker()
        tracker.update(self.frame, observation(2, 2.5), 10.0)
        tracker.update(self.frame, observation(2, 2.4), 10.1)
        self.assertEqual((), tracker.snapshot(12.6))

    def test_rejects_inconsistent_confirmation(self) -> None:
        tracker = CooldownTracker()
        tracker.update(self.frame, observation(2, 8.7), 10.0)
        result = tracker.update(self.frame, observation(2, 83.1), 10.1)
        self.assertEqual((), result)
        self.assertEqual("rejected", tracker.last_decisions[0].action)


if __name__ == "__main__":
    unittest.main()
