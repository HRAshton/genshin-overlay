from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genshin_overlay.diagnostics import DebugRecorder
from genshin_overlay.models import (
    CooldownReading,
    CooldownState,
    CooldownView,
    GlyphMatch,
    PartyObservation,
    Rect,
    ScreenObservation,
    SkillHud,
    TrackerDecision,
)


class DiagnosticRecorderTests(unittest.TestCase):
    def test_writes_reproducible_bundle(self) -> None:
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        frame[75:82, 145:150] = 255
        glyph = GlyphMatch(Rect(5, 5, 5, 7), "8", 0.81, "6", 0.62, True)
        reading = CooldownReading(
            CooldownState.COOLDOWN,
            8.7,
            0.81,
            "8.7",
            (glyph,),
        )
        hud = SkillHud(
            Rect(120, 90, 10, 8),
            Rect(150, 90, 10, 8),
            Rect(115, 65, 30, 25),
            Rect(140, 65, 40, 25),
            0.25,
            0.9,
        )
        observation = ScreenObservation(
            Rect(0, 0, 200, 120),
            hud,
            PartyObservation(2, Rect(150, 15, 30, 30), 0.8),
            reading,
            reading,
        )
        decision = TrackerDecision(2, "Q", "pending", "confirmation 1/2", 8.7)
        tracked = (
            CooldownView(2, "Q", 8.7, 8.7, frame[15:45, 150:180].copy()),
        )

        with tempfile.TemporaryDirectory() as directory:
            recorder = DebugRecorder(Path(directory))
            recorder.record(frame, observation, (decision,), (), 10.0)
            recorder.record(frame, observation, (decision,), tracked, 10.1)
            bundle = recorder.dump(frame, observation, "manual test", 10.1)

            self.assertTrue((bundle / "frame.png").exists())
            self.assertTrue((bundle / "annotated.png").exists())
            self.assertTrue((bundle / "000_e_mask.png").exists())
            self.assertTrue((bundle / "001_q_glyphs.png").exists())
            metadata = json.loads((bundle / "metadata.json").read_text("utf-8"))
            self.assertEqual("manual test", metadata["reason"])
            self.assertEqual(2, len(metadata["frames"]))
            first = metadata["frames"][0]
            self.assertEqual("pending", first["tracker_decisions"][0]["action"])
            self.assertEqual([], first["tracked_items"])
            self.assertEqual("Q", metadata["frames"][1]["tracked_items"][0]["skill"])
            match = first["observation"]["q"]["glyphs"][0]
            self.assertEqual("8", match["best_digit"])
            self.assertAlmostEqual(0.19, match["margin"])


if __name__ == "__main__":
    unittest.main()
