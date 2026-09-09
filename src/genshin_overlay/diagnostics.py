from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from genshin_overlay.debug import annotate_frame
from genshin_overlay.models import (
    CooldownReading,
    CooldownView,
    Rect,
    ScreenObservation,
    TrackerDecision,
)
from genshin_overlay.vision.digits import white_text_mask


@dataclass(slots=True)
class _Sample:
    timestamp: float
    observation: ScreenObservation
    decisions: tuple[TrackerDecision, ...]
    tracked_items: tuple[CooldownView, ...]
    e_crop: np.ndarray | None
    q_crop: np.ndarray | None


def _rect(rect: Rect | None) -> dict[str, int] | None:
    if rect is None:
        return None
    return {
        "x": rect.x,
        "y": rect.y,
        "width": rect.width,
        "height": rect.height,
    }


def _reading(reading: CooldownReading) -> dict[str, object]:
    return {
        "state": reading.state.value,
        "seconds": reading.seconds,
        "confidence": reading.confidence,
        "text": reading.text,
        "glyphs": [
            {
                "rect": _rect(match.rect),
                "best_digit": match.best_digit,
                "best_score": match.best_score,
                "second_digit": match.second_digit,
                "second_score": match.second_score,
                "margin": match.best_score - match.second_score,
                "accepted": match.accepted,
            }
            for match in reading.glyphs
        ],
    }


def _observation(observation: ScreenObservation) -> dict[str, object]:
    hud = observation.hud
    party = observation.party
    return {
        "viewport": _rect(observation.viewport),
        "hud": None
        if hud is None
        else {
            "confidence": hud.confidence,
            "scale": hud.scale,
            "e_key": _rect(hud.e_key),
            "q_key": _rect(hud.q_key),
            "e_number": _rect(hud.e_number),
            "q_number": _rect(hud.q_number),
        },
        "party": None
        if party is None
        else {
            "active_slot": party.active_slot,
            "confidence": party.confidence,
            "portrait_rect": _rect(party.portrait_rect),
        },
        "e": _reading(observation.e),
        "q": _reading(observation.q),
    }


def _decision(decision: TrackerDecision) -> dict[str, object]:
    return {
        "party_slot": decision.party_slot,
        "skill": decision.skill,
        "action": decision.action,
        "reason": decision.reason,
        "observed_seconds": decision.observed_seconds,
        "expected_seconds": decision.expected_seconds,
    }


def _tracked_item(item: CooldownView) -> dict[str, object]:
    return {
        "party_slot": item.party_slot,
        "skill": item.skill,
        "remaining_seconds": item.remaining_seconds,
        "total_seconds": item.total_seconds,
    }


def _annotate_glyphs(crop: np.ndarray, reading: CooldownReading) -> np.ndarray:
    output = crop.copy()
    for match in reading.glyphs:
        color = (0, 220, 0) if match.accepted else (0, 0, 255)
        rect = match.rect
        cv2.rectangle(output, (rect.x, rect.y), (rect.right, rect.bottom), color, 1)
        label = f"{match.best_digit}:{match.best_score:.2f}"
        cv2.putText(
            output,
            label,
            (rect.x, max(9, rect.y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


class DebugRecorder:
    """Keep small native-resolution HUD history and write reproducible bundles."""

    def __init__(self, output_root: Path, history_seconds: float = 2.0):
        self.output_root = output_root
        self.history_seconds = history_seconds
        self._samples: deque[_Sample] = deque()

    def record(
        self,
        frame: np.ndarray,
        observation: ScreenObservation,
        decisions: tuple[TrackerDecision, ...],
        tracked_items: tuple[CooldownView, ...],
        now: float,
    ) -> None:
        hud = observation.hud
        e_crop = None if hud is None else hud.e_number.crop(frame).copy()
        q_crop = None if hud is None else hud.q_number.crop(frame).copy()
        self._samples.append(
            _Sample(now, observation, decisions, tracked_items, e_crop, q_crop)
        )
        cutoff = now - self.history_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def dump(
        self,
        frame: np.ndarray,
        observation: ScreenObservation,
        reason: str,
        now: float,
    ) -> Path:
        safe_reason = re.sub(r"[^a-zA-Z0-9_-]+", "-", reason).strip("-") or "manual"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = self.output_root / f"{stamp}_{safe_reason}"
        output.mkdir(parents=True, exist_ok=False)
        cv2.imwrite(str(output / "frame.png"), frame)
        cv2.imwrite(str(output / "annotated.png"), annotate_frame(frame, observation))

        party = observation.party
        if party is not None:
            cv2.imwrite(str(output / "portrait.png"), party.portrait_rect.crop(frame))

        metadata: dict[str, object] = {
            "schema_version": 1,
            "reason": reason,
            "captured_at": datetime.now().astimezone().isoformat(),
            "frame_shape": list(frame.shape),
            "frames": [],
        }
        frames = metadata["frames"]
        assert isinstance(frames, list)
        for index, sample in enumerate(self._samples):
            prefix = f"{index:03d}"
            if sample.e_crop is not None:
                cv2.imwrite(str(output / f"{prefix}_e.png"), sample.e_crop)
                cv2.imwrite(
                    str(output / f"{prefix}_e_mask.png"),
                    white_text_mask(sample.e_crop),
                )
                cv2.imwrite(
                    str(output / f"{prefix}_e_glyphs.png"),
                    _annotate_glyphs(sample.e_crop, sample.observation.e),
                )
            if sample.q_crop is not None:
                cv2.imwrite(str(output / f"{prefix}_q.png"), sample.q_crop)
                cv2.imwrite(
                    str(output / f"{prefix}_q_mask.png"),
                    white_text_mask(sample.q_crop),
                )
                cv2.imwrite(
                    str(output / f"{prefix}_q_glyphs.png"),
                    _annotate_glyphs(sample.q_crop, sample.observation.q),
                )
            frames.append(
                {
                    "index": index,
                    "age_seconds": now - sample.timestamp,
                    "observation": _observation(sample.observation),
                    "tracker_decisions": [
                        _decision(decision) for decision in sample.decisions
                    ],
                    "tracked_items": [
                        _tracked_item(item) for item in sample.tracked_items
                    ],
                }
            )
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output
