from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True, slots=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.bottom, self.x : self.right]


class CooldownState(str, Enum):
    READY = "READY"
    COOLDOWN = "COOLDOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class GlyphMatch:
    rect: Rect
    best_digit: str
    best_score: float
    second_digit: str | None
    second_score: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class SkillHud:
    e_key: Rect
    q_key: Rect
    e_number: Rect
    q_number: Rect
    scale: float
    confidence: float


@dataclass(frozen=True, slots=True)
class CooldownReading:
    state: CooldownState
    seconds: float | None
    confidence: float
    text: str | None = None
    glyphs: tuple[GlyphMatch, ...] = ()


@dataclass(frozen=True, slots=True)
class PartyObservation:
    active_slot: int
    portrait_rect: Rect
    confidence: float


@dataclass(frozen=True, slots=True)
class ScreenObservation:
    viewport: Rect
    hud: SkillHud | None
    party: PartyObservation | None
    e: CooldownReading
    q: CooldownReading


@dataclass(frozen=True, slots=True)
class CooldownView:
    party_slot: int
    skill: str
    remaining_seconds: float
    total_seconds: float
    portrait_bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class TrackerDecision:
    party_slot: int | None
    skill: str
    action: str
    reason: str
    observed_seconds: float | None
    expected_seconds: float | None = None
