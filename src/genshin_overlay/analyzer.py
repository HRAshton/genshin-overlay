from __future__ import annotations

import cv2
import numpy as np

from genshin_overlay.models import (
    CooldownReading,
    CooldownState,
    ScreenObservation,
    Rect,
    SkillHud,
)
from genshin_overlay.vision.digits import DigitRecognizer
from genshin_overlay.vision.party import locate_active_party
from genshin_overlay.vision.skill_hud import locate_skill_hud
from genshin_overlay.vision.viewport import locate_viewport


class ScreenAnalyzer:
    def __init__(self, recognizer: DigitRecognizer | None = None):
        self.recognizer = recognizer or DigitRecognizer()
        self._frame_shape: tuple[int, ...] | None = None
        self._viewport: Rect | None = None
        self._hud: SkillHud | None = None

    def analyze(
        self, frame: np.ndarray, *, refresh_geometry: bool = True
    ) -> ScreenObservation:
        shape_changed = self._frame_shape != frame.shape
        if refresh_geometry or shape_changed or self._viewport is None:
            self._frame_shape = frame.shape
            self._viewport = locate_viewport(frame)
        if refresh_geometry or shape_changed or self._hud is None:
            # self._viewport is set above when refresh_geometry or shape_changed is True
            assert self._viewport is not None
            self._hud = locate_skill_hud(frame, self._viewport)
        assert self._viewport is not None
        viewport = self._viewport
        hud = self._hud
        party = locate_active_party(frame, viewport)
        if hud is None:
            unknown = CooldownReading(CooldownState.UNKNOWN, None, 0.0)
            return ScreenObservation(viewport, None, party, unknown, unknown)
        e = self.recognizer.read(hud.e_number.crop(frame))
        q = self.recognizer.read(hud.q_number.crop(frame))
        return ScreenObservation(viewport, hud, party, e, q)

    def analyze_file(self, path: str) -> tuple[np.ndarray, ScreenObservation]:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return frame, self.analyze(frame)
