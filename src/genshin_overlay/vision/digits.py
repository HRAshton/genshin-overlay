from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from genshin_overlay.models import CooldownReading, CooldownState, GlyphMatch, Rect


CANVAS_SHAPE = (40, 32)
COOLDOWN_TEXT = re.compile(r"^\d{1,2}\.\d$")


def white_text_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 180)).astype(np.uint8) * 255


def normalize_glyph(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    canvas = np.zeros(CANVAS_SHAPE, dtype=np.uint8)
    if xs.size == 0:
        return canvas
    glyph = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    factor = min(28 / glyph.shape[1], 36 / glyph.shape[0])
    resized = cv2.resize(
        glyph,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_NEAREST,
    )
    y = (CANVAS_SHAPE[0] - resized.shape[0]) // 2
    x = (CANVAS_SHAPE[1] - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    left_on = left > 0
    right_on = right > 0
    union = np.count_nonzero(left_on | right_on)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(left_on & right_on) / union)


class DigitRecognizer:
    def __init__(self, template_dir: Path | None = None, minimum_score: float = 0.55):
        root = template_dir or Path(__file__).parents[1] / "assets" / "digits"
        self.minimum_score = minimum_score
        self.templates: dict[str, np.ndarray] = {}
        if root.exists():
            for path in root.glob("*.png"):
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    self.templates[path.stem] = normalize_glyph(image)

    def read(self, image: np.ndarray) -> CooldownReading:
        if image.size == 0 or not self.templates:
            return CooldownReading(CooldownState.UNKNOWN, None, 0.0)

        mask = white_text_mask(image)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components: list[tuple[int, int, int, int, int, np.ndarray]] = []
        for index in range(1, count):
            x, y, width, height, area = map(int, stats[index])
            if area < 2 or width < 2 or height < 1:
                continue
            if width > image.shape[1] * 0.4 or height > image.shape[0] * 0.65:
                continue
            component = (labels[y : y + height, x : x + width] == index).astype(
                np.uint8
            ) * 255
            components.append((x, y, width, height, area, component))

        if not components:
            return CooldownReading(CooldownState.READY, 0.0, 0.92)

        likely_digits = [item for item in components if item[3] >= image.shape[0] * 0.18]
        if not likely_digits:
            return CooldownReading(CooldownState.READY, 0.0, 0.85)
        typical_height = float(np.median([item[3] for item in likely_digits]))
        baseline = float(np.median([item[1] + item[3] for item in likely_digits]))

        tokens: list[tuple[int, str, float]] = []
        dot_candidates: list[tuple[int, int, int, int, int]] = []
        matches: list[GlyphMatch] = []
        for x, y, width, height, area, component in components:
            if height <= typical_height * 0.4 and width <= typical_height * 0.4:
                if y + height >= baseline - 3:
                    dot_candidates.append((x, y, width, height, area))
                continue
            if height < typical_height * 0.72:
                continue
            normalized = normalize_glyph(component)
            scores = {
                digit: _iou(normalized, template)
                for digit, template in self.templates.items()
            }
            ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            digit, score = ranking[0]
            second_digit, second_score = ranking[1] if len(ranking) > 1 else (None, 0.0)
            accepted = score >= self.minimum_score
            matches.append(
                GlyphMatch(
                    Rect(x, y, width, height),
                    digit,
                    score,
                    second_digit,
                    second_score,
                    accepted,
                )
            )
            if accepted:
                tokens.append((x, digit, score))

        tokens.sort(key=lambda item: item[0])
        digit_positions = [item[0] for item in tokens]
        valid_dots = [
            candidate
            for candidate in dot_candidates
            if any(
                left < candidate[0] < right
                for left, right in zip(digit_positions, digit_positions[1:])
            )
        ]
        if valid_dots:
            x, y, width, height, _ = max(
                valid_dots, key=lambda candidate: candidate[4]
            )
            tokens.append((x, ".", 0.95))
            matches.append(GlyphMatch(Rect(x, y, width, height), ".", 0.95, None, 0.0, True))
        tokens.sort(key=lambda item: item[0])
        text = "".join(item[1] for item in tokens)
        if not text or not any(char.isdigit() for char in text):
            return CooldownReading(CooldownState.READY, 0.0, 0.8, glyphs=tuple(matches))
        if COOLDOWN_TEXT.fullmatch(text) is None:
            return CooldownReading(
                CooldownState.UNKNOWN, None, 0.2, text, tuple(matches)
            )
        try:
            seconds = float(text)
        except ValueError:
            return CooldownReading(
                CooldownState.UNKNOWN, None, 0.2, text, tuple(matches)
            )
        confidence = float(np.mean([item[2] for item in tokens]))
        return CooldownReading(
            CooldownState.COOLDOWN, seconds, confidence, text, tuple(matches)
        )
