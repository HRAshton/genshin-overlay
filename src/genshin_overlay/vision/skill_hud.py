from __future__ import annotations

import cv2
import numpy as np

from genshin_overlay.models import Rect, SkillHud


REFERENCE_HEIGHT = 1440.0
Q_KEY_X = (3749 - 1120) / 2880
Q_KEY_Y = 1369 / 1440
Q_KEY_WIDTH = 41
Q_KEY_HEIGHT = 33
E_TO_Q_DISTANCE = 159


def _clamp(rect: Rect, bounds: Rect) -> Rect:
    x0 = max(bounds.x, rect.x)
    y0 = max(bounds.y, rect.y)
    x1 = min(bounds.right, rect.right)
    y1 = min(bounds.bottom, rect.bottom)
    return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def locate_skill_hud(frame: np.ndarray, viewport: Rect) -> SkillHud | None:
    """Locate Q keycap visually, then derive skill-relative regions."""
    scale_guess = viewport.height / REFERENCE_HEIGHT
    predicted_x = viewport.x + int(round(viewport.width * Q_KEY_X))
    predicted_y = viewport.y + int(round(viewport.height * Q_KEY_Y))
    margin_x = max(35, int(round(80 * scale_guess)))
    margin_y = max(25, int(round(55 * scale_guess)))
    search = _clamp(
        Rect(
            predicted_x - margin_x,
            predicted_y - margin_y,
            margin_x * 2 + int(70 * scale_guess),
            margin_y * 2 + int(55 * scale_guess),
        ),
        viewport,
    )
    if search.width == 0 or search.height == 0:
        return None

    crop = search.crop(frame)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 175)).astype(np.uint8) * 255
    # OpenCV stubs cause overload-resolution issues for uint8 arrays; ignore here
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)  # type: ignore[call-overload]

    expected_w = Q_KEY_WIDTH * scale_guess
    expected_h = Q_KEY_HEIGHT * scale_guess
    best: tuple[float, Rect] | None = None
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if not (0.65 * expected_w <= width <= 1.45 * expected_w):
            continue
        if not (0.65 * expected_h <= height <= 1.45 * expected_h):
            continue
        fill = area / (width * height)
        if fill < 0.45:
            continue
        rect = Rect(search.x + x, search.y + y, width, height)
        distance = abs(rect.x - predicted_x) / max(expected_w, 1) + abs(
            rect.y - predicted_y
        ) / max(expected_h, 1)
        size_error = abs(width / expected_w - 1) + abs(height / expected_h - 1)
        score = max(0.0, 1.0 - 0.18 * distance - 0.25 * size_error)
        if best is None or score > best[0]:
            best = score, rect

    if best is None or best[0] < 0.55:
        return None

    confidence, q_key = best
    scale = q_key.width / Q_KEY_WIDTH
    e_key = Rect(
        int(round(q_key.x - E_TO_Q_DISTANCE * scale)),
        q_key.y,
        q_key.width,
        q_key.height,
    )
    e_number = _clamp(
        Rect(
            int(round(e_key.x - 25 * scale)),
            int(round(e_key.y - 84 * scale)),
            int(round(90 * scale)),
            int(round(65 * scale)),
        ),
        viewport,
    )
    q_number = _clamp(
        Rect(
            int(round(q_key.x - 34 * scale)),
            int(round(q_key.y - 119 * scale)),
            int(round(110 * scale)),
            int(round(90 * scale)),
        ),
        viewport,
    )
    return SkillHud(e_key, q_key, e_number, q_number, scale, confidence)
