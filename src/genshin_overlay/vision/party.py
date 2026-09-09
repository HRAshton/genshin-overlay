from __future__ import annotations

import cv2
import numpy as np

from genshin_overlay.models import PartyObservation, Rect


REFERENCE_HEIGHT = 1440.0
KEY_X = (3805 - 1120) / 2880
ROW_Y = (351, 473, 594, 716)
KEY_WIDTH = 41
KEY_HEIGHT = 34


def locate_active_party(frame: np.ndarray, viewport: Rect) -> PartyObservation | None:
    """Identify active slot from dimmed 1-4 keycap and return portrait crop."""
    scale = viewport.height / REFERENCE_HEIGHT
    key_x = viewport.x + int(round(viewport.width * KEY_X))
    means: list[float] = []
    rects: list[Rect] = []
    for reference_y in ROW_Y:
        rect = Rect(
            key_x,
            viewport.y + int(round(reference_y * scale)),
            max(1, int(round(KEY_WIDTH * scale))),
            max(1, int(round(KEY_HEIGHT * scale))),
        )
        if rect.right > viewport.right or rect.bottom > viewport.bottom:
            return None
        rects.append(rect)
        means.append(float(cv2.cvtColor(rect.crop(frame), cv2.COLOR_BGR2GRAY).mean()))

    order = np.argsort(means)
    active_index = int(order[0])
    separation = means[int(order[1])] - means[active_index]
    if separation < 25:
        return None

    key = rects[active_index]
    portrait = Rect(
        int(round(key.x - 130 * scale)),
        int(round(key.y - 50 * scale)),
        max(1, int(round(125 * scale))),
        max(1, int(round(110 * scale))),
    )
    if portrait.x < viewport.x or portrait.y < viewport.y:
        return None
    confidence = min(1.0, 0.65 + separation / 180)
    return PartyObservation(active_index + 1, portrait, confidence)
