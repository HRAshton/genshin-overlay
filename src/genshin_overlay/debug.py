from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from genshin_overlay.models import Rect, ScreenObservation
from genshin_overlay.vision.digits import white_text_mask


def _box(image: np.ndarray, rect: Rect, color: tuple[int, int, int], label: str) -> None:
    cv2.rectangle(image, (rect.x, rect.y), (rect.right, rect.bottom), color, 3)
    cv2.putText(
        image,
        label,
        (rect.x, max(25, rect.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def save_debug_images(
    frame: np.ndarray,
    observation: ScreenObservation,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = annotate_frame(frame, observation)
    if observation.hud is not None:
        for skill, rect in (("e", observation.hud.e_number), ("q", observation.hud.q_number)):
            crop = rect.crop(frame)
            cv2.imwrite(str(output_dir / f"{stem}_{skill}_crop.png"), crop)
            cv2.imwrite(str(output_dir / f"{stem}_{skill}_mask.png"), white_text_mask(crop))
    if observation.party is not None:
        cv2.imwrite(
            str(output_dir / f"{stem}_portrait.png"),
            observation.party.portrait_rect.crop(frame),
        )
    cv2.imwrite(str(output_dir / f"{stem}_annotated.png"), annotated)


def annotate_frame(frame: np.ndarray, observation: ScreenObservation) -> np.ndarray:
    annotated = frame.copy()
    _box(annotated, observation.viewport, (255, 180, 0), "viewport")
    if observation.hud is not None:
        _box(annotated, observation.hud.e_number, (0, 255, 0), f"E {observation.e.text or observation.e.state.value}")
        _box(annotated, observation.hud.q_number, (0, 220, 255), f"Q {observation.q.text or observation.q.state.value}")
    if observation.party is not None:
        _box(annotated, observation.party.portrait_rect, (255, 0, 255), f"active {observation.party.active_slot}")
    return annotated
