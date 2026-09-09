from __future__ import annotations

import cv2
import numpy as np

from genshin_overlay.models import Rect


def _largest_run(indices: np.ndarray) -> tuple[int, int] | None:
    if indices.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, indices.size - 1]
    lengths = indices[ends] - indices[starts] + 1
    best = int(np.argmax(lengths))
    return int(indices[starts[best]]), int(indices[ends[best]]) + 1


def locate_viewport(frame: np.ndarray, black_threshold: float = 2.0) -> Rect:
    """Find largest rendered region separated by true black letterboxing."""
    if frame is None or frame.ndim != 3 or frame.size == 0:
        raise ValueError("frame must be a non-empty BGR image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    column_run = _largest_run(np.flatnonzero(gray.mean(axis=0) > black_threshold))
    if column_run is None or column_run[1] - column_run[0] < frame.shape[1] * 0.3:
        column_run = (0, frame.shape[1])

    x0, x1 = column_run
    row_run = _largest_run(
        np.flatnonzero(gray[:, x0:x1].mean(axis=1) > black_threshold)
    )
    if row_run is None or row_run[1] - row_run[0] < frame.shape[0] * 0.3:
        row_run = (0, frame.shape[0])

    y0, y1 = row_run
    return Rect(x0, y0, x1 - x0, y1 - y0)
