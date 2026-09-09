from __future__ import annotations

from typing import Any

import numpy as np


class DxcamCapture:
    """Small lifecycle boundary around Windows Desktop Duplication."""

    def __init__(self, output_index: int, target_fps: int):
        self.output_index = output_index
        self.target_fps = target_fps
        self._camera: Any = None

    def start(self) -> None:
        import dxcam

        self._camera = dxcam.create(
            output_idx=self.output_index,
            output_color="BGR",
            max_buffer_len=2,
        )
        self._camera.start(target_fps=self.target_fps, video_mode=True)

    def latest(self) -> np.ndarray | None:
        if self._camera is None:
            raise RuntimeError("capture is not started")
        return self._camera.get_latest_frame(copy=True)

    def close(self) -> None:
        if self._camera is None:
            return
        camera, self._camera = self._camera, None
        try:
            camera.stop()
        finally:
            camera.release()

    def __enter__(self) -> "DxcamCapture":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
