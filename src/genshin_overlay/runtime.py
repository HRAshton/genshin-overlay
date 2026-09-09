from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from genshin_overlay.analyzer import ScreenAnalyzer
from genshin_overlay.capture import DxcamCapture
from genshin_overlay.config import AppConfig
from genshin_overlay.debug import annotate_frame
from genshin_overlay.diagnostics import DebugRecorder
from genshin_overlay.tracker import CooldownTracker


class VisionWorker(QThread):
    snapshot_ready = Signal(object)
    debug_ready = Signal(object, str)
    bundle_saved = Signal(str)
    fatal_error = Signal(str)

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._dump_requested = threading.Event()

    def request_debug_dump(self) -> None:
        self._dump_requested.set()

    def run(self) -> None:
        analyzer = ScreenAnalyzer()
        tracker = CooldownTracker()
        analysis_period = 1.0 / self.config.analysis_fps
        next_analysis = 0.0
        next_geometry = 0.0
        recorder = (
            DebugRecorder(Path.cwd() / "debug_sessions") if self.config.debug else None
        )
        tracked_keys: set[tuple[int, str]] = set()
        try:
            with DxcamCapture(self.config.output_index, self.config.capture_fps) as capture:
                while not self.isInterruptionRequested():
                    now = time.monotonic()
                    if now < next_analysis:
                        self.msleep(2)
                        continue
                    frame = capture.latest()
                    if frame is None:
                        self.msleep(5)
                        continue
                    refresh_geometry = now >= next_geometry
                    observation = analyzer.analyze(
                        frame,
                        refresh_geometry=refresh_geometry,
                    )
                    if refresh_geometry:
                        next_geometry = now + self.config.geometry_interval_seconds
                    snapshot = tracker.update(frame, observation, now)
                    self.snapshot_ready.emit(snapshot)
                    if self.config.debug:
                        assert recorder is not None
                        recorder.record(
                            frame,
                            observation,
                            tracker.last_decisions,
                            snapshot,
                            now,
                        )
                        current_keys = {
                            (item.party_slot, item.skill) for item in snapshot
                        }
                        new_keys = current_keys - tracked_keys
                        reasons = [
                            "cooldown-start-"
                            + "-".join(f"slot{slot}{skill}" for slot, skill in sorted(new_keys))
                        ] if new_keys else []
                        if self._dump_requested.is_set():
                            self._dump_requested.clear()
                            reasons.append("manual")
                        if reasons:
                            bundle = recorder.dump(
                                frame,
                                observation,
                                "+".join(reasons),
                                now,
                            )
                            self.bundle_saved.emit(str(bundle))
                        tracked_keys = current_keys
                        annotated = annotate_frame(frame, observation)
                        if annotated.shape[1] > 1280:
                            ratio = 1280 / annotated.shape[1]
                            annotated = cv2.resize(
                                annotated,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_AREA,
                            )
                        party = (
                            "unknown"
                            if observation.party is None
                            else str(observation.party.active_slot)
                        )
                        status = (
                            f"active={party} "
                            f"E={observation.e.text or observation.e.state.value} "
                            f"Q={observation.q.text or observation.q.state.value} "
                            f"tracked={len(snapshot)}"
                        )
                        self.debug_ready.emit(annotated, status)
                    next_analysis = now + analysis_period
        except Exception as error:
            logging.exception("capture worker failed")
            self.fatal_error.emit(str(error))

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(3000)
