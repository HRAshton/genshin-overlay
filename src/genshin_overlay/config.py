from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    output_index: int = 0
    capture_fps: int = 30
    analysis_fps: int = 12
    geometry_interval_seconds: float = 1.0
    overlay_x: float = 0.5
    overlay_top_margin: int = 16
    debug: bool = False
