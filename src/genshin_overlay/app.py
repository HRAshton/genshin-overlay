from __future__ import annotations

import argparse
import logging
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from genshin_overlay.config import AppConfig
from genshin_overlay.overlay import CooldownOverlay, DebugWindow
from genshin_overlay.runtime import VisionWorker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Screen-only Genshin cooldown overlay")
    parser.add_argument("--debug", action="store_true", help="show annotated detector view")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--analysis-fps", type=int, default=12)
    args = parser.parse_args(argv)
    if args.capture_fps <= 0 or args.analysis_fps <= 0:
        parser.error("FPS values must be positive")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("dxcam").setLevel(logging.INFO)
    config = AppConfig(
        output_index=args.output_index,
        capture_fps=args.capture_fps,
        analysis_fps=args.analysis_fps,
        debug=args.debug,
    )

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    screens = app.screens()
    if not 0 <= config.output_index < len(screens):
        raise RuntimeError(f"Qt cannot find output index {config.output_index}")
    screen = screens[config.output_index]

    overlay = CooldownOverlay(config.overlay_x, config.overlay_top_margin)
    overlay.setGeometry(screen.geometry())
    overlay.show()
    QTimer.singleShot(0, overlay.exclude_from_capture)

    debug_window = DebugWindow() if config.debug else None
    if debug_window is not None:
        debug_window.show()
        QTimer.singleShot(0, debug_window.exclude_from_capture)

    worker = VisionWorker(config)
    worker.snapshot_ready.connect(overlay.set_items)
    if debug_window is not None:
        worker.debug_ready.connect(debug_window.set_frame)
        worker.bundle_saved.connect(debug_window.set_saved_bundle)
        debug_window.save_requested.connect(worker.request_debug_dump)

    exit_code = 0

    def fail(message: str) -> None:
        nonlocal exit_code
        exit_code = 1
        logging.error("live capture stopped: %s", message)
        app.quit()

    worker.fatal_error.connect(fail)
    app.aboutToQuit.connect(worker.stop)
    worker.start()
    qt_code = app.exec()
    worker.stop()
    return exit_code or qt_code
