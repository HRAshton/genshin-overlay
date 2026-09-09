from __future__ import annotations

import ctypes
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from genshin_overlay.models import CooldownView


def _qimage_from_bgr(image: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width, _ = rgb.shape
    return QImage(
        rgb.data,
        width,
        height,
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()


def _exclude_window_from_capture(widget: QWidget) -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetWindowDisplayAffinity(int(widget.winId()), 0x11)
    except (AttributeError, OSError):
        pass


class CooldownOverlay(QWidget):
    def __init__(self, horizontal_position: float = 0.5, top_margin: int = 16):
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._horizontal_position = horizontal_position
        self._top_margin = top_margin
        self._items: tuple[CooldownView, ...] = ()
        self._received_at = time.monotonic()
        self._portraits: dict[tuple[int, str], QImage] = {}
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    def set_items(self, items: tuple[CooldownView, ...]) -> None:
        self._items = items
        self._received_at = time.monotonic()
        active_keys = {(item.party_slot, item.skill) for item in items}
        self._portraits = {
            key: image for key, image in self._portraits.items() if key in active_keys
        }
        for item in items:
            key = item.party_slot, item.skill
            self._portraits[key] = _qimage_from_bgr(item.portrait_bgr)
        self.update()

    def exclude_from_capture(self) -> None:
        _exclude_window_from_capture(self)

    def paintEvent(self, _event) -> None:
        if not self._items:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        elapsed = time.monotonic() - self._received_at
        size = 84.0
        gap = 14.0
        total_width = len(self._items) * size + (len(self._items) - 1) * gap
        start_x = self.width() * self._horizontal_position - total_width / 2
        top = float(self._top_margin)
        for index, item in enumerate(self._items):
            remaining = max(0.0, item.remaining_seconds - elapsed)
            if remaining <= 0:
                continue
            rect = QRectF(start_x + index * (size + gap), top, size, size)
            self._paint_item(painter, rect, item, remaining)

    def _paint_item(
        self,
        painter: QPainter,
        rect: QRectF,
        item: CooldownView,
        remaining: float,
    ) -> None:
        portrait = self._portraits[item.party_slot, item.skill]
        circle = QPainterPath()
        circle.addEllipse(rect)

        painter.save()
        painter.setClipPath(circle)
        painter.setOpacity(0.2)
        painter.drawImage(rect, portrait)
        painter.restore()

        fraction = min(1.0, remaining / max(item.total_seconds, 0.01))
        visible = QRectF(
            rect.x(),
            rect.bottom() - rect.height() * fraction,
            rect.width(),
            rect.height() * fraction,
        )
        painter.save()
        painter.setClipPath(circle)
        painter.setClipRect(visible, Qt.ClipOperation.IntersectClip)
        painter.drawImage(rect, portrait)
        painter.restore()

        color = QColor("#67e8f9") if item.skill == "E" else QColor("#facc15")
        painter.setPen(QPen(color, 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)

        badge = QRectF(rect.right() - 25, rect.top() - 3, 27, 27)
        painter.setPen(QPen(QColor(10, 15, 24, 230), 2))
        painter.setBrush(color)
        painter.drawEllipse(badge)
        painter.setPen(QColor("#111827"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, item.skill)

        text = f"{remaining:.1f}"
        painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(0, 0, 0, 220), 5))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.setPen(QColor("white"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class DebugWindow(QWidget):
    save_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genshin Overlay - detector debug")
        self.resize(1280, 420)
        self._image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._status = QLabel("Waiting for capture...")
        self._saved = QLabel("No diagnostic bundle saved yet.")
        self._save = QPushButton("Save diagnostic bundle")
        self._save.clicked.connect(self.save_requested)
        layout = QVBoxLayout(self)
        layout.addWidget(self._image, 1)
        layout.addWidget(self._status)
        layout.addWidget(self._saved)
        layout.addWidget(self._save)

    def exclude_from_capture(self) -> None:
        _exclude_window_from_capture(self)

    def set_frame(self, frame: np.ndarray, status: str) -> None:
        image = _qimage_from_bgr(frame)
        pixmap = QPixmap.fromImage(image).scaled(
            self._image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image.setPixmap(pixmap)
        self._status.setText(status)

    def set_saved_bundle(self, path: str) -> None:
        self._saved.setText(f"Saved: {path}")
