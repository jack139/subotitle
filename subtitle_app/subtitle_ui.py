"""PyQt6 透明置顶字幕层。"""

from __future__ import annotations

import collections
import logging

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from .config import UiConfig

log = logging.getLogger(__name__)


class SubtitleWindow(QWidget):
    # 内部信号：把任意线程的更新编组到 GUI 主线程
    _push = pyqtSignal(str)

    def __init__(self, cfg: UiConfig):
        super().__init__()
        self.cfg = cfg
        self._history: collections.deque[str] = collections.deque(
            maxlen=cfg.max_lines
        )

        self.setWindowTitle("Live Subtitle")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._label = QLabel("", self)
        self._label.setWordWrap(True)
        self._label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
        )
        self._label.setFont(QFont("Noto Sans CJK SC", cfg.font_size))
        self._label.setStyleSheet(
            f"color: {cfg.text_color};"
            f"background-color: {cfg.background};"
            "border-radius: 12px;"
            "padding: 12px 20px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._push.connect(self._on_push)
        self._reposition()

    def _reposition(self) -> None:
        screens = QApplication.screens()
        if not screens:
            return
        screen = screens[min(self.cfg.screen, len(screens) - 1)]
        geo = screen.availableGeometry()
        w = int(geo.width() * self.cfg.width_ratio)
        h = self.cfg.band_height
        x = geo.x() + (geo.width() - w) // 2
        y = geo.y() + geo.height() - h - self.cfg.bottom_margin
        self.setGeometry(x, y, w, h)

    def push_subtitle(self, text: str) -> None:
        """线程安全：可从流水线线程直接调用。"""
        self._push.emit(text)

    @pyqtSlot(str)
    def _on_push(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        self._label.setText("\n".join(self._history))
