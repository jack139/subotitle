"""PyQt6 透明置顶字幕层。"""

from __future__ import annotations

import collections
import logging

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from .config import UiConfig

log = logging.getLogger(__name__)

# 关闭按钮与字幕条边缘的间距
_CLOSE_BUTTON_MARGIN = 8


class _CloseButton(QPushButton):
    """字幕窗右上角的退出按钮。

    必须是 SubtitleWindow 的子控件（而不是独立顶层窗口）：
    独立顶层窗口要靠 move() 定位，而 Wayland 下顶层窗口的 move() 会被
    组合器忽略，按钮会跑到别处，看起来就是“点了没反应”。
    """

    def __init__(self, parent: QWidget, size: int):
        super().__init__("✕", parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("退出程序")
        self.setFixedSize(size, size)

        radius = size // 2
        self.setStyleSheet(
            "QPushButton {"
            " color: #ffffff;"
            " background-color: rgba(255, 255, 255, 45);"
            f" border-radius: {radius}px;"
            f" font-size: {max(12, size // 2)}px;"
            " border: none;"
            "}"
            "QPushButton:hover { background-color: rgba(255, 80, 80, 200); }"
            "QPushButton:pressed { background-color: rgba(220, 40, 40, 230); }"
        )
        self.clicked.connect(self._quit)

    def _quit(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.quit()


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
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # 注意：这里不能再设置 WA_TransparentForMouseEvents。
        # Qt 的语义是该属性会连同子控件一起屏蔽鼠标事件，
        # 一旦开启，右上角的关闭按钮就永远收不到点击。

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

        # 关闭按钮：子控件，绝对定位在字幕条右上角
        self._close_button = _CloseButton(self, max(28, cfg.font_size + 8))
        self._close_button.raise_()

        self._push.connect(self._on_push)
        self._place_button()
        self._reposition()

    def _place_button(self) -> None:
        """把关闭按钮摆到当前窗口的右上角，并保证它在字幕之上。"""
        btn = self._close_button
        size = btn.width()
        btn.move(
            self.width() - size - _CLOSE_BUTTON_MARGIN,
            _CLOSE_BUTTON_MARGIN,
        )
        btn.raise_()

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
        self._place_button()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        super().resizeEvent(event)
        self._place_button()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._reposition()

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
