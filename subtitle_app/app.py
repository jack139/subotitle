"""应用入口：组装 UI 与流水线。"""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from .config import load_config
from .pipeline import SubtitlePipeline
from .subtitle_ui import SubtitleWindow

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    cfg = load_config(config_path)
    log.info("使用配置: %s", config_path)

    app = QApplication(sys.argv)
    window = SubtitleWindow(cfg.ui)
    window.show()

    def on_subtitle(english: str, chinese: str) -> None:
        log.debug("英文: %s", english)
        window.push_subtitle(chinese)

    pipeline = SubtitlePipeline(
        cfg,
        on_subtitle=on_subtitle,
        on_error=lambda msg: log.error("流水线错误: %s", msg),
    )
    pipeline.start()

    app.aboutToQuit.connect(pipeline.stop)
    return app.exec()
