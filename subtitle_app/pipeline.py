"""流水线：捕获 → 定长分块 → STT（内置 VAD）→ 翻译 → 回调。"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

from .audio_capture import AudioCapture
from .config import AppConfig
from .stt import Transcriber
from .translator import Translator
from .vad import AudioChunker

log = logging.getLogger(__name__)

SubtitleCallback = Callable[[str, str], None]  # (english, chinese)
ErrorCallback = Callable[[str], None]


class SubtitlePipeline:
    def __init__(
        self,
        cfg: AppConfig,
        on_subtitle: SubtitleCallback,
        on_error: ErrorCallback | None = None,
    ):
        self.cfg = cfg
        self.on_subtitle = on_subtitle
        self.on_error = on_error or (lambda msg: None)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="subtitle-pipeline", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        capture = AudioCapture(self.cfg.audio)
        translator: Translator | None = None
        last_sentence: str | None = None

        try:
            chunker = AudioChunker(
                sample_rate=self.cfg.audio.sample_rate,
                frame_ms=self.cfg.audio.frame_ms,
                max_segment_ms=self.cfg.vad.max_segment_ms,
            )
            transcriber = Transcriber(self.cfg.stt, self.cfg.vad)
            translator = Translator(self.cfg.translate)

            for frame in capture.frames():
                if self._stop.is_set():
                    break
                for segment in chunker.process(frame):
                    if self._stop.is_set():
                        break
                    last_sentence = self._handle(
                        segment, transcriber, translator, last_sentence
                    )

            tail = chunker.flush()
            if tail is not None and not self._stop.is_set():
                self._handle(tail, transcriber, translator, last_sentence)

        except Exception as e:  # noqa: BLE001
            log.exception("流水线异常")
            self.on_error(str(e))
        finally:
            capture.close()
            if translator:
                translator.close()

    def _handle(
        self,
        audio: np.ndarray,
        transcriber: Transcriber,
        translator: Translator,
        context: str | None,
    ) -> str | None:
        english = transcriber.transcribe(audio)
        if not english:
            return context
        chinese = translator.translate(english, context=context)
        if chinese:
            self.on_subtitle(english, chinese)
        return english
