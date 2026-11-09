"""流水线：捕获 → VAD 分段 → STT → 翻译 → 回调。"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from .audio_capture import AudioCapture
from .config import AppConfig
from .stt import Transcriber
from .translator import Translator
from .vad import VadSegmenter

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
            segmenter = VadSegmenter(
                self.cfg.vad,
                self.cfg.audio.sample_rate,
                self.cfg.audio.frame_ms,
            )
            transcriber = Transcriber(self.cfg.stt)
            translator = Translator(self.cfg.translate)

            for frame in capture.frames():
                if self._stop.is_set():
                    break
                for segment in segmenter.process(frame):
                    if self._stop.is_set():
                        break
                    last_sentence = self._handle(
                        segment, transcriber, translator, last_sentence
                    )
                    # Silero 模型与 faster-whisper 的 vad_filter 共享，
                    # 识别过程会改动其流式状态，这里复位以免影响后续帧
                    segmenter.reset_model_state()

            tail = segmenter.flush()
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
        audio,
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
