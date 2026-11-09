"""faster-whisper 语音识别封装（GPU 友好）。"""

from __future__ import annotations

import logging

import numpy as np

from .config import SttConfig

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, cfg: SttConfig):
        self.cfg = cfg

        # 延迟导入，避免未安装 faster-whisper 时影响其它模块
        from faster_whisper import WhisperModel

        log.info(
            "加载 Whisper 模型 %s (%s/%s)",
            cfg.model,
            cfg.device,
            cfg.compute_type,
        )
        self.model = WhisperModel(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str | None:
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
        )

        parts: list[str] = []
        for seg in segments:
            # 两个条件同时满足才丢弃，避免误杀“安静但真实的人声”
            if (
                seg.no_speech_prob > self.cfg.no_speech_threshold
                and seg.avg_logprob < self.cfg.log_prob_threshold
            ):
                log.debug(
                    "丢弃疑似幻觉段: %r (no_speech=%.2f, logprob=%.2f)",
                    seg.text,
                    seg.no_speech_prob,
                    seg.avg_logprob,
                )
                continue
            text = seg.text.strip()
            if text:
                parts.append(text)

        text = " ".join(parts).strip()
        return text or None
