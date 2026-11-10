"""faster-whisper 语音识别封装（GPU 友好），VAD 由模型自带的 Silero filter 完成。"""

from __future__ import annotations

import logging

import numpy as np

from .config import SttConfig, VadConfig

log = logging.getLogger(__name__)


def build_vad_parameters(cfg: VadConfig) -> dict:
    """把 [vad] 配置翻译成 faster-whisper 的 VadOptions 字段。

    传入的 key 必须是 VadOptions 的合法字段，否则 VadOptions(**...) 会报错。
    """
    return {
        "threshold": cfg.threshold,
        "min_speech_duration_ms": cfg.min_speech_ms,
        "min_silence_duration_ms": cfg.min_silence_ms,
        "speech_pad_ms": cfg.padding_ms,
    }


class Transcriber:
    def __init__(self, cfg: SttConfig, vad_cfg: VadConfig | None = None):
        self.cfg = cfg
        self.vad_cfg = vad_cfg or VadConfig()
        self.vad_parameters = build_vad_parameters(self.vad_cfg)

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
            vad_parameters=self.vad_parameters if self.cfg.vad_filter else None,
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
