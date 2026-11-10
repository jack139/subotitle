"""定长音频切分。

VAD 已交给 faster-whisper 内置的 Silero VAD filter（见 stt.py），
本模块只负责把音频流切成不超过 max_segment_ms 的窗口，
避免把无限长的音频一次性喂给模型。
"""

from __future__ import annotations

import numpy as np


class AudioChunker:
    """逐帧喂入 16kHz 单声道 float32，按最大时长切分成段。"""

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int,
        max_segment_ms: int,
        min_flush_ms: int = 320,
    ):
        if sample_rate != 16000:
            raise ValueError(
                "Whisper 要求 16kHz 单声道，请把 [audio].sample_rate 设为 16000"
            )
        if frame_ms <= 0:
            raise ValueError(f"frame_ms 必须为正数，当前为 {frame_ms}")
        if max_segment_ms <= 0:
            raise ValueError(f"max_segment_ms 必须为正数，当前为 {max_segment_ms}")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.max_segment_ms = max_segment_ms
        self.min_flush_ms = min_flush_ms

        self._frame_samples = sample_rate * frame_ms // 1000
        self._max_frames = max(1, max_segment_ms // frame_ms)
        self._segment: list[np.ndarray] = []

    def process(self, frame: np.ndarray) -> list[np.ndarray]:
        """接收一帧音频，如果累积到最大段长则返回一个完整的音频段。"""
        if len(frame) != self._frame_samples:
            return []

        self._segment.append(frame)

        if len(self._segment) >= self._max_frames:
            segment = self._finalize()
            if segment is not None:
                return [segment]

        return []

    def flush(self) -> np.ndarray | None:
        """返回尚未切分的剩余音频；过短的尾巴直接丢弃，避免模型产生幻觉。"""
        if not self._segment:
            return None

        remaining_ms = len(self._segment) * self.frame_ms
        if remaining_ms < self.min_flush_ms:
            self._segment = []
            return None

        return self._finalize()

    def _finalize(self) -> np.ndarray | None:
        if not self._segment:
            return None

        segment = np.concatenate(self._segment)
        self._segment = []
        return segment
