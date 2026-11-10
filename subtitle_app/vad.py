"""流式语音分段器。

当前版本已完全禁用 Silero VAD，改为按照固定时长切分音频段，
后续直接交给 Whisper 识别。该模块保留了与原 VAD 版本相同的接口，
避免影响 pipeline.py 等调用方。
"""

from __future__ import annotations

import numpy as np

from .config import VadConfig


class VadSegmenter:
    """逐帧喂入 16kHz 单声道 float32，返回固定时长的语音段。"""

    def __init__(self, cfg: VadConfig, sample_rate: int, frame_ms: int):
        if sample_rate != 16000:
            raise ValueError(
                "当前音频管线按 16kHz 设计，请把 [audio].sample_rate 设为 16000"
            )

        frame_samples = sample_rate * frame_ms // 1000
        if frame_samples != 512:
            raise ValueError(
                f"每帧需要 512 个采样点，"
                f"当前 frame_ms={frame_ms} 得到 {frame_samples}。"
                "请把 [audio].frame_ms 设为 32。"
            )

        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms

        # 每个语音段的最大帧数
        self.max_segment_frames = max(1, cfg.max_segment_ms // frame_ms)

        self._segment: list[np.ndarray] = []

    def reset_model_state(self) -> None:
        """当前没有需要复位的模型状态，保留接口以兼容 pipeline。"""
        pass

    def process(self, frame: np.ndarray) -> list[np.ndarray]:
        """接收一帧音频，如果累积到最大段长则返回一个完整的语音段。"""
        expected = self.sample_rate * self.frame_ms // 1000
        if len(frame) != expected:
            return []

        self._segment.append(frame)

        if len(self._segment) >= self.max_segment_frames:
            segment = self._finalize()
            if segment is not None:
                return [segment]

        return []

    def flush(self) -> np.ndarray | None:
        """返回尚未被切分的剩余音频，通常在流结束时调用。"""
        if self._segment:
            return self._finalize()
        return None

    def _finalize(self) -> np.ndarray | None:
        if not self._segment:
            return None

        segment = np.concatenate(self._segment)
        self._segment = []
        return segment
