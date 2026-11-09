"""基于 webrtcvad 的流式语音分段。"""

from __future__ import annotations

import collections
import logging

import numpy as np

from .config import VadConfig

try:
    import webrtcvad
except ImportError:  # pragma: no cover
    webrtcvad = None

log = logging.getLogger(__name__)


class VadSegmenter:
    """逐帧喂入 16kHz 单声道 float32，返回切好的语音段（np.float32 数组）。

    状态机：检测到语音后开始累积，静音达到阈值或段长超限即输出。
    """

    def __init__(self, cfg: VadConfig, sample_rate: int, frame_ms: int):
        if webrtcvad is None:
            raise RuntimeError("缺少 webrtcvad，请安装 webrtcvad-wheels")
        if frame_ms not in (10, 20, 30):
            raise ValueError("webrtcvad 仅支持 10/20/30ms 帧长")

        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(cfg.aggressiveness)

        self.min_speech_frames = max(1, cfg.min_speech_ms // frame_ms)
        self.max_silence_frames = max(1, cfg.min_silence_ms // frame_ms)
        self.max_segment_frames = max(1, cfg.max_segment_ms // frame_ms)
        self.padding_frames = max(0, cfg.padding_ms // frame_ms)

        self._preroll = collections.deque(maxlen=self.padding_frames or None)
        self._segment: list[np.ndarray] = []
        self._in_speech = False
        self._silence_frames = 0
        self._speech_frames = 0

    def _is_speech(self, frame: np.ndarray) -> bool:
        pcm = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        return self.vad.is_speech(pcm, self.sample_rate)

    def process(self, frame: np.ndarray) -> list[np.ndarray]:
        expected = self.sample_rate * self.frame_ms // 1000
        if len(frame) != expected:
            return []

        completed: list[np.ndarray] = []
        if self._is_speech(frame):
            if not self._in_speech:
                self._in_speech = True
                self._segment = list(self._preroll)
                self._preroll.clear()
                self._speech_frames = 0
                self._silence_frames = 0
            self._segment.append(frame)
            self._speech_frames += 1
            self._silence_frames = 0
            if len(self._segment) >= self.max_segment_frames:
                seg = self._finalize()
                if seg is not None:
                    completed.append(seg)
        else:
            if self._in_speech:
                self._segment.append(frame)
                self._silence_frames += 1
                if self._silence_frames >= self.max_silence_frames:
                    seg = self._finalize()
                    if seg is not None:
                        completed.append(seg)
            elif self.padding_frames:
                self._preroll.append(frame)
        return completed

    def flush(self) -> np.ndarray | None:
        if self._in_speech:
            return self._finalize()
        return None

    def _finalize(self) -> np.ndarray | None:
        seg = np.concatenate(self._segment) if self._segment else None
        enough = self._speech_frames >= self.min_speech_frames

        self._segment = []
        self._in_speech = False
        self._silence_frames = 0
        self._speech_frames = 0

        if seg is None or not enough:
            return None
        return seg
