"""基于 Silero VAD 的流式语音分段。

复用 faster-whisper 内置的 Silero VAD v5（ONNX）模型，
因此不需要额外安装 `silero-vad` / PyTorch。
"""

from __future__ import annotations

import collections
import logging

import numpy as np

from .config import VadConfig

log = logging.getLogger(__name__)

try:
    from faster_whisper.vad import get_vad_model
except ImportError:  # pragma: no cover
    get_vad_model = None

# Silero VAD v5 在 16kHz 下的固定最小处理单元（512 采样 = 32ms）
_SILERO_SAMPLES_16K = 512


class VadSegmenter:
    """逐帧喂入 16kHz 单声道 float32，返回切好的语音段（np.float32 数组）。

    模型来自 faster-whisper 自带的 Silero VAD v5 ONNX 实现：
    每次喂入 512 个采样点（32ms），模型内部维护状态并返回该帧的语音概率。
    """

    def __init__(self, cfg: VadConfig, sample_rate: int, frame_ms: int):
        if get_vad_model is None:
            raise RuntimeError(
                "缺少 faster-whisper，无法使用其内置 Silero VAD，"
                "请 `pip install -r requirements.txt`"
            )
        if sample_rate != 16000:
            raise ValueError(
                "Silero VAD v5 流式接口按 16kHz 设计，"
                "请把 [audio].sample_rate 设为 16000"
            )
        frame_samples = sample_rate * frame_ms // 1000
        if frame_samples != _SILERO_SAMPLES_16K:
            raise ValueError(
                f"Silero VAD 在 16kHz 下需要每帧 {_SILERO_SAMPLES_16K} 个采样点，"
                f"当前 frame_ms={frame_ms} 得到 {frame_samples}。"
                "请把 [audio].frame_ms 设为 32。"
            )

        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.threshold = float(cfg.threshold)

        log.info("加载 faster-whisper 内置 Silero VAD (ONNX)")
        try:
            self.model = get_vad_model()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"加载内置 Silero VAD 失败: {e}"
                "（该模型依赖 onnxruntime，请确认已安装）"
            ) from e
        self.reset_model_state()

        self.min_speech_frames = max(1, cfg.min_speech_ms // frame_ms)
        self.max_silence_frames = max(1, cfg.min_silence_ms // frame_ms)
        self.max_segment_frames = max(1, cfg.max_segment_ms // frame_ms)
        self.padding_frames = max(0, cfg.padding_ms // frame_ms)

        self._preroll = collections.deque(maxlen=self.padding_frames or None)
        self._segment: list[np.ndarray] = []
        self._in_speech = False
        self._silence_frames = 0
        self._speech_frames = 0

    def reset_model_state(self) -> None:
        """复位 Silero 模型的流式状态。

        该模型实例由 faster-whisper 的 `lru_cache` 共享，
        `transcribe(vad_filter=True)` 会改动它的内部状态，
        因此每次识别结束后需要复位，避免影响后续帧的概率。
        """
        reset = getattr(self.model, "reset_states", None)
        if reset is None:
            return
        try:
            reset()
        except TypeError:
            # 个别版本要求显式 batch_size
            reset(1)

    def _prob(self, chunk: np.ndarray) -> float:
        n = int(chunk.shape[0])
        try:
            out = self.model(chunk, n, self.sample_rate)
        except TypeError:
            # 较旧版本的 __call__ 只有 (audio, num_samples)。
            # 注意：若 TypeError 来自函数体内部，这一帧会被重复喂入一次，
            # 概率略受影响，但不影响分段逻辑。
            out = self.model(chunk, n)
        return float(np.asarray(out, dtype=np.float32).reshape(-1)[-1])

    def _is_speech(self, frame: np.ndarray) -> bool:
        chunk = np.ascontiguousarray(frame, dtype=np.float32)
        return self._prob(chunk) >= self.threshold

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
