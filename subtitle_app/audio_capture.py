"""捕获系统正在播放的声音（PipeWire/PulseAudio 输出 sink 的 monitor）。"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading

import numpy as np

from .config import AudioConfig

log = logging.getLogger(__name__)


def _require(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise RuntimeError(
            f"未找到命令 {cmd}，请确认已安装 pipewire-pulse / pulseaudio-utils"
        )


def default_monitor_source() -> str:
    """通过 pactl 获取默认输出 sink，并拼出对应的 monitor source 名称。"""
    _require("pactl")
    try:
        sink = subprocess.check_output(
            ["pactl", "get-default-sink"], text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("无法获取默认 sink，请检查 PipeWire/PulseAudio 状态") from e
    if not sink:
        raise RuntimeError("默认 sink 名称为空")
    return f"{sink}.monitor"


class AudioCapture:
    """启动 parec 子进程，按固定帧长产出 float32 numpy 数组。"""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()

        self.bytes_per_sample = 2  # s16le
        self.frame_samples = cfg.sample_rate * cfg.frame_ms // 1000
        self.frame_bytes = self.frame_samples * cfg.channels * self.bytes_per_sample

    def _cmd(self) -> list[str]:
        source = self.cfg.source or default_monitor_source()
        log.info("捕获音频源: %s", source)
        return [
            "parec",
            f"--device={source}",
            f"--rate={self.cfg.sample_rate}",
            f"--channels={self.cfg.channels}",
            f"--format={self.cfg.format}",
            f"--latency-msec={self.cfg.latency_msec}",
        ]

    @staticmethod
    def _read_exact(stream, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def frames(self):
        _require("parec")
        self._proc = subprocess.Popen(
            self._cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert self._proc.stdout is not None
        try:
            while not self._stop.is_set():
                data = self._read_exact(self._proc.stdout, self.frame_bytes)
                if data is None:
                    log.warning("parec 输出结束")
                    break

                pcm = np.frombuffer(data, dtype=np.int16)
                if self.cfg.channels > 1:
                    pcm = pcm.reshape(-1, self.cfg.channels).mean(axis=1)
                yield pcm.astype(np.float32) / 32768.0
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
