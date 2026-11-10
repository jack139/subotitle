"""配置加载：从 TOML 读取，缺省值定义在各 dataclass 中。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    format: str = "s16le"
    # 每帧长度。Whisper 只要求整段音频是 16kHz 单声道，
    # 这里只要保证 frame_ms 是 max_segment_ms 的整除因子即可。
    frame_ms: int = 32
    latency_msec: int = 200
    # 显式指定 monitor source；None 表示用默认 sink 自动推导
    source: str | None = None


@dataclass
class VadConfig:
    # 本节参数全部传给 faster-whisper 内置的 Silero VAD filter。
    # 开关在 [stt].vad_filter；关掉后这些参数不生效。
    # 语音概率阈值：越大越保守
    threshold: float = 0.5
    # 少于该时长的语音片段会被丢弃
    min_speech_ms: int = 250
    # 静音超过该时长才认为一句话结束（越小切得越碎、延迟越低）
    min_silence_ms: int = 500
    # 每段语音前后补的静音，避免切掉字头字尾
    padding_ms: int = 200
    # 送给 Whisper 的单个音频块最大长度，实际决定调用 Whisper 的频率
    max_segment_ms: int = 8000


@dataclass
class SttConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 5
    # 启用 faster-whisper 自带的 Silero VAD filter，参数见 [vad]
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0


@dataclass
class TranslateConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "not-needed"
    model: str = "qwen2.5-7b-instruct"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: float = 30.0


@dataclass
class UiConfig:
    font_size: int = 28
    width_ratio: float = 0.8
    band_height: int = 220
    bottom_margin: int = 60
    max_lines: int = 3
    background: str = "rgba(0, 0, 0, 160)"
    text_color: str = "#ffffff"
    screen: int = 0


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    ui: UiConfig = field(default_factory=UiConfig)


_SECTION_TYPES = {
    "audio": AudioConfig,
    "vad": VadConfig,
    "stt": SttConfig,
    "translate": TranslateConfig,
    "ui": UiConfig,
}


def _from_dict(cls, data: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig()
    if path is None:
        return cfg

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")

    with p.open("rb") as f:
        raw = tomllib.load(f)

    for section, cls in _SECTION_TYPES.items():
        if section in raw:
            setattr(cfg, section, _from_dict(cls, raw[section]))
    return cfg
