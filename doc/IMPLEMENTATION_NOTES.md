# 实现补充说明（DESIGN.md 附录）

本文是 `doc/DESIGN.md` 的补充说明，**不修改 DESIGN.md 本体**。
DESIGN.md 描述“为什么这样设计”，本文描述“代码落在哪里、有哪些实现细节与已修复的问题”。

运行步骤、系统依赖、CUDA/cuDNN 安装、LLM 服务配置见 `doc/QUICKSTART.md`。

---

## 1. DESIGN.md 模块与代码文件的对应关系

| DESIGN.md 模块 | 代码文件 | 关键实现 |
| --- | --- | --- |
| 模块一：系统音频捕获 | `subtitle_app/audio_capture.py` | `pactl get-default-sink` 拼 `<sink>.monitor`；`parec` 显式指定 16kHz/单声道/s16le；按 32ms（512 采样）定长从 stdout 切帧 |
| 模块二：流式 STT | `subtitle_app/stt.py`、`subtitle_app/vad.py` | Silero VAD v5 做流式切分；`faster-whisper` 做识别；`no_speech_prob` 与 `avg_logprob` 双条件门控 |
| 模块三：英→中翻译 | `subtitle_app/translator.py` | `requests` 直连 OpenAI 兼容接口，**不使用 openai 包**；带上一句作为上下文 |
| 模块四：字幕渲染 | `subtitle_app/subtitle_ui.py` | PyQt6 透明、无边框、置顶、点击穿透；跨线程用 `pyqtSignal` 编组到 GUI 线程 |
| 流水线编排 | `subtitle_app/pipeline.py` | 捕获 → 分段 → 识别 → 翻译 → 回调，全部在独立线程 |
| 配置 | `subtitle_app/config.py`、`config.toml` | TOML + dataclass，未知字段忽略 |
| 入口 | `subtitle_app/app.py`、`subtitle_app/__main__.py` | 组装 UI 与流水线 |

---

## 2. requirements.txt 是什么

`requirements.txt` 不是程序代码，而是 **pip 的依赖清单**：
每行一个包名加版本约束，`#` 开头是注释。安装命令：

    pip install -r requirements.txt

当前清单包含：`faster-whisper`、`onnxruntime`、`numpy`、`requests`、`PyQt6`。
**不再包含 `silero-vad`**：VAD 直接复用 faster-whisper 自带的
Silero VAD v5 ONNX 模型（`faster_whisper/assets/silero_encoder_v5.onnx`），
后端就是 `onnxruntime`，因此不会引入 PyTorch，安装体积大幅下降。

文件末尾注释掉的两行（`nvidia-cublas-cu12`、`nvidia-cudnn-cu12`）是可选的
pip 版 CUDA/cuDNN 运行时，只在系统没有可用 CUDA/cuDNN 时按需启用，
启用方式和 `LD_LIBRARY_PATH` 设置见 `doc/QUICKSTART.md` 第 4 节。

需要 Python 3.11+，因为 `config.py` 使用了标准库 `tomllib`（Debian 12 自带 3.11）。

---

## 3. 与 DESIGN.md 略有出入的实现取舍

以下不是对设计的否定，而是落地时为了让程序能跑起来做的具体选择：

- **VAD 用 Silero VAD v5，复用 faster-whisper 内置的那一份**。Silero 对噪声和
  非语音段更稳，且与 `faster-whisper` 的 `vad_filter` 是同一模型，行为一致。
  代价是 16kHz 下每帧必须是 512 采样（32ms），`[audio].frame_ms` 默认随之改为 32。
- **上下文只带一句**。DESIGN.md 提到“滑动窗口”，当前实现只把
  `last_sentence` 作为上一句传给翻译接口（`_user_content` 拼成
  “上一句 / 当前句”）。窗口再长会增加延迟，先保证端到端可用。
- **丢弃幻觉用“与”而非“或”**。`no_speech_prob` 与 `avg_logprob`
  两道阈值同时命中才丢弃，避免误杀“安静但真实的人声”，与
  DESIGN.md 模块二给出的配置一致。
- **STT 不开 `condition_on_previous_text`**。流式分段场景下开启它更容易
  把上一段的幻觉传染给下一段。
- **捕获分帧由程序做，不由 `parec` 做**。`parec` 只负责重采样和混单，
  定长切帧在 `AudioCapture._read_exact` 中保证，便于 VAD 逐帧判断。
- **UI 只显示中文**。英文原文通过 `on_subtitle(english, chinese)` 回调
  传给上层，当前 `app.py` 只把中文推进字幕窗；需要双语时改 `on_subtitle`
  一行即可。

---

## 4. Silero VAD 的接入细节（faster-whisper 内置模型）

### 4.1 为什么可以省掉 silero-vad 包

faster-whisper 把 Silero VAD v5 的 ONNX 模型作为包数据一起发布，
`faster_whisper.vad.get_speech_timestamps()` 在 `vad_filter=True` 时用的就是它。
我们的 `VadSegmenter` 通过 `faster_whisper.vad.get_vad_model()` 拿到同一个
模型实例，按 512 采样逐块喂入，从而得到每帧语音概率，用于端点检测。

### 4.2 两条必须保留的约束

1. **`[audio].frame_ms` 必须是 32**。Silero VAD v5 在 16kHz 下的流式接口
   只接受 512 采样的块。`vad.py` 启动时会显式校验，不满足直接报错，
   避免静默降级成错误结果。
2. **每次识别后要复位模型状态**。`get_vad_model()` 带 `functools.lru_cache`，
   是进程级单例，与 `transcribe(vad_filter=True)` 共用同一份内部状态。
   识别过程会改动它，所以 `pipeline.py` 在每次 `_handle()` 之后调用
   `segmenter.reset_model_state()`。如果观察到切句抖动，也可以直接把
   `[stt].vad_filter` 设为 `false`（外部 VAD 已经做过一次过滤）。

### 4.3 接口稳定性风险

`faster_whisper.vad.get_vad_model` 属于半公开接口，不同版本 `__call__`
的签名有差异。`vad.py` 里先按 `(audio, num_samples, sample_rate)` 调用，
遇到 `TypeError` 回退到 `(audio, num_samples)`。升级 faster-whisper 后
若报 `调用 Silero VAD 失败`，优先检查这里，并按新版签名调整 `_prob()`。

---

## 5. 已修复的实现问题（避免回退）

这几处是早期版本直接导致程序跑不起来的缺陷，已修复，改动时不要退回：

1. **`pipeline.py` 曾使用 `transcriber.last_text`**，但 `Transcriber`
   没有这个属性，第一段语音就会 `AttributeError`。
   现在由 `_handle()` 返回英文，`_run()` 维护 `last_sentence`。
2. **`app.py` 曾把 `window.push_subtitle` 直接当作 `on_subtitle`**，
   而流水线按 `on_subtitle(english, chinese)` 两个参数调用、
   `push_subtitle(text)` 只接收一个参数，会 `TypeError`。
   现在用包装函数 `on_subtitle(english, chinese)` 只把中文推给 UI。
3. **`[vad].use_onnx` 已删除**。改为复用 faster-whisper 内置模型后，
   这个开关没有意义；`_from_dict()` 会忽略旧配置里的该字段，不影响启动。

---

## 6. 调参速查

| 目标 | 调整 |
| --- | --- |
| 字幕出得太慢 | 降 `[vad].min_silence_ms`（如 500 → 350）；`[stt].beam_size` 降到 1 |
| 句子被切得太碎 | 升 `[vad].min_silence_ms`、`[vad].min_speech_ms` |
| 一直没有字幕 | 降 `[vad].threshold`（如 0.5 → 0.35，更敏感）；先用 `parec` 验证 monitor 有数据 |
| 静音/噪声被识别出字 | 升 `[vad].threshold`（如 0.5 → 0.7，更保守） |
| 切句位置抖动 | `[stt].vad_filter = false`，避免与外部 VAD 共用模型状态 |
| 幻觉太多 | 降 `[stt].no_speech_threshold`（如 0.6 → 0.5） |
| 显存不足 | `[stt].compute_type = "int8_float16"`，或换 `small.en` |
| 翻译太慢 | 降 `[translate].max_tokens`；确认 LLM 服务已开 GPU 卸载 |
| 译文带解释或引号 | `translator.py` 的 `SYSTEM_PROMPT` 再加约束，或前端补一次清洗 |

---

## 7. 尚未实现、可作为后续项

- 双语字幕（同时显示英文原文与中文译文）。
- 翻译上下文的滑动窗口（目前是一句）。
- 模型常驻进程 / 预热，减少首帧延迟。
- Wayland 下用 `wlr-layer-shell` 做可靠置顶与点击穿透。
- 字幕行淡出与打字机效果（DESIGN.md 提到的 `translation-overlay` 特性）。
