# 设计方案

## 设计目标

用 Python 编写一个本地程序，使用 PC 的系统音频作为输入（即当前系统播放的音频、耳机里听到的声音），调用后台的音频识别模型或音频翻译模型，把**英文**翻译为**中文**，在桌面显示字幕。

---

## 整体架构判断

可行架构是一条“流水线”：

```
系统音频 monitor 捕获
        ↓
VAD 语音活动检测 + 分段
        ↓
流式/准实时 STT
        ↓
英→中机器翻译
        ↓
桌面字幕渲染
```

---

## 模块一：系统音频捕获（Debian 12）

Debian 12 桌面默认使用 PipeWire，同时提供 PulseAudio 兼容层。捕获“当前系统正在播放的声音”最直接的方式是抓取默认输出 sink 的 **monitor source**。

**关键兼容性问题**：`@DEFAULT_MONITOR@` 在 `parec -d` 中不被支持，在 PipeWire 0.3.65（Debian 12）上会静默退回到麦克风输入。不能依赖 `parec -d @DEFAULT_MONITOR@`，而应先通过 `pactl list sources short` 找到默认 sink 对应的 `.monitor` 源名称，再显式传入。

**推荐实现顺序：**

1. **先用 `pactl get-default-sink` 拿到默认 sink 名**，拼接为 `<sink_name>.monitor`，作为 `parec -d` 的参数。这是最可控的方式。
2. **`parec` 的输出参数显式指定**：`--rate=16000 --channels=1 --format=s16le`。直接在 `parec` 参数中指定 16kHz 单声道可以省去后续一次重采样，也减少格式转换的歧义。如果 `parec` 不接受，再用 `librosa` 或 `scipy` 做客户端重采样和混单。

**额外注意点**：`parec` 在 PipeWire 上停止录制时可能丢失最后几秒的数据。如果分段策略依赖精确的截止点，需要确认这一点是否影响 VAD 分段；若影响，改用 `flexaudio`。


---

## 模块二：流式语音识别（STT）

STT 是整条流水线中延迟和资源消耗的大头。必须按 CPU / GPU 分两条配置线，延迟预期不同。

**引擎选择：`faster-whisper`**。基于 CTranslate2，在相同准确率下比原版 Whisper 快 4～6 倍，CPU 和 NVIDIA GPU 都能用，INT8 量化后模型体积和内存占用显著降低。

### CPU 线（默认）

基准参考：i7-1370P（20 逻辑核）、`faster-whisper` 1.2.1、int8、无 GPU，对中位时长 7.3 秒的单条语音：

| 模型 | 中位解码时间 | RTF | WER（LibriSpeech clean） | 体积 |
|---|---|---|---|---|
| `tiny.en` | 0.89 s | 0.154（6.5×实时） | 4.82% | 78 MB |
| `base.en` | 1.56 s | 0.283（3.5×实时） | 4.07% | 148 MB |
| `small.en` | 5.05 s | 0.520（1.9×实时） | 2.59% | 486 MB |

更弱的 CPU 会等比例变慢。纯 CPU 上 `small.en` 做实时字幕的延迟是秒级偏高的，端到端会落在 5～8 秒量级；`base.en` 的中位 1.56 秒才是 CPU 实时字幕更现实的起点，代价是 WER 从 2.59% 升到 4.07%——对于“看视频理解内容”，4% 左右的 WER 通常仍可接受。

**CPU 线推荐配置**：`base.en` + `compute_type="int8"` + Silero VAD。若实际体验延迟仍不理想，降到 `tiny.en`；若更看重准确率且能接受更长滞后，再考虑 `small.en`。

### GPU 线（有 CUDA 时）

`small.en` 或 `large-v3-turbo`，`device="cuda"`，`compute_type="float16"` 或 `int8_float16`。GPU 上 `faster-whisper` 的延迟比 CPU 低一个数量级，`small.en` 可以轻松做到接近实时，此时瓶颈会转移到翻译和渲染，而不是 STT。

### VAD 与幻觉抑制

Whisper 在静音和噪声段会编造出 “You” “Thank you” 这类输出，实测约 1.4% 的片段出现幻觉，其中约 38% 有害，VAD 加门控是必要的，不是可选项。

一层可落地的配置：

- Silero VAD 开启：`vad_filter=True`，`vad_min_silence_ms` 默认 500。
- `condition_on_previous_text=False`。
- 置信度门控：当 `no_speech_prob > 0.6` **且** `avg_logprob < -1.0` 时丢弃该段。两个条件同时满足才丢弃，是为了让“安静但真实的人声”不被误杀。

Silero VAD 与能量过滤结合的方案可以把误触发降低约 34%。


---

## 模块三：英→中机器翻译

本地翻译的可行选择是 Opus-MT，可以通过 CTranslate2 加速。

**策略：** 先用 **Opus-MT + CTranslate2** 跑通端到端延迟。翻译这一步的延迟通常不是瓶颈，STT 才是；不要在翻译模型上过早过度投入。

Opus-MT en-zh 作为更轻的第一选择仍然成立，但如果实测 Opus-MT 的译文质量在常看的视频类型上明显不够用，可以考虑小尺寸的Qwen模型。

有 GPU 可用时，优先考虑使用小尺寸Qwen模型的方案，可以使用OpenAI类的接口实现，方便更换后端模型。

### 上下文问题

逐句翻译会丢失跨句上下文，代词、省略、时态在长对话中可能译得不自然。但对于“看视频理解内容”的目标，逐句翻译是可接受的起点。如果后续要优化，可以维护一个滑动窗口的上下文，把前一两句作为翻译模型的参考前缀。

---

## 模块四：桌面字幕渲染


### PyQt6 通用 OSD

用 PyQt6 创建一个透明、无边框、总在最前的窗口，把翻译文本渲染在屏幕底部。已有的 `translation-overlay` 项目正是这个思路：捕获 PipeWire 系统音频 → 本地翻译模型 → PyQt 透明置顶字幕层，带打字机效果和旧行淡出。它目前调的是 Meta SeamlessM4T / faster-whisper，语言对可配置，架构可以直接借鉴。

**Wayland 下的实际限制**：Wayland 出于安全设计，不允许应用程序自行修改窗口的屏幕堆叠顺序，`WindowStaysOnTopHint` 和 `raise_()` 在部分 compositor 上可能不生效或行为不一致。透明窗口本身在 PyQt6 6.9 上可以创建，问题不在透明度，而在置顶和点击穿透的可靠性。点击穿透还需要额外设置 `Qt.WA_TransparentForMouseEvents` 或 `WindowTransparentForInput`，且同样依赖 compositor 支持。

因此需要在实际桌面环境（GNOME Wayland / KDE Wayland / X11 回退）上尽早验证置顶和点击穿透，不要等到第二阶段末尾才发现不可用。

**多显示器注意**：先限定单显示器，或读取 `QScreen` 的 geometry 显式定位。


---

## 硬件环境

开发环境主要为 Debian 系列（例如 `bookworm`）。

**区分两种使用环境：**

- **有 GPU 和 CUDA 的环境**：STT 用 `small.en` 或 `large-v3-turbo`（fp16/int8_float16），翻译用 Qwen3-1.7B，延迟主要来自分段策略而非推理。显存需同时容纳 STT + 翻译模型，桌面级 GPU 需留意余量。
- **普通 CPU 环境**：STT 用 `base.en`（int8）为起点，翻译用 Opus-MT 。端到端延迟预计在 2～5 秒量级，取决于 CPU 性能和分段长度。

---

## 容易踩的坑

- **Whisper 幻觉**：静音段会被“翻译”出无关文本。务必加 VAD，并对极短片段或低置信度结果做过滤。
- **音频重采样**：Whisper 期望 16kHz 单声道。`parec` 出来的可能是 48kHz 立体声，需要在送入模型前做重采样和混单，否则会影响识别质量。优先在 `parec` 参数中直接指定 16kHz 单声道。
- **PipeWire 与 PulseAudio 的工具混用**：Debian 12 上 `pactl` / `parec` 通常指向 PipeWire 兼容层，行为基本一致，但如果遇到 monitor source 名称异常，用 `pactl list sources short` 和 `pw-cli` 交叉确认。
- **逐句翻译的上下文丢失**：可以接受，但如果发现质量明显下降，考虑在翻译时带上前一句作为上下文。
- **不要过早追求“流式”**：Whisper 的“流式”本质是分块批处理，分块太短会牺牲准确率，太长会增加延迟。先用 2～3 秒的 VAD 段跑通，再根据体感调参。
