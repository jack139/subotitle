# 快速开始：Debian 12 + NVIDIA GPU 实时字幕

本文按"从零到看到字幕"的顺序说明。命令默认在项目根目录执行。

项目根目录至少需要：

    subtitle_app/          # Python 包
    config.toml            # 配置文件
    requirements.txt       # pip 依赖清单

如果你是从旧版本升级，请确认已经使用本次提供的
`subtitle_app/pipeline.py` 和 `subtitle_app/app.py`，否则会遇到
`last_text` 或 `push_subtitle` 参数不匹配的错误。

## 1. requirements.txt 是什么

它不是程序，而是 pip 依赖清单。每行一个包，`#` 开头是注释。
安装：

    pip install -r requirements.txt

本项目需要 Python 3.11+，因为 `config.py` 使用了标准库 `tomllib`。
Debian 12 自带的 `python3` 就是 3.11。

## 2. 安装系统依赖

    sudo apt update
    sudo apt install -y \
      python3 python3-venv python3-pip \
      pipewire-pulse pulseaudio-utils \
      libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 \
      libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
      libxcb-shape0 libxcb-xinerama0 \
      fonts-noto-cjk

说明：

- `pactl` / `parec` 来自 `pulseaudio-utils`，PipeWire 环境下也需要它。
- PyQt6 在 X11 下可能还需要一批 `libxcb-*`，上面已列出常用项。
- `fonts-noto-cjk` 用于正常显示中文字幕。

## 3. 创建虚拟环境并安装 Python 依赖

    cd /path/to/project
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install -r requirements.txt

## 4. 准备 GPU 运行时（CUDA / cuDNN）

先确认驱动正常：

    nvidia-smi

`faster-whisper` 底层是 CTranslate2。CTranslate2 4.5 及以后通常要求
CUDA 12 + cuDNN 9；更早的 4.x 可能要求 cuDNN 8。先看版本：

    python -c "import ctranslate2; print(ctranslate2.__version__)"

如果 `nvidia-smi` 正常，但启动时报：

- `Could not load library libcudnn_ops.so...`
- `Could not load library libcublas.so...`

说明 CUDA/cuDNN 运行时缺失或版本不匹配。两种处理方式：

### 方式 A：使用系统级 CUDA/cuDNN

安装与 CTranslate2 匹配的 CUDA 12 运行时和 cuDNN，确保
`libcublas.so.12`、`libcudnn.so.9`（或对应版本）在 `ldconfig` 路径中。
NVIDIA 官方 apt 源或 CUDA Toolkit 安装包都可以。

### 方式 B：用 pip 提供运行时

    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

启动前设置动态库路径：

    export LD_LIBRARY_PATH="$(python - <<'PY'
    import os
    import nvidia.cublas
    import nvidia.cudnn
    print(":".join(os.path.join(os.path.dirname(m.__file__), "lib")
                   for m in (nvidia.cublas, nvidia.cudnn)))
    PY
    ):$LD_LIBRARY_PATH"

可以把它写进 `run.sh` 或 `~/.bashrc`。

### CPU 回退

如果暂时不想折腾 CUDA，把 `config.toml` 改成：

    [stt]
    model = "small.en"
    device = "cpu"
    compute_type = "int8"

## 5. 确认音频源

程序捕获的是默认输出 sink 的 monitor source。先检查：

    pactl info
    pactl get-default-sink
    pactl list sources short

正常情况下程序会自动拼出 `<sink>.monitor`。如果名称异常，在
`config.toml` 的 `[audio]` 里显式指定：

    [audio]
    source = "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

手动验证 monitor 是否有声音：

    parec --device="$(pactl get-default-sink).monitor" \
          --rate=16000 --channels=1 --format=s16le \
          --latency-msec=200 > /tmp/monitor.raw

播放一段声音，然后 Ctrl+C 停止。`/tmp/monitor.raw` 明显变大说明
捕获正常。也可以用 `pw-play` 或 `aplay` 回放测试：

    pw-play --format s16 --rate 16000 --channels 1 /tmp/monitor.raw

## 6. 配置 config.toml

项目根目录已经提供一份可用的 `config.toml`。关键项：

| 配置项 | 说明 |
| --- | --- |
| `[audio].source` | monitor source；留空自动推导 |
| `[stt].model` | `large-v3-turbo` / `small.en` / `base.en` |
| `[stt].device` | `cuda` 或 `cpu` |
| `[stt].compute_type` | GPU 用 `float16` 或 `int8_float16`，CPU 用 `int8` |
| `[vad].aggressiveness` | 0–3，越大越激进；听不清人声时可降到 1 |
| `[translate].base_url` | 必须带 `/v1`，程序会追加 `/chat/completions` |
| `[translate].model` | 必须与本地 LLM 服务 `/v1/models` 返回的 id 一致 |
| `[ui].screen` | 显示到第几个显示器，从 0 开始 |

## 7. 确认本地 LLM 服务

程序通过 `requests` 直接 POST 到 OpenAI 兼容接口，不使用 `openai` 包。
先确认服务在跑：

    curl -s http://127.0.0.1:8000/v1/models | python -m json.tool

再测试一次对话：

    curl -s http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": "把 hello 翻译成中文"}],
        "temperature": 0.2
      }'

如果报 model not found，把 `[translate].model` 改成 `/v1/models` 返回的 id。

常见启动方式示例：

    # llama.cpp
    llama-server -m /path/to/qwen2.5-7b-instruct-q4_k_m.gguf \
      --host 127.0.0.1 --port 8000 -c 4096 -ngl 99

    # vLLM
    vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000

## 8. 运行

在项目根目录：

    source .venv/bin/activate
    python -m subtitle_app config.toml

如果仓库里有 `run.py`，也可以：

    python run.py config.toml

首次运行会从 Hugging Face 下载 Whisper 模型到 `~/.cache/huggingface`，
需要网络和磁盘空间。

Wayland 下如果字幕不置顶或点击穿透无效，可以试 X11 平台插件：

    QT_QPA_PLATFORM=xcb python -m subtitle_app config.toml

## 9. 常见问题

| 现象 | 处理 |
| --- | --- |
| `FileNotFoundError: 配置文件不存在` | 在项目根目录创建 `config.toml`，或传入正确路径 |
| `未找到命令 parec/pactl` | `sudo apt install pulseaudio-utils` |
| `无法获取默认 sink` | 检查 `systemctl --user status pipewire pipewire-pulse` 和 `pactl info` |
| `AttributeError: ... last_text` | 使用本次修复后的 `pipeline.py` |
| `TypeError: push_subtitle() ...` | 使用本次修复后的 `app.py` |
| CUDA/cuDNN 加载失败 | 按第 4 节安装匹配版本，或先切 CPU |
| 翻译请求失败/超时 | 检查 LLM 服务端口、`base_url`、`model`，必要时加大 `timeout` |
| 一直没有字幕 | 确认系统确实在播放声音；把 `[vad].aggressiveness` 降到 1；看终端日志 |
| 字幕不置顶/点击穿透 | Wayland compositor 限制，试 `QT_QPA_PLATFORM=xcb` 或 X11 会话 |
| Qt 报 xcb 插件错误 | 安装第 2 节的 `libxcb-*`，或切换 `QT_QPA_PLATFORM` |
| 显存不足 | `compute_type = "int8_float16"`，或换 `small.en` |

## 10. 一次性命令清单

    # 1) 系统依赖
    sudo apt update
    sudo apt install -y \
      python3 python3-venv python3-pip \
      pipewire-pulse pulseaudio-utils \
      libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 \
      libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
      libxcb-shape0 libxcb-xinerama0 \
      fonts-noto-cjk

    # 2) Python 依赖
    cd /path/to/project
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install -r requirements.txt

    # 3) 确认音频
    pactl get-default-sink

    # 4) 确认 LLM
    curl -s http://127.0.0.1:8000/v1/models

    # 5) 运行
    python -m subtitle_app config.toml
