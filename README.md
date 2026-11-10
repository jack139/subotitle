# Linux桌面自动音频字幕

## 功能

抓取当前系统的音频输出，识别并翻译为中文，然后在屏幕生成字幕。主要方便观看没有字幕的英文视频。目前支持英译中，翻译其他语言需要使用相应的ASR模型。

使用的模型：
- faster-whisper `small.en`
- Qwen-2B-Q5_K_M.gguf


## 运行环境：

- debian 12 `bookworm`
- X11、 Qt6
- CUDA，显存至少6G（faster-whisper使用 800M，Qwen-2B 使用 4G）


## 运行

```bash
# 需先启动 LLM 进程

source .venv/bin/activate
python3.12 -m subtitle_app config.toml
```

## 说明

- 这是我的第一个完整用模型生成的代码
- 使用 `paulgauthier/aider + deepseek-flash`
- 我只参与：模型选型、环境搭建、配置参数调整等。代码开发都是 aider 在做
- 问题：过程中也有反复，一开始生成的版本中，模型想要使用自己的vad，但调不通。期间我换了pro的模型，改了几次代码也是调不通。我让它先停用vad，直接做ASR，这次调通了。然后再加vad，它还是想自己做的vad，我提示它不要自己写VAD，用 `faster-whisper` 自带的vad，然后就调通了。小结一下就是：还是需要有人帮模型做决定，否则它可能会一直钻牛角尖。
- 总花费：¥3.39 （`deepseek-flash`）
