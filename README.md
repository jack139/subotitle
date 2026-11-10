# Linux桌面自动音频字幕

## 运行环境：

- debian 12 bookworm
- X11、 Qt6
- CUDA，显存至少6G（含Qwen-2B-Q5KM）


## 运行

```bash
# 需先启动 LLM 进程

source .venv/bin/activate
python3.12 -m subtitle_app config.toml
```

## 说明

- 这是我的第一个完整用模型生成的代码
- 使用 paulgauthier/aider + deepseek-flash
- 我只参与：模型选型、环境搭建、配置参数调整等。代码开发都是 aider 在做
- 问题：过程中也有反复，一开始生成的版本模型要使用自己的vad，调不同。期间我换了pro的模型，改了几次代码也是调不通。我让它先停用vad，直接做ASR，这次调通了。然后再加vad，它还是向自己做vad，我提示它不要自己写VAD，用faster-whisper自带的vad，然后就调通了。小结一下就是：还是需要有人帮模型做决定，否则它可能会一直钻牛角尖。
- 总花费：¥3.39 （deepseek-flash）