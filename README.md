# Linux桌面自动音频字幕

运行环境：

- debian 12 bookworm
- X11、 Qt6
- CUDA，显存至少6G（含Qwen-2B-Q5KM）


```bash
# 需先启动 LLM 进程

source .venv/bin/activate
python3.12 -m subtitle_app config.toml
```
