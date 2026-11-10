"""通过 OpenAI 兼容的 HTTP 接口调用本地 LLM 做英→中翻译（使用 requests）。"""

from __future__ import annotations

import logging

import requests

from .config import TranslateConfig

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一个实时字幕翻译引擎。把用户给出的英文翻译成简洁、自然、口语化的简体中文。"
    "只输出译文本身，不要输出解释、引号、原文或任何多余内容。"
)


class Translator:
    def __init__(self, cfg: TranslateConfig):
        self.cfg = cfg
        self._url = cfg.base_url.rstrip("/") + "/chat/completions"
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            }
        )

    @staticmethod
    def _user_content(text: str, context: str | None) -> str:
        if context:
            return f"上一句：{context}\n当前句：{text}\n请只翻译当前句。"
        return text

    def translate(self, text: str, context: str | None = None) -> str | None:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._user_content(text, context)},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }

        try:
            resp = self._session.post(
                self._url, json=payload, timeout=self.cfg.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            log.error("翻译请求失败: %s", e)
            return None
        except (KeyError, IndexError, ValueError) as e:
            log.error("翻译响应格式异常: %s", e)
            return None

        translated = (content or "").strip().strip("“”\"'").strip()
        return translated or None

    def close(self) -> None:
        self._session.close()
