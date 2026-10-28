"""LLMGateway：以 OpenAI SDK 兼容方式调用 DeepSeek，强制流式输出。"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from openai import AsyncOpenAI, OpenAIError

from .config import Settings


class GatewayError(RuntimeError):
    """LLM 网关调用失败。"""


class LLMGateway:
    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings = settings
        if client is None:
            if not settings.api_key:
                raise GatewayError(
                    "未配置 DEEPSEEK_API_KEY，请先设置环境变量后再运行。"
                )
            client = AsyncOpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=settings.request_timeout,
                max_retries=1,
            )
        self._client = client

    async def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        """流式调用模型，返回完整文本。"""
        chunks: list[str] = []
        try:
            stream = await self._client.chat.completions.create(
                model=self.settings.model,
                messages=list(messages),
                stream=True,
                temperature=self.settings.temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
            async for event in stream:
                if not event.choices:
                    continue
                content = event.choices[0].delta.content
                if content:
                    chunks.append(content)
                    if on_delta is not None:
                        on_delta(content)
        except OpenAIError as exc:
            raise GatewayError(f"DeepSeek API 调用失败: {exc}") from exc

        return "".join(chunks)

    async def close(self) -> None:
        await self._client.close()
