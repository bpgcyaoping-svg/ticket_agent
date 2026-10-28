"""模型接入层：声明逻辑模型，并把 Provider 的流式调用注入 Agent Loop。

对应 pi 的 ``model.ts``。本层不含任何 Agent 业务逻辑（无工具、无提示词、
无循环控制），只负责：
1. 声明一个 OpenAI-completions 兼容的逻辑模型；
2. 把 ``LLMGateway.stream_chat`` 适配成 Agent Loop 需要的 ``stream_fn``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from .config import Settings
from .gateway import LLMGateway

PROVIDER_ID = "deepseek"
API = "openai-completions"


@dataclass(frozen=True)
class ModelSpec:
    """逻辑模型声明（对应 pi 的 ``Model`` 描述）。"""

    id: str
    provider: str
    api: str
    base_url: str
    context_window: int = 128_000
    max_tokens: int = 4096
    temperature: float = 0.0
    reasoning: bool = False


StreamFn = Callable[
    [Sequence[dict[str, Any]], Callable[[str], None] | None],
    Awaitable[str],
]


def create_model(settings: Settings) -> ModelSpec:
    return ModelSpec(
        id=settings.model,
        provider=PROVIDER_ID,
        api=API,
        base_url=settings.base_url,
        temperature=settings.temperature,
    )


def create_stream_fn(gateway: LLMGateway) -> StreamFn:
    """把网关的流式能力包装成 Loop 可注入的 ``stream_fn``。"""

    async def stream(
        messages: Sequence[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        return await gateway.stream_chat(messages, on_delta=on_delta)

    return stream
