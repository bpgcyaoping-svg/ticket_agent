from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Sequence


class ScriptedGateway:
    """按脚本返回模型文本的假网关，用于确定性集成测试。"""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.settings = SimpleNamespace(
            model="scripted-deepseek",
            base_url="http://scripted.local/v1",
            temperature=0.0,
        )

    async def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("脚本响应已用尽，Agent 仍在请求模型")
        text = self._responses.pop(0)
        if on_delta is not None:
            for index in range(0, len(text), 7):
                on_delta(text[index : index + 7])
        return text


def tool_context_messages(messages: Sequence[dict[str, Any]]) -> list[str]:
    return [
        str(message["content"])
        for message in messages
        if message.get("role") == "user"
        and str(message.get("content", "")).startswith("<tool_result")
    ]
