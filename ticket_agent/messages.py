"""内部消息结构。

Harness 内部不直接使用 OpenAI 的 dict 消息，而是使用强类型消息，便于
工具调用与工具结果通过 ``tool_call_id`` 精确关联。真正请求模型时再由
``to_openai_messages`` 转换成 OpenAI Chat Completions 所需的格式。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, Union


@dataclass(frozen=True)
class ToolCall:
    """模型发起的一次工具调用。"""

    tool_call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass(frozen=True)
class ToolResult:
    """ToolRuntime 统一产出的结构化工具结果。"""

    tool_call_id: str
    name: str
    success: bool
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float = 0.0

    @classmethod
    def ok(
        cls,
        call: ToolCall,
        data: Any,
        duration_ms: float = 0.0,
    ) -> "ToolResult":
        return cls(
            tool_call_id=call.tool_call_id,
            name=call.name,
            success=True,
            data=data,
            duration_ms=duration_ms,
        )

    @classmethod
    def fail(
        cls,
        call: ToolCall,
        error_code: str,
        error_message: str,
        duration_ms: float = 0.0,
    ) -> "ToolResult":
        return cls(
            tool_call_id=call.tool_call_id,
            name=call.name,
            success=False,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "success": self.success,
            "data": self.data,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "duration_ms": round(self.duration_ms, 2),
        }

    def to_context(self) -> str:
        """序列化为回填给模型的上下文文本。

        ``tool_call_id`` 始终保留，模型据此把结果与上一轮工具调用对应起来。
        """
        body = json.dumps(self.to_dict(), ensure_ascii=False, default=str)
        return (
            f'<tool_result tool_call_id="{self.tool_call_id}" '
            f'name="{self.name}" success="{str(self.success).lower()}">\n'
            f"{body}\n</tool_result>"
        )


@dataclass(frozen=True)
class TextBlock:
    """助手消息中的文本块。"""

    text: str


@dataclass(frozen=True)
class ToolCallBlock:
    """助手消息中的工具调用块。"""

    tool_call: ToolCall


MessageBlock = Union[TextBlock, ToolCallBlock]


@dataclass(frozen=True)
class UserMessage:
    content: str
    role: Literal["user"] = field(default="user", init=False)

    def to_openai(self) -> dict[str, Any]:
        return {"role": "user", "content": self.content}


@dataclass
class AssistantMessage:
    """助手消息：原始文本 + 结构化块（文本块 / toolCall 块）。"""

    raw_text: str
    blocks: list[MessageBlock] = field(default_factory=list)
    role: Literal["assistant"] = field(default="assistant", init=False)

    @property
    def text(self) -> str:
        """只拼接文本块，不包含工具调用 JSON。"""
        return "".join(block.text for block in self.blocks if isinstance(block, TextBlock)).strip()

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [
            block.tool_call
            for block in self.blocks
            if isinstance(block, ToolCallBlock)
        ]

    def to_openai(self) -> dict[str, Any]:
        # 保留原始输出，模型下一轮能看到自己上一轮发出的工具调用 JSON。
        return {"role": "assistant", "content": self.raw_text}


@dataclass(frozen=True)
class ToolResultMessage:
    """工具结果消息。"""

    result: ToolResult
    role: Literal["tool"] = field(default="tool", init=False)

    def to_openai(self) -> dict[str, Any]:
        # 本项目使用文本协议而非原生 function calling，因此回填为带标签的
        # user 消息；tool_call_id 保存在 <tool_result> 标签中，不会丢失。
        return {"role": "user", "content": self.result.to_context()}


ChatMessage = Union[UserMessage, AssistantMessage, ToolResultMessage]


def to_openai_messages(
    system_prompt: str,
    messages: Sequence[ChatMessage],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    converted.extend(message.to_openai() for message in messages)
    return converted
