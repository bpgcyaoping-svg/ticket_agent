"""AgentContext：一次 Agent 运行所需的系统提示词、工具与会话消息。

对应 pi 的 ``AgentContext``（systemPrompt + messages + tools）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .agent_tool import AgentTool
from .messages import ChatMessage, to_openai_messages


@dataclass
class AgentContext:
    system_prompt: str
    tools: Sequence[AgentTool] = ()
    messages: list[ChatMessage] = field(default_factory=list)

    def get_tool(self, name: str) -> AgentTool | None:
        return next((tool for tool in self.tools if tool.name == name), None)

    def llm_messages(self) -> list[dict[str, Any]]:
        return to_openai_messages(self.system_prompt, self.messages)
