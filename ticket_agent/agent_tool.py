"""AgentTool：pi ``AgentTool`` 的 Python 适配层。

模型只看到 ``name / description / parameters``；真正的执行统一走
``ToolRuntime.invoke()``。这一层不承载业务逻辑，只负责把 ToolSpec
适配成 Agent Loop 可调用的工具对象。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .runtime import ManagedToolResult, ToolInvocation, ToolRuntime


@dataclass(frozen=True)
class AgentTool:
    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    runtime: ToolRuntime

    def schema_view(self) -> dict[str, Any]:
        """模型可见视图：不包含 runtime、handler 或任何可执行引用。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
    ) -> ManagedToolResult:
        return await self.runtime.invoke(
            ToolInvocation(
                tool_call_id=tool_call_id,
                model_name=self.name,
                args=dict(params),
                signal=signal,
            )
        )


def describe_tools(tools: Sequence[AgentTool]) -> list[dict[str, Any]]:
    return [tool.schema_view() for tool in tools]


def describe_tools_json(tools: Sequence[AgentTool]) -> str:
    return json.dumps(describe_tools(tools), ensure_ascii=False, indent=2)
