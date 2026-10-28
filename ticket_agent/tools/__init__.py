"""业务工具注册入口与 AgentTool 适配。

- ``build_runtime``：注册 6 个业务工具到 ToolRuntime；
- ``build_agent_tools``：把 ToolSpec 适配成 pi 风格 AgentTool（对应 pi-tools.ts）。
"""

from __future__ import annotations

from typing import Sequence

from ..agent_tool import AgentTool
from ..rag import KnowledgeBase
from ..runtime import ToolRuntime
from . import complaint, fault, intent, knowledge, order, refund

TOOL_NAMES = [
    "classify_intent",
    "query_order",
    "query_refund",
    "search_knowledge_base",
    "report_system_fault",
    "file_complaint",
]


def build_runtime(knowledge_base: KnowledgeBase) -> ToolRuntime:
    runtime = ToolRuntime()
    specs = [
        intent.SPEC,
        order.SPEC,
        refund.SPEC,
        knowledge.build_spec(knowledge_base),
        fault.SPEC,
        complaint.SPEC,
    ]
    for spec in specs:
        runtime.register(spec)
    return runtime


def build_agent_tools(runtime: ToolRuntime) -> Sequence[AgentTool]:
    """模型侧工具视图：只有 schema，执行统一委托给 Runtime.invoke。"""
    return [
        AgentTool(
            name=spec.name,
            label=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            runtime=runtime,
        )
        for spec in runtime.specs()
    ]


__all__ = ["build_runtime", "build_agent_tools", "TOOL_NAMES"]
