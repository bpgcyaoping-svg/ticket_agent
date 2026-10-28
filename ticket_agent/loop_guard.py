"""LoopGuard：挂在 Agent Loop Hook 上的循环保护与协议纠偏。

对应 pi 的 ``loop-guard.ts``：
- ``before_tool_call``：未知工具拦截、相同工具调用重复 3 次拦截；
- ``after_tool_call``：预留观测点（本项目的业务不需要额外处理）；
- ``should_stop_after_turn``：协议纠偏一次、无工具调用即结束、最大轮次保护；
- ``get_follow_up_messages``：把纠偏提示在下一轮前回填给模型。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from .agent_context import AgentContext
from .agent_loop import (
    BeforeToolCallResult,
    StopDecision,
    ToolCallContext,
    ToolResultContext,
    TurnContext,
)
from .messages import ChatMessage, UserMessage


class StopReason:
    ANSWER = "answer"
    MAX_STEPS = "max_steps"
    PROTOCOL_ERROR = "protocol_error"
    REPEATED_ACTION = "repeated_action"


@dataclass
class LoopState:
    max_steps: int
    protocol_correction_used: bool = False
    stop_reason: str | None = None
    tool_fingerprints: Counter[str] = field(default_factory=Counter)


class LoopGuard:
    def __init__(self, max_steps: int) -> None:
        self.state = LoopState(max_steps=max_steps)
        self._follow_ups: list[ChatMessage] = []

    async def before_tool_call(
        self,
        context: ToolCallContext,
    ) -> BeforeToolCallResult | None:
        if context.tool is None:
            return BeforeToolCallResult(
                block=True,
                code="TOOL_NOT_FOUND",
                reason=f"未注册的工具: {context.tool_call.name}",
            )

        fingerprint = json.dumps(
            {
                "name": context.tool_call.name,
                "arguments": context.tool_call.arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        self.state.tool_fingerprints[fingerprint] += 1
        if self.state.tool_fingerprints[fingerprint] >= 3:
            self.state.stop_reason = StopReason.REPEATED_ACTION
            return BeforeToolCallResult(
                block=True,
                code="REPEATED_ACTION",
                reason="相同工具调用已重复 3 次，为避免死循环已阻止执行",
            )
        return None

    async def after_tool_call(self, context: ToolResultContext) -> None:
        return None

    async def prepare_next_turn(self, context: TurnContext) -> AgentContext:
        return context.context

    async def should_stop_after_turn(self, context: TurnContext) -> StopDecision:
        if self.state.stop_reason == StopReason.REPEATED_ACTION:
            return StopDecision(True, StopReason.REPEATED_ACTION)

        if context.parse_error and not context.assistant_message.tool_calls:
            if not self.state.protocol_correction_used:
                self.state.protocol_correction_used = True
                self._follow_ups.append(
                    UserMessage(
                        content=(
                            f"上一条输出不是合法的工具调用：{context.parse_error}。\n"
                            '请二选一：1) 输出一个合法 JSON 对象 '
                            '{"type":"tool_call","name":"...","arguments":{...}}；'
                            "2) 如果不需要工具，直接用中文给出最终答复。"
                        )
                    )
                )
                return StopDecision(False)
            self.state.stop_reason = StopReason.PROTOCOL_ERROR
            return StopDecision(True, StopReason.PROTOCOL_ERROR)

        if not context.tool_results and not context.assistant_message.tool_calls:
            self.state.stop_reason = StopReason.ANSWER
            return StopDecision(True, StopReason.ANSWER)

        if context.turn >= self.state.max_steps:
            self.state.stop_reason = StopReason.MAX_STEPS
            return StopDecision(True, StopReason.MAX_STEPS)

        return StopDecision(False)

    async def drain_follow_ups(self) -> list[ChatMessage]:
        messages = list(self._follow_ups)
        self._follow_ups.clear()
        return messages
