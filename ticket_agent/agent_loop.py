"""Agent Loop 引擎：pi ``runAgentLoop`` 的 Python 等价实现。

本层只负责推进循环与触发 Hook，不含工单业务逻辑：
    turn_start → 流式调用模型 → 解析 ToolCall → before_tool_call
    → AgentTool.execute → after_tool_call → 回填 ToolResultMessage
    → should_stop_after_turn / get_follow_up_messages → 下一轮

停止原因、协议纠正与重复调用保护由 ``LoopGuard`` 通过 Hook 提供；
模型与 Provider 由 ``model.py`` 注入；工具由 ``AgentTool`` 提供。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Sequence

from .agent_context import AgentContext
from .agent_tool import AgentTool
from .events import AgentEvent
from .messages import (
    AssistantMessage,
    ChatMessage,
    MessageBlock,
    TextBlock,
    ToolCall,
    ToolCallBlock,
    ToolResult,
    ToolResultMessage,
)
from .model import StreamFn
from .protocol import parse_model_output


@dataclass(frozen=True)
class BeforeToolCallResult:
    block: bool = False
    code: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str | None = None


@dataclass(frozen=True)
class ToolCallContext:
    turn: int
    tool_call: ToolCall
    tool: AgentTool | None
    context: AgentContext
    signal: Any = None


@dataclass(frozen=True)
class ToolResultContext:
    turn: int
    tool_call: ToolCall
    tool: AgentTool | None
    context: AgentContext
    result: ToolResult
    signal: Any = None


@dataclass(frozen=True)
class TurnContext:
    turn: int
    context: AgentContext
    assistant_message: AssistantMessage
    tool_results: list[ToolResult]
    parse_error: str | None


@dataclass
class AgentLoopConfig:
    model: str
    stream_fn: StreamFn
    max_steps: int = 8
    tool_execution: Literal["sequential"] = "sequential"
    before_tool_call: Callable[[ToolCallContext], Awaitable[BeforeToolCallResult | None]] | None = None
    after_tool_call: Callable[[ToolResultContext], Awaitable[None]] | None = None
    prepare_next_turn: Callable[[TurnContext], Awaitable[AgentContext]] | None = None
    should_stop_after_turn: Callable[[TurnContext], Awaitable[StopDecision]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[Sequence[ChatMessage]]] | None = None
    on_text: Callable[[str], None] | None = None


@dataclass
class AgentRunResult:
    final_answer: str
    steps: int
    stop_reason: str
    tool_results: list[ToolResult]
    duration_ms: float
    messages: list[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
            "tool_results": [result.to_dict() for result in self.tool_results],
            "duration_ms": round(self.duration_ms, 2),
        }


def _fallback_answer(stop_reason: str, max_steps: int, message: AssistantMessage | None) -> str:
    if stop_reason in {"answer", "protocol_error"} and message is not None:
        return message.text or message.raw_text.strip()
    if stop_reason == "max_steps":
        return (
            f"已达到最大推理轮次（{max_steps}），为避免无限循环已停止。"
            "请补充信息或联系人工客服。"
        )
    if stop_reason == "repeated_action":
        return (
            "检测到重复的工具调用，为避免死循环已停止。"
            "请补充信息或联系人工客服。"
        )
    return message.text if message is not None else ""


async def run_agent_loop(
    prompts: Sequence[ChatMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    on_event: Callable[[AgentEvent], None] | None = None,
    signal: Any = None,
) -> AgentRunResult:
    context.messages.extend(prompts)
    started = time.perf_counter()
    tool_results: list[ToolResult] = []
    steps = 0
    stop_reason = "max_steps"
    last_assistant: AssistantMessage | None = None

    def emit(event_type: str, **data: Any) -> None:
        if on_event is not None:
            on_event(AgentEvent(type=event_type, turn=steps, data=data))

    try:
        for turn in range(1, config.max_steps + 1):
            steps = turn
            emit("turn_start", step=turn)

            messages = context.llm_messages()
            emit("llm_request", step=turn, messages=len(messages))
            emit("assistant_stream", step=turn)

            def on_delta(delta: str) -> None:
                emit("message_update", delta=delta)
                if config.on_text is not None:
                    config.on_text(delta)

            text = await config.stream_fn(messages, on_delta)
            if text and not text.endswith("\n") and config.on_text is not None:
                config.on_text("\n")

            parsed = parse_model_output(text)
            blocks: list[MessageBlock] = []
            if parsed.prose:
                blocks.append(TextBlock(text=parsed.prose))
            blocks.extend(ToolCallBlock(tool_call=call) for call in parsed.tool_calls)

            assistant = AssistantMessage(raw_text=text, blocks=blocks)
            context.messages.append(assistant)
            last_assistant = assistant
            emit(
                "message_end",
                step=turn,
                chars=len(text),
                tool_calls=len(parsed.tool_calls),
                parse_error=parsed.error or "",
            )

            turn_results: list[ToolResult] = []
            for call in parsed.tool_calls:
                tool = context.get_tool(call.name)
                call_context = ToolCallContext(
                    turn=turn,
                    tool_call=call,
                    tool=tool,
                    context=context,
                    signal=signal,
                )
                before = None
                if config.before_tool_call is not None:
                    before = await config.before_tool_call(call_context)

                emit(
                    "tool_execution_start",
                    tool_name=call.name,
                    tool_call_id=call.tool_call_id,
                    arguments=call.arguments,
                )

                if before is not None and before.block:
                    result = ToolResult.fail(
                        call,
                        before.code or "TOOL_CALL_BLOCKED",
                        before.reason or "工具调用被拦截",
                    )
                elif tool is not None:
                    managed = await tool.execute(call.tool_call_id, call.arguments, signal)
                    result = managed.result
                else:
                    result = ToolResult.fail(
                        call,
                        "TOOL_NOT_FOUND",
                        f"未注册的工具: {call.name}",
                    )

                context.messages.append(ToolResultMessage(result))
                turn_results.append(result)
                tool_results.append(result)
                emit(
                    "tool_execution_end",
                    tool_name=result.name,
                    tool_call_id=result.tool_call_id,
                    success=result.success,
                    error_code=result.error_code or "",
                    duration_ms=round(result.duration_ms, 2),
                )

                if config.after_tool_call is not None:
                    await config.after_tool_call(
                        ToolResultContext(
                            turn=turn,
                            tool_call=call,
                            tool=tool,
                            context=context,
                            result=result,
                            signal=signal,
                        )
                    )

            turn_context = TurnContext(
                turn=turn,
                context=context,
                assistant_message=assistant,
                tool_results=turn_results,
                parse_error=parsed.error,
            )

            if config.should_stop_after_turn is not None:
                decision = await config.should_stop_after_turn(turn_context)
            elif not parsed.tool_calls:
                decision = StopDecision(
                    True,
                    "protocol_error" if parsed.error else "answer",
                )
            elif turn >= config.max_steps:
                decision = StopDecision(True, "max_steps")
            else:
                decision = StopDecision(False)

            if not decision.stop and turn >= config.max_steps:
                decision = StopDecision(True, "max_steps")

            if decision.stop:
                stop_reason = decision.reason or "answer"
                break

            if config.get_follow_up_messages is not None:
                follow_ups = await config.get_follow_up_messages()
                context.messages.extend(follow_ups)

            if config.prepare_next_turn is not None:
                context = await config.prepare_next_turn(turn_context)

            emit("turn_end", step=turn, tool_calls=len(turn_results))
    finally:
        duration_ms = (time.perf_counter() - started) * 1000

    return AgentRunResult(
        final_answer=_fallback_answer(stop_reason, config.max_steps, last_assistant),
        steps=steps,
        stop_reason=stop_reason,
        tool_results=tool_results,
        duration_ms=duration_ms,
        messages=context.messages,
    )
