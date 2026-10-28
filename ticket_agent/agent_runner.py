"""装配层：把 Model、AgentTool、Runtime、LoopGuard 接入 Agent Loop。

对应 pi 的 ``agent-runner.ts``。本层只做组装与事件观察，不实现循环规则，
也不直接执行任何业务 handler。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from .agent_context import AgentContext
from .agent_loop import AgentLoopConfig, AgentRunResult, run_agent_loop
from .agent_tool import AgentTool
from .events import AgentEvent, EventLogger
from .gateway import LLMGateway
from .loop_guard import LoopGuard
from .messages import UserMessage
from .model import create_model, create_stream_fn
from .protocol import build_system_prompt
from .runtime import ToolRuntime
from .tools import build_agent_tools


@dataclass
class AgentRunner:
    gateway: LLMGateway
    runtime: ToolRuntime
    events: EventLogger | None = None
    system_prompt: str | None = None
    max_steps: int = 8
    output: TextIO = sys.stdout

    def __post_init__(self) -> None:
        if self.events is None:
            self.events = EventLogger()
        self.model = create_model(self.gateway.settings)
        self.stream_fn = create_stream_fn(self.gateway)
        self.tools: list[AgentTool] = build_agent_tools(self.runtime)
        if self.system_prompt is None:
            self.system_prompt = build_system_prompt(self.tools)

    def _write(self, delta: str) -> None:
        self.output.write(delta)
        self.output.flush()

    def _observe(self, event: AgentEvent) -> None:
        events = self.events
        data = event.data
        if event.type == "llm_request":
            events.emit(
                "llm_request",
                step=event.turn,
                messages=data.get("messages", 0),
            )
        elif event.type == "assistant_stream":
            events.emit("assistant_stream", step=event.turn)
        elif event.type == "message_end":
            events.emit(
                "message_end",
                step=event.turn,
                chars=data.get("chars", 0),
                tool_calls=data.get("tool_calls", 0),
                parse_error=data.get("parse_error", ""),
            )
        elif event.type == "tool_execution_start":
            events.emit(
                "tool_start",
                name=data.get("tool_name"),
                tool_call_id=data.get("tool_call_id"),
                arguments=data.get("arguments"),
            )
        elif event.type == "tool_execution_end":
            events.emit(
                "tool_end",
                name=data.get("tool_name"),
                tool_call_id=data.get("tool_call_id"),
                success=data.get("success"),
                error_code=data.get("error_code", ""),
                duration_ms=data.get("duration_ms", 0),
            )

    async def run(self, user_input: str) -> AgentRunResult:
        context = AgentContext(
            system_prompt=self.system_prompt or "",
            tools=self.tools,
        )
        guard = LoopGuard(self.max_steps)
        config = AgentLoopConfig(
            model=self.model.id,
            stream_fn=self.stream_fn,
            max_steps=self.max_steps,
            tool_execution="sequential",
            before_tool_call=guard.before_tool_call,
            after_tool_call=guard.after_tool_call,
            prepare_next_turn=guard.prepare_next_turn,
            should_stop_after_turn=guard.should_stop_after_turn,
            get_follow_up_messages=guard.drain_follow_ups,
            on_text=self._write,
        )

        self.events.emit("agent_start", input=user_input, model=self.model.id)
        result = await run_agent_loop(
            [UserMessage(content=user_input)],
            context,
            config,
            on_event=self._observe,
        )
        self.events.emit(
            "agent_end",
            steps=result.steps,
            stop_reason=result.stop_reason,
            tool_calls=len(result.tool_results),
            duration_ms=round(result.duration_ms, 2),
        )
        return result
