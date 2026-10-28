"""可观测性事件。

流式正文写入 stdout；verbose 模式下事件写入 stderr，避免与正文交织。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass(frozen=True)
class AgentEvent:
    """Agent Loop 内部事件（对应 pi 的 AgentEvent）。"""

    type: str
    turn: int
    data: dict[str, Any] = field(default_factory=dict)


def _format_value(value: Any, limit: int = 100) -> str:
    text = str(value).replace("\n", " ")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return repr(text) if " " in text else text


@dataclass
class EventLogger:
    verbose: bool = False
    stream: TextIO = sys.stderr
    records: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: str, **fields: Any) -> None:
        self.records.append({"event": event, "ts": time.time(), **fields})
        if not self.verbose:
            return
        detail = " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
        suffix = f" {detail}" if detail else ""
        print(f"[{event}]{suffix}", file=self.stream, flush=True)

    def names(self) -> list[str]:
        return [record["event"] for record in self.records]

    def clear(self) -> None:
        self.records.clear()
