"""ToolRuntime：工具注册、参数校验、统一执行与结构化结果。

对应 pi 的 ``runtime.ts``：所有业务工具都通过 ``invoke()`` 进入，返回
``ManagedToolResult``（ok / code / model_view）。模型只能看到
``ToolSpec.schema_view()``（名称 / 描述 / 参数 schema），handler 永远不会
被序列化进提示词，也不能被模型直接调用。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .messages import ToolCall, ToolResult

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolError(Exception):
    """业务 handler 主动抛出的可预期错误。"""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    timeout_seconds: float = 15.0

    def schema_view(self) -> dict[str, Any]:
        """只暴露给模型的字段，不包含 handler。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class ToolInvocation:
    """一次工具调用请求，``tool_call_id`` 是调用与结果的连接键。"""

    tool_call_id: str
    model_name: str
    args: dict[str, Any] = field(default_factory=dict)
    signal: Any = None


@dataclass(frozen=True)
class ManagedToolResult:
    """Runtime 的统一产物：结构化结果 + 模型可见视图。"""

    result: ToolResult

    @property
    def ok(self) -> bool:
        return self.result.success

    @property
    def success(self) -> bool:
        return self.result.success

    @property
    def code(self) -> str | None:
        return self.result.error_code

    @property
    def model_view(self) -> dict[str, Any]:
        return self.result.to_dict()

    # 便捷访问，避免调用方反复拆 result。
    @property
    def tool_call_id(self) -> str:
        return self.result.tool_call_id

    @property
    def name(self) -> str:
        return self.result.name

    @property
    def data(self) -> Any:
        return self.result.data

    @property
    def error_code(self) -> str | None:
        return self.result.error_code

    @property
    def error_message(self) -> str | None:
        return self.result.error_message

    @property
    def duration_ms(self) -> float:
        return self.result.duration_ms


def _type_matches(expected: str, value: Any) -> bool:
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, _JSON_TYPES[expected])


def validate_arguments(schema: dict[str, Any], arguments: Any) -> list[str]:
    """最小可用的 JSON Schema 校验：required / type / enum / 未知字段。"""
    if not isinstance(arguments, dict):
        return ["arguments 必须是 JSON 对象"]

    properties = schema.get("properties", {})
    errors: list[str] = []

    for key in schema.get("required", []):
        if key not in arguments:
            errors.append(f"缺少必填参数: {key}")

    if schema.get("additionalProperties") is False:
        for key in arguments:
            if key not in properties:
                errors.append(f"未知参数: {key}")

    for key, value in arguments.items():
        spec = properties.get(key)
        if not spec:
            continue
        expected = spec.get("type")
        if expected in _JSON_TYPES and not _type_matches(expected, value):
            errors.append(
                f"参数 {key} 类型应为 {expected}，实际为 {type(value).__name__}"
            )
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"参数 {key} 取值必须是 {spec['enum']} 之一")

    return errors


@dataclass
class ToolRuntime:
    """所有工具调用的唯一入口。"""

    default_timeout_seconds: float = 15.0
    _specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"工具重复注册: {spec.name}")
        if not spec.name or not callable(spec.handler):
            raise ValueError("ToolSpec 缺少 name 或 handler")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def tool_names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in self.tool_names()]

    def describe_tools(self) -> list[dict[str, Any]]:
        return [spec.schema_view() for spec in self.specs()]

    def describe_tools_json(self) -> str:
        return json.dumps(self.describe_tools(), ensure_ascii=False, indent=2)

    async def invoke(self, invocation: ToolInvocation) -> ManagedToolResult:
        """唯一执行入口：校验参数、执行 handler、封装结构化结果。"""
        started = time.perf_counter()

        def elapsed_ms() -> float:
            return (time.perf_counter() - started) * 1000

        call = ToolCall(
            tool_call_id=invocation.tool_call_id,
            name=invocation.model_name,
            arguments=invocation.args if isinstance(invocation.args, dict) else {},
        )

        spec = self._specs.get(invocation.model_name)
        if spec is None:
            return ManagedToolResult(
                ToolResult.fail(
                    call,
                    "TOOL_NOT_FOUND",
                    f"未注册的工具: {invocation.model_name}",
                    elapsed_ms(),
                )
            )

        errors = validate_arguments(spec.parameters, invocation.args)
        if errors:
            return ManagedToolResult(
                ToolResult.fail(
                    call,
                    "INVALID_ARGUMENTS",
                    "; ".join(errors),
                    elapsed_ms(),
                )
            )

        timeout = spec.timeout_seconds or self.default_timeout_seconds
        try:
            if inspect.iscoroutinefunction(spec.handler):
                data = await asyncio.wait_for(
                    spec.handler(**invocation.args),
                    timeout=timeout,
                )
            else:
                data = await asyncio.wait_for(
                    asyncio.to_thread(spec.handler, **invocation.args),
                    timeout=timeout,
                )
        except ToolError as exc:
            return ManagedToolResult(
                ToolResult.fail(call, exc.error_code, exc.message, elapsed_ms())
            )
        except (asyncio.TimeoutError, TimeoutError):
            return ManagedToolResult(
                ToolResult.fail(
                    call,
                    "TOOL_TIMEOUT",
                    f"工具 {invocation.model_name} 执行超时（>{timeout:.0f}s）",
                    elapsed_ms(),
                )
            )
        except Exception as exc:  # 隔离 handler 异常，避免打断 Agent Loop
            return ManagedToolResult(
                ToolResult.fail(
                    call,
                    "TOOL_EXECUTION_FAILED",
                    f"{type(exc).__name__}: {exc}",
                    elapsed_ms(),
                )
            )

        try:
            json.dumps(data, ensure_ascii=False)
        except TypeError as exc:
            return ManagedToolResult(
                ToolResult.fail(
                    call,
                    "TOOL_RESULT_NOT_SERIALIZABLE",
                    f"工具 {invocation.model_name} 返回值无法 JSON 序列化: {exc}",
                    elapsed_ms(),
                )
            )

        return ManagedToolResult(ToolResult.ok(call, data, elapsed_ms()))
