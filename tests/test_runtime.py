from __future__ import annotations

import asyncio

import pytest

from ticket_agent.runtime import (
    ToolError,
    ToolInvocation,
    ToolRuntime,
    ToolSpec,
    validate_arguments,
)
from ticket_agent.tools import TOOL_NAMES, build_agent_tools


def _call(name: str, **arguments: object) -> ToolInvocation:
    return ToolInvocation("call_test", name, dict(arguments))


def test_registry_exposes_schema_only(runtime) -> None:
    assert runtime.tool_names() == sorted(TOOL_NAMES)

    catalog = runtime.describe_tools()
    assert len(catalog) == len(TOOL_NAMES)
    for item in catalog:
        assert set(item) == {"name", "description", "parameters"}

    rendered = runtime.describe_tools_json()
    assert "handler" not in rendered
    assert "function" not in rendered


def test_agent_tools_expose_schema_only(runtime) -> None:
    tools = build_agent_tools(runtime)

    assert [tool.name for tool in tools] == sorted(TOOL_NAMES)
    for tool in tools:
        assert set(tool.schema_view()) == {"name", "description", "parameters"}
        assert not hasattr(tool, "handler")


async def test_successful_execution(runtime) -> None:
    result = await runtime.invoke(
        _call("query_order", order_id="ORD-20261001")
    )

    assert result.success is True
    assert result.error_code is None
    assert result.tool_call_id == "call_test"
    assert result.data["order"]["status"] == "运输中"
    assert result.duration_ms >= 0


async def test_unknown_tool(runtime) -> None:
    result = await runtime.invoke(_call("drop_database"))

    assert result.success is False
    assert result.error_code == "TOOL_NOT_FOUND"


async def test_missing_required_argument(runtime) -> None:
    result = await runtime.invoke(_call("query_order"))

    assert result.error_code == "INVALID_ARGUMENTS"
    assert "order_id" in (result.error_message or "")


async def test_wrong_argument_type(runtime) -> None:
    result = await runtime.invoke(_call("query_order", order_id=123))

    assert result.error_code == "INVALID_ARGUMENTS"
    assert "类型" in (result.error_message or "")


async def test_enum_validation(runtime) -> None:
    result = await runtime.invoke(
        _call(
            "report_system_fault",
            title="登录失败",
            description="持续 500",
            severity="urgent",
        )
    )

    assert result.error_code == "INVALID_ARGUMENTS"


async def test_business_error_is_structured(runtime) -> None:
    result = await runtime.invoke(_call("query_order", order_id="ORD-NOT-EXIST"))

    assert result.success is False
    assert result.error_code == "ORDER_NOT_FOUND"
    assert "ORD-NOT-EXIST" in (result.error_message or "")


async def test_both_refund_arguments_missing_is_business_error(runtime) -> None:
    result = await runtime.invoke(_call("query_refund"))

    assert result.error_code == "INVALID_ARGUMENTS"


def test_validate_arguments_rejects_unknown_fields() -> None:
    schema = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    }

    assert validate_arguments(schema, {"order_id": "ORD-1", "extra": 1})


async def test_sync_handler_and_business_error() -> None:
    runtime = ToolRuntime()

    def sync_handler(value: int) -> dict[str, int]:
        if value < 0:
            raise ToolError("NEGATIVE", "不允许负数")
        return {"value": value}

    runtime.register(
        ToolSpec(
            name="sync",
            description="sync",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            handler=sync_handler,
        )
    )

    ok = await runtime.invoke(_call("sync", value=3))
    assert ok.success is True and ok.data == {"value": 3}

    bad = await runtime.invoke(_call("sync", value=-1))
    assert bad.error_code == "NEGATIVE"


async def test_timeout() -> None:
    runtime = ToolRuntime()

    async def slow() -> dict[str, bool]:
        await asyncio.sleep(0.2)
        return {"done": True}

    runtime.register(
        ToolSpec(
            name="slow",
            description="slow",
            parameters={"type": "object", "properties": {}},
            handler=slow,
            timeout_seconds=0.05,
        )
    )

    result = await runtime.invoke(_call("slow"))
    assert result.error_code == "TOOL_TIMEOUT"


async def test_unexpected_exception_is_isolated() -> None:
    runtime = ToolRuntime()

    def boom() -> dict[str, bool]:
        raise RuntimeError("boom")

    runtime.register(
        ToolSpec(
            name="boom",
            description="boom",
            parameters={"type": "object", "properties": {}},
            handler=boom,
        )
    )

    result = await runtime.invoke(_call("boom"))
    assert result.error_code == "TOOL_EXECUTION_FAILED"
    assert "boom" in (result.error_message or "")


async def test_non_serializable_result() -> None:
    runtime = ToolRuntime()

    def weird() -> dict[str, object]:
        return {"value": {1, 2, 3}}

    runtime.register(
        ToolSpec(
            name="weird",
            description="weird",
            parameters={"type": "object", "properties": {}},
            handler=weird,
        )
    )

    result = await runtime.invoke(_call("weird"))
    assert result.error_code == "TOOL_RESULT_NOT_SERIALIZABLE"


def test_duplicate_registration(runtime) -> None:
    with pytest.raises(ValueError):
        runtime.register(runtime.get("query_order"))
