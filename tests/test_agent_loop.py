from __future__ import annotations

import json
from io import StringIO

from ticket_agent.agent_runner import AgentRunner
from ticket_agent.events import EventLogger
from tests.helpers import ScriptedGateway, tool_context_messages


def _tool_call(name: str, call_id: str, **arguments: object) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "tool_call_id": call_id,
            "name": name,
            "arguments": arguments,
        },
        ensure_ascii=False,
    )


async def test_single_tool_then_answer(runtime) -> None:
    gateway = ScriptedGateway(
        [
            _tool_call("query_order", "call_1", order_id="ORD-20261001"),
            "您的订单正在运输中，顺丰单号 SF1234567890，预计 10 月 29 日送达。",
        ]
    )
    events = EventLogger()
    loop = AgentRunner(gateway, runtime, events=events, output=StringIO())

    result = await loop.run("帮我查订单 ORD-20261001")

    assert result.stop_reason == "answer"
    assert result.steps == 2
    assert len(result.tool_results) == 1
    assert result.tool_results[0].tool_call_id == "call_1"
    assert result.tool_results[0].success is True
    assert "运输中" in result.final_answer
    assert result.duration_ms > 0

    contexts = tool_context_messages(gateway.calls[1])
    assert len(contexts) == 1
    assert 'tool_call_id="call_1"' in contexts[0]
    assert "ORD-20261001" in contexts[0]

    names = events.names()
    assert names[0] == "agent_start"
    assert names[-1] == "agent_end"
    for event in ("assistant_stream", "message_end", "tool_start", "tool_end"):
        assert event in names


async def test_sequential_tools_across_turns(runtime) -> None:
    gateway = ScriptedGateway(
        [
            _tool_call("classify_intent", "call_a", text="查订单物流"),
            _tool_call("query_order", "call_b", order_id="ORD-20261001"),
            "订单 ORD-20261001 正在运输中，预计 10 月 29 日送达。",
        ]
    )
    loop = AgentRunner(gateway, runtime, output=StringIO())

    result = await loop.run("查一下订单物流")

    assert result.steps == 3
    assert [item.name for item in result.tool_results] == [
        "classify_intent",
        "query_order",
    ]
    assert [item.tool_call_id for item in result.tool_results] == ["call_a", "call_b"]
    assert result.final_answer.startswith("订单 ORD-20261001")


async def test_tool_error_is_fed_back_to_model(runtime) -> None:
    gateway = ScriptedGateway(
        [
            _tool_call("query_order", "call_bad", order_id="ORD-00000000"),
            "未查到该订单，请核对订单号后重试。",
        ]
    )
    loop = AgentRunner(gateway, runtime, output=StringIO())

    result = await loop.run("查订单 ORD-00000000")

    assert result.tool_results[0].success is False
    assert result.tool_results[0].error_code == "ORDER_NOT_FOUND"
    assert result.stop_reason == "answer"
    context = tool_context_messages(gateway.calls[1])[0]
    assert "ORDER_NOT_FOUND" in context
    assert 'tool_call_id="call_bad"' in context


async def test_unknown_tool_does_not_break_loop(runtime) -> None:
    gateway = ScriptedGateway(
        [
            _tool_call("delete_everything", "call_x"),
            "抱歉，我无法执行该操作，请告诉我具体诉求。",
        ]
    )
    loop = AgentRunner(gateway, runtime, output=StringIO())

    result = await loop.run("执行危险操作")

    assert result.tool_results[0].error_code == "TOOL_NOT_FOUND"
    assert result.final_answer


async def test_rag_scenario(runtime) -> None:
    gateway = ScriptedGateway(
        [
            _tool_call(
                "search_knowledge_base",
                "call_kb",
                question="耳机怎么配对，一直连不上",
                top_k=2,
            ),
            "请先删除旧蓝牙记录，再长按充电盒按键 3 秒重新配对。",
        ]
    )
    loop = AgentRunner(gateway, runtime, output=StringIO())

    result = await loop.run("耳机连不上怎么办")

    assert result.tool_results[0].success is True
    assert result.tool_results[0].data["results"][0]["source"] == "产品使用-耳机.md"
    assert "重新配对" in result.final_answer


async def test_fault_and_complaint_scenarios(runtime) -> None:
    gateway = ScriptedGateway(
        [
            _tool_call(
                "report_system_fault",
                "call_flt",
                title="登录 500",
                description="Android 登录持续报 500",
                severity="high",
                platform="Android",
            ),
            _tool_call(
                "file_complaint",
                "call_cmp",
                summary="客服态度差",
                detail="多次沟通无人处理",
                order_id="ORD-20261002",
            ),
            "已为您登记故障单和投诉单，相关团队会尽快联系您。",
        ]
    )
    loop = AgentRunner(gateway, runtime, output=StringIO())

    result = await loop.run("登录报错，同时我要投诉")

    assert result.tool_results[0].data["fault_id"] == "FLT-0001"
    assert result.tool_results[1].data["complaint_id"] == "CMP-0001"
    assert result.steps == 3


async def test_protocol_error_triggers_single_correction(runtime) -> None:
    gateway = ScriptedGateway(
        [
            '{"type":"tool_call","name":"query_order","arguments":',
            _tool_call("query_order", "call_ok", order_id="ORD-20261001"),
            "订单正在运输中。",
        ]
    )
    loop = AgentRunner(gateway, runtime, output=StringIO())

    result = await loop.run("查订单")

    assert result.steps == 3
    assert result.tool_results[0].tool_call_id == "call_ok"
    assert result.stop_reason == "answer"

    correction = [
        message["content"]
        for message in gateway.calls[1]
        if message["role"] == "user"
        and str(message["content"]).startswith("上一条输出不是合法的工具调用")
    ]
    assert correction


async def test_max_steps_guard(runtime) -> None:
    gateway = ScriptedGateway(
        [
            _tool_call("query_order", "call_1", order_id="ORD-20261001"),
            _tool_call("query_order", "call_2", order_id="ORD-20261002"),
        ]
    )
    loop = AgentRunner(gateway, runtime, max_steps=2, output=StringIO())

    result = await loop.run("一直查订单")

    assert result.steps == 2
    assert result.stop_reason == "max_steps"
    assert "最大推理轮次" in result.final_answer


async def test_repeated_tool_call_is_blocked(runtime) -> None:
    repeated = _tool_call("query_order", "call_r", order_id="ORD-20261001")
    gateway = ScriptedGateway([repeated, repeated, repeated])
    loop = AgentRunner(gateway, runtime, output=StringIO())

    result = await loop.run("一直重复查订单")

    assert result.steps == 3
    assert result.stop_reason == "repeated_action"
    assert result.tool_results[-1].error_code == "REPEATED_ACTION"
    assert "重复" in result.final_answer


async def test_streaming_output_and_to_dict(runtime) -> None:
    answer = "订单 ORD-20261001 正在运输中。"
    gateway = ScriptedGateway(
        [
            _tool_call("query_order", "call_s", order_id="ORD-20261001"),
            answer,
        ]
    )
    output = StringIO()
    loop = AgentRunner(gateway, runtime, output=output)

    result = await loop.run("查订单")

    assert answer in output.getvalue()
    payload = result.to_dict()
    assert payload["steps"] == 2
    assert payload["tool_results"][0]["success"] is True
