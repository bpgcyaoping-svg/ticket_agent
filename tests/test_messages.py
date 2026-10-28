from __future__ import annotations

from ticket_agent.protocol import parse_model_output
from ticket_agent.messages import (
    AssistantMessage,
    TextBlock,
    ToolCall,
    ToolCallBlock,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    to_openai_messages,
)


def test_tool_result_context_keeps_tool_call_id() -> None:
    call = ToolCall("call_1", "query_order", {"order_id": "ORD-20261001"})
    result = ToolResult.ok(call, {"status": "运输中"}, duration_ms=12.5)

    context = result.to_context()
    assert 'tool_call_id="call_1"' in context
    assert '"tool_call_id": "call_1"' in context
    assert "query_order" in context
    assert "运输中" in context


def test_assistant_message_supports_text_and_tool_call_blocks() -> None:
    call = ToolCall("call_9", "query_order", {"order_id": "ORD-20261001"})
    message = AssistantMessage(
        raw_text='{"type":"tool_call"}',
        blocks=[TextBlock("我先查一下订单。"), ToolCallBlock(call)],
    )

    assert message.text == "我先查一下订单。"
    assert message.tool_calls == [call]


def test_openai_message_roles_and_tool_id_round_trip() -> None:
    call = ToolCall("call_2", "query_order", {"order_id": "ORD-20261001"})
    result = ToolResult.ok(call, {"status": "运输中"})
    messages = [
        UserMessage("查订单"),
        AssistantMessage(raw_text="", blocks=[ToolCallBlock(call)]),
        ToolResultMessage(result),
    ]

    converted = to_openai_messages("system", messages)
    assert [item["role"] for item in converted] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    # 文本协议下工具结果回填为 user 消息，但 tool_call_id 仍保留
    assert "call_2" in converted[-1]["content"]


def test_parse_tool_call_from_fenced_block() -> None:
    text = """```json
{"type":"tool_call","tool_call_id":"call_f","name":"query_order","arguments":{"order_id":"ORD-20261001"}}
```"""
    parsed = parse_model_output(text)

    assert parsed.error is None
    assert parsed.prose == ""
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].tool_call_id == "call_f"
    assert parsed.tool_calls[0].arguments == {"order_id": "ORD-20261001"}


def test_parse_tool_call_with_surrounding_prose() -> None:
    text = (
        "我需要先查订单。\n"
        '{"type":"tool_call","name":"query_order","arguments":{"order_id":"ORD-20261001"}}\n'
        "请稍等。"
    )
    parsed = parse_model_output(text)

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].tool_call_id.startswith("call_")
    assert parsed.prose == "我需要先查订单。\n\n请稍等。"


def test_parse_arguments_as_json_string() -> None:
    text = (
        '{"type":"tool_call","name":"query_order",'
        '"arguments":"{\\"order_id\\":\\"ORD-20261001\\"}"}'
    )
    parsed = parse_model_output(text)

    assert parsed.tool_calls[0].arguments == {"order_id": "ORD-20261001"}


def test_plain_text_has_no_tool_call() -> None:
    parsed = parse_model_output("您的订单正在运输中，预计明天送达。")

    assert parsed.tool_calls == []
    assert parsed.error is None
    assert parsed.prose == "您的订单正在运输中，预计明天送达。"


def test_non_tool_json_is_ignored() -> None:
    parsed = parse_model_output('结果如下：{"status":"ok"}')

    assert parsed.tool_calls == []
    assert parsed.error is None


def test_broken_tool_call_is_reported() -> None:
    parsed = parse_model_output('{"type":"tool_call","name":"query_order","arguments":')

    assert parsed.tool_calls == []
    assert parsed.error is not None
