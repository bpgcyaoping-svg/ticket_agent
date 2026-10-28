from __future__ import annotations

from ticket_agent.runtime import ToolInvocation
from ticket_agent.tools import data
from ticket_agent.tools.intent import classify


def _call(name: str, **arguments: object) -> ToolInvocation:
    return ToolInvocation("call_tool", name, dict(arguments))


async def test_classify_intent_categories() -> None:
    cases = {
        "我的订单什么时候发货": "订单物流",
        "我要申请退款": "退款",
        "耳机怎么配对": "产品咨询",
        "App 闪退打不开": "系统故障",
        "我要投诉客服态度太差": "投诉",
        "今天天气不错": "其他",
    }
    for text, expected in cases.items():
        result = classify(text)
        assert result["category"] == expected, text
        assert 0 < result["confidence"] <= 0.95


async def test_classify_intent_tool(runtime) -> None:
    result = await runtime.invoke(_call("classify_intent", text="我想退款"))

    assert result.success is True
    assert result.data["category"] == "退款"
    assert "query_refund" in result.data["suggested_tools"]


async def test_query_order_success(runtime) -> None:
    result = await runtime.invoke(
        _call("query_order", order_id="ORD-20261001")
    )

    order = result.data["order"]
    assert order["product"] == "无线降噪耳机 Pro"
    assert order["tracking_no"] == "SF1234567890"


async def test_query_order_not_found(runtime) -> None:
    result = await runtime.invoke(
        _call("query_order", order_id="ORD-00000000")
    )

    assert result.error_code == "ORDER_NOT_FOUND"


async def test_query_refund_by_refund_id(runtime) -> None:
    result = await runtime.invoke(
        _call("query_refund", refund_id="RF-20261001")
    )

    assert result.data["refund"]["status"] == "审核中"
    assert result.data["order"]["order_id"] == "ORD-20261001"


async def test_query_refund_by_order_id(runtime) -> None:
    result = await runtime.invoke(
        _call("query_refund", order_id="ORD-20261002")
    )

    assert result.data["refund"]["refund_id"] == "RF-20261002"
    assert result.data["refund"]["status"] == "已退款"


async def test_query_refund_not_found(runtime) -> None:
    result = await runtime.invoke(
        _call("query_refund", refund_id="RF-99999999")
    )

    assert result.error_code == "REFUND_NOT_FOUND"


async def test_search_knowledge_base(runtime) -> None:
    result = await runtime.invoke(
        _call("search_knowledge_base", question="手环怎么连接手机 App")
    )

    assert result.success is True
    assert result.data["count"] >= 1
    assert result.data["results"][0]["source"] == "产品使用-手环.md"


async def test_search_knowledge_base_no_match(runtime) -> None:
    result = await runtime.invoke(
        _call("search_knowledge_base", question="量子力学入门")
    )

    assert result.error_code == "KB_NO_MATCH"


async def test_report_system_fault(runtime) -> None:
    result = await runtime.invoke(
        _call(
            "report_system_fault",
            title="登录报错 500",
            description="Android 客户端登录持续返回 500，已持续两天",
            severity="critical",
            affected_service="登录",
            contact="138****0000",
            platform="Android",
            app_version="6.2.1",
        )
    )

    payload = result.data
    assert payload["fault_id"] == "FLT-0001"
    assert payload["assigned_team"] == "账号与安全组"
    assert payload["sla_hours"] == 4
    assert payload["contact_registered"] is True
    assert payload["diagnostics"]["trace_id"] == "trace-flt-0001"
    assert "FLT-0001" in data.FAULTS


async def test_file_complaint_with_order(runtime) -> None:
    result = await runtime.invoke(
        _call(
            "file_complaint",
            summary="客服态度差",
            detail="咨询退款时被反复挂断电话",
            order_id="ORD-20261002",
        )
    )

    payload = result.data
    assert payload["complaint_id"] == "CMP-0001"
    assert payload["priority"] == "P2"
    assert payload["linked_order"]["found"] is True
    assert payload["linked_order"]["product"] == "智能手环 Lite"
    assert "CMP-0001" in data.COMPLAINTS


async def test_file_complaint_high_priority_and_unknown_order(runtime) -> None:
    result = await runtime.invoke(
        _call(
            "file_complaint",
            summary="要求赔偿",
            detail="商品损坏且客服不处理",
            order_id="ORD-UNKNOWN",
            expectation="赔偿损失，否则向 12315 投诉",
        )
    )

    payload = result.data
    assert payload["priority"] == "P1"
    assert payload["sla_hours"] == 4
    assert payload["linked_order"]["found"] is False
