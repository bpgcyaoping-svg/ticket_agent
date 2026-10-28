"""业务模块：用户投诉登记。"""

from __future__ import annotations

from typing import Any

from ..runtime import ToolSpec
from . import data

_HIGH_PRIORITY_WORDS = ("赔偿", "曝光", "12315", "起诉", "媒体", "监管", "律师")


async def file_complaint(
    summary: str,
    detail: str,
    order_id: str | None = None,
    contact: str | None = None,
    expectation: str | None = None,
) -> dict[str, Any]:
    combined = f"{summary} {detail} {expectation or ''}"
    priority = "P1" if any(word in combined for word in _HIGH_PRIORITY_WORDS) else "P2"
    complaint_id = data.next_complaint_id()

    order = data.get_order(order_id) if order_id else None
    linked_order: dict[str, Any] | None = None
    if order_id:
        linked_order = {
            "order_id": order_id,
            "found": order is not None,
            "product": order["product"] if order else None,
            "status": order["status"] if order else None,
        }

    record = {
        "complaint_id": complaint_id,
        "summary": summary,
        "detail": detail,
        "expectation": expectation,
        "priority": priority,
        "status": "已登记",
        "owner": "客户体验组",
        "order_id": order_id,
        "contact_registered": bool(contact),
        "created_at": data.now_iso(),
    }
    data.record_complaint(record)

    return {
        "complaint_id": complaint_id,
        "status": "已登记",
        "priority": priority,
        "owner": "客户体验组",
        "sla_hours": 4 if priority == "P1" else 24,
        "created_at": record["created_at"],
        "linked_order": linked_order,
        "contact_registered": bool(contact),
        "next_steps": [
            "客户体验组将在 SLA 内回访并给出处理方案",
            "如需升级，请补充订单号、沟通记录或凭证",
        ],
    }


SPEC = ToolSpec(
    name="file_complaint",
    description=(
        "登记用户投诉，返回投诉单号、优先级、责任团队与 SLA。"
        "涉及态度、服务体验、赔偿诉求时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "投诉摘要"},
            "detail": {"type": "string", "description": "投诉详情与经过"},
            "order_id": {
                "type": "string",
                "description": "关联订单号（可选）",
            },
            "contact": {"type": "string", "description": "用户联系方式"},
            "expectation": {
                "type": "string",
                "description": "用户期望的处理结果（可选）",
            },
        },
        "required": ["summary", "detail"],
        "additionalProperties": False,
    },
    handler=file_complaint,
)
