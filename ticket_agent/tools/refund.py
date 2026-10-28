"""业务模块：退款进度查询。"""

from __future__ import annotations

from typing import Any

from ..runtime import ToolError, ToolSpec
from . import data


async def query_refund(
    refund_id: str | None = None,
    order_id: str | None = None,
) -> dict[str, Any]:
    if not refund_id and not order_id:
        raise ToolError(
            "INVALID_ARGUMENTS",
            "refund_id 与 order_id 至少提供一个",
        )

    if refund_id:
        refund = data.get_refund(refund_id)
        if refund is None:
            raise ToolError(
                "REFUND_NOT_FOUND",
                f"未找到退款单 {refund_id}",
            )
    else:
        refunds = data.find_refunds_by_order(order_id or "")
        if not refunds:
            raise ToolError(
                "REFUND_NOT_FOUND",
                f"订单 {order_id} 暂无退款记录",
            )
        refund = sorted(refunds, key=lambda item: item["applied_at"])[-1]

    order = data.get_order(refund["order_id"])
    return {
        "refund": refund,
        "order": order,
        "source": "after-sales",
        "queried_at": data.now_iso(),
    }


SPEC = ToolSpec(
    name="query_refund",
    description=(
        "查询退款单进度与时间线。可以用退款单号查询，也可以用订单号查询该订单"
        "最新的退款记录。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "refund_id": {
                "type": "string",
                "description": "退款单号，例如 RF-20261001",
            },
            "order_id": {
                "type": "string",
                "description": "订单号，例如 ORD-20261001",
            },
        },
        "additionalProperties": False,
    },
    handler=query_refund,
)
