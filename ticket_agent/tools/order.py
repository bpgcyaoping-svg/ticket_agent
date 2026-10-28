"""业务模块：订单查询。"""

from __future__ import annotations

from typing import Any

from ..runtime import ToolError, ToolSpec
from . import data


async def query_order(order_id: str) -> dict[str, Any]:
    order = data.get_order(order_id)
    if order is None:
        raise ToolError(
            "ORDER_NOT_FOUND",
            f"未找到订单 {order_id}，请核对订单号后重试",
        )
    return {
        "order": order,
        "source": "order-center",
        "queried_at": data.now_iso(),
    }


SPEC = ToolSpec(
    name="query_order",
    description=(
        "根据订单号查询订单详情，包括商品、金额、订单状态、物流单号、"
        "预计送达时间与售后政策。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "订单号，例如 ORD-20261001",
            },
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    handler=query_order,
)
