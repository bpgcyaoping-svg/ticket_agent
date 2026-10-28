"""演示用业务数据仓储（内存实现，可替换为真实订单/售后/工单服务）。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

ORDERS: dict[str, dict[str, Any]] = {
    "ORD-20261001": {
        "order_id": "ORD-20261001",
        "user_id": "U-10086",
        "user_name": "张先生",
        "product": "无线降噪耳机 Pro",
        "sku": "AUD-NC-PRO",
        "quantity": 1,
        "amount": 899.00,
        "currency": "CNY",
        "status": "运输中",
        "created_at": "2026-10-20 10:12:00",
        "paid_at": "2026-10-20 10:13:24",
        "shipped_at": "2026-10-21 18:40:00",
        "carrier": "顺丰速运",
        "tracking_no": "SF1234567890",
        "eta": "2026-10-29",
        "address_masked": "上海市浦东新区 **** 号",
        "after_sales": "已签收后支持 7 天无理由退货",
    },
    "ORD-20261002": {
        "order_id": "ORD-20261002",
        "user_id": "U-10086",
        "user_name": "张先生",
        "product": "智能手环 Lite",
        "sku": "WEA-LITE-02",
        "quantity": 1,
        "amount": 299.00,
        "currency": "CNY",
        "status": "已完成",
        "created_at": "2026-10-08 09:30:00",
        "paid_at": "2026-10-08 09:31:10",
        "shipped_at": "2026-10-09 11:00:00",
        "carrier": "京东物流",
        "tracking_no": "JDVA00012345678",
        "eta": "2026-10-11",
        "delivered_at": "2026-10-11 15:22:00",
        "address_masked": "上海市浦东新区 **** 号",
        "after_sales": "已退款 299.00 元",
    },
    "ORD-20261003": {
        "order_id": "ORD-20261003",
        "user_id": "U-20001",
        "user_name": "李女士",
        "product": "便携意式咖啡机",
        "sku": "COF-ESP-01",
        "quantity": 2,
        "amount": 2598.00,
        "currency": "CNY",
        "status": "待发货",
        "created_at": "2026-10-27 20:05:00",
        "paid_at": "2026-10-27 20:06:32",
        "carrier": None,
        "tracking_no": None,
        "eta": "预计 2026-10-30 前发出",
        "address_masked": "北京市朝阳区 **** 号",
        "after_sales": "支持 7 天无理由退货",
    },
}

REFUNDS: dict[str, dict[str, Any]] = {
    "RF-20261001": {
        "refund_id": "RF-20261001",
        "order_id": "ORD-20261001",
        "status": "审核中",
        "amount": 899.00,
        "currency": "CNY",
        "reason": "物流长时间未更新",
        "applied_at": "2026-10-27 14:20:00",
        "expected_arrival": "2026-10-31",
        "timeline": [
            {"at": "2026-10-27 14:20:00", "event": "用户提交退款申请"},
            {"at": "2026-10-27 16:05:00", "event": "系统受理，进入审核"},
        ],
    },
    "RF-20261002": {
        "refund_id": "RF-20261002",
        "order_id": "ORD-20261002",
        "status": "已退款",
        "amount": 299.00,
        "currency": "CNY",
        "reason": "七天无理由退货",
        "applied_at": "2026-10-12 10:00:00",
        "refunded_at": "2026-10-14 09:30:00",
        "expected_arrival": "2026-10-14",
        "timeline": [
            {"at": "2026-10-12 10:00:00", "event": "用户提交退款申请"},
            {"at": "2026-10-13 11:20:00", "event": "仓库签收退货"},
            {"at": "2026-10-14 09:30:00", "event": "退款已原路退回"},
        ],
    },
}

FAULTS: dict[str, dict[str, Any]] = {}
COMPLAINTS: dict[str, dict[str, Any]] = {}

_sequence = {"fault": 0, "complaint": 0}


def reset_business_state() -> None:
    """测试辅助：清空工单类数据并重置自增编号。"""
    FAULTS.clear()
    COMPLAINTS.clear()
    _sequence["fault"] = 0
    _sequence["complaint"] = 0


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_order(order_id: str) -> dict[str, Any] | None:
    order = ORDERS.get(order_id)
    return deepcopy(order) if order else None


def get_refund(refund_id: str) -> dict[str, Any] | None:
    refund = REFUNDS.get(refund_id)
    return deepcopy(refund) if refund else None


def find_refunds_by_order(order_id: str) -> list[dict[str, Any]]:
    return [
        deepcopy(refund)
        for refund in REFUNDS.values()
        if refund["order_id"] == order_id
    ]


def next_fault_id() -> str:
    _sequence["fault"] += 1
    return f"FLT-{_sequence['fault']:04d}"


def next_complaint_id() -> str:
    _sequence["complaint"] += 1
    return f"CMP-{_sequence['complaint']:04d}"


def record_fault(record: dict[str, Any]) -> None:
    FAULTS[str(record["fault_id"])] = deepcopy(record)


def record_complaint(record: dict[str, Any]) -> None:
    COMPLAINTS[str(record["complaint_id"])] = deepcopy(record)
