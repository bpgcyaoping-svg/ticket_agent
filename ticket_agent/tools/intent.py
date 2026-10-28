"""业务模块：用户意图识别分类。"""

from __future__ import annotations

from typing import Any

from ..runtime import ToolSpec

# 顺序即同等命中数下的优先级
INTENT_RULES: list[tuple[str, list[str], list[str]]] = [
    ("退款", ["退款", "退货", "退钱", "退单", "申请退"], ["query_refund", "query_order"]),
    ("订单物流", ["订单", "物流", "快递", "发货", "到货", "签收", "配送", "运费"], ["query_order"]),
    (
        "产品咨询",
        ["怎么用", "如何使用", "如何", "怎么", "功能", "设置", "配对", "连接", "充电", "说明书", "保修", "发票", "固件"],
        ["search_knowledge_base"],
    ),
    (
        "系统故障",
        ["故障", "报错", "错误", "打不开", "闪退", "崩溃", "卡死", "无法登录", "异常", "bug", "500"],
        ["report_system_fault"],
    ),
    (
        "投诉",
        ["投诉", "差评", "态度", "维权", "欺骗", "生气", "太差", "垃圾", "曝光"],
        ["file_complaint"],
    ),
]


def classify(text: str) -> dict[str, Any]:
    lowered = text.lower()
    best_category = "其他"
    best_keywords: list[str] = []
    best_tools: list[str] = []

    for category, keywords, tools in INTENT_RULES:
        matched = [keyword for keyword in keywords if keyword.lower() in lowered]
        if len(matched) > len(best_keywords):
            best_category = category
            best_keywords = matched
            best_tools = tools

    if not best_keywords:
        return {
            "category": "其他",
            "confidence": 0.30,
            "matched_keywords": [],
            "suggested_tools": [],
            "reason": "未命中业务关键词，建议先向用户澄清诉求",
        }

    confidence = min(0.95, 0.55 + 0.1 * len(best_keywords))
    return {
        "category": best_category,
        "confidence": round(confidence, 2),
        "matched_keywords": best_keywords,
        "suggested_tools": best_tools,
        "reason": "命中关键词: " + "、".join(best_keywords),
    }


async def classify_intent(text: str) -> dict[str, Any]:
    return classify(text)


SPEC = ToolSpec(
    name="classify_intent",
    description=(
        "对用户诉求做意图分类，返回类别、置信度和建议工具。"
        "类别包括：退款、订单物流、产品咨询、系统故障、投诉、其他。"
        "当用户诉求不明确时优先调用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "用户原始问题或工单内容"},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    handler=classify_intent,
)
