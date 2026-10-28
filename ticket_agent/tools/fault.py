"""业务模块：系统故障反馈与诊断信息采集。"""

from __future__ import annotations

from typing import Any

from ..runtime import ToolSpec
from . import data

SEVERITY_SLA = {
    "low": 72,
    "medium": 24,
    "high": 8,
    "critical": 4,
}


def _pick_team(text: str) -> str:
    lowered = text.lower()
    rules = [
        (("登录", "账号", "密码", "验证码", "认证"), "账号与安全组"),
        (("支付", "退款", "订单", "结算", "发票"), "交易平台组"),
        (("闪退", "崩溃", "客户端", "app", "页面", "加载"), "客户端研发组"),
    ]
    for keywords, team in rules:
        if any(keyword in lowered for keyword in keywords):
            return team
    return "基础平台组"


async def report_system_fault(
    title: str,
    description: str,
    severity: str = "medium",
    affected_service: str | None = None,
    contact: str | None = None,
    platform: str = "Web",
    app_version: str = "unknown",
) -> dict[str, Any]:
    fault_id = data.next_fault_id()
    assigned_team = _pick_team(f"{title} {description} {affected_service or ''}")
    signature = description.strip().replace("\n", " ")[:60]

    diagnostics = {
        "trace_id": f"trace-{fault_id.lower()}",
        "collected_at": data.now_iso(),
        "platform": platform,
        "app_version": app_version,
        "affected_service": affected_service or "未指定",
        "error_signature": signature,
        "log_snapshot": [
            f"[ERROR] {platform}/{app_version} {signature}",
            "[WARN] 近 10 分钟错误率 3.2%（演示数据）",
            "[INFO] 已采集请求链路、设备环境与客户端版本",
        ],
        "signals": ["用户描述已归档", "复现路径待验证"],
    }

    record = {
        "fault_id": fault_id,
        "title": title,
        "description": description,
        "severity": severity,
        "status": "已受理",
        "assigned_team": assigned_team,
        "affected_service": affected_service,
        "platform": platform,
        "app_version": app_version,
        "contact_registered": bool(contact),
        "created_at": data.now_iso(),
        "diagnostics": diagnostics,
    }
    data.record_fault(record)

    return {
        "fault_id": fault_id,
        "status": "已受理",
        "severity": severity,
        "assigned_team": assigned_team,
        "sla_hours": SEVERITY_SLA[severity],
        "diagnostics": diagnostics,
        "contact_registered": bool(contact),
        "next_steps": [
            f"{assigned_team} 将在 {SEVERITY_SLA[severity]} 小时内首次响应",
            "请保留故障发生时间与操作路径，便于复现",
        ],
    }


SPEC = ToolSpec(
    name="report_system_fault",
    description=(
        "登记系统故障并采集诊断信息（错误签名、日志快照、客户端环境），"
        "返回故障单号、优先级、处理团队与 SLA。适用于报错、闪退、无法登录等场景。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "故障标题"},
            "description": {"type": "string", "description": "故障现象与复现步骤"},
            "severity": {
                "type": "string",
                "description": "严重程度，默认 medium",
                "enum": ["low", "medium", "high", "critical"],
            },
            "affected_service": {
                "type": "string",
                "description": "受影响的业务或服务，例如 登录、支付、订单",
            },
            "contact": {"type": "string", "description": "用户联系方式"},
            "platform": {"type": "string", "description": "平台，例如 iOS、Android、Web"},
            "app_version": {"type": "string", "description": "客户端版本号"},
        },
        "required": ["title", "description"],
        "additionalProperties": False,
    },
    handler=report_system_fault,
)
