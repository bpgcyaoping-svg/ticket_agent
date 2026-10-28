from __future__ import annotations

import os
from io import StringIO

import pytest

from ticket_agent.agent_runner import AgentRunner
from ticket_agent.config import KNOWLEDGE_DIR, load_settings
from ticket_agent.gateway import LLMGateway
from ticket_agent.rag import KnowledgeBase
from ticket_agent.tools import build_runtime

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE") != "1",
    reason="需要真实 DEEPSEEK_API_KEY，设置 RUN_LIVE=1 后运行",
)


@pytest.mark.live
async def test_live_agent_calls_tool_and_answers() -> None:
    settings = load_settings()
    gateway = LLMGateway(settings)
    loop = AgentRunner(
        gateway,
        build_runtime(KnowledgeBase.load(KNOWLEDGE_DIR)),
        output=StringIO(),
    )

    try:
        result = await loop.run(
            "帮我查一下订单 ORD-20261001 的物流状态，现在到哪了？"
        )
    finally:
        await gateway.close()

    assert result.final_answer
    assert result.tool_results, "真实模型应至少调用一次工具"
