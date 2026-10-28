"""业务模块：产品使用咨询（RAG 检索）。"""

from __future__ import annotations

from typing import Any

from ..rag import KnowledgeBase
from ..runtime import ToolError, ToolSpec


def build_spec(knowledge_base: KnowledgeBase) -> ToolSpec:
    async def search_knowledge_base(question: str, top_k: int = 3) -> dict[str, Any]:
        top_k = min(max(top_k, 1), 5)
        hits = knowledge_base.search(question, top_k=top_k)
        if not hits:
            raise ToolError(
                "KB_NO_MATCH",
                "知识库未检索到相关内容，请补充产品型号或转人工处理",
            )
        return {
            "question": question,
            "count": len(hits),
            "results": [hit.to_dict() for hit in hits],
            "source": "product-knowledge-base",
        }

    return ToolSpec(
        name="search_knowledge_base",
        description=(
            "检索产品使用知识库，覆盖耳机/手环使用、账号登录、保修、发票与"
            "退款政策。回答产品使用咨询时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户的产品使用问题",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回段落数量，默认 3，最大 5",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        handler=search_knowledge_base,
    )
