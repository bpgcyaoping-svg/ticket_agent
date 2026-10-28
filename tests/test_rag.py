from __future__ import annotations

from ticket_agent.rag import tokenize


def test_chinese_bigram_tokenizer() -> None:
    tokens = tokenize("耳机怎么配对")

    assert "耳机" in tokens
    assert "怎么" in tokens
    assert "配对" in tokens


def test_headphone_faq_is_retrieved(knowledge_base) -> None:
    hits = knowledge_base.search("耳机怎么配对，一直连不上", top_k=3)

    assert hits
    assert hits[0].source == "产品使用-耳机.md"
    assert hits[0].score > 0


def test_refund_policy_is_retrieved(knowledge_base) -> None:
    hits = knowledge_base.search("退款多久到账", top_k=3)

    assert hits
    assert hits[0].source == "售后与退款政策.md"
    assert "到账" in hits[0].content


def test_invoice_faq_is_retrieved(knowledge_base) -> None:
    hits = knowledge_base.search("发票怎么开", top_k=3)

    assert hits
    assert hits[0].source == "发票与保修.md"


def test_top_k_is_respected(knowledge_base) -> None:
    assert len(knowledge_base.search("耳机", top_k=1)) == 1
    assert len(knowledge_base.search("耳机", top_k=5)) <= 5


def test_empty_or_unrelated_query(knowledge_base) -> None:
    assert knowledge_base.search("") == []
    assert knowledge_base.search("量子力学入门") == []


def test_knowledge_stats(knowledge_base) -> None:
    stats = knowledge_base.stats()

    assert stats["documents"] == 5
    assert stats["chunks"] >= stats["documents"]
