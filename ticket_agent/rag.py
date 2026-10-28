"""轻量 RAG：Markdown 知识库 + 字符级 TF-IDF 检索（零额外依赖）。

中文没有分词依赖时，使用「汉字二元组 + 英文/数字词」作为 token，
对客服 FAQ 这类短文本检索效果稳定且可解释。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []

    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    tokens.extend(_WORD_RE.findall(text))
    return tokens


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    section: str
    content: str


@dataclass(frozen=True)
class SearchHit:
    source: str
    section: str
    score: float
    content: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "section": self.section,
            "score": round(self.score, 4),
            "content": self.content,
        }


def load_markdown_chunks(root: Path) -> list[KnowledgeChunk]:
    """按标题与空行切分 Markdown，得到可检索段落。"""
    chunks: list[KnowledgeChunk] = []

    for path in sorted(root.glob("*.md")):
        heading = path.stem
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            buffer.clear()
            if text:
                chunks.append(
                    KnowledgeChunk(
                        source=path.name,
                        section=heading,
                        content=f"{heading}\n{text}",
                    )
                )

        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                flush()
                heading = stripped.lstrip("#").strip() or path.stem
            elif not stripped:
                flush()
            else:
                buffer.append(stripped)
        flush()

    return chunks


class TfidfIndex:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks
        self._term_freqs = [Counter(tokenize(chunk.content)) for chunk in chunks]

        document_freq: Counter[str] = Counter()
        for freq in self._term_freqs:
            document_freq.update(freq.keys())

        total = max(len(chunks), 1)
        self._idf = {
            term: math.log((total + 1) / (df + 1)) + 1
            for term, df in document_freq.items()
        }
        self._vectors = [self._document_vector(freq) for freq in self._term_freqs]
        self._norms = [self._norm(vector) for vector in self._vectors]

    def _document_vector(self, freq: Counter[str]) -> dict[str, float]:
        return {
            term: (1 + math.log(count)) * self._idf[term]
            for term, count in freq.items()
        }

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(weight * weight for weight in vector.values()))

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.02,
    ) -> list[SearchHit]:
        query_terms = Counter(tokenize(query))
        query_vector = {
            term: (1 + math.log(count)) * self._idf[term]
            for term, count in query_terms.items()
            if term in self._idf
        }
        if not query_vector:
            return []

        query_norm = self._norm(query_vector)
        if query_norm == 0:
            return []

        scored: list[tuple[float, int]] = []
        for index, vector in enumerate(self._vectors):
            denominator = self._norms[index] * query_norm
            if denominator == 0:
                continue
            dot = sum(
                weight * vector.get(term, 0.0)
                for term, weight in query_vector.items()
            )
            score = dot / denominator
            if score >= min_score:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))

        hits: list[SearchHit] = []
        for score, index in scored[:top_k]:
            chunk = self.chunks[index]
            hits.append(
                SearchHit(
                    source=chunk.source,
                    section=chunk.section,
                    score=score,
                    content=chunk.content,
                )
            )
        return hits


@dataclass
class KnowledgeBase:
    chunks: list[KnowledgeChunk]
    index: TfidfIndex

    @classmethod
    def load(cls, root: Path) -> "KnowledgeBase":
        if not root.is_dir():
            raise FileNotFoundError(f"知识库目录不存在: {root}")
        chunks = load_markdown_chunks(root)
        if not chunks:
            raise ValueError(f"知识库为空: {root}")
        return cls(chunks=chunks, index=TfidfIndex(chunks))

    def search(self, question: str, top_k: int = 3) -> list[SearchHit]:
        return self.index.search(question, top_k=top_k)

    def stats(self) -> dict[str, object]:
        return {
            "documents": len({chunk.source for chunk in self.chunks}),
            "chunks": len(self.chunks),
        }
