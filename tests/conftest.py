from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ticket_agent.config import KNOWLEDGE_DIR  # noqa: E402
from ticket_agent.rag import KnowledgeBase  # noqa: E402
from ticket_agent.tools import build_runtime  # noqa: E402
from ticket_agent.tools import data  # noqa: E402


@pytest.fixture
def knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.load(KNOWLEDGE_DIR)


@pytest.fixture
def runtime(knowledge_base: KnowledgeBase):
    return build_runtime(knowledge_base)


@pytest.fixture(autouse=True)
def clean_business_state():
    data.reset_business_state()
    yield
    data.reset_business_state()
