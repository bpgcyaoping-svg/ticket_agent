"""运行配置：全部从环境变量读取，``DEEPSEEK_API_KEY`` 为必填。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def _env_candidates() -> list[Path]:
    """项目 .env 优先，随后向上最多 3 层查找仓库根的 .env。"""
    return [root / ".env" for root in (PROJECT_ROOT, *PROJECT_ROOT.parents[:3])]


def _load_env() -> None:
    """按优先级加载 .env，不覆盖已存在的进程变量。"""
    for path in _env_candidates():
        if path.is_file():
            load_dotenv(path, override=False)


def resolve_base_url(raw: str | None) -> str:
    """规整 base_url。

    Anthropic 兼容网关（例如 ``.../anthropic``）不提供 OpenAI 风格的
    ``/chat/completions``，遇到时回退到 DeepSeek 官方 OpenAI 兼容地址。
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        return DEFAULT_BASE_URL
    if url.endswith("/anthropic"):
        return DEFAULT_BASE_URL
    return url


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_steps: int = 8
    request_timeout: float = 60.0
    verbose: bool = False


def load_settings(verbose: bool | None = None) -> Settings:
    _load_env()

    env_verbose = os.getenv("AGENT_VERBOSE", "0").strip().lower() in {"1", "true", "yes"}
    return Settings(
        api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        base_url=resolve_base_url(os.getenv("DEEPSEEK_BASE_URL")),
        model=(os.getenv("DEEPSEEK_MODEL", "").strip() or DEFAULT_MODEL),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0")),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
        request_timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", "60")),
        verbose=env_verbose if verbose is None else verbose,
    )
