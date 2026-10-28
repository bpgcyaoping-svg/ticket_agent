"""智能工单处理平台 Agent 入口。

用法：
    python main.py                      # 交互模式（等价于 asyncio.run(agent_loop("")))
    python main.py "查询订单 ORD-20261001" # 单轮问答
    python main.py --demo               # 跑内置业务场景
    python main.py --verbose            # 打印关键事件
    python main.py --show-tools         # 查看模型可见的工具 schema
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Sequence

from ticket_agent.agent_loop import AgentRunResult
from ticket_agent.agent_runner import AgentRunner
from ticket_agent.agent_tool import describe_tools_json
from ticket_agent.config import KNOWLEDGE_DIR, Settings, load_settings
from ticket_agent.events import EventLogger
from ticket_agent.gateway import GatewayError, LLMGateway
from ticket_agent.rag import KnowledgeBase
from ticket_agent.tools import build_agent_tools, build_runtime

DEMO_QUESTIONS = [
    "你好，我 10 月 20 日买的无线降噪耳机 Pro，订单号 ORD-20261001，物流好几天没更新，帮我查一下订单详情和退款进度。",
    "无线降噪耳机 Pro 怎么和手机配对？一直连不上。",
    "登录 App 一直报 500 错误，已经两天了，帮我登记系统故障，平台 Android，版本 6.2.1。",
    "我要投诉客服态度太差，订单 ORD-20261002 的退款问题一直没人处理，希望尽快解决。",
]

BANNER = """\
============================================================
 智能工单处理平台 · Agent Demo
 能力：意图识别 / 订单查询 / 退款查询 / 产品咨询(RAG)
       系统故障登记 / 用户投诉
 命令：输入问题开始；demo 跑示例；help 帮助；exit 退出
============================================================"""


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def build_agent(
    settings: Settings,
    events: EventLogger,
) -> tuple[AgentRunner, LLMGateway]:
    knowledge_base = KnowledgeBase.load(KNOWLEDGE_DIR)
    runtime = build_runtime(knowledge_base)
    gateway = LLMGateway(settings)
    runner = AgentRunner(
        gateway=gateway,
        runtime=runtime,
        events=events,
        max_steps=settings.max_steps,
    )
    return runner, gateway


async def _run_question(agent: AgentRunner, question: str) -> AgentRunResult:
    print(f"\n{'=' * 60}\n用户: {question}\n{'-' * 60}", flush=True)
    result = await agent.run(question)
    print("\n最终答案:", flush=True)
    print(result.final_answer or "（模型未返回文本答案）", flush=True)
    print(
        f"\n运行时长: {result.duration_ms / 1000:.2f}s | "
        f"轮次: {result.steps} | 工具调用: {len(result.tool_results)}",
        flush=True,
    )
    return result


async def agent_loop(
    user_input: str = "",
    *,
    verbose: bool | None = None,
) -> list[AgentRunResult]:
    """单轮问答或交互式 REPL；``agent_loop("")`` 即交互模式。"""
    settings = load_settings(verbose=verbose)
    # verbose 事件与正文同流输出，保证重定向到日志时顺序一致
    events = EventLogger(verbose=settings.verbose, stream=sys.stdout)

    try:
        agent, gateway = build_agent(settings, events)
    except (GatewayError, FileNotFoundError, ValueError) as exc:
        print(f"[启动失败] {exc}", file=sys.stderr)
        return []

    try:
        if user_input.strip():
            return [await _run_question(agent, user_input.strip())]

        print(BANNER, flush=True)
        results: list[AgentRunResult] = []
        while True:
            try:
                line = await asyncio.to_thread(input, "\n用户> ")
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break

            line = line.strip()
            if not line:
                continue
            if line.lower() in {"exit", "quit", "q"}:
                break
            if line.lower() in {"help", "/help"}:
                print(BANNER)
                continue
            if line.lower() in {"demo", "/demo"}:
                for question in DEMO_QUESTIONS:
                    results.append(await _run_question(agent, question))
                continue

            try:
                results.append(await _run_question(agent, line))
            except GatewayError as exc:
                print(f"\n[LLM 调用失败] {exc}", file=sys.stderr)

        return results
    finally:
        await gateway.close()


async def run_demo(*, verbose: bool = False) -> None:
    settings = load_settings(verbose=verbose)
    events = EventLogger(verbose=settings.verbose, stream=sys.stdout)

    try:
        agent, gateway = build_agent(settings, events)
    except (GatewayError, FileNotFoundError, ValueError) as exc:
        print(f"[启动失败] {exc}", file=sys.stderr)
        return

    try:
        for question in DEMO_QUESTIONS:
            try:
                await _run_question(agent, question)
            except GatewayError as exc:
                print(f"\n[LLM 调用失败] {exc}", file=sys.stderr)
                break
    finally:
        await gateway.close()


def main(argv: Sequence[str] | None = None) -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="智能工单处理平台 Agent")
    parser.add_argument("question", nargs="*", help="要处理的用户问题")
    parser.add_argument("--verbose", action="store_true", help="打印关键事件")
    parser.add_argument("--demo", action="store_true", help="运行内置业务场景")
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="打印模型可见的工具 schema（不调用模型）",
    )
    args = parser.parse_args(argv)

    if args.show_tools:
        knowledge_base = KnowledgeBase.load(KNOWLEDGE_DIR)
        runtime = build_runtime(knowledge_base)
        print(describe_tools_json(build_agent_tools(runtime)))
        return

    if args.demo:
        asyncio.run(run_demo(verbose=args.verbose))
        return

    question = " ".join(args.question).strip()
    if question:
        asyncio.run(agent_loop(question, verbose=args.verbose))
        return

    # 无参数时进入交互模式，等价于 asyncio.run(agent_loop(""))
    asyncio.run(agent_loop("", verbose=args.verbose))


if __name__ == "__main__":
    _configure_stdio()
    if len(sys.argv) == 1:
        # 题目要求的最小运行入口：无参数时进入交互模式
        asyncio.run(agent_loop(""))
    else:
        main()
