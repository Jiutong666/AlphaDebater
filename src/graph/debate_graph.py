"""
LangGraph 辩论状态机 — AlphaDebater 核心编排模块。

定义红蓝军对抗辩论的完整状态流转图：
    START → FETCH_DATA → BULL → BEAR → (循环N轮) → CIO → END

每一轮 Bull 和 Bear 各运行一次。达到最大轮数后，
自动路由至 CIO 节点进行最终裁判。

状态管理：
    - messages 字段使用 LangGraph 原生 Reducer (debate_message_reducer)
    - 节点返回 {"messages": [msg]} → LangGraph 自动追加到列表
    - 每个 DebateMessage 是不可变的 Pydantic 对象
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from src.agents.bull_agent import run_bull_agent
from src.agents.bear_agent import run_bear_agent
from src.agents.cio_agent import run_cio_agent
from src.config.settings import settings
from src.data.base import DataSource
from src.data.yfinance_source import YFinanceSource
from src.data.fmp_source import FMPSource
from src.data.sample_source import SampleSource
from src.models.messages import DebateMessage
from src.models.state import DebateState, create_initial_state
from src.utils.printer import (
    print_header,
    print_data_summary,
    print_round_header,
    print_agent_message,
    print_cio_verdict,
    print_section,
)


# ── 数据源工厂 ─────────────────────────────────────────────────

def _get_data_source() -> DataSource:
    """根据配置动态加载数据源。

    支持: fmp (默认), yfinance, sample

    Returns:
        DataSource 实例。
    """
    source_name: str = settings.data_source.lower()
    if source_name == "fmp":
        return FMPSource()
    if source_name == "yfinance":
        return YFinanceSource()
    if source_name == "sample":
        return SampleSource()
    raise ValueError(f"不支持的数据源: {source_name}")


# ── 图节点定义 ─────────────────────────────────────────────────

def fetch_data_node(state: DebateState) -> dict[str, object]:
    """节点: 抓取财务数据。

    Args:
        state: 当前状态（此时只有 ticker 有值）。

    Returns:
        包含 financial_data 的状态更新字典。
    """
    ticker: str = state["ticker"]
    print_section("📡", "正在抓取财务数据...")

    source: DataSource = _get_data_source()

    try:
        raw_data: dict = source.fetch(ticker)
    except (RuntimeError, ValueError) as exc:
        msg = str(exc).lower()
        should_fallback = any(kw in msg for kw in [
            "rate limited", "too many requests", "retry", "限流",
            "403", "401", "forbidden", "unauthorized", "api key",
        ])
        # 数据源不可用 → 自动降级到 sample 数据
        if should_fallback and isinstance(source, (YFinanceSource, FMPSource)):
            print_section("⚠️", "数据源不可用，自动切换为离线样本数据...")
            print(f"     原因: {exc}")
            source = SampleSource()
            raw_data = source.fetch(ticker)
        else:
            raise

    financial_text: str = source.to_text(raw_data)

    # 构建时间维度隔离的结构化数据
    try:
        from src.models.financial_data import TemporalFinancialData
        info = raw_data.get("info", {})
        structured = TemporalFinancialData.from_flat_dict(info)
        structured_dict = structured.model_dump()
    except Exception:
        structured_dict = {}

    print_data_summary(ticker, raw_data.get("info", {}))
    return {
        "financial_data": financial_text,
        "structured_data": structured_dict,
    }


def bull_node(state: DebateState) -> dict[str, object]:
    """节点: 🐂 红军多头发言。

    Args:
        state: 当前辩论状态。

    Returns:
        包含 messages 的状态更新。LangGraph Reducer 自动追加。
    """
    round_num: int = state["current_round"]
    print_round_header(round_num)
    print_section("🐂", f"红军多头 — 第 {round_num} 轮陈述")

    result: dict[str, object] = run_bull_agent(state)

    # 从返回的 messages 中提取最新内容用于终端打印
    msgs: list[DebateMessage] = result.get("messages", [])
    if msgs:
        print_agent_message("Bull / 红军多头", "🐂", msgs[-1].content)

    return result


def bear_node(state: DebateState) -> dict[str, object]:
    """节点: 🐻 蓝军空头发言。

    Args:
        state: 当前辩论状态。

    Returns:
        包含 messages 的状态更新。LangGraph Reducer 自动追加。
    """
    round_num: int = state["current_round"]
    print_section("🐻", f"蓝军空头 — 第 {round_num} 轮陈述")

    result: dict[str, object] = run_bear_agent(state)

    # 从返回的 messages 中提取最新内容用于终端打印
    msgs: list[DebateMessage] = result.get("messages", [])
    if msgs:
        print_agent_message("Bear / 蓝军空头", "🐻", msgs[-1].content)

    return result


def increment_round_node(state: DebateState) -> dict[str, object]:
    """节点: 轮次计数器 +1。

    在每轮辩论结束后调用，为下一轮做轮次递增。

    Args:
        state: 当前状态。

    Returns:
        轮次 +1 的状态更新。
    """
    next_round: int = state["current_round"] + 1
    return {"current_round": next_round}


def cio_node(state: DebateState) -> dict[str, object]:
    """节点: 🎯 CIO 最终裁判。

    读取完整的辩论记录 (messages)，对各方的论证质量打分，
    计算加权后的最终目标价。

    Args:
        state: 辩论完成后的状态。

    Returns:
        包含最终目标价、信心分、判词的状态更新。
    """
    print_section("🎯", "CIO 正在评估辩论质量，计算最终目标价...")

    result: dict[str, object] = run_cio_agent(state)

    # 提取结果用于终端美化输出
    final_price: float = float(result.get("final_target_price", 0.0))
    bull_conf: float = float(result.get("bull_confidence", 50.0))
    bear_conf: float = float(result.get("bear_confidence", 50.0))
    verdict: str = str(result.get("final_verdict", "N/A"))

    print_cio_verdict(final_price, bull_conf, bear_conf, verdict)

    return result


# ── 条件边：决定下一跳 ─────────────────────────────────────────

def should_continue_debate(state: DebateState) -> Literal["bull", "cio"]:
    """条件边：判断辩论是否继续。

    在 increment_round 节点后调用。
    如果当前轮次 <= 最大轮数，继续下一轮 bull 发言。
    否则，路由至 CIO 最终裁判。

    Args:
        state: increment_round 更新后的状态。

    Returns:
        "bull" → 继续辩论 / "cio" → 进入最终裁判。
    """
    if state["current_round"] <= settings.max_debate_rounds:
        return "bull"
    return "cio"


# ── 构建辩论图 ─────────────────────────────────────────────────

def build_debate_graph() -> StateGraph:
    """构建完整的 LangGraph 辩论状态图。

    Returns:
        编译好的 StateGraph 实例，可直接执行 invoke()。
    """
    graph = StateGraph(DebateState)

    # 注册节点
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("bull", bull_node)
    graph.add_node("bear", bear_node)
    graph.add_node("increment_round", increment_round_node)
    graph.add_node("cio", cio_node)

    # 设置入口
    graph.set_entry_point("fetch_data")

    # 添加边
    graph.add_edge("fetch_data", "bull")
    graph.add_edge("bull", "bear")
    graph.add_edge("bear", "increment_round")

    # 条件边: increment_round → bull (继续) 或 cio (结束)
    graph.add_conditional_edges(
        "increment_round",
        should_continue_debate,
        {
            "bull": "bull",
            "cio": "cio",
        },
    )

    graph.add_edge("cio", END)

    return graph.compile()


# ── 公开入口 ───────────────────────────────────────────────────

def run_debate(ticker: str) -> DebateState:
    """执行一次完整的对抗式辩论。

    Args:
        ticker: 股票代码（如 "AAPL", "TSLA"）。

    Returns:
        包含完整辩论记录和最终目标价的 DebateState。

    Raises:
        ValueError: 数据抓取失败时抛出。
    """
    print_header(ticker)

    app = build_debate_graph()
    initial_state: DebateState = create_initial_state(ticker)
    final_state: DebateState = app.invoke(initial_state)

    return final_state
