"""
LangGraph 状态定义模块。

定义了 DebateState TypedDict —— 红蓝军辩论全生命周期的共享状态。

核心设计：
    - messages 字段使用 LangGraph 原生 Reducer 机制，
      只追加、不覆盖、不压缩。
    - 上下文切片逻辑 (Context Slicer) 负责从 messages 中
      精确提取对手上一轮的完整发言，不产生任何数据失真。
"""

from __future__ import annotations

from typing import TypedDict

from src.models.messages import MessageList, DebateMessage


class DebateState(TypedDict):
    """辩论状态，在 LangGraph 节点间流转。

    该状态被红军(Bull)、蓝军(Bear)、CIO 三个代理读写，
    是辩论图(Graph)的唯一数据源 (Single Source of Truth)。

    Attributes:
        ticker:             股票代码，如 "AAPL"。
        financial_data:     序列化为文本的财务数据。红蓝军共享同一份数据，
                            确保辩论基于相同事实。
        structured_data:    时间维度隔离的结构化财务数据 (dict)。
                            TemporalFinancialData.model_dump() 的结果。
                            用于计算工具访问，不直接注入 LLM prompt。
        current_round:      当前辩论轮次 (1-indexed)。
        messages:           LangGraph 原生 Reducer 管理的消息列表。
                            每轮 Bull/Bear 发言后自动追加，不可删除/修改。
        final_target_price: CIO 计算的最终目标价。初始为 0.0。
        bull_confidence:    CIO 评定的红军论证置信度 (0-100)。
        bear_confidence:    CIO 评定的蓝军论证置信度 (0-100)。
        final_verdict:      CIO 最终判词全文。
    """

    ticker: str
    financial_data: str
    structured_data: dict
    current_round: int
    messages: MessageList  # ← LangGraph 原生 Reducer，只追加不覆盖
    final_target_price: float
    bull_confidence: float
    bear_confidence: float
    final_verdict: str

def create_initial_state(ticker: str) -> DebateState:
    """创建初始化的辩论状态。

    Args:
        ticker: 股票代码。

    Returns:
        填充了默认值的 DebateState。
    """
    return DebateState(
        ticker=ticker.upper(),
        financial_data="",
        structured_data={},
        current_round=1,
        messages=[],
        final_target_price=0.0,
        bull_confidence=0.0,
        bear_confidence=0.0,
        final_verdict="",
    )
