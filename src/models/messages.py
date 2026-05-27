"""
Pydantic 消息模型 — 强制类型契约。

AlphaDebater 的每一条辩论发言都是不可篡改的 Pydantic 对象。
通过 LangGraph 原生 Reducer 机制追加到 State 中，
任何下游代码都不得修改已记录的发言内容。

设计原则：
    - 不可变性：消息一经记录，只读不写
    - 强类型：所有字段经过 Pydantic 校验
    - 原生 Reducer：利用 LangGraph add_messages 语义，
      自动将新消息追加到列表末尾
"""

from __future__ import annotations

from typing import Literal, Annotated
from pydantic import BaseModel, Field


class DebateMessage(BaseModel):
    """单条辩论发言 — 不可变记录。

    每条消息代表 Bull 或 Bear 在一轮辩论中的一次完整发言。
    消息一旦创建，内容 (content) 不得修改。
    历史消息列表只追加、不删除、不压缩。

    Attributes:
        round:   辩论轮次 (1-indexed)。
        speaker: 发言者身份，"bull" 或 "bear"。
        content: 完整发言原文，未经任何截断或摘要。
    """

    round: int = Field(..., ge=1, le=10, description="辩论轮次 (1-10)")
    speaker: Literal["bull", "bear"] = Field(..., description="发言者身份")
    content: str = Field(..., min_length=1, description="完整发言原文 (不可截断)")


# ── LangGraph 原生 Reducer ─────────────────────────────────────

def debate_message_reducer(
    existing: list[DebateMessage],
    new: list[DebateMessage],
) -> list[DebateMessage]:
    """LangGraph 消息追加 Reducer。

    模仿 LangGraph 内置 add_messages 的语义：
        绝不修改已有消息，只将新消息追加到列表末尾。

    这个 Reducer 被绑定到 DebateState.messages 字段上，
    确保每次节点返回 {"messages": [msg]} 时，
    LangGraph 自动执行 existing + new 而非 replace。

    Args:
        existing: State 中已有的消息列表。
        new:      本轮节点返回的新消息列表。

    Returns:
        追加后的完整消息列表。
    """
    if existing is None:
        existing = []
    if new is None:
        new = []
    return existing + new


# 类型别名：带 Reducer 的消息列表
MessageList = Annotated[
    list[DebateMessage],
    debate_message_reducer,
]
