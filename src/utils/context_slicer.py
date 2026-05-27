"""
物理上下文切片器 (Context Slicer) — 替代摘要压缩。

AlphaDebater 的核心设计哲学：交叉质询需要 100% 精确的数值打脸。
任何形式的摘要压缩都会导致数据丢失和幻觉。

本模块实现"物理切片过滤"而非"语义压缩"：
    - 财经数据：始终完整注入（一份数据，红蓝军共享）
    - 对手发言：注入对手全部轮次的完整原话（跨轮记忆，追踪论点演进）
    - 数据引用：强制要求引用具体数值（防空洞论证）
    - 本轮指令：根据轮次动态注入任务描述

切出来的 Prompt 结构：
    [财务数据] + [对手论点演进轨迹（全部轮次）] + [数据引用硬性要求] + [本轮任务指令]

严格保证：
    - 不截断任何一条发言
    - 不将多个角色发言混在一起（防止角色混淆）
    - 不注入自己过去的发言（防止自我抄袭）
    - 跨轮记忆完整（第3轮 agent 不会忘记第1轮对手的核心论点）
"""

from __future__ import annotations

from typing import Literal

from src.models.messages import DebateMessage
from src.models.state import DebateState


def slice_opponent_last_speech(
    messages: list[DebateMessage],
    opponent: Literal["bull", "bear"],
) -> str:
    """从消息列表中切片提取对手的上一轮完整发言。

    只取对手 (opponent) 的最近一条消息。不返回任何其他内容。

    Args:
        messages: 完整的辩论消息列表。
        opponent: 目标对手身份 ("bull" 或 "bear")。

    Returns:
        对手上一轮的完整发言原文。如果对手尚未发言，返回提示文本。
    """
    opponent_msgs = [m for m in messages if m.speaker == opponent]

    if not opponent_msgs:
        opponent_name = "红军多头" if opponent == "bull" else "蓝军空头"
        return f"（{opponent_name}尚未发言，这是第 1 轮，请直接建立你的逻辑。）"

    # 取对手最近一条发言（物理切片，一字不改）
    last_msg = opponent_msgs[-1]
    return last_msg.content


def build_opponent_timeline(
    messages: list[DebateMessage],
    opponent: Literal["bull", "bear"],
) -> str:
    """构建对手全部论点的演进轨迹——跨轮记忆。

    与 slice_opponent_last_speech 不同，此函数返回对手在**所有轮次**
    中的完整发言，按时间顺序排列。这解决了"第 3 轮 agent 忘记第 1 轮
    对手核心论点"的跨轮记忆断裂问题。

    保持物理切片哲学：不做语义压缩，逐条注入完整原文。

    Args:
        messages: 完整的辩论消息列表。
        opponent: 目标对手身份 ("bull" 或 "bear")。

    Returns:
        对手全部发言的时序追踪文本。
    """
    opponent_msgs = [m for m in messages if m.speaker == opponent]

    if not opponent_msgs:
        opponent_name = "红军多头" if opponent == "bull" else "蓝军空头"
        return f"（{opponent_name}尚未发言，这是第 1 轮，请直接建立你的逻辑。）"

    opponent_emoji = "🐂" if opponent == "bull" else "🐻"
    opponent_label = "红军多头" if opponent == "bull" else "蓝军空头"

    parts: list[str] = []
    parts.append(
        f"## {opponent_emoji} {opponent_label}论点演进轨迹（完整追踪）\n"
    )

    for msg in opponent_msgs:
        round_label = _round_label(msg.round)
        parts.append(f"### 第 {msg.round} 轮 — {round_label}\n")
        parts.append(msg.content)
        parts.append("")

    # 跨轮追踪提示
    if len(opponent_msgs) > 1:
        parts.append("---")
        parts.append(
            "⚠️ **跨轮追踪提示**：上方是对方在**所有轮次**中的完整论点链。你必须：\n"
            "1. 逐条回应对方每一轮的核心指控，尤其是前几轮中你尚未充分回应的论点\n"
            "2. 揭露对方在论点演进中的矛盾或退让（如对方第1轮强调X，第2轮却回避X转而谈Y）\n"
            "3. 如果对方某一论点被你摧毁后未再提起，明确指出其已默认放弃该论点\n"
        )

    return "\n".join(parts)


def _round_label(round_num: int) -> str:
    """返回轮次的描述标签。"""
    labels = {1: "初始立论", 2: "交叉质询", 3: "总结陈词"}
    return labels.get(round_num, f"第{round_num}轮")


# ── 数据引用硬性要求（注入任务指令）───────────────────────────────

_DATA_CITATION_REQUIREMENT = """## 数据引用硬性要求（必须遵守）

你必须在发言中引用**至少3个**来自上方财务数据的具体数值。
未引用具体数据的论证将被 CIO 降权评分。

注意时间维度标注：数据已按 [LTM/TTM]、[MRQ]、[NTM/Forward] 分类。
LTM/TTM = 历史已审计数据，MRQ = 最近季度趋势，NTM/Forward = 未来预测。
不同时间维度的数据不可直接对比或混合计算。

正确引用示例：
- "当前PE为35.2倍（见上方[LTM/TTM 估值倍数]），但..."
- "财务数据显示ROE仅16.8%（见上方[LTM/TTM 盈利能力]），远低于行业水平"
- "分析师预估下一财年EPS为$6.50（见上方[NTM/Forward 预期差]），对应远期PE仅28.6倍"
- "营收增长率已从35%骤降至2.5%（见上方[MRQ 增长指标]），趋势堪忧"

错误示例（将被扣分）：
- "估值偏高" — 未引用具体数值
- "盈利能力不错" — 模糊描述，无数据支撑
- 将 LTM 数据当作 NTM 数据引用 — 时间维度混淆
"""


def build_bull_context(state: DebateState) -> str:
    """为红军多头构建本轮上下文。

    切片策略：
        - 财务数据 (完整)
        - 蓝军空头全部轮次发言 (完整原话，跨轮记忆)
        - 数据引用硬性要求
        - 本轮任务指令

    Args:
        state: 当前辩论状态。

    Returns:
        组装好的 Prompt 字符串。
    """
    round_num: int = state["current_round"]
    financial_data: str = state["financial_data"]
    messages: list[DebateMessage] = state.get("messages", [])

    parts: list[str] = []

    # ── 1. 财务数据（完整）──
    parts.append("## 📊 财务数据（红蓝军共享事实依据）\n")
    parts.append(financial_data)
    parts.append("\n" + "─" * 50 + "\n")

    # ── 2. 对手全部论点演进轨迹（跨轮记忆）──
    bear_timeline = build_opponent_timeline(messages, opponent="bear")
    parts.append(bear_timeline)
    parts.append("\n" + "─" * 50 + "\n")

    # ── 3. 数据引用硬性要求 ──
    parts.append(_DATA_CITATION_REQUIREMENT)
    parts.append("\n" + "─" * 50 + "\n")

    # ── 4. 本轮任务指令 ──
    parts.append(f"## ⚔️ 第 {round_num} 轮任务\n")
    if round_num == 1:
        parts.append(
            "建立完整的多头投资逻辑。基于以上财务数据，"
            "给出目标价预估（必须是具体数字如 $XXX.XX）。"
            "预判空头可能的攻击点并提前防御。"
        )
    elif round_num == 2:
        parts.append(
            "**交叉质询模式**：逐条引用上方蓝军空头的每个论点（用 > 引用原话），"
            "用数据和逻辑逐一毁灭性反驳。必须揭露对方的选择性失明和数据误读。"
            "语气必须攻击性。"
        )
    elif round_num >= 3:
        parts.append(
            "**总结陈词 + 致命一击**：利用上方完整的论点演进轨迹，"
            "指出空头逻辑中仍然站不住脚的核心谬误。"
            "揭示对方在论点演进中的自相矛盾（如第1轮强调X被摧毁后，"
            "第2轮转而谈Y回避X）。"
            "总结最有力的三个多头论点，给出最终目标价（具体数字），"
            "明确投资建议（📈 重仓买入 / 买入 / 观望偏多）。"
        )

    return "\n".join(parts)


def build_bear_context(state: DebateState) -> str:
    """为蓝军空头构建本轮上下文。

    切片策略：
        - 财务数据 (完整)
        - 红军多头全部轮次发言 (完整原话，跨轮记忆)
        - 数据引用硬性要求
        - 本轮任务指令

    Args:
        state: 当前辩论状态。

    Returns:
        组装好的 Prompt 字符串。
    """
    round_num: int = state["current_round"]
    financial_data: str = state["financial_data"]
    messages: list[DebateMessage] = state.get("messages", [])

    parts: list[str] = []

    # ── 1. 财务数据（完整）──
    parts.append("## 📊 财务数据（红蓝军共享事实依据）\n")
    parts.append(financial_data)
    parts.append("\n" + "─" * 50 + "\n")

    # ── 2. 对手全部论点演进轨迹（跨轮记忆）──
    bull_timeline = build_opponent_timeline(messages, opponent="bull")
    parts.append(bull_timeline)
    parts.append("\n" + "─" * 50 + "\n")

    # ── 3. 数据引用硬性要求 ──
    parts.append(_DATA_CITATION_REQUIREMENT)
    parts.append("\n" + "─" * 50 + "\n")

    # ── 4. 本轮任务指令 ──
    parts.append(f"## ⚔️ 第 {round_num} 轮任务\n")
    if round_num == 1:
        parts.append(
            "建立完整的做空逻辑。基于以上财务数据，"
            "给出合理目标价（必须是具体数字如 $XXX.XX）。"
            "预判多头可能的辩护策略并提前准备反驳。"
        )
    elif round_num == 2:
        parts.append(
            "**交叉质询模式**：逐条引用上方红军多头的每个论点（用 > 引用原话），"
            "用数据和逻辑逐一毁灭性反驳。必须揭露对方的过度乐观和选择性失明。"
            "语气必须锋利不留情面。"
        )
    elif round_num >= 3:
        parts.append(
            "**总结陈词 + 致命一击**：利用上方完整的论点演进轨迹，"
            "指出多头逻辑中逻辑不自洽或刻意回避的核心问题。"
            "揭示对方在论点演进中的自相矛盾（如第1轮强调X被摧毁后，"
            "第2轮转而谈Y回避X）。"
            "总结最有力的三个空头论点，给出最终目标价（具体数字），"
            "明确做空建议（📉 重仓做空 / 做空 / 减持）。"
            "用一句话总结多头的分析为何是'散户思维'。"
        )

    return "\n".join(parts)


def build_cio_context(state: DebateState) -> str:
    """为 CIO 裁判构建完整辩论上下文。

    CIO 需要看到全部辩论记录，因为其任务是审判整体辩论质量。
    但同样不压缩，每条发言用 XML 标签物理隔离。

    Args:
        state: 辩论完成后的状态。

    Returns:
        组装好的 CIO Prompt 字符串。
    """
    messages: list[DebateMessage] = state.get("messages", [])

    parts: list[str] = []
    parts.append("# 📜 红蓝军辩论完整记录（原文，无压缩）\n")

    bull_speeches: list[str] = []
    bear_speeches: list[str] = []

    for msg in messages:
        if msg.speaker == "bull":
            bull_speeches.append(
                f"### 🐂 红军多头 — 第 {msg.round} 轮\n\n{msg.content}"
            )
        else:
            bear_speeches.append(
                f"### 🐻 蓝军空头 — 第 {msg.round} 轮\n\n{msg.content}"
            )

    parts.append("## 红军多头 (Bull) 各轮论点\n")
    for speech in bull_speeches:
        parts.append(speech)
        parts.append("")

    parts.append("## 蓝军空头 (Bear) 各轮论点\n")
    for speech in bear_speeches:
        parts.append(speech)
        parts.append("")

    parts.append(
        "\n---\n"
        "## 🎯 裁判任务\n\n"
        "请严格按系统提示中的评分标准和纯 JSON 格式进行裁判。\n"
        "输出纯 JSON，不要用 ```json 包裹。\n"
    )

    return "\n".join(parts)
