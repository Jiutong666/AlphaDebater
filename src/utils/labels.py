"""
Centralised display labels for i18n readiness.

All Chinese UI strings used across the project are defined here.
When internationalisation is needed, replace this module with gettext / babel.
"""

from __future__ import annotations

# ── Temporal dimension labels ─────────────────────────────────────

LTM_LABEL = "[LTM/TTM — 过去12个月已审计/已报告数据]"
MRQ_LABEL = "[MRQ — 最近季度同比数据 (趋势检测，不可线性外推)]"
NTM_LABEL = "[NTM/Forward — 未来12个月预测数据 (仅此维度可用于推导目标价)]"

# ── Agent role labels ─────────────────────────────────────────────

BULL_DISPLAY_NAME = "Bull / 红军多头"
BULL_EMOJI = "🐂"
BEAR_DISPLAY_NAME = "Bear / 蓝军空头"
BEAR_EMOJI = "🐻"

# ── Temporal discipline footer ────────────────────────────────────

TEMPORAL_DISCIPLINE_WARNING = [
    "⚠️ 时间维度纪律 (Temporal Discipline):",
    "  - [LTM/TTM] = 过去12个月已审计数据 → 用于历史估值、盈利能力、财务健康",
    "  - [MRQ] = 最近季度同比 → 用于趋势检测、转折点识别",
    "  - [NTM/Forward] = 未来12个月预测 → 仅此维度可用于推导目标价",
    "  - 禁止跨维度混合计算 (如用 LTM 增长率论证 NTM 估值倍数)",
]

# ── Data citation requirement ─────────────────────────────────────

DATA_CITATION_REQUIREMENT = """## 数据引用硬性要求（必须遵守）

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

# ── Round labels ──────────────────────────────────────────────────

ROUND_LABELS: dict[int, str] = {
    1: "初始立论",
    2: "交叉质询",
    3: "总结陈词",
}
