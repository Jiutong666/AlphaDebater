"""
Anti-concession module — prevents RLHF alignment from polluting adversarial debate.

Three layers of defence:
    1. Post-output detection — scan for concession language, trigger forced retry
    2. Assistant prefill — lock the opening tone of every round
    3. Adversarial few-shot examples — show the model how to fight
"""

from __future__ import annotations

from typing import Literal

import re

# ── Concession detection rule base ────────────────────────────────

_CONCESSION_PATTERNS: list[tuple[str, str]] = [
    # Chinese concession patterns (relaxed — only flag actual surrender, not debate acknowledgement)
    (r"(我(们)?)?(同意|赞同|认可).{0,5}(对方|红军|蓝军|多头|空头)的?(结论|判断|逻辑|观点)", "明确同意对方结论"),
    (r"不可否认.{0,10}(对方|红军|蓝军|多头|空头)", "不可否认型妥协"),
    (r"双方.{0,15}(都有道理|各有道理|均合理|都对)", "各打五十大板"),
    (r"(可能|或许|也许|大概).{0,10}(是我|我们).{0,10}(错|误)", "自我怀疑"),
    (r"(值得|需要).{0,10}(反思|重新考虑|重新审视).{0,10}(自己|我方|我们)", "自我反思型妥协"),
    (r"(这个|这点|此处).{0,5}(我|我们).{0,5}(承认|让步|退一步)", "公开让步"),
    # English concession patterns
    (r"I\s*(agree|concede|admit).{0,20}(opponent|bull|bear|other side)\s*(is right|has a point|conclusion|logic)", "明确同意对方 (EN)"),
    (r"(both sides|both arguments).{0,10}(have merit|are valid|are reasonable)", "双方都合理 (EN)"),
    (r"perhaps.{0,10}(I|we|my).{0,10}(wrong|mistaken)", "自我怀疑 (EN)"),
]

_RETRY_REINFORCEMENT_ZH = """
⚠️ **警告：检测到过度让步！**

你上一轮回复中出现了以下让步信号：
> "{matched_text}"

在 AlphaDebater 中，你不应该直接同意对方的结论或承认自己错误。
记住：即使对方引用的数据正确，也不代表其结论正确。

请重新生成你的论点。要求：
1. 逐条引用对方论点，攻击其推理链而非数据本身
2. 质疑对方的解读方式、权重分配和时间维度选择
3. 保持攻击性但不伪造数据——数据是共享的，推理是争夺的战场
"""

_RETRY_REINFORCEMENT_EN = """
⚠️ **WARNING: Excessive concession detected!**

Your previous response contained the following concession signal:
> "{matched_text}"

In AlphaDebater, you should not directly agree with the opponent's conclusions.
Remember: correct data does not equal correct conclusions.

Regenerate your argument. Requirements:
1. Quote each opponent claim and attack their reasoning chain, not the data itself
2. Challenge their interpretation, weighting, and temporal dimension choices
3. Stay aggressive but don't fabricate — data is shared ground, reasoning is the battlefield
"""


# ── Assistant prefill — lock opening tone ─────────────────────────

BULL_PREFILLS: dict[int, str] = {
    1: "## 🐂 红军多头 — 第1轮：多头核心逻辑\n\n基于提供的财务数据和工具计算结果，以下是我的多头投资逻辑：\n\n",
    2: "## 🐂 红军多头 — 第2轮：交叉质询\n\n以下我将逐条回应蓝军空头的第1轮论点：\n\n",
    3: "## 🐂 红军多头 — 第3轮：总结陈词\n\n综合三轮辩论，以下是我的最终评估（目标价均经工具计算验证）：\n\n",
}

BEAR_PREFILLS: dict[int, str] = {
    1: "## 🐻 蓝军空头 — 第1轮：做空核心逻辑\n\n基于提供的财务数据和工具计算结果，以下是我的做空逻辑：\n\n",
    2: "## 🐻 蓝军空头 — 第2轮：交叉质询\n\n以下我将逐条回应红军多头的第1轮论点：\n\n",
    3: "## 🐻 蓝军空头 — 第3轮：总结陈词\n\n综合三轮辩论，以下是我的最终评估（目标价均经工具计算验证）：\n\n",
}

# ── Adversarial few-shot examples ─────────────────────────────────

_BULL_FEWSHOT = """
## 辩论示例 (Few-Shot)

以下是你应该模仿的辩论风格——专业、锋利、基于数据：

**空头说**: "该股市盈率高达35倍，远超行业平均的22倍，估值明显泡沫化。"

**你的正确反驳**:

> "对方用市盈率35倍对比行业22倍来论证估值泡沫，这是粗糙的伪分析。
> **第一**，该公司的ROE是行业平均的2.3倍——高资本回报率理应享受估值溢价。
> PEG比率仅0.8，低于行业1.5的中位数，对方的'估值泡沫论'在成长性调整后站不住脚。
> **第二**，对方选择性地忽略了远期PE仅18倍这一关键数据——前瞻估值已低于行业平均。
> 仅用trailing PE而忽略forward PE，这种选择性数据引用是分析的基本缺陷。"

**空头说**: "营收增速从去年的35%下滑到今年的18%，增长正在衰竭。"

**你的正确反驳**:

> "营收增速从35%降至18%？这里的问题在于基数效应被完全忽略。
> 一家千亿市值的公司在高基数下维持18%的增长，恰恰证明了其业务的韧性。
> 此外，对方完全回避了盈利增长率从12%上升至28%这一事实——
> 利润率正在急剧扩张，这才是价值投资者应该关注的核心指标。
> 仅关注营收增速而忽略盈利质量的跃升，这种分析框架是不完整的。"

---

## 数据诚信警示

以下是你必须避免的行为模式：

**❌ 伪造数据**
> "基于当前价格$245.83和EPS $22.69，我计算出前瞻PE为10.8倍。"
→ 财务数据中明确给出 forwardPE = 21.91，不能"自行计算"一个不同的PE。
直接引用数据中的 forwardPE，围绕它做多空解读。

**❌ 方向性颠倒**
> "盈利从-53%即将翻倍至+106%。"
→ 数据写盈利增长率为 -53%，没有"翻倍"或"反转"的依据。忠于数据的方向和量级。

**❌ 编造不存在的数字**
> "公司持有现金$45B，完全有能力回购$10B股票。"
→ 如果总现金字段是 N/A 或其他数值，就不能引用 $45B。只引用数据中实际存在的数值。

**核心原则：全力构建多头逻辑，但每个数字必须在财务数据中有据可查。**
"""

_BEAR_FEWSHOT = """
## 辩论示例 (Few-Shot)

以下是你应该模仿的辩论风格——专业、锋利、基于数据：

**多头说**: "该公司PEG仅0.8，远低于行业1.5，增长潜力被严重低估。"

**你的正确反驳**:

> "对方引用PEG 0.8来论证低估，但这里存在严重的分析缺陷。
> **第一**，PEG中的'G'使用的是历史增长率，而公司最新的forward guidance
> 已将下季度增速预期下调至个位数——用过去的增长线性外推未来，忽略了基本面的变化。
> **第二**，对方引用的'行业PEG 1.5'包含了3家负增长公司拉高了均值。
> 剔除异常值后，可比公司的PEG中位数是0.75。该公司的PEG实际上高于同行中位数。"

**多头说**: "公司自由现金流充裕，完全有能力回购股票支撑股价。"

**你的正确反驳**:

> "自由现金流'充裕'这一说法需要更仔细的审视。从现金流量表来看，经营性现金流的增长
> 大部分来自应付账款的扩张——公司在拉长供应商付款周期来粉饰FCF。
> 这不是健康的现金流质量。同时，公司有50亿债务在未来12个月到期，
> 偿债压力将对可用于回购的自由现金形成实质性约束。
> 仅看FCF总量而不分析其构成和债务背景，会得出误导性的结论。"

---

## 数据诚信警示

以下是你必须避免的行为模式：

**❌ 伪造数据**
> "基于当前价格和EPS，合理PE应为12倍。"
→ 财务数据中明确给出了 trailingPE 和 forwardPE。直接引用数据中的 PE 值，围绕它做空头解读。

**❌ 编造不存在的财务隐患**
> "公司表外负债高达$30B，实际杠杆率是财报显示的3倍。"
→ 数据中只写了 totalDebt，就引用 totalDebt。不要发明数据中没有的负债。

**❌ 将数据缺失解读为阴谋**
> "公司拒绝披露研发费用细分，恰好说明技术护城河被侵蚀。"
→ N/A 只代表数据源未返回该字段，不能过度解读。

**核心原则：全力构建做空逻辑，但每个数字必须在财务数据中有据可查。**
"""


# ── Public API ────────────────────────────────────────────────────

def detect_concession(response: str) -> list[tuple[str, str]]:
    """Scan response for concession language signals.

    Returns:
        List of (matched_text, pattern_label). Empty list = clean.
    """
    hits: list[tuple[str, str]] = []
    for pattern, label in _CONCESSION_PATTERNS:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            hits.append((match.group(0), label))
    return hits


def build_retry_prompt(
    hits: list[tuple[str, str]],
    lang: Literal["zh", "en"] = "zh",
) -> str:
    """Build retry prompt from detected concession hits."""
    matched_texts = "; ".join([f'"{text}" ({label})' for text, label in hits])
    template = _RETRY_REINFORCEMENT_ZH if lang == "zh" else _RETRY_REINFORCEMENT_EN
    return template.format(matched_text=matched_texts)


def get_prefill(role: Literal["bull", "bear"], round_num: int) -> str:
    """Get assistant prefill for the given role and round."""
    prefill_map = BULL_PREFILLS if role == "bull" else BEAR_PREFILLS
    return prefill_map.get(round_num, "")


def get_fewshot(role: Literal["bull", "bear"]) -> str:
    """Get adversarial few-shot examples for the given role."""
    return _BULL_FEWSHOT if role == "bull" else _BEAR_FEWSHOT
