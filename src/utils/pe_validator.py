"""
PE validator — distinguishes Fact Falsification from Valuation Assumption.

CRITICAL DISTINCTION (金融基本概念):
    - Fact Falsification: Claiming a data point IS in the financial data when it's not.
      Example: "数据显示 forwardPE = 10.8" when data says 21.91. THIS IS FABRICATION.
    - Valuation Assumption: Proposing a PE multiple as an investment opinion.
      Example: "基于 ROE 111%, 我认为应给予 20x 合理PE". THIS IS LEGITIMATE ANALYSIS.

The validator ONLY catches Fact Falsification. Valuation Assumptions are free speech.
PE multiples in calculate_target_price tool calls are NEVER validated — they are
inherently opinions, and the tool exists precisely to let agents express them.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_pe_anchors(financial_data: str) -> dict[str, float]:
    """Extract known PE-related values from financial data text.

    Returns:
        Dict mapping anchor name -> value, e.g. {"trailingPE": 12.67, "forwardPE": 21.91}
    """
    anchors: dict[str, float] = {}

    patterns = [
        (r'trailingPE\s*[=:：]\s*(\d+(?:\.\d+)?)', "trailingPE"),
        (r'forwardPE\s*[=:：]\s*(\d+(?:\.\d+)?)', "forwardPE"),
        (r'[Pp][Ee]\s*[Rr]atio\s*[=:：]\s*(\d+(?:\.\d+)?)', "peRatio"),
        (r'[Pp][Ee][Gg]\s*[=:：]\s*(\d+(?:\.\d+)?)', "pegRatio"),
        (r'Forward PE[^0-9]*(\d+(?:\.\d+)?)', "forwardPE"),
        (r'Trailing PE[^0-9]*(\d+(?:\.\d+)?)', "trailingPE"),
        (r'PEG[^0-9]*(\d+(?:\.\d+)?)', "pegRatio"),
        (r'(?:^|\n)\s*PE[^G].*?(\d+(?:\.\d+)?)', "pe"),
    ]

    for pattern, name in patterns:
        for match in re.finditer(pattern, financial_data):
            try:
                val = float(match.group(1))
                if 0.5 < val < 1000:
                    if name not in anchors:
                        anchors[name] = val
            except (ValueError, IndexError):
                pass

    if "trailingPE" in anchors or "forwardPE" in anchors:
        anchors.pop("pe", None)

    return anchors


# ── Fact-claim language patterns ───────────────────────────────────
# These phrases indicate the agent is claiming a number IS in the data,
# not proposing it as an opinion. If the number doesn't match anchors,
# it's fabrication.

_FACT_CLAIM_PATTERNS = [
    r'数据显示.{0,10}(?:PE|市盈率).{0,5}[为是=]',
    r'数据中.{0,10}(?:PE|市盈率).{0,5}[为是=]',
    r'(?:forwardPE|trailingPE|Forward PE|Trailing PE)\s*[=:：为]\s*',
    r'(?:forward\s*PE|trailing\s*PE)\s*[=:：为]\s*',
    r'当前(?:PE|市盈率)[为是=]',
    r'(?:PE|市盈率)\s*[=:：]\s*\d',
    r'实际(?:PE|市盈率)[为是=仅]',
    r'财务数据(?:显示|表明).{0,10}(?:PE|市盈率).{0,5}[为是=]',
]

# ── Opinion/assumption language patterns ───────────────────────────
# These phrases indicate the agent is expressing an opinion about what
# PE SHOULD be, not claiming it IS in the data. ALWAYS legitimate.

_OPINION_PATTERNS = [
    r'(?:我认为|我主张|我方认为|我们给予|应给予|应该给予|应当给予)',
    r'(?:合理|公允|适当|目标)(?:PE|市盈率|估值)',
    r'(?:给予|赋予|适用)(?:\d+(?:\.\d+)?)\s*(?:x|倍)\s*(?:PE|市盈率|估值)',
    r'(?:如果|假设|若|基于.{0,10}给予).{0,15}(?:PE|市盈率)',
    r'(?:场景|压力测试|敏感性|敏感性分析)',
    r'(?:按|按照|采用)\s*(?:\d+(?:\.\d+)?)\s*(?:x|倍)',
    r'(?:应[享该]有|值得)(?:.{0,10}(?:估值|溢价))',
    r'(?:I\s*(?:believe|think|propose|argue|assign|apply))',
    r'(?:deserves|warrants|justifies|should trade at)',
    r'(?:reasonable|fair|appropriate|target)\s*(?:PE|multiple|valuation)',
]


def detect_fact_fabrication(
    response: str,
    anchors: dict[str, float],
    tolerance: float = 0.10,
) -> list[tuple[str, str, float]]:
    """Detect when an agent falsely claims a PE value IS in the financial data.

    ONLY flags cases where the agent presents a number as a DATA FACT
    (not an opinion) and that number doesn't match known anchors.

    Valuation assumptions (e.g. "我认为应给予 20x PE") are NEVER flagged.

    Args:
        response: Agent's full response text.
        anchors: Known PE values from financial data.
        tolerance: Tight tolerance for fact claims (default 10%).

    Returns:
        List of (matched_text, context, value) for fabricated fact claims.
    """
    if not anchors:
        return []

    fabricated: list[tuple[str, str, float]] = []

    # Find all PE-like numbers: "Nx", "N倍", "PE = N", "PE: N"
    pe_number_patterns = [
        (re.compile(r'(\d+(?:\.\d+)?)\s*(?:x|倍)'), 1),
        (re.compile(r'(?:PE|P/E|市盈率)\s*[=:：]\s*(\d+(?:\.\d+)?)'), 1),
    ]

    for pattern, group_idx in pe_number_patterns:
        for m_match in re.finditer(pattern, response):
            pe_value = float(m_match.group(group_idx))
            if pe_value < 3 or pe_value > 500:
                continue

            # Check if this is an opinion statement — if so, skip
            # Check both text before the match AND a window including the match
            before_start = max(0, m_match.start() - 60)
            before_text = response[before_start:m_match.start()]
            ctx_start = max(0, m_match.start() - 20)
            ctx_end = min(len(response), m_match.end() + 5)
            ctx_text = response[ctx_start:ctx_end]

            is_opinion = any(
                re.search(pat, before_text, re.IGNORECASE)
                or re.search(pat, ctx_text, re.IGNORECASE)
                for pat in _OPINION_PATTERNS
            )
            if is_opinion:
                continue

            # Check if this is a fact claim — only flag if the agent
            # asserts this number IS in the data and it's not
            is_fact_claim = any(
                re.search(pat, before_text, re.IGNORECASE)
                for pat in _FACT_CLAIM_PATTERNS
            )

            if not is_fact_claim:
                # Also check if the match + surrounding context forms a fact claim
                # like "forwardPE = 20" or "trailingPE: 20"
                # Include the match itself (not just text before) to correctly
                # handle cases where "PE = N" is part of "forwardPE = N"
                ctx_start = max(0, m_match.start() - 20)
                ctx_end = min(len(response), m_match.end())
                context_with_match = response[ctx_start:ctx_end]
                if re.search(
                    r'(?:forwardPE|trailingPE|forward\s*PE|trailing\s*PE)\s*[=:：]\s*\d',
                    context_with_match,
                    re.IGNORECASE,
                ):
                    is_fact_claim = True

            if not is_fact_claim:
                continue

            # It's a fact claim — verify against anchors
            if not _is_anchored(pe_value, anchors, tolerance):
                ctx_start = max(0, m_match.start() - 25)
                ctx_end = min(len(response), m_match.end() + 25)
                ctx = response[ctx_start:ctx_end].strip().replace('\n', ' ')
                fabricated.append((m_match.group(0), ctx, pe_value))

    return fabricated


def validate_tool_call_pe(
    tool_log: list[dict[str, Any]],
    anchors: dict[str, float],
    tolerance: float = 0.15,
) -> list[tuple[float, float | None]]:
    """DEPRECATED — PE multiples in tool calls are opinions, not facts.

    This function now always returns an empty list. The calculate_target_price
    tool exists precisely to let agents express their valuation opinions.
    Validating pe_multiple against data anchors is a category error:
    PE multiples are assumptions, not factual data points.
    """
    return []


def _is_anchored(
    value: float,
    anchors: dict[str, float],
    tolerance: float,
) -> bool:
    """Check if value is within tolerance of any known anchor."""
    for anchor_val in anchors.values():
        if anchor_val == 0:
            continue
        if abs(value - anchor_val) / anchor_val <= tolerance:
            return True
    return False


def build_pe_retry_prompt(
    text_fabricated: list[tuple[str, str, float]],
    tool_invalid: list[tuple[float, float | None]],
    anchors: dict[str, float],
) -> str:
    """Build retry prompt explaining the Fact vs Assumption boundary.

    The tool_invalid parameter is accepted for backwards compatibility
    but ignored (tool PE validation is deprecated).
    """
    parts: list[str] = []

    parts.append(
        "⚠️ **事实性校验失败 — 检测到将估值假设错误陈述为数据事实！**\n\n"
        "**关键区分 (Fact vs. Assumption):**\n"
        "- 🔴 事实捏造 (禁止): \"数据显示 forwardPE = 10.8x\" 但数据中 forwardPE = 21.91\n"
        "- 🟢 估值假设 (合法): \"基于 ROE 111%，我认为应给予 20x 合理PE\"\n\n"
    )

    if text_fabricated:
        parts.append("**你的回复中将以下 PE 倍数陈述为数据事实，但它们在数据中不存在：**\n")
        for matched_text, ctx, value in text_fabricated[:5]:
            parts.append(
                f'- **{matched_text}** ({value:.1f}x) '
                f'(上下文: "...{ctx}...")\n'
            )

    # List valid anchors
    anchor_items = [f"  - **{name}** = {val:.2f}x" for name, val in sorted(anchors.items())]
    parts.append(
        "\n**财务数据中实际存在的 PE 值：**\n"
        + "\n".join(anchor_items)
        + "\n"
    )

    parts.append(
        "\n请重新生成你的论点。要求：\n"
        "1. 引用数据中实际存在的 PE 值时，**必须使用上述锚点值**，不得改写\n"
        "2. 如果你想使用不同的 PE 倍数，请**明确标注为估值假设**\n"
        "   (如 \"基于 X 原因，我认为应给予 Yx PE\")——这完全合法\n"
        "3. 禁止说 \"数据显示 PE 为 X\" 当 X 不在上述锚点列表中\n"
        "4. 禁止说 \"forwardPE = X\" 当 X 与数据中的 forwardPE 不一致\n"
    )

    return "".join(parts)
