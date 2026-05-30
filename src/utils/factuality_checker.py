"""
Post-hoc factuality verification — prevents hallucinated numbers in agent responses.

Extracts every number from the response and checks whether it appears
in the financial data. Numbers that can't be matched trigger a retry.
"""

from __future__ import annotations

import re


def _extract_known_numbers(financial_data: str) -> set[float]:
    """Extract all known numeric values from financial data text."""
    known: set[float] = set()
    for match in re.finditer(r'(?:\$)?(\d+(?:\.\d+)?)', financial_data):
        try:
            val = float(match.group(1))
            if val > 0:
                known.add(val)
        except ValueError:
            pass
    return known


def _number_in_text(value: float, raw: str, text: str) -> bool:
    """Check if a number appears as an independent token in text (word-boundary match)."""
    escaped_raw = re.escape(raw)
    if re.search(r'(?<!\d)' + escaped_raw + r'(?!\d)', text):
        return True
    for fmt_str in (f"{value:.2f}", f"{value:.1f}", f"{value:.0f}"):
        escaped = re.escape(fmt_str)
        if re.search(r'(?<!\d)' + escaped + r'(?!\d)', text):
            return True
    if value == int(value):
        int_str = str(int(value))
        escaped_int = re.escape(int_str)
        if re.search(r'(?<!\d|\.)' + escaped_int + r'(?!\d|\.)', text):
            return True
    return False


def _approx_match(value: float, known: set[float], tolerance: float = 0.015) -> bool:
    """Check if value is within tolerance of any known value (allows minor rounding)."""
    for k in known:
        if k == 0:
            continue
        if abs(value - k) / k < tolerance:
            return True
    return False


def verify_numbers_in_response(
    response: str,
    financial_data: str,
) -> list[tuple[str, str]]:
    """Check that all numbers cited in the response exist in the financial data.

    Automatically skips:
        - Target prices / estimates (agent's own judgment)
        - Small integers <= 10 (round numbers, structural numbers)
        - Integer dollar amounts (likely target prices, not data citations)

    Returns:
        List of (raw_number, context) for suspicious numbers.
    """
    known = _extract_known_numbers(financial_data)
    if not known:
        return []

    suspicious: list[tuple[str, str]] = []

    for match in re.finditer(
        r'(?:\$)?\d+(?:\.\d+)?\s*(?:[%％]|倍|[亿万亿]|[TBMK])?',
        response,
    ):
        full_match = match.group(0).strip()
        num_match = re.search(r'(\d+(?:\.\d+)?)', full_match)
        if not num_match:
            continue
        value = float(num_match.group(1))

        # Skip small integers (round numbers, list indices)
        if value == int(value) and value <= 10:
            continue

        # Skip target prices / estimates (check preceding 5 chars only)
        before_start = max(0, match.start() - 5)
        before_text = response[before_start:match.start()]
        if re.search(
            r'目标价|target|预估|估计|推算|建议|买入|卖出|做空|做多|'
            r'重仓|减持|观望|持有|工具计算|经工具计算',
            before_text,
            re.IGNORECASE,
        ):
            continue

        # Skip integer dollar amounts likely to be target prices
        if full_match.startswith('$') and value == int(value):
            if not _number_in_text(value, full_match, financial_data):
                continue

        # Two-level matching
        if _number_in_text(value, full_match, financial_data):
            continue
        if _approx_match(value, known, tolerance=0.015):
            continue

        # Build context for display
        ctx_start = max(0, match.start() - 15)
        ctx_end = min(len(response), match.end() + 15)
        ctx = response[ctx_start:ctx_end].strip().replace('\n', ' ')
        suspicious.append((full_match, ctx))

    return suspicious


def build_factuality_retry_prompt(
    suspicious: list[tuple[str, str]],
) -> str:
    """Build a retry prompt listing the hallucinated numbers."""
    items: list[str] = []
    for value, context in suspicious[:5]:
        items.append(f'- **"{value}"** (上下文: "...{context}...")')

    item_block = "\n".join(items)

    return f"""⚠️ **事实性校验失败 — 检测到未在财务数据中出现的数值！**

你的回复中引用了以下无法在财务数据中找到的数值：

{item_block}

这些数值不属于提供的财务数据，属于**幻觉 (hallucination)**。

请重新生成你的论点。要求：
1. 仅引用上方**财务数据中实际存在**的数值
2. 如果某项数据在财务数据中显示为 N/A，请如实说明"该数据缺失"，不得编造
3. 引用数据时保持原始精度（如数据中 PE 为 56.72，请写 56.72 而非"约57"）
4. 如果需要推算目标价，请明确标注为"基于当前数据推算"并展示推算过程，而非当作已有事实数据直接引用"""
