"""
Neuro-Symbolic Calculation Tools — 确定性财务计算函数。

提供纯 Python 实现的财务计算工具，通过 OpenAI 兼容的 Tool Calling
接口暴露给 LLM。LLM 负责从数据中提取参数（神经），Python 负责计算（符号），
消除算术幻觉。

工具设计原则：
    - 每个工具返回 dict（公式 + 结果），供 LLM 包装为叙事话术
    - 所有输入参数由 LLM 从财务数据中提取并传入
    - 工具不访问任何外部状态，纯函数，可独立单元测试
"""

from __future__ import annotations

from typing import Any, Callable


# ── 计算函数 ────────────────────────────────────────────────────

def calculate_target_price(eps: float, pe_multiple: float) -> dict[str, Any]:
    """目标价 = 每股收益 × 市盈率倍数。

    Args:
        eps:         每股收益 (TTM 或 Forward EPS)。
        pe_multiple: 目标市盈率倍数 (如 25 表示 25x)。

    Returns:
        {"target_price": float, "formula": str}
    """
    target = round(eps * pe_multiple, 2)
    return {
        "target_price": target,
        "formula": f"EPS ${eps:.2f} × PE {pe_multiple:.1f}x = ${target:.2f}",
    }


def calculate_upside_downside(
    current_price: float,
    target_price: float,
) -> dict[str, Any]:
    """涨跌幅 = (目标价 - 当前价格) / 当前价格 × 100%。

    正值代表上涨空间 (upside)，负值代表下跌空间 (downside)。

    Args:
        current_price: 当前市场价格。
        target_price:  目标价格。

    Returns:
        {"percentage": float, "direction": "上涨" | "下跌", "formula": str}
    """
    if current_price <= 0:
        return {"percentage": None, "direction": "N/A", "error": "当前价格必须大于0"}
    pct = round((target_price - current_price) / current_price * 100, 2)
    direction = "上涨" if pct >= 0 else "下跌"
    return {
        "percentage": pct,
        "direction": direction,
        "formula": (
            f"(${target_price:.2f} - ${current_price:.2f}) / "
            f"${current_price:.2f} × 100% = {pct:+.2f}%"
        ),
    }


def calculate_peg_ratio(
    pe_ratio: float,
    earnings_growth_pct: float,
) -> dict[str, Any]:
    """DEPRECATED — 系统已预计算 PEG，此函数不再暴露为工具。

    PEG = Forward PE / 盈利增长率(%)，公式已锁死在数据层。
    Agent 必须直接引用数据中预计算的 PEG 值，禁止自行计算。

    保留此函数仅为测试兼容性。
    """
    if earnings_growth_pct == 0:
        return {
            "peg_ratio": None,
            "error": "盈利增长率为0%，PEG无定义（除数为零）。",
        }
    peg = round(pe_ratio / earnings_growth_pct, 2)
    return {
        "peg_ratio": peg,
        "pe_used": pe_ratio,
        "growth_rate_used": earnings_growth_pct,
        "formula": f"PE {pe_ratio:.2f}x / 增长率 {earnings_growth_pct:.1f}% = PEG {peg:.2f}",
    }


def calculate_growth_rate(
    past_value: float,
    current_value: float,
    periods: int = 1,
) -> dict[str, Any]:
    """复合年增长率 (CAGR) = ((现值 / 过去值) ^ (1/期数) - 1) × 100%。

    适用于计算营收、盈利、EPS 等指标的多年复合增长率。

    Args:
        past_value:    过去的值。
        current_value: 当前的值。
        periods:       时间跨度 (年数)，默认 1 年。

    Returns:
        {"growth_rate_pct": float, "past": float, "current": float, "periods": int, "formula": str}
    """
    if past_value == 0:
        return {
            "growth_rate_pct": None,
            "error": "过去值为0，无法计算增长率。",
        }
    if past_value < 0 and current_value > 0:
        return {
            "growth_rate_pct": None,
            "error": "正负号变化使 CAGR 无定义。",
        }
    ratio = current_value / past_value
    if ratio < 0:
        return {
            "growth_rate_pct": None,
            "error": "负增长比率使 CAGR 无定义。",
        }
    cagr = round((ratio ** (1.0 / periods) - 1) * 100, 2)
    return {
        "growth_rate_pct": cagr,
        "past": past_value,
        "current": current_value,
        "periods": periods,
        "formula": (
            f"((${current_value:.2f} / ${past_value:.2f})^(1/{periods}) - 1) "
            f"× 100% = {cagr:+.2f}%"
        ),
    }


def calculate_price_from_ps(
    revenue_per_share: float,
    ps_multiple: float,
) -> dict[str, Any]:
    """目标价 = 每股营收 × 市销率倍数。

    适用于评估营收驱动型公司的估值 (如早期成长股、SaaS)。

    Args:
        revenue_per_share: 每股营收 (TTM 或 Forward)。
        ps_multiple:       目标市销率倍数。

    Returns:
        {"target_price": float, "formula": str}
    """
    target = round(revenue_per_share * ps_multiple, 2)
    return {
        "target_price": target,
        "formula": (
            f"每股营收 ${revenue_per_share:.2f} × PS {ps_multiple:.1f}x = "
            f"${target:.2f}"
        ),
    }


# ── 工具注册表 ──────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "calculate_target_price": calculate_target_price,
    "calculate_upside_downside": calculate_upside_downside,
    "calculate_growth_rate": calculate_growth_rate,
    "calculate_price_from_ps": calculate_price_from_ps,
}

# ── OpenAI 兼容的工具定义 ───────────────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate_target_price",
            "description": (
                "用市盈率(PE)估值法计算目标价。目标价 = 每股收益(EPS) × 市盈率(PE)。"
                "当你需要给出目标价时，必须调用此工具进行计算，禁止心算。"
                "EPS 使用财务数据中的 trailingEps 或 Forward EPS，PE 使用你认为合理的倍数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "eps": {
                        "type": "number",
                        "description": "每股收益 (EPS)，从财务数据中获取。",
                    },
                    "pe_multiple": {
                        "type": "number",
                        "description": "目标市盈率倍数 (如 25 表示 25x earnings)。",
                    },
                },
                "required": ["eps", "pe_multiple"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_upside_downside",
            "description": (
                "计算从当前价格到目标价格的涨跌幅百分比。"
                "正值 = 上涨空间，负值 = 下跌空间。"
                "当你需要给出涨跌幅时，必须调用此工具，禁止心算百分比。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_price": {
                        "type": "number",
                        "description": "当前市场价格，从财务数据中获取。",
                    },
                    "target_price": {
                        "type": "number",
                        "description": "目标价格 (可来自 calculate_target_price 的结果)。",
                    },
                },
                "required": ["current_price", "target_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_growth_rate",
            "description": (
                "计算复合年增长率 (CAGR) = ((现值/过去值)^(1/期数) - 1) × 100%。"
                "适用于多年营收/盈利/EPS 增长率的精确计算。"
                "禁止心算复合增长率，必须调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "past_value": {
                        "type": "number",
                        "description": "过去的值 (如 3 年前的营收)。",
                    },
                    "current_value": {
                        "type": "number",
                        "description": "当前的值 (如当前营收)。",
                    },
                    "periods": {
                        "type": "integer",
                        "description": "时间跨度 (年数)，默认 1。",
                        "default": 1,
                    },
                },
                "required": ["past_value", "current_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_price_from_ps",
            "description": (
                "用市销率(PS)估值法计算目标价。目标价 = 每股营收 × 市销率(PS)。"
                "适用于营收驱动型公司的估值。"
                "当你需要基于 PS 给出目标价时，必须调用此工具，禁止心算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "revenue_per_share": {
                        "type": "number",
                        "description": "每股营收，从财务数据中计算或获取。",
                    },
                    "ps_multiple": {
                        "type": "number",
                        "description": "目标市销率倍数。",
                    },
                },
                "required": ["revenue_per_share", "ps_multiple"],
            },
        },
    },
]
