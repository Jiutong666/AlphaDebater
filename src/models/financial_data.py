"""
Temporal Financial Data Models — 时间维度隔离。

将财务数据严格划分为三个不可跨越的维度：
    - LTM/TTM (过去12个月已审计数据): 历史估值、盈利能力、财务健康
    - MRQ (最近一个季度): 趋势拐点检测
    - NTM/Forward (未来12个月预测): 前瞻估值、目标价推导

Pydantic 模型 + from_flat_dict 工厂方法，将数据源的扁平字典
转换为时间维度隔离的结构化数据。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class LTMTTMData(BaseModel):
    """LTM/TTM — 过去12个月已审计/已报告数据 ('铁证')。

    用于评估历史盈利能力、现有估值水平、财务健康状况。
    禁止与 NTM/Forward 预测数据混合计算。
    """

    trailing_pe: Optional[float] = None
    trailing_eps: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_to_ebitda: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    gross_margin: Optional[float] = None
    net_profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    free_cashflow_per_share: Optional[float] = None
    free_cashflow_total: Optional[float] = None
    total_cash: Optional[float] = None
    total_debt: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None


class MRQData(BaseModel):
    """MRQ — 最近一个季度环比/同比数据 ('趋势拐点')。

    用于捕捉近期趋势变化（加速/减速）、转折点识别。
    单一季度数据可能包含季节性因素，不可线性外推。
    """

    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    earnings_quarterly_growth: Optional[float] = None


class NTMForwardData(BaseModel):
    """NTM/Forward — 未来12个月预测数据 ('预期')。

    仅此维度的数据可用于推导目标价和远期估值。
    分析师预测存在系统性乐观偏差，需与历史数据对比验证。
    """

    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    est_eps_next_year_avg: Optional[float] = None
    est_eps_next_year_high: Optional[float] = None
    est_eps_next_year_low: Optional[float] = None
    est_eps_next_quarter_avg: Optional[float] = None
    est_revenue_next_year_avg: Optional[float] = None
    analyst_target_mean: Optional[float] = None
    analyst_target_high: Optional[float] = None
    analyst_target_low: Optional[float] = None
    analyst_count: Optional[int] = None


class TemporalFinancialData(BaseModel):
    """三层时间维度严格隔离的财务数据容器。

    通过 from_flat_dict() 工厂方法从数据源的扁平字典构建。
    """

    ticker: str
    company_name: str = "N/A"
    industry: Optional[str] = None
    sector: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    data_source: str = "Unknown"

    ltm_ttm: LTMTTMData
    mrq: MRQData
    ntm_forward: NTMForwardData

    @classmethod
    def from_flat_dict(cls, info: dict[str, Any]) -> "TemporalFinancialData":
        """从数据源扁平字典构建时间维度隔离模型。

        自动将各字段归类到 LTM/MRQ/NTM 三个维度。
        无法归类的字段将被忽略（不影响辩论核心逻辑）。
        """
        ltm = LTMTTMData(
            trailing_pe=_float_or_none(info.get("trailingPE")),
            trailing_eps=_float_or_none(info.get("trailingEps")),
            price_to_book=_float_or_none(info.get("priceToBook")),
            price_to_sales=_float_or_none(info.get("priceToSalesTrailing12Months")),
            enterprise_to_ebitda=_float_or_none(info.get("enterpriseToEbitda")),
            return_on_equity=_float_or_none(info.get("returnOnEquity")),
            return_on_assets=_float_or_none(info.get("returnOnAssets")),
            gross_margin=_float_or_none(info.get("grossMargins")),
            net_profit_margin=_float_or_none(
                info.get("profitMargins") or info.get("netProfitMargin")
            ),
            debt_to_equity=_float_or_none(info.get("debtToEquity")),
            current_ratio=_float_or_none(info.get("currentRatio")),
            quick_ratio=_float_or_none(info.get("quickRatio")),
            free_cashflow_per_share=_float_or_none(info.get("freeCashflowPerShare") or info.get("freeCashflow")),
            free_cashflow_total=_float_or_none(info.get("freeCashflowTotal") or info.get("freeCashflow")),
            total_cash=_float_or_none(info.get("totalCash")),
            total_debt=_float_or_none(info.get("totalDebt")),
            dividend_yield=_float_or_none(info.get("dividendYield")),
            payout_ratio=_float_or_none(info.get("payoutRatio")),
        )

        mrq = MRQData(
            revenue_growth_yoy=_float_or_none(info.get("revenueGrowth")),
            earnings_growth_yoy=_float_or_none(info.get("earningsGrowth")),
            earnings_quarterly_growth=_float_or_none(info.get("earningsQuarterlyGrowth")),
        )

        ntm = NTMForwardData(
            forward_pe=_float_or_none(info.get("forwardPE")),
            peg_ratio=_float_or_none(info.get("pegRatio")),
            est_eps_next_year_avg=_float_or_none(info.get("estEpsNextY_avg")),
            est_eps_next_year_high=_float_or_none(info.get("estEpsNextY_high")),
            est_eps_next_year_low=_float_or_none(info.get("estEpsNextY_low")),
            est_eps_next_quarter_avg=_float_or_none(info.get("estEpsNextQ_avg")),
            est_revenue_next_year_avg=_float_or_none(info.get("estRevenueNextY_avg")),
            analyst_target_mean=_float_or_none(info.get("targetMeanPrice")),
            analyst_target_high=_float_or_none(info.get("targetHighPrice")),
            analyst_target_low=_float_or_none(info.get("targetLowPrice")),
            analyst_count=_int_or_none(info.get("numberOfAnalystOpinions")),
        )

        return cls(
            ticker=str(info.get("symbol", "")),
            company_name=str(info.get("shortName", "N/A")),
            industry=info.get("industry"),
            sector=info.get("sector"),
            current_price=_float_or_none(info.get("currentPrice")),
            market_cap=_float_or_none(info.get("marketCap")),
            beta=_float_or_none(info.get("beta")),
            data_source=str(info.get("dataSource", "Unknown")),
            ltm_ttm=ltm,
            mrq=mrq,
            ntm_forward=ntm,
        )


def _float_or_none(val: object) -> Optional[float]:
    """安全转换为 float，不可转换时返回 None。"""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _int_or_none(val: object) -> Optional[int]:
    """安全转换为 int，不可转换时返回 None。"""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
