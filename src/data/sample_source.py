"""
离线样本数据源 — 开发调试用。

当 yfinance 被限流时，使用预置的真实财务数据样本，
确保辩论链路可以完整跑通，不依赖外部网络。

数据来源：手动整理的公开财务数据，仅供开发测试。
"""

from __future__ import annotations

from typing import Any

from src.data.base import DataSource


# ── 预置样本数据 ────────────────────────────────────────────────
# 数据接近真实公开信息，用于验证辩论逻辑

_SAMPLE_DATA: dict[str, dict[str, Any]] = {
    "TSLA": {
        "info": {
            "shortName": "Tesla, Inc.",
            "symbol": "TSLA",
            "industry": "Auto Manufacturers",
            "sector": "Consumer Cyclical",
            "currentPrice": 245.83,
            "targetMeanPrice": 218.75,
            "targetHighPrice": 350.00,
            "targetLowPrice": 85.00,
            "recommendationKey": "hold",
            "numberOfAnalystOpinions": 42,
            "trailingPE": 56.72,
            "forwardPE": 68.97,
            "priceToBook": 8.91,
            "priceToSalesTrailing12Months": 7.51,
            "pegRatio": 2.83,
            "enterpriseToEbitda": 41.25,
            "returnOnEquity": 0.168,
            "returnOnAssets": 0.074,
            "grossMargins": 0.178,
            "profitMargins": 0.132,
            "trailingEps": 4.33,
            "estEpsNextY_avg": 3.57,
            "estEpsNextY_high": 4.20,
            "estEpsNextY_low": 2.80,
            "estRevenueNextY_avg": 108000000000,
            "revenueGrowth": 0.025,
            "earningsGrowth": -0.53,
            "earningsQuarterlyGrowth": -0.46,
            "debtToEquity": 24.65,
            "currentRatio": 1.86,
            "quickRatio": 1.29,
            "freeCashflow": 3472000000,
            "totalCash": 33400000000,
            "totalDebt": 13427000000,
            "marketCap": 787000000000,
            "fiftyTwoWeekHigh": 488.54,
            "fiftyTwoWeekLow": 138.80,
            "beta": 2.39,
            "dividendYield": None,
            "payoutRatio": 0.0,
        }
    },
    "AAPL": {
        "info": {
            "shortName": "Apple Inc.",
            "symbol": "AAPL",
            "industry": "Consumer Electronics",
            "sector": "Technology",
            "currentPrice": 195.87,
            "targetMeanPrice": 216.34,
            "targetHighPrice": 275.00,
            "targetLowPrice": 125.00,
            "recommendationKey": "buy",
            "numberOfAnalystOpinions": 38,
            "trailingPE": 30.12,
            "forwardPE": 28.57,
            "priceToBook": 45.83,
            "priceToSalesTrailing12Months": 7.89,
            "pegRatio": 2.15,
            "enterpriseToEbitda": 22.68,
            "returnOnEquity": 1.54,
            "returnOnAssets": 0.281,
            "grossMargins": 0.459,
            "profitMargins": 0.263,
            "trailingEps": 6.50,
            "estEpsNextY_avg": 6.85,
            "estEpsNextY_high": 7.50,
            "estEpsNextY_low": 6.00,
            "estRevenueNextY_avg": 420000000000,
            "revenueGrowth": 0.05,
            "earningsGrowth": 0.081,
            "earningsQuarterlyGrowth": 0.10,
            "debtToEquity": 1.89,
            "currentRatio": 1.04,
            "quickRatio": 1.02,
            "freeCashflow": 92000000000,
            "totalCash": 65000000000,
            "totalDebt": 108000000000,
            "marketCap": 3000000000000,
            "fiftyTwoWeekHigh": 260.10,
            "fiftyTwoWeekLow": 164.08,
            "beta": 1.24,
            "dividendYield": 0.0051,
            "payoutRatio": 0.152,
        }
    },
    "NVDA": {
        "info": {
            "shortName": "NVIDIA Corporation",
            "symbol": "NVDA",
            "industry": "Semiconductors",
            "sector": "Technology",
            "currentPrice": 980.45,
            "targetMeanPrice": 1050.20,
            "targetHighPrice": 1400.00,
            "targetLowPrice": 680.00,
            "recommendationKey": "buy",
            "numberOfAnalystOpinions": 45,
            "trailingPE": 54.32,
            "forwardPE": 35.71,
            "priceToBook": 28.50,
            "priceToSalesTrailing12Months": 16.42,
            "pegRatio": 1.18,
            "enterpriseToEbitda": 37.50,
            "returnOnEquity": 0.932,
            "returnOnAssets": 0.495,
            "grossMargins": 0.756,
            "profitMargins": 0.552,
            "trailingEps": 18.05,
            "estEpsNextY_avg": 27.45,
            "estEpsNextY_high": 30.00,
            "estEpsNextY_low": 24.00,
            "estRevenueNextY_avg": 160000000000,
            "revenueGrowth": 0.94,
            "earningsGrowth": 1.12,
            "earningsQuarterlyGrowth": 0.80,
            "debtToEquity": 28.30,
            "currentRatio": 3.52,
            "quickRatio": 3.05,
            "freeCashflow": 42000000000,
            "totalCash": 31400000000,
            "totalDebt": 11300000000,
            "marketCap": 2450000000000,
            "fiftyTwoWeekHigh": 1150.00,
            "fiftyTwoWeekLow": 392.00,
            "beta": 1.65,
            "dividendYield": 0.0001,
            "payoutRatio": 0.005,
        }
    },
}


class SampleSource(DataSource):
    """离线样本数据源。

    使用预置的公开财务数据，完全不依赖网络，毫秒级响应。
    适用于 yfinance 被限流时的开发调试和演示。

    支持的标的: TSLA, AAPL, NVDA

    Examples:
        >>> source = SampleSource()
        >>> data = source.fetch("TSLA")
        >>> text = source.to_text(data)
    """

    def fetch(self, ticker: str) -> dict[str, Any]:
        """返回预置的样本数据。

        Args:
            ticker: 股票代码。

        Returns:
            包含 info 字典的复合字典。

        Raises:
            ValueError: 不支持的股票代码。
        """
        key = ticker.upper()
        if key not in _SAMPLE_DATA:
            supported = ", ".join(sorted(_SAMPLE_DATA.keys()))
            raise ValueError(
                f"SampleSource 不支持 '{key}'。"
                f"当前支持的标的: {supported}"
            )
        return {"info": _SAMPLE_DATA[key]["info"]}

    def to_text(self, data: dict[str, Any]) -> str:
        """将样本数据转为 LLM 可读文本。

        格式与 YFinanceSource.to_text() 完全一致。

        Args:
            data: fetch() 返回的数据字典。

        Returns:
            格式化后的文本。
        """
        info: dict[str, Any] = data.get("info", {})

        def _fmt(key: str, fmt_spec: str = ".2f") -> str:
            val = info.get(key)
            if val is None:
                return "N/A"
            if isinstance(val, (int, float)):
                return f"{val:{fmt_spec}}"
            return str(val)

        def _money(val: object, compact: bool = False) -> str:
            if val is None:
                return "N/A"
            try:
                v = float(val)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                return str(val)
            if compact:
                if abs(v) >= 1e12:
                    return f"${v / 1e12:.2f}T"
                if abs(v) >= 1e9:
                    return f"${v / 1e9:.2f}B"
                if abs(v) >= 1e6:
                    return f"${v / 1e6:.2f}M"
            return f"${v:,.0f}"

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"股票名称: {info.get('shortName', 'N/A')} ({info.get('symbol', 'N/A')})")
        lines.append(f"行业: {info.get('industry', 'N/A')} | 板块: {info.get('sector', 'N/A')}")
        lines.append("=" * 60)

        # ── 维度 1: 基础与估值 ──
        lines.append("\n[维度一: 基础与估值]")
        lines.append("  [LTM/TTM — 过去12个月已审计/已报告数据]")
        lines.append(f"  当前价格:              ${_fmt('currentPrice')}")
        lines.append(f"  市值:                   {_money(info.get('marketCap'), compact=True)}")
        lines.append(f"  市盈率 (TTM):           {_fmt('trailingPE')}")
        lines.append(f"  市净率:                 {_fmt('priceToBook')}")
        lines.append(f"  市销率 (TTM):           {_fmt('priceToSalesTrailing12Months')}")
        lines.append(f"  企业价值/EBITDA:        {_fmt('enterpriseToEbitda')}")
        lines.append("  [NTM/Forward — 未来12个月预测数据 (仅此维度可用于推导目标价)]")
        lines.append(f"  远期市盈率 (Forward PE):{_fmt('forwardPE')}")
        # PEG: 标注防篡改
        peg_display = _fmt('pegRatio')
        if info.get('pegRatio') is not None:
            lines.append(f"  PEG 比率 (经系统计算，禁止自行修改): {peg_display}")
        else:
            lines.append(f"  PEG 比率:               {peg_display}")
        lines.append(f"  华尔街一致预期目标价 (仅供参考，主观预测): ${_fmt('targetMeanPrice')}")
        lines.append(f"  华尔街目标价上限 (仅供参考): ${_fmt('targetHighPrice')}")
        lines.append(f"  华尔街目标价下限 (仅供参考): ${_fmt('targetLowPrice')}")
        lines.append(f"  分析师共识评级:         {_fmt('recommendationKey', 's')}")
        lines.append(f"  覆盖分析师人数:         {_fmt('numberOfAnalystOpinions', 'd')}")

        # ── 维度 2: 预期差 ──
        lines.append("\n[维度二: 预期差 — NTM/Forward 分析师预测]")
        lines.append("  [NTM/Forward — 未来12个月预测数据，仅此维度可用于推导目标价]")
        lines.append(f"  Forward EPS 预估 (均值): ${_fmt('estEpsNextY_avg')}")
        lines.append(f"  Forward EPS 预估 (高值): ${_fmt('estEpsNextY_high')}")
        lines.append(f"  Forward EPS 预估 (低值): ${_fmt('estEpsNextY_low')}")
        lines.append(f"  下一财年营收预估 (均值): {_money(info.get('estRevenueNextY_avg'), compact=True)}")
        # EPS 预期差
        ttm_eps = info.get('trailingEps')
        fwd_eps = info.get('estEpsNextY_avg')
        if ttm_eps is not None and fwd_eps is not None and float(ttm_eps) != 0:
            try:
                diff_pct = (float(fwd_eps) / float(ttm_eps) - 1) * 100
                direction = "增长" if diff_pct >= 0 else "下滑"
                lines.append(f"  → Forward EPS vs TTM EPS: {direction} {abs(diff_pct):.1f}% "
                           f"(${_fmt('trailingEps')} → ${_fmt('estEpsNextY_avg')})")
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        # ── 维度 3: 盈利质量 ──
        lines.append("\n[维度三: 盈利质量]")
        lines.append("  [LTM/TTM — 过去12个月已审计/已报告数据]")
        lines.append(f"  毛利率 (Gross Margin):   {_fmt('grossMargins', '.2%')}")
        lines.append(f"  净利率 (Net Margin):     {_fmt('profitMargins', '.2%')}")
        lines.append(f"  ROE (净资产收益率):      {_fmt('returnOnEquity', '.2%')}")
        lines.append(f"  ROA (总资产收益率):      {_fmt('returnOnAssets', '.2%')}")
        lines.append(f"  每股收益 (TTM EPS):     ${_fmt('trailingEps')}")

        # ── 维度 4: 增长趋势 ──
        lines.append("\n[维度四: 增长趋势]")
        lines.append("  [MRQ — 最近季度同比数据 (趋势检测，不可线性外推)]")
        lines.append(f"  营收增长率 (YoY):       {_fmt('revenueGrowth', '.2%')}")
        lines.append(f"  盈利增长率 (YoY):       {_fmt('earningsGrowth', '.2%')}")
        lines.append(f"  EPS 增速 (YoY):         {_fmt('earningsQuarterlyGrowth', '.2%')}")

        # ── 维度 5: 财务健康 ──
        lines.append("\n[维度五: 财务健康]")
        lines.append("  [LTM/TTM — 过去12个月已审计/已报告数据]")
        lines.append(f"  负债权益比:             {_fmt('debtToEquity')}")
        lines.append(f"  流动比率:               {_fmt('currentRatio')}")
        lines.append(f"  速动比率:               {_fmt('quickRatio')}")
        fcf = info.get('freeCashflow')
        if fcf is not None:
            fcf_val = float(fcf)
            if fcf_val < 1e6:
                # 小数值 → 可能是每股 FCF
                lines.append(f"  每股自由现金流 (FCF/Share): ${fcf_val:.2f}")
                if info.get('marketCap') and info.get('currentPrice'):
                    try:
                        shares = float(info['marketCap']) / float(info['currentPrice'])
                        lines.append(f"  推算总自由现金流 (FCF):  {_money(fcf_val * shares, compact=True)}")
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
            else:
                lines.append(f"  总自由现金流 (FCF):      {_money(fcf, compact=True)}")
        else:
            lines.append(f"  自由现金流:             N/A")
        lines.append(f"  总现金:                 {_money(info.get('totalCash'), compact=True)}")
        lines.append(f"  总债务:                 {_money(info.get('totalDebt'), compact=True)}")

        # ── 市场数据 ──
        lines.append("\n[市场数据 — 仅供背景参考，不可作为估值论据]")
        lines.append(f"  52周最高:               ${_fmt('fiftyTwoWeekHigh')}")
        lines.append(f"  52周最低:               ${_fmt('fiftyTwoWeekLow')}")
        lines.append(f"  Beta (5Y):              {_fmt('beta')}")

        # ── 股息 ──
        lines.append("\n[股息与回购]")
        lines.append(f"  股息率:                 {_fmt('dividendYield', '.2%')}")
        lines.append(f"  派息比率:               {_fmt('payoutRatio', '.2%')}")

        lines.append("\n" + "=" * 60)
        lines.append("⚠️  数据来源: 离线样本 (开发调试模式)")
        lines.append("以上为本次辩论的全部依据数据。红蓝双方必须严格基于上述数据展开论证。")
        lines.append("")
        lines.append("⚠️ 时间维度纪律 (Temporal Discipline):")
        lines.append("  - [LTM/TTM] = 过去12个月已审计数据 → 用于历史估值、盈利能力、财务健康")
        lines.append("  - [MRQ]     = 最近季度同比 → 用于趋势检测、转折点识别")
        lines.append("  - [NTM/Forward] = 未来12个月预测 → 仅此维度可用于推导目标价")
        lines.append("  - 禁止跨维度混合计算 (如用 LTM 增长率论证 NTM 估值倍数)")
        lines.append("=" * 60)

        return "\n".join(lines)
