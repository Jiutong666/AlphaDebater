"""
Financial Modeling Prep (FMP) 数据源 V2 — 机构级数据源。

基于 FMP Stable API 抓取四大维度数据，实现 DataSource 接口。

FMP 于 2025 年 8 月废弃了 /api/v3 传统端点，全面迁移至 Stable API：
    - URL 格式: /stable/{endpoint}?symbol={ticker}&apikey=...
    - 字段名: 统一为 camelCase (如 peRatioTTM → priceToEarningsRatioTTM)
    - 增长指标: 独立为 /financial-growth 端点

四大维度：
    1. 基础与估值:  当前价格、市值、PE / PB / PS / PEG / EV/EBITDA
    2. 预期差:      分析师 EPS 预估 (下一财年 / 下一季度)
    3. 聪明钱动向:  内部人士买卖比例
    4. 财务健康:    营收同比增速、毛利率、ROE、现金流

特性：
    - requests_cache 本地 SQLite 缓存 (24h TTL，保护 API 额度)
    - 合并 7 个 FMP 端点
    - 自动重试 + 指数退避
    - 输出格式与现有数据源完全兼容
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
import requests_cache

from src.config.settings import settings
from src.data.base import DataSource

logger = logging.getLogger(__name__)

# ── requests_cache 模块级会话 (24h TTL) ─────────────────────────
# 绝对不允许每次运行都消耗真实的 API 额度！
# CachedSession 自动将 HTTP 响应持久化到 fmp_cache.sqlite，
# 同 URL 24h 内直接读磁盘，零网络请求。

_CACHE_NAME = "fmp_cache"
_CACHE_TTL = 86400  # 24 小时

_session = requests_cache.CachedSession(
    cache_name=_CACHE_NAME,
    backend="sqlite",
    expire_after=_CACHE_TTL,
    allowable_codes=(200,),
)

# ── 常量 ────────────────────────────────────────────────────────

_BASE_URL = "https://financialmodelingprep.com/stable"
_MAX_RETRIES: int = 3
_BASE_DELAY: float = 2.0
_REQUEST_TIMEOUT: int = 15


# ── API 请求层 ────────────────────────────────────────────────

def _fmp_get(
    endpoint: str,
    ticker: str,
    extra_params: dict | None = None,
    *,
    optional: bool = False,
) -> list[dict[str, Any]]:
    """调用 FMP Stable API，带自动重试与 requests_cache 缓存。

    Stable API 统一使用 ?symbol={ticker} 查询参数格式。

    Args:
        endpoint:     API 路径 (如 "/profile", "/analyst-estimates")。
        ticker:       股票代码。
        extra_params: 额外查询参数 (如 {"period": "annual", "limit": "5"})。
        optional:     True 时 404 返回空列表而不抛异常 (用于免费版不可用的端点)。

    Returns:
        JSON 响应解析后的列表。

    Raises:
        RuntimeError: 网络错误或 API 限流 / Key 无效。
    """
    api_key = settings.fmp_api_key
    params: dict[str, Any] = {"symbol": ticker, "apikey": api_key}
    if extra_params:
        params.update(extra_params)

    url = f"{_BASE_URL}{endpoint}"
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _session.get(url, params=params, timeout=_REQUEST_TIMEOUT)

            # 可选端点: 404 → 静默返回空列表 (免费版无权限)
            if optional and resp.status_code == 404:
                logger.debug(f"FMP 端点不可用 (404): {endpoint} (免费版限制，已跳过)")
                return []

            resp.raise_for_status()
            data: list[dict[str, Any]] = resp.json()

            if not isinstance(data, list):
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"FMP 返回非列表数据 (attempt {attempt + 1})，{delay:.0f}s 后重试..."
                    )
                    time.sleep(delay)
                    continue
                return []

            if not data:
                return []

            return data

        except requests.HTTPError as exc:
            status = exc.response.status_code if hasattr(exc, 'response') else 0

            if status == 429 and attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning(f"FMP 限流 (attempt {attempt + 1})，{delay:.0f}s 后重试...")
                time.sleep(delay)
                continue

            if status in (401, 403):
                raise RuntimeError(
                    f"FMP API Key 无效或无权访问 (HTTP {status})。\n"
                    f"请到 https://financialmodelingprep.com 注册获取 API Key，\n"
                    f"然后更新 .env 中的 FMP_API_KEY。"
                ) from exc

            # 402: 付费端点或无效标的 → 返回空列表让上层判为无效 ticker
            if status == 402:
                logger.debug(f"FMP 402 (需要付费订阅或标的无效): {endpoint}")
                return []

            raise RuntimeError(f"FMP API 返回 HTTP {status}: {exc}") from exc

        except requests.RequestException as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning(f"FMP 网络错误 (attempt {attempt + 1})，{delay:.0f}s 后重试...")
                time.sleep(delay)
                continue

    raise RuntimeError(
        f"FMP API 请求失败 (已重试 {_MAX_RETRIES} 次): {last_error}"
    )


# ── 数据组装 ──────────────────────────────────────────────────

def _parse_52w_range(range_str: str) -> tuple[float | None, float | None]:
    """解析 profile.range 字段 (格式: \"193.46-305.08\")。

    Returns:
        (low_52w, high_52w)
    """
    if not range_str or "-" not in range_str:
        return None, None
    parts = range_str.split("-")
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None, None


def _summarize_insider_trading(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总内部人士交易记录。

    区分买入 / 卖出 / 授予，计算买卖股数比和信号方向。
    免费版 FMP 可能返回空列表，此时所有字段为 0/None。

    Args:
        transactions: /insider-trading 端点返回的原始列表。

    Returns:
        扁平化的汇总字典。
    """
    buy_shares = 0.0
    sell_shares = 0.0
    award_shares = 0.0
    buy_count = 0
    sell_count = 0
    latest_transactions: list[dict[str, Any]] = []

    # 提取最近 20 条交易明细
    for t in transactions[:20]:
        txn_type = t.get("transactionType", "")
        try:
            shares = float(t.get("securitiesTransacted", 0))
        except (ValueError, TypeError):
            shares = 0.0

        latest_transactions.append({
            "date": t.get("transactionDate") or t.get("filingDate"),
            "type": txn_type,
            "shares": int(shares),
            "price": t.get("price"),
            "officer": t.get("reportingName") or t.get("reportingCik"),
        })

    # 汇总统计
    for t in transactions:
        txn_type = t.get("transactionType", "")
        try:
            shares = float(t.get("securitiesTransacted", 0))
        except (ValueError, TypeError):
            shares = 0.0

        if "Purchase" in txn_type:
            buy_shares += shares
            buy_count += 1
        elif "Sale" in txn_type:
            sell_shares += shares
            sell_count += 1
        elif "Award" in txn_type or "Grant" in txn_type:
            award_shares += shares

    total_buy_sell = buy_shares + sell_shares
    buy_ratio = (buy_shares / total_buy_sell) if total_buy_sell > 0 else None

    return {
        "insiderBuyShares": int(buy_shares),
        "insiderSellShares": int(sell_shares),
        "insiderAwardShares": int(award_shares),
        "insiderBuyCount": buy_count,
        "insiderSellCount": sell_count,
        "insiderBuyRatio": buy_ratio,          # > 0.5 = 净买入
        "insiderHasData": bool(transactions),   # 免费版是否有数据
        "insiderRecentTransactions": latest_transactions,
    }


def _extract_analyst_estimates(estimates: list[dict[str, Any]]) -> dict[str, Any]:
    """从分析师预估列表中提取 Forward EPS / Revenue 预估。

    FMP /analyst-estimates 按 date 降序返回年度预估。
    每个条目包含 epsAvg / epsHigh / epsLow / revenueAvg 等字段。

    Args:
        estimates: /analyst-estimates?period=annual 端点返回的列表。

    Returns:
        扁平化的汇总字典，可直接合并到 info。
    """
    result: dict[str, Any] = {
        "estEpsNextY_avg": None,
        "estEpsNextY_high": None,
        "estEpsNextY_low": None,
        "estEpsNextY_date": None,
        "estEpsNextY_analysts": None,
        "estEpsNextQ_avg": None,
        "estEpsNextQ_high": None,
        "estEpsNextQ_low": None,
        "estEpsNextQ_date": None,
        "estEpsNextQ_analysts": None,
        "estRevenueNextY_avg": None,
        "estRevenueNextY_high": None,
        "estRevenueNextY_low": None,
        "estRevenueNextQ_avg": None,
    }

    if not estimates:
        return result

    # 按 date 排序 (最新在前)
    sorted_est = sorted(
        estimates,
        key=lambda x: x.get("date", ""),
        reverse=True,
    )

    # 最新一条 → "下一财年"预估
    if sorted_est:
        y = sorted_est[0]
        result["estEpsNextY_avg"] = y.get("epsAvg")
        result["estEpsNextY_high"] = y.get("epsHigh")
        result["estEpsNextY_low"] = y.get("epsLow")
        result["estEpsNextY_date"] = y.get("date")
        result["estEpsNextY_analysts"] = y.get("numAnalystsEps")
        result["estRevenueNextY_avg"] = y.get("revenueAvg")
        result["estRevenueNextY_high"] = y.get("revenueHigh")
        result["estRevenueNextY_low"] = y.get("revenueLow")

    # 第二条 → "下一季度" (短期) 预估；只有一条时复用 Year 数据
    if len(sorted_est) >= 2:
        q = sorted_est[1]
        result["estEpsNextQ_avg"] = q.get("epsAvg")
        result["estEpsNextQ_high"] = q.get("epsHigh")
        result["estEpsNextQ_low"] = q.get("epsLow")
        result["estEpsNextQ_date"] = q.get("date")
        result["estEpsNextQ_analysts"] = q.get("numAnalystsEps")
        result["estRevenueNextQ_avg"] = q.get("revenueAvg")
    else:
        result["estEpsNextQ_avg"] = result["estEpsNextY_avg"]
        result["estEpsNextQ_high"] = result["estEpsNextY_high"]
        result["estEpsNextQ_low"] = result["estEpsNextY_low"]
        result["estEpsNextQ_date"] = result["estEpsNextY_date"]
        result["estEpsNextQ_analysts"] = result["estEpsNextY_analysts"]
        result["estRevenueNextQ_avg"] = result["estRevenueNextY_avg"]

    return result


def _build_info(
    ticker: str,
    profile: dict[str, Any],
    ratios: dict[str, Any],
    key_metrics: dict[str, Any],
    growth: dict[str, Any],
    analyst_est: dict[str, Any],
    insider_summary: dict[str, Any],
    balance_sheet: dict[str, Any],
    price_target: dict[str, Any],
) -> dict[str, Any]:
    """将 FMP 8 个端点的原始数据合并为标准 info 字典 (扁平化)。

    字段名与 yfinance info 保持一致，确保下游兼容。

    Args:
        ticker:         股票代码。
        profile:        /profile 端点响应第一条。
        ratios:         /ratios-ttm 端点响应第一条。
        key_metrics:    /key-metrics-ttm 端点响应第一条。
        growth:         /financial-growth 端点响应第一条。
        analyst_est:    /analyst-estimates 汇总结果。
        insider_summary: /insider-trading 汇总结果。
        balance_sheet:  /balance-sheet-statement 端点响应第一条。
        price_target:   /price-target-consensus 端点响应第一条。
    """
    p = profile or {}
    r = ratios or {}
    k = key_metrics or {}
    g = growth or {}
    bs = balance_sheet or {}
    pt = price_target or {}

    # 解析 52 周高低
    low_52w, high_52w = _parse_52w_range(p.get("range", ""))

    info: dict[str, Any] = {
        # ── 基本信息 ──
        "shortName": p.get("companyName", ticker),
        "symbol": p.get("symbol", ticker),
        "industry": p.get("industry"),
        "sector": p.get("sector"),

        # ── 维度 1: 基础与估值 ──
        "currentPrice": p.get("price"),
        "marketCap": p.get("marketCap"),
        "trailingPE": r.get("priceToEarningsRatioTTM"),
        "forwardPE": None,  # 将在下方根据分析师预估计算
        "priceToBook": r.get("priceToBookRatioTTM"),
        "priceToSalesTrailing12Months": r.get("priceToSalesRatioTTM"),
        "pegRatio": r.get("priceToEarningsGrowthRatioTTM"),
        "enterpriseToEbitda": k.get("evToEBITDATTM") or r.get("enterpriseValueMultipleTTM"),

        # 分析师目标价 (来自 /price-target-consensus)
        "targetMeanPrice": pt.get("targetConsensus"),
        "targetHighPrice": pt.get("targetHigh"),
        "targetLowPrice": pt.get("targetLow"),
        "targetMedianPrice": pt.get("targetMedian"),
        "recommendationKey": None,
        "numberOfAnalystOpinions": analyst_est.get("estEpsNextY_analysts"),

        # ── 维度 2: 预期差 (Forward 指引) ──
        # 来自 /analyst-estimates
        **analyst_est,

        # ── 维度 3: 聪明钱动向 ──
        # 来自 /insider-trading 汇总
        **insider_summary,

        # ── 维度 4: 财务健康 ──
        "revenueGrowth": g.get("revenueGrowth"),                # 营收同比增速
        "grossMargins": r.get("grossProfitMarginTTM"),          # 毛利率
        "netProfitMargin": r.get("netProfitMarginTTM"),         # 净利率
        "returnOnEquity": k.get("returnOnEquityTTM"),           # ROE
        "returnOnAssets": k.get("returnOnAssetsTTM"),           # ROA
        "trailingEps": r.get("netIncomePerShareTTM"),           # EPS (TTM)
        "earningsGrowth": g.get("netIncomeGrowth"),             # 净利润增速
        "earningsQuarterlyGrowth": g.get("epsgrowth"),          # EPS 增速

        # ── 财务健康: 资产负债表 ──
        "debtToEquity": r.get("debtToEquityRatioTTM"),
        "currentRatio": r.get("currentRatioTTM"),
        "quickRatio": r.get("quickRatioTTM"),
        "freeCashflowPerShare": r.get("freeCashFlowPerShareTTM"),  # 每股自由现金流
        "freeCashflowTotal": None,  # 下方根据每股FCF和市值推算
        "totalCash": bs.get("cashAndCashEquivalents"),
        "totalDebt": bs.get("totalDebt"),

        # ── 市场数据 ──
        "fiftyTwoWeekHigh": high_52w,
        "fiftyTwoWeekLow": low_52w,
        "fiftyDayAverage": None,
        "twoHundredDayAverage": None,
        "beta": p.get("beta"),
        "averageVolume": p.get("averageVolume"),

        # ── 股息 ──
        "dividendYield": r.get("dividendYieldTTM"),
        "payoutRatio": r.get("dividendPayoutRatioTTM"),

        # ── 元数据 ──
        "dataSource": "Financial Modeling Prep (FMP) Stable API",
    }

    # ── 计算 forwardPE: currentPrice / Forward EPS ──
    current_price = info.get("currentPrice")
    fwd_eps = info.get("estEpsNextY_avg")
    if current_price is not None and fwd_eps is not None:
        try:
            if float(fwd_eps) > 0:
                info["forwardPE"] = round(float(current_price) / float(fwd_eps), 2)
        except (ValueError, TypeError):
            pass

    # ── 推算总自由现金流: 每股FCF × (市值/当前价) ──
    fcf_per_share = info.get("freeCashflowPerShare")
    mkt_cap = info.get("marketCap")
    if fcf_per_share and mkt_cap and current_price:
        try:
            if float(current_price) > 0:
                shares = float(mkt_cap) / float(current_price)
                info["freeCashflowTotal"] = round(float(fcf_per_share) * shares, 0)
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    return info


# ── DataSource 实现 ─────────────────────────────────────────────

class FMPSource(DataSource):
    """Financial Modeling Prep 机构级数据源。

    一次 fetch() 调用合并 **8 个** Stable API 端点:

    ==========================  ============================
    端点                         对应维度
    ==========================  ============================
    /profile                     公司概况 (价格、市值、行业)
    /ratios-ttm                  估值倍数 (PE/PB/PS/PEG) + 盈利质量
    /key-metrics-ttm             ROE / ROA / EV/EBITDA
    /financial-growth            营收/盈利增长率
    /analyst-estimates           预期差 (Forward EPS)
    /price-target-consensus      分析师目标价 (高/低/共识)
    /insider-trading             聪明钱动向
    /balance-sheet-statement     现金与债务
    ==========================  ============================

    requests_cache 提供 24h SQLite 缓存，保护 API 额度。

    Examples:
        >>> source = FMPSource()
        >>> data = source.fetch("AAPL")
        >>> text = source.to_text(data)
    """

    def fetch(self, ticker: str) -> dict[str, Any]:
        """从 FMP 抓取四大维度数据并组装为扁平化字典。

        Args:
            ticker: 股票代码 (如 "AAPL", "TSLA")。

        Returns:
            {"info": {...}} 格式的字典，包含四大维度所有字段。

        Raises:
            ValueError:   股票代码无效或 FMP 不支持该标的。
            RuntimeError: 网络错误或 API Key 无效 (多次重试后)。
        """
        key = ticker.upper()

        # ── 抓取 8 个端点 (requests_cache 保证 24h 内零网络请求) ──
        try:
            profile_arr = _fmp_get("/profile", key)
            ratios_arr = _fmp_get("/ratios-ttm", key)
            metrics_arr = _fmp_get("/key-metrics-ttm", key)
            growth_arr = _fmp_get("/financial-growth", key, extra_params={"limit": "1"})
            estimates_arr = _fmp_get(
                "/analyst-estimates", key,
                extra_params={"period": "annual", "limit": "5"},
            )
            price_target_arr = _fmp_get("/price-target-consensus", key)
            insider_arr = _fmp_get(
                "/insider-trading", key,
                extra_params={"limit": "100"},
                optional=True,
            )
            bs_arr = _fmp_get(
                "/balance-sheet-statement", key,
                extra_params={"limit": "1"},
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"FMP 数据抓取失败: {exc}") from exc

        # ── 有效性校验 ──
        if not profile_arr and not ratios_arr:
            raise ValueError(
                f"股票代码 '{key}' 无效或 FMP 不支持该标的。\n"
                f"请检查代码拼写 (如 AAPL、TSLA)。"
            )

        profile = profile_arr[0] if profile_arr else {}
        ratios = ratios_arr[0] if ratios_arr else {}
        key_metrics = metrics_arr[0] if metrics_arr else {}
        growth = growth_arr[0] if growth_arr else {}
        balance_sheet = bs_arr[0] if bs_arr else {}
        price_target = price_target_arr[0] if price_target_arr else {}

        # ── 二级数据处理 ──
        analyst_est = _extract_analyst_estimates(estimates_arr)
        insider_summary = _summarize_insider_trading(insider_arr)

        # ── 组装扁平化 info ──
        info = _build_info(
            key, profile, ratios, key_metrics, growth,
            analyst_est, insider_summary, balance_sheet, price_target,
        )
        return {"info": info}

    def to_text(self, data: dict[str, Any]) -> str:
        """将 FMP 四大维度数据序列化为 LLM 可读文本。

        格式与 YFinanceSource / SampleSource 保持一致，
        新增 [预期差] 和 [聪明钱动向] 板块。

        Args:
            data: fetch() 返回的数据字典。

        Returns:
            格式化后的 Markdown 兼容文本。
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
            """将大数值转为可读格式 (T/B/M)。"""
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
        lines.append(
            f"股票名称: {info.get('shortName', 'N/A')} "
            f"({info.get('symbol', 'N/A')})"
        )
        lines.append(
            f"行业: {info.get('industry', 'N/A')} "
            f"| 板块: {info.get('sector', 'N/A')}"
        )
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
        # PEG: 优先展示系统预计算值，标注防篡改
        peg_display = _fmt('pegRatio')
        if info.get('pegRatio') is not None:
            lines.append(f"  PEG 比率 (经系统计算，禁止自行修改): {peg_display}")
        else:
            lines.append(f"  PEG 比率:               {peg_display}")
        lines.append(f"  华尔街一致预期目标价 (仅供参考，主观预测): ${_fmt('targetMeanPrice')}")
        lines.append(f"  华尔街目标价上限 (仅供参考): ${_fmt('targetHighPrice')}")
        lines.append(f"  华尔街目标价下限 (仅供参考): ${_fmt('targetLowPrice')}")

        # ── 维度 2: 预期差 (Forward 指引) ──
        lines.append("\n[维度二: 预期差 — NTM/Forward 分析师预测]")
        lines.append("  [NTM/Forward — 未来12个月预测数据，仅此维度可用于推导目标价]")
        est_y_date = info.get("estEpsNextY_date", "N/A")
        est_y_analysts = info.get("estEpsNextY_analysts")
        est_y_label = f"下一财年 ({est_y_date})"
        lines.append(f"  {est_y_label}:")
        lines.append(f"    EPS 预估 (均值):      ${_fmt('estEpsNextY_avg')}  "
                     f"({est_y_analysts or 'N/A'} 位分析师)")
        lines.append(f"    EPS 预估 (高值):      ${_fmt('estEpsNextY_high')}")
        lines.append(f"    EPS 预估 (低值):      ${_fmt('estEpsNextY_low')}")
        lines.append(f"    营收预估 (均值):       {_money(info.get('estRevenueNextY_avg'), compact=True)}")

        # 如果下一季度与下一财年不同，单独展示
        if info.get("estEpsNextQ_date") != info.get("estEpsNextY_date"):
            est_q_date = info.get("estEpsNextQ_date", "N/A")
            est_q_analysts = info.get("estEpsNextQ_analysts")
            lines.append(f"  下一季度 ({est_q_date}):")
            lines.append(f"    EPS 预估 (均值):      ${_fmt('estEpsNextQ_avg')}  "
                         f"({est_q_analysts or 'N/A'} 位分析师)")
            lines.append(f"    EPS 预估 (高值):      ${_fmt('estEpsNextQ_high')}")

        # EPS 预期差分析: TTM EPS vs Forward EPS
        ttm_eps = info.get("trailingEps")
        fwd_eps = info.get("estEpsNextY_avg")
        if ttm_eps is not None and fwd_eps is not None and float(ttm_eps) != 0:
            try:
                diff_pct = (float(fwd_eps) / float(ttm_eps) - 1) * 100
                direction = "增长" if diff_pct >= 0 else "下滑"
                lines.append(
                    f"  → Forward EPS vs TTM EPS: {direction} {abs(diff_pct):.1f}% "
                    f"(${_fmt('trailingEps')} → ${_fmt('estEpsNextY_avg')})"
                )
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        # ── 维度 3: 聪明钱动向 ──
        lines.append("\n[维度三: 聪明钱动向 — 内部人士交易]")

        if not info.get("insiderHasData"):
            lines.append("  (免费版 FMP API 不返回内部人士交易数据)")
        else:
            lines.append(f"  内部人士买入笔数:       {info.get('insiderBuyCount', 0)}")
            lines.append(f"  内部人士卖出笔数:       {info.get('insiderSellCount', 0)}")
            lines.append(f"  买入股数:               {info.get('insiderBuyShares', 0):,}")
            lines.append(f"  卖出股数:               {info.get('insiderSellShares', 0):,}")
            if info.get("insiderAwardShares"):
                lines.append(f"  授予股数:               {info.get('insiderAwardShares', 0):,}")

            buy_ratio = info.get("insiderBuyRatio")
            if buy_ratio is not None:
                ratio_pct = float(buy_ratio) * 100
                signal = "[净买入 - 看多信号]" if buy_ratio > 0.5 else "[净卖出 - 看空信号]"
                lines.append(f"  买入占比:               {ratio_pct:.1f}%  {signal}")

        # ── 维度 4: 财务健康 ──
        lines.append("\n[维度四: 财务健康]")
        lines.append("  [LTM/TTM — 过去12个月已审计/已报告数据]")
        lines.append(f"  毛利率:                 {_fmt('grossMargins', '.2%')}")
        lines.append(f"  净利率:                 {_fmt('netProfitMargin', '.2%')}")
        lines.append(f"  ROE (净资产收益率):      {_fmt('returnOnEquity', '.2%')}")
        lines.append(f"  ROA (总资产收益率):      {_fmt('returnOnAssets', '.2%')}")
        lines.append(f"  每股收益 (TTM):         ${_fmt('trailingEps')}")
        lines.append(f"  负债权益比:             {_fmt('debtToEquity')}")
        lines.append(f"  流动比率:               {_fmt('currentRatio')}")
        lines.append(f"  速动比率:               {_fmt('quickRatio')}")
        lines.append(f"  每股自由现金流 (FCF/Share): ${_fmt('freeCashflowPerShare')}")
        total_fcf = info.get('freeCashflowTotal')
        total_display = _money(total_fcf, compact=True) if total_fcf else "N/A"
        lines.append(f"  推算总自由现金流 (Total FCF): {total_display}")
        lines.append(f"  总现金:                 {_money(info.get('totalCash'))}")
        lines.append(f"  总债务:                 {_money(info.get('totalDebt'))}")
        lines.append("  [MRQ — 最近季度同比数据 (趋势检测)]")
        lines.append(f"  营收增长率 (YoY):       {_fmt('revenueGrowth', '.2%')}")
        lines.append(f"  盈利增长率 (YoY):       {_fmt('earningsGrowth', '.2%')}")
        lines.append(f"  EPS 增速 (YoY):         {_fmt('earningsQuarterlyGrowth', '.2%')}")

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
        lines.append("数据来源: Financial Modeling Prep (FMP) Stable API — 机构级数据")
        lines.append("以上四大维度为本次辩论的全部依据数据。红蓝双方必须严格基于上述数据展开论证。")
        lines.append("")
        lines.append("⚠️ 时间维度纪律 (Temporal Discipline):")
        lines.append("  - [LTM/TTM] = 过去12个月已审计数据 → 用于历史估值、盈利能力、财务健康")
        lines.append("  - [MRQ]     = 最近季度同比 → 用于趋势检测、转折点识别")
        lines.append("  - [NTM/Forward] = 未来12个月预测 → 仅此维度可用于推导目标价")
        lines.append("  - 禁止跨维度混合计算 (如用 LTM 增长率论证 NTM 估值倍数)")
        lines.append("=" * 60)

        return "\n".join(lines)
