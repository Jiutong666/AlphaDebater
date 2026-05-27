"""
Yahoo Finance 数据源实现 (V1.0)。

基于 yfinance 库抓取美股/全球股票数据，实现 DataSource 接口。

特性：
    - 本地 JSON 文件缓存（同标的 1 小时内直接读磁盘，零网络请求）
    - 与 yfinance 内部 HTTP 库完全解耦（不劫持 session，兼容 curl_cffi）
    - 随机抖动重试（应对限流）
"""

from __future__ import annotations

import json
import time
import random
import logging
from pathlib import Path
from typing import Any

import yfinance as yf

from src.data.base import DataSource

logger = logging.getLogger(__name__)

# ── 缓存配置 ────────────────────────────────────────────────────

_CACHE_DIR = Path(".cache")
_CACHE_TTL: int = 3600  # 1 小时

# ── 重试配置 ────────────────────────────────────────────────────

_MAX_RETRIES: int = 3
_BASE_DELAY: float = 3.0


def _cache_path(ticker: str) -> Path:
    """返回某个 ticker 的缓存文件路径。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{ticker.upper()}.json"


def _load_from_cache(ticker: str) -> dict[str, Any] | None:
    """尝试从本地 JSON 缓存加载数据。

    Args:
        ticker: 股票代码。

    Returns:
        缓存数据字典，或 None（缓存不存在/已过期/损坏）。
    """
    path = _cache_path(ticker)

    if not path.exists():
        return None

    # 检查是否过期
    age = time.time() - path.stat().st_mtime
    if age > _CACHE_TTL:
        logger.info(f"缓存已过期 ({age:.0f}s)，重新抓取 {ticker}")
        return None

    # 读取并校验
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("info", {}).get("shortName"):
            logger.info(f"命中缓存: {ticker} ({(age):.0f}s 前)")
            return data
    except (json.JSONDecodeError, KeyError, OSError):
        path.unlink(missing_ok=True)

    return None


def _save_to_cache(ticker: str, data: dict[str, Any]) -> None:
    """将数据写入本地 JSON 缓存。

    Args:
        ticker: 股票代码。
        data:   要缓存的数据字典。
    """
    path = _cache_path(ticker)
    try:
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except OSError as exc:
        logger.warning(f"缓存写入失败 (非致命): {exc}")


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否可重试（限流/网络类）。"""
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "too many requests", "rate limited", "try after a while",
        "timeout", "timed out", "connection", "429", "403",
    ])


class YFinanceSource(DataSource):
    """Yahoo Finance 数据源。

    本地 JSON 缓存 + 随机抖动重试。
    不劫持 yfinance 的 HTTP session，完全兼容 curl_cffi。

    Examples:
        >>> source = YFinanceSource()
        >>> data = source.fetch("AAPL")
        >>> text = source.to_text(data)
    """

    def fetch(self, ticker: str) -> dict[str, Any]:
        """从 Yahoo Finance 抓取股票数据（优先读缓存）。

        缓存策略：
        1. 检查 .cache/{TICKER}.json 是否存在且未过期
        2. 命中 → 直接返回，零网络请求
        3. 未命中 → 调用 yfinance → 成功则写入缓存

        Args:
            ticker: 股票代码。

        Returns:
            包含 info 字典的复合字典。

        Raises:
            ValueError:   股票代码无效。
            RuntimeError: 多次重试后仍失败。
        """
        key = ticker.upper()

        # ── 优先读缓存 ──
        cached = _load_from_cache(key)
        if cached is not None:
            return cached

        # ── 缓存未命中，请求 yfinance ──
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                stock = yf.Ticker(key)
                info: dict[str, Any] = stock.info or {}

                if info and info.get("shortName"):
                    data = {"info": info}
                    _save_to_cache(key, data)
                    return data

                # 空数据：可能是限流导致的假空响应
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"yfinance 返回空数据 (attempt {attempt + 1}/{_MAX_RETRIES + 1})，"
                        f"{delay:.0f}s 后重试..."
                    )
                    time.sleep(delay)
                    continue

                raise ValueError(
                    f"股票代码 '{key}' 无效或数据不可用。"
                    f"请检查代码拼写（如 AAPL、TSLA、600519.SS）。"
                )

            except ValueError:
                raise

            except Exception as exc:
                last_error = exc

                if not _is_retryable(exc):
                    raise ValueError(
                        f"获取股票 '{key}' 数据时发生不可恢复的错误: {exc}"
                    ) from exc

                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.5)
                    logger.warning(
                        f"yfinance 限流 (attempt {attempt + 1}/{_MAX_RETRIES + 1})，"
                        f"{delay + jitter:.0f}s 后重试..."
                    )
                    time.sleep(delay + jitter)
                    continue

        raise RuntimeError(
            f"无法获取股票 '{key}' 的数据：已重试 {_MAX_RETRIES} 次仍失败。\n"
            f"建议：等待 5-10 分钟后重试，或更换网络环境。\n"
            f"最后错误: {last_error}"
        )

    def to_text(self, data: dict[str, Any]) -> str:
        """将 yfinance 原始数据转为 LLM 可读文本。"""
        info: dict[str, Any] = data.get("info", {})

        def _fmt(key: str, fmt_spec: str = ".2f") -> str:
            val = info.get(key)
            if val is None:
                return "N/A"
            if isinstance(val, (int, float)):
                return f"{val:{fmt_spec}}"
            return str(val)

        lines: list[str] = []
        lines.append("=" * 50)
        lines.append(f"股票名称: {info.get('shortName', 'N/A')} ({info.get('symbol', 'N/A')})")
        lines.append(f"行业: {info.get('industry', 'N/A')} | 板块: {info.get('sector', 'N/A')}")
        lines.append("=" * 50)

        lines.append("\n[估值指标]")
        lines.append("  [LTM/TTM — 过去12个月已审计/已报告数据]")
        lines.append(f"  当前价格:              ${_fmt('currentPrice')}")
        lines.append(f"  市盈率 (TTM):           {_fmt('trailingPE')}")
        lines.append(f"  市净率:                 {_fmt('priceToBook')}")
        lines.append(f"  市销率 (TTM):           {_fmt('priceToSalesTrailing12Months')}")
        lines.append(f"  企业价值/EBITDA:        {_fmt('enterpriseToEbitda')}")
        lines.append("  [NTM/Forward — 未来12个月预测数据]")
        lines.append(f"  远期市盈率:             {_fmt('forwardPE')}")
        lines.append(f"  PEG 比率:               {_fmt('pegRatio')}")
        lines.append(f"  分析师目标均价:         ${_fmt('targetMeanPrice')}")
        lines.append(f"  分析师共识 (1强买-5强卖): {_fmt('recommendationKey', 's')}")
        lines.append(f"  目标价上限:             ${_fmt('targetHighPrice')}")
        lines.append(f"  目标价下限:             ${_fmt('targetLowPrice')}")
        lines.append(f"  分析师人数:             {_fmt('numberOfAnalystOpinions', 'd')}")

        lines.append("\n[盈利能力]")
        lines.append("  [LTM/TTM — 过去12个月已审计/已报告数据]")
        lines.append(f"  ROE (净资产收益率):      {_fmt('returnOnEquity', '.2%')}")
        lines.append(f"  ROA (总资产收益率):      {_fmt('returnOnAssets', '.2%')}")
        lines.append(f"  毛利率:                 {_fmt('grossMargins', '.2%')}")
        lines.append(f"  净利率:                 {_fmt('profitMargins', '.2%')}")
        lines.append(f"  每股收益 (TTM):         {_fmt('trailingEps')}")

        lines.append("\n[增长指标]")
        lines.append("  [MRQ — 最近季度同比数据 (趋势检测)]")
        lines.append(f"  营收增长率 (YoY):       {_fmt('revenueGrowth', '.2%')}")
        lines.append(f"  盈利增长率 (YoY):       {_fmt('earningsGrowth', '.2%')}")
        lines.append(f"  季度盈利增长 (YoY):     {_fmt('earningsQuarterlyGrowth', '.2%')}")

        lines.append("\n[财务健康]")
        lines.append("  [LTM/TTM — 过去12个月已审计/已报告数据]")
        lines.append(f"  负债权益比:             {_fmt('debtToEquity')}")
        lines.append(f"  流动比率:               {_fmt('currentRatio')}")
        lines.append(f"  速动比率:               {_fmt('quickRatio')}")
        lines.append(f"  自由现金流:             {_fmt('freeCashflow', '.0f')}")
        lines.append(f"  总现金:                 {_fmt('totalCash', '.0f')}")
        lines.append(f"  总债务:                 {_fmt('totalDebt', '.0f')}")

        lines.append("\n[市场数据]")
        lines.append(f"  市值:                   {_fmt('marketCap', '.0f')}")
        lines.append(f"  52周最高:               ${_fmt('fiftyTwoWeekHigh')}")
        lines.append(f"  52周最低:               ${_fmt('fiftyTwoWeekLow')}")
        lines.append(f"  50日均价:               ${_fmt('fiftyDayAverage')}")
        lines.append(f"  200日均价:              ${_fmt('twoHundredDayAverage')}")
        lines.append(f"  Beta (5Y):              {_fmt('beta')}")
        lines.append(f"  平均成交量:             {_fmt('averageVolume', '.0f')}")

        lines.append("\n[股息与回购]")
        lines.append(f"  股息率:                 {_fmt('dividendYield', '.2%')}")
        lines.append(f"  派息比率:               {_fmt('payoutRatio', '.2%')}")

        lines.append("\n" + "=" * 50)
        lines.append("以上为本次辩论的全部依据数据。红蓝双方必须严格基于上述数据展开论证。")
        lines.append("")
        lines.append("⚠️ 时间维度纪律 (Temporal Discipline):")
        lines.append("  - [LTM/TTM] = 过去12个月已审计数据 → 用于历史估值、盈利能力、财务健康")
        lines.append("  - [MRQ]     = 最近季度同比 → 用于趋势检测、转折点识别")
        lines.append("  - [NTM/Forward] = 未来12个月预测 → 仅此维度可用于推导目标价")
        lines.append("  - 禁止跨维度混合计算 (如用 LTM 增长率论证 NTM 估值倍数)")
        lines.append("=" * 50)

        return "\n".join(lines)
