"""
News sentiment data source — adds a non-fundamental dimension to debate ammunition.

Fetches recent news headlines via FMP /stock-news endpoint and performs
lightweight keyword-based sentiment classification. No new dependencies required.

Design:
    - News data is marked as [NEWS — 非基本面数据，仅供参考] to distinguish
      it from audited financial data
    - Sentiment is keyword-based: bullish/bearish/neutral counts
    - Agents can use news sentiment as supplementary arguments when financial
      data alone doesn't support their assigned position
"""

from __future__ import annotations

import logging
from typing import Any

from src.utils.financial_formatter import fmt_money

logger = logging.getLogger(__name__)

# ── Sentiment keyword lexicons ──────────────────────────────────────

_BULLISH_KEYWORDS = [
    "beat", "beats", "upgrade", "upgrades", "raised", "raise", "outperform",
    "strong", "growth", "surge", "surges", "rally", "record", "positive",
    "buyback", "dividend", "approval", "breakthrough", "expansion",
    "partnership", "launch", "guidance raised", "beat estimates",
    "盈利超预期", "超预期", "上调", "增长", "突破", "创新高",
    "回购", "分红", "合作", "获批", "扩张", "利好",
    "上调评级", "强于大盘", "买入", "增持",
]

_BEARISH_KEYWORDS = [
    "miss", "misses", "downgrade", "downgrades", "cut", "cuts", "underperform",
    "weak", "decline", "plunge", "plunges", "drop", "sell-off", "selloff",
    "probe", "investigation", "lawsuit", "fine", "penalty", "layoff",
    "layoffs", "restructuring", "write-down", "impairment", "warning",
    "guidance cut", "miss estimates", "recall", "short report",
    "盈利不及预期", "不及预期", "下调", "下滑", "暴跌", "下跌",
    "调查", "罚款", "诉讼", "裁员", "减记", "警告",
    "下调评级", "弱于大盘", "卖出", "减持",
]

# ── Public API ──────────────────────────────────────────────────────


def fetch_news(key: str, limit: int = 20) -> dict[str, Any]:
    """Fetch recent news for a ticker from FMP /stock-news endpoint.

    Args:
        key: FMP API key.
        limit: Max number of news articles to fetch.

    Returns:
        Dict with keys: headlines (list), sentiment_counts, summary_text.
        Empty dict if the endpoint is unavailable or returns no data.
    """
    try:
        import requests

        url = (
            f"https://financialmodelingprep.com/stable/stock-news"
            f"?symbol={key}&limit={limit}&apikey={settings_fmp_key()}"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"News API returned {resp.status_code}")
            return {}

        articles: list[dict[str, Any]] = resp.json()
        if not articles:
            return {}

        return _classify_news(articles, limit)

    except Exception as exc:
        logger.warning(f"News fetch failed (non-critical): {exc}")
        return {}


def settings_fmp_key() -> str:
    """Lazy import to avoid circular dependency."""
    from src.config.settings import settings
    return settings.fmp_api_key


def _classify_news(
    articles: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    """Classify articles by sentiment using keyword matching.

    Args:
        articles: Raw article list from FMP API.
        limit: Max articles to include.

    Returns:
        Structured news data dict.
    """
    headlines: list[dict[str, str]] = []
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    for article in articles[:limit]:
        title: str = article.get("title", "")
        text: str = article.get("text", "") or ""
        source: str = article.get("site", "") or article.get("source", "")
        date: str = article.get("publishedDate", "") or ""

        combined = (title + " " + (text[:300] if text else "")).lower()

        is_bullish = any(kw.lower() in combined for kw in _BULLISH_KEYWORDS)
        is_bearish = any(kw.lower() in combined for kw in _BEARISH_KEYWORDS)

        if is_bullish and not is_bearish:
            sentiment = "bullish"
            bullish_count += 1
        elif is_bearish and not is_bullish:
            sentiment = "bearish"
            bearish_count += 1
        else:
            sentiment = "neutral"
            neutral_count += 1

        headlines.append({
            "title": title,
            "source": source,
            "date": date,
            "sentiment": sentiment,
        })

    total = bullish_count + bearish_count + neutral_count
    bullish_pct = round(bullish_count / total * 100) if total > 0 else 0
    bearish_pct = round(bearish_count / total * 100) if total > 0 else 0

    # Determine overall tone
    if bullish_count > bearish_count * 1.5:
        overall = "偏正面"
    elif bearish_count > bullish_count * 1.5:
        overall = "偏负面"
    else:
        overall = "中性/混合"

    # Build summary text for injection into financial data
    summary_parts = [
        f"  总新闻数:               {total}",
        f"  正面 (bullish):         {bullish_count} ({bullish_pct}%)",
        f"  负面 (bearish):         {bearish_count} ({bearish_pct}%)",
        f"  中性 (neutral):         {neutral_count} ({total - bullish_count - bearish_count})",
        f"  整体情绪:               {overall}",
        "",
        "  近期标题 (按情绪分类):",
    ]

    for h in headlines[:15]:
        emoji = {"bullish": "+", "bearish": "-", "neutral": "o"}.get(h["sentiment"], "o")
        summary_parts.append(
            f"    [{emoji}] {h['title'][:120]}"
            f"  — {h.get('source', '?')}, {h.get('date', '?')}"
        )

    return {
        "headlines": headlines,
        "sentiment_counts": {
            "bullish": bullish_count,
            "bearish": bearish_count,
            "neutral": neutral_count,
            "overall": overall,
        },
        "summary_text": "\n".join(summary_parts),
    }


def format_news_section(news_data: dict[str, Any]) -> list[str]:
    """Format news data as a dimension section for to_text() output.

    Args:
        news_data: Result from fetch_news().

    Returns:
        List of formatted lines to append to financial data text.
    """
    lines: list[str] = []
    lines.append("\n[维度五: 近期新闻与市场情绪]")
    lines.append(
        "  [NEWS — 非基本面数据，基于新闻标题关键字情绪分类，仅供参考。"
        "可作为补充论据，但不应替代基本面分析。]"
    )

    summary = news_data.get("summary_text", "")
    if summary:
        lines.append(summary)
    else:
        lines.append("  (新闻数据不可得)")

    return lines
