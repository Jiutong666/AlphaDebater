from __future__ import annotations

"""
终端美化输出模块。

提供带 Emoji 状态标识的结构化打印函数，提升 CLI 可读性。
"""

import shutil
from typing import Any


# ── 终端宽度 ──────────────────────────────────────────────────

def _get_terminal_width() -> int:
    """获取终端宽度，默认 100 列。"""
    return shutil.get_terminal_size().columns or 100


def _sep(char: str = "─") -> str:
    """生成一条横跨终端的水平分割线。"""
    return char * _get_terminal_width()


# ── 状态打印函数 ──────────────────────────────────────────────

def print_header(ticker: str) -> None:
    """打印程序启动横幅。

    Args:
        ticker: 股票代码。
    """
    print()
    print(_sep("▬"))
    print(f"  🏦  AlphaDebater — 对抗式股票预测系统")
    print(f"  📊  分析标的: {ticker.upper()}")
    print(_sep("▬"))
    print()


def print_section(emoji: str, title: str) -> None:
    """打印章节标题。

    Args:
        emoji: 章节图标。
        title: 章节名称。
    """
    print(f"\n  {emoji}  {title}")
    print(f"  {_sep('─')}")


def print_data_summary(ticker: str, info: dict[str, Any]) -> None:
    """打印抓取到的数据摘要。

    Args:
        ticker: 股票代码。
        info:   数据字典。
    """
    print_section("📡", f"数据抓取完成 — {ticker.upper()}")

    # 尝试展示关键字段
    key_fields = [
        ("currentPrice",          "当前价格"),
        ("targetMeanPrice",       "分析师目标均价"),
        ("recommendationKey",     "分析师共识"),
        ("marketCap",             "市值"),
        ("trailingPE",            "市盈率 (TTM)"),
        ("forwardPE",             "远期市盈率"),
        ("priceToBook",           "市净率"),
        ("returnOnEquity",        "ROE"),
        ("revenueGrowth",         "营收增长率"),
        ("earningsGrowth",        "盈利增长率"),
        ("debtToEquity",          "负债权益比"),
        ("freeCashflow",          "自由现金流"),
        ("fiftyTwoWeekHigh",      "52周最高"),
        ("fiftyTwoWeekLow",       "52周最低"),
        ("beta",                  "Beta"),
    ]

    for field, label in key_fields:
        value = info.get(field, "N/A")
        if value is not None:
            print(f"     • {label}: {value}")

    print()


def print_round_header(round_num: int) -> None:
    """打印辩论轮次标题。

    Args:
        round_num: 当前轮次编号 (1-indexed)。
    """
    print(f"\n  {'▬' * 40}")
    print(f"  ⚔️  第 {round_num} 轮 交叉质询")
    print(f"  {'▬' * 40}\n")


def print_agent_message(agent_name: str, emoji: str, content: str) -> None:
    """打印某个 Agent 的发言。

    Args:
        agent_name: Agent 名称 (如 "Bull / 红军多头")。
        emoji:      Agent 图标。
        content:    发言内容。
    """
    print(f"  {emoji}  [{agent_name}]")
    print(f"  {'─' * (len(agent_name) + 6)}")
    # 为每行添加缩进
    for line in content.strip().split("\n"):
        print(f"     {line}")
    print()


def print_cio_verdict(
    target_price: float,
    bull_conf: float,
    bear_conf: float,
    verdict: str,
) -> None:
    """打印 CIO 最终裁决。

    Args:
        target_price: 最终目标价。
        bull_conf:    红军信心分。
        bear_conf:    蓝军信心分。
        verdict:      判词全文。
    """
    print(f"\n  {'▬' * 40}")
    print(f"  🎯  CIO 最终裁决")
    print(f"  {'▬' * 40}\n")

    print(f"     🐂 红军 (多头) 信心分: {bull_conf:.1f}%")
    print(f"     🐻 蓝军 (空头) 信心分: {bear_conf:.1f}%")
    print(f"     🎯 最终目标价: ${target_price:.2f}")
    print()
    print(f"  📋 判词:")
    for line in verdict.strip().split("\n"):
        print(f"     {line}")
    print()
    print(_sep("▬"))
    print(f"  ✅  分析完成")
    print(_sep("▬"))


def print_error(message: str) -> None:
    """打印错误信息。

    Args:
        message: 错误描述。
    """
    print(f"\n  ❌  错误: {message}\n")
