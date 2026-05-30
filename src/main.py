from __future__ import annotations

"""
AlphaDebater CLI 入口。

用法:
    python -m src.main AAPL
    python -m src.main TSLA --rounds 3
"""

import argparse
import sys
import warnings
from typing import Sequence

# 抑制 LangGraph / LangChain 内部 deprecation warning
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", module="langgraph")
warnings.filterwarnings("ignore", module="langchain")

from src.graph.debate_graph import run_debate
from src.config.settings import settings
from src.utils.printer import print_error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 命令行参数列表，None 则使用 sys.argv。

    Returns:
        解析后的命名空间。
    """
    parser = argparse.ArgumentParser(
        prog="alphadebater",
        description="🏦  AlphaDebater — 对抗式股票预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.main AAPL
  python -m src.main TSLA --rounds 5
        """,
    )

    parser.add_argument(
        "ticker",
        type=str,
        help="股票代码 (例如: AAPL, TSLA, 600519.SS)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=settings.max_debate_rounds,
        help=f"辩论轮数 (默认: {settings.max_debate_rounds})",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="强制使用离线样本数据 (跳过 yfinance)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖 LLM 模型 (如 deepseek-chat, gpt-4o, claude-sonnet-4-6)",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """主入口函数。

    Args:
        argv: 命令行参数，None 则使用 sys.argv。
    """
    args = parse_args(argv)

    # 覆盖配置中的轮数
    if args.rounds != settings.max_debate_rounds:
        settings.max_debate_rounds = args.rounds
    # 强制使用离线样本数据
    if args.sample:
        settings.data_source = "sample"
    # 覆盖 LLM 模型
    if args.model:
        from src.utils.llm_client import llm_client
        settings.llm_model = args.model
        llm_client.model = args.model
        print(f"\n  🤖 使用模型: {args.model}\n")

    ticker: str = args.ticker.strip().upper()

    try:
        final_state = run_debate(ticker)

        # 最终输出已在 run_debate 内部通过 printer 完成
        # 此处可以添加后续逻辑（如写入文件、推送通知等）

    except KeyboardInterrupt:
        print("\n\n  ⏹️  用户中断辩论。\n")
        sys.exit(0)
    except ValueError as exc:
        print_error(str(exc))
        sys.exit(1)
    except Exception as exc:
        print_error(f"未预期的错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
