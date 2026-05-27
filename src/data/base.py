from __future__ import annotations

"""
数据源抽象基类模块。

定义 DataSource 接口，所有数据源（yfinance、Bloomberg、Wind 等）
均需实现此接口，实现"即插即用"的数据源切换。
"""

from abc import ABC, abstractmethod
from typing import Any


class DataSource(ABC):
    """金融数据源抽象基类。

    子类必须实现 fetch() 方法，返回标准化的财务数据字典。
    """

    @abstractmethod
    def fetch(self, ticker: str) -> dict[str, Any]:
        """获取指定股票的全部财务数据。

        Args:
            ticker: 股票代码 (如 "AAPL", "600519.SS")。

        Returns:
            包含所有可用财务指标的字典。

        Raises:
            ValueError:   股票代码无效或数据源不支持该标的。
            RuntimeError: 网络错误或数据源不可用。
        """
        ...

    @abstractmethod
    def to_text(self, data: dict[str, Any]) -> str:
        """将数据字典序列化为 LLM 可读的文本格式。

        Args:
            data: fetch() 返回的数据字典。

        Returns:
            格式化后的文本字符串，将作为 Prompt 注入 LLM 上下文。
        """
        ...
