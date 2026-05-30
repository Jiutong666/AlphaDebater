from __future__ import annotations

"""
全局配置管理模块。

使用 pydantic-settings 从 .env 文件和环境变量中加载配置，
提供类型校验和默认值。
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """AlphaDebater 全局配置。

    所有配置项可通过 .env 文件或环境变量覆盖。
    """

    # ── LLM 配置 ─────────────────────────────────────────────
    llm_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="LLM API 地址 (OpenAI 兼容格式)",
    )
    llm_api_key: str = Field(
        default="",
        description="API 密钥 (从 .env 或环境变量加载)",
    )
    llm_model: str = Field(
        default="deepseek-chat",
        description="模型名称",
    )

    # ── 辩论配置 ─────────────────────────────────────────────
    max_debate_rounds: int = Field(
        default=3,
        ge=1,
        le=5,
        description="最大辩论轮数 (1-5)",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM 采样温度 (低温度提升指令遵循度和数值准确性)",
    )
    max_tokens: int = Field(
        default=8192,
        ge=256,
        le=8192,
        description="单次 LLM 调用最大 Token 数",
    )

    # ── 数据源配置 ───────────────────────────────────────────
    data_source: str = Field(
        default="fmp",
        description="数据源: fmp, yfinance, sample",
    )
    fmp_api_key: str = Field(
        default="",
        description="Financial Modeling Prep API Key (从 .env 或环境变量加载)",
    )
    enable_news: bool = Field(
        default=False,
        description="是否启用新闻情绪数据 (维度五)。需要 FMP API Key。",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# 全局单例
settings = Settings()
