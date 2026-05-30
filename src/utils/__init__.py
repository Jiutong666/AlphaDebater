"""
Utility modules: LLM client, formatting, anti-concession, context slicing.
"""

from __future__ import annotations

from src.utils.llm_client import LLMClient, llm_client
from src.utils.financial_formatter import fmt_val, fmt_money

__all__ = ["LLMClient", "llm_client", "fmt_val", "fmt_money"]
