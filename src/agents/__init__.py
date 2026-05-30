"""
Debate agents: Bull (long), Bear (short), CIO (judge).
"""

from __future__ import annotations

from src.agents.debate_agent import run_debate_agent, run_bull_agent, run_bear_agent
from src.agents.cio_agent import run_cio_agent, CIOVerdict

__all__ = [
    "run_debate_agent",
    "run_bull_agent",
    "run_bear_agent",
    "run_cio_agent",
    "CIOVerdict",
]
