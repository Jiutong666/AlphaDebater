"""
Parameterized debate agent — single implementation for both Bull and Bear roles.

Replaces the previous copy-pasted bull_agent.py and bear_agent.py.
Role-specific behavior (prompt, fewshot, prefill, context) is selected
by the `role` parameter at call time.

Industrial-grade features:
    - Adversarial few-shot injection (anti-alignment pollution)
    - Concession detection + auto retry (anti-politeness concession)
    - Physical context slicing (full opponent speech, no compression)
    - Factuality verification (hallucinated number detection)
    - PE anchor validation (fabricated PE multiple detection)
"""

from __future__ import annotations

from typing import Literal

from src.models.state import DebateState
from src.models.messages import DebateMessage
from src.utils.llm_client import llm_client
from src.utils.anti_concession import (
    detect_concession,
    build_retry_prompt,
    get_prefill,
    get_fewshot,
)
from src.utils.factuality_checker import (
    verify_numbers_in_response,
    build_factuality_retry_prompt,
)
from src.utils.calculation_tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from src.utils.context_slicer import build_bull_context, build_bear_context
from src.utils.prompt_loader import load_prompt
from src.utils.pe_validator import (
    extract_pe_anchors,
    detect_fact_fabrication,
    validate_tool_call_pe,
    build_pe_retry_prompt,
)

Role = Literal["bull", "bear"]

_ROLE_META: dict[Role, dict[str, str]] = {
    "bull": {
        "prompt_file": "bull_system",
        "display_name": "Bull / 红军多头",
        "display_emoji": "🐂",
        "error_name": "红军多头",
    },
    "bear": {
        "prompt_file": "bear_system",
        "display_name": "Bear / 蓝军空头",
        "display_emoji": "🐻",
        "error_name": "蓝军空头",
    },
}


def _build_system_prompt(role: Role) -> str:
    """Build the full system prompt: base prompt + few-shot examples."""
    meta = _ROLE_META[role]
    base = load_prompt(meta["prompt_file"])
    return base + get_fewshot(role)


def run_debate_agent(
    state: DebateState,
    role: Role,
    *,
    max_concession_retries: int = 2,
) -> dict[str, object]:
    """Execute one round of debate reasoning for the given role.

    Context slicing: injects financial data + opponent's full prior speech.
    Concession detection: if output contains concession language, auto-retry.
    Factuality check: if output contains hallucinated numbers, auto-retry.

    Args:
        state: Current debate state.
        role:  "bull" or "bear".
        max_concession_retries: Max retries after concession/factuality detection.

    Returns:
        State update dict containing a DebateMessage list under "messages".
    """
    meta = _ROLE_META[role]
    round_num: int = state["current_round"]
    context_builder = build_bull_context if role == "bull" else build_bear_context

    user_message: str = context_builder(state)
    full_system_prompt: str = _build_system_prompt(role)
    financial_data: str = state["financial_data"]

    prefill = get_prefill(role, round_num)
    response: str = ""
    retry_warning: str = ""

    for attempt in range(max_concession_retries + 1):
        try:
            msg = user_message
            if retry_warning:
                msg = user_message + "\n\n" + retry_warning

            response, tool_log = llm_client.chat_with_tools(
                system_prompt=full_system_prompt,
                user_message=msg,
                tools=TOOL_DEFINITIONS,
                tool_registry=TOOL_REGISTRY,
                prefill=prefill if attempt == 0 else "",
            )
            if tool_log:
                tool_names = [t["tool"] for t in tool_log]
                print(f"  [{meta['display_name']} 工具调用]: {', '.join(tool_names)}")
        except Exception as exc:
            response = (
                f"⚠️ {meta['error_name']}推理出错: {exc}\n\n"
                "基于已知数据维持此前立场不变。"
            )
            break

        # Check 1: concession detection
        concession_hits = detect_concession(response)
        if concession_hits:
            retry_warning = build_retry_prompt(concession_hits, "zh")
            continue

        # Check 2: factuality verification (hallucinated numbers)
        suspicious_nums = verify_numbers_in_response(response, financial_data)
        if suspicious_nums:
            retry_warning = build_factuality_retry_prompt(suspicious_nums)
            continue

        # Check 3: Fact vs Assumption validation
        # Only flags fact falsification (claiming a number IS in the data
        # when it's not). Valuation assumptions (e.g. "I think PE should be 20x")
        # are legitimate and NEVER blocked.
        pe_anchors = extract_pe_anchors(financial_data)
        text_fab = detect_fact_fabrication(response, pe_anchors)
        tool_inv = validate_tool_call_pe(tool_log, pe_anchors)
        if text_fab or tool_inv:
            retry_warning = build_pe_retry_prompt(text_fab, tool_inv, pe_anchors)
            continue

        break

    new_message = DebateMessage(
        round=round_num,
        speaker=role,
        content=response,
    )

    return {"messages": [new_message]}


# ── Backwards-compatible thin wrappers ────────────────────────────

def run_bull_agent(state: DebateState, **kwargs: object) -> dict[str, object]:
    """Thin wrapper — delegates to run_debate_agent(role='bull')."""
    return run_debate_agent(state, "bull", **kwargs)  # type: ignore[arg-type]


def run_bear_agent(state: DebateState, **kwargs: object) -> dict[str, object]:
    """Thin wrapper — delegates to run_debate_agent(role='bear')."""
    return run_debate_agent(state, "bear", **kwargs)  # type: ignore[arg-type]
