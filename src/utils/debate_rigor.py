"""
Backwards-compatibility re-exports from the split modules.

New code should import directly from:
    - src.utils.anti_concession    (detect_concession, build_retry_prompt, get_prefill, get_fewshot)
    - src.utils.factuality_checker (verify_numbers_in_response, build_factuality_retry_prompt)
"""

from __future__ import annotations

from src.utils.anti_concession import (
    detect_concession,
    build_retry_prompt,
    get_prefill,
    get_fewshot,
)
from src.utils.factuality_checker import (
    verify_numbers_in_response,
    build_factuality_retry_prompt,
    _extract_known_numbers,
)

__all__ = [
    "detect_concession",
    "build_retry_prompt",
    "get_prefill",
    "get_fewshot",
    "verify_numbers_in_response",
    "build_factuality_retry_prompt",
    "_extract_known_numbers",
]
