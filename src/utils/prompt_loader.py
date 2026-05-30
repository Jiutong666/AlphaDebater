"""
Prompt loader — reads prompt templates from the prompts/ directory.

Prompts are stored as .txt files so non-engineers can review and edit them.
The loader resolves paths relative to the project root.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_CACHE: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Load a prompt template from prompts/<name>.txt.

    Results are cached in memory so repeated calls hit zero disk I/O.
    """
    if name not in _CACHE:
        path = _PROMPTS_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        _CACHE[name] = path.read_text(encoding="utf-8")
    return _CACHE[name]


def clear_cache() -> None:
    """Clear the in-memory prompt cache (useful during development)."""
    _CACHE.clear()
