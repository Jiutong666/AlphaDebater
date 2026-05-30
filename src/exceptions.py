"""
AlphaDebater custom exception hierarchy.

All library-specific exceptions inherit from AlphaDebaterError,
giving callers a single root to catch.
"""

from __future__ import annotations


class AlphaDebaterError(Exception):
    """Root of the AlphaDebater exception hierarchy."""


# ── Data layer ────────────────────────────────────────────────────

class DataSourceError(AlphaDebaterError):
    """Raised when a data source fails to fetch or parse data."""


class InvalidTickerError(DataSourceError):
    """Raised when the ticker symbol is invalid or unsupported."""


class DataSourceUnavailableError(DataSourceError):
    """Raised when the data source is temporarily unavailable (rate limit, network)."""


# ── LLM layer ─────────────────────────────────────────────────────

class LLMError(AlphaDebaterError):
    """Raised when an LLM API call fails after all retries."""


class LLMParseError(LLMError):
    """Raised when the LLM response cannot be parsed (e.g. invalid JSON)."""


# ── Validation layer ──────────────────────────────────────────────

class ValidationError(AlphaDebaterError):
    """Raised when data fails Pydantic or business-logic validation."""


class ConcessionDetectedError(ValidationError):
    """Raised when an agent response contains forbidden concession language."""


class FactualityError(ValidationError):
    """Raised when an agent response contains hallucinated numbers."""
