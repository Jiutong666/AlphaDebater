"""
Shared financial data formatter — eliminates copy-paste across data sources.

Provides `fmt_val`, `fmt_money`, and `build_sections` so every DataSource
declares its layout as a list of sections and delegates rendering here.
"""

from __future__ import annotations

from typing import Any, Callable


def fmt_val(info: dict[str, Any], key: str, fmt_spec: str = ".2f") -> str:
    """Format a single info-field value for display.

    Returns "N/A" when the key is missing or None.
    """
    val = info.get(key)
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:{fmt_spec}}"
    return str(val)


def fmt_money(val: object, compact: bool = False) -> str:
    """Format a large monetary value with T/B/M suffixes when compact=True."""
    if val is None:
        return "N/A"
    try:
        v = float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return str(val)
    if compact:
        if abs(v) >= 1e12:
            return f"${v / 1e12:.2f}T"
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


# ── Declarative section builder ───────────────────────────────────

FieldSpec = tuple[str, str, str]  # (label, info-key, format-spec)
PostProcessor = Callable[[dict[str, Any], list[str]], None]


class Section:
    """A named group of fields rendered together with a time-dimension label."""

    def __init__(
        self,
        title: str,
        fields: list[FieldSpec],
        *,
        time_dim: str = "",
        post: PostProcessor | None = None,
    ) -> None:
        self.title = title
        self.fields = fields
        self.time_dim = time_dim
        self.post = post

    def render(self, info: dict[str, Any]) -> list[str]:
        lines: list[str] = [self.title]
        if self.time_dim:
            lines.append(f"  {self.time_dim}")
        for label, key, fmt_spec in self.fields:
            if fmt_spec == "money":
                lines.append(f"  {label}: {fmt_money(info.get(key), compact=True)}")
            elif fmt_spec == "money_raw":
                lines.append(f"  {label}: {fmt_money(info.get(key))}")
            elif fmt_spec == "s":
                lines.append(f"  {label}: {fmt_val(info, key, 's')}")
            elif fmt_spec == "d":
                lines.append(f"  {label}: {fmt_val(info, key, 'd')}")
            else:
                lines.append(f"  {label}: {fmt_val(info, key, fmt_spec)}")
        if self.post:
            self.post(info, lines)
        return lines


def build_header(info: dict[str, Any]) -> list[str]:
    """Standard header block for every data source."""
    return [
        "=" * 60,
        f"股票名称: {info.get('shortName', 'N/A')} ({info.get('symbol', 'N/A')})",
        f"行业: {info.get('industry', 'N/A')} | 板块: {info.get('sector', 'N/A')}",
        "=" * 60,
    ]


def build_footer(source_label: str, extra_lines: list[str] | None = None) -> list[str]:
    """Standard footer with temporal discipline warning."""
    lines = [
        "",
        "=" * 60,
        f"数据来源: {source_label}",
        "以上为本次辩论的全部依据数据。红蓝双方必须严格基于上述数据展开论证。",
        "",
        "⚠️ 时间维度纪律 (Temporal Discipline):",
        "  - [LTM/TTM] = 过去12个月已审计数据 → 用于历史估值、盈利能力、财务健康",
        "  - [MRQ] = 最近季度同比 → 用于趋势检测、转折点识别",
        "  - [NTM/Forward] = 未来12个月预测 → 仅此维度可用于推导目标价",
        "  - 禁止跨维度混合计算 (如用 LTM 增长率论证 NTM 估值倍数)",
        "=" * 60,
    ]
    if extra_lines:
        lines = lines[:-1]  # drop last "="
        lines.extend(extra_lines)
        lines.append("=" * 60)
    return lines


def render_sections(
    info: dict[str, Any],
    sections: list[Section],
    *,
    header_info: dict[str, Any] | None = None,
    source_label: str = "Unknown",
) -> str:
    """Render a complete financial data text from declarative sections."""
    info_for_header = header_info if header_info is not None else info
    parts: list[str] = []
    parts.extend(build_header(info_for_header))
    for section in sections:
        parts.append("")
        parts.extend(section.render(info))
    parts.extend(build_footer(source_label))
    return "\n".join(parts)
