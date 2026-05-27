from __future__ import annotations

"""
CIO Agent — 首席投资官 / 最终裁判。

角色定位：不持任何预设立场，像法官一样严格评估红蓝双方
的论证质量、逻辑自洽性和数据引用准确度，给出加权目标价。

工业级特性：
    - Pydantic 严格输出 Schema (拒绝脏数据进入下游)
    - 四层降级 JSON 解析 (json → strip_markdown → manual_repair → regex)
    - 解析失败自动 Retry (将错误反馈给 LLM 要求修正)
    - 所有数值字段自带范围校验
"""

import json
import re
import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.models.state import DebateState
from src.utils.llm_client import llm_client
from src.utils.context_slicer import build_cio_context

logger = logging.getLogger(__name__)

# ── Pydantic 输出 Schema ────────────────────────────────────────

class CIOVerdict(BaseModel):
    """CIO 裁判结果的严格 Schema。

    所有字段带有 Pydantic 校验，确保下游消费者
    (前端看板、数据库写入、API 返回) 拿到的是干净数据。
    """

    bull_total_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="红军多头总分 (0-100)",
    )
    bear_total_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="蓝军空头总分 (0-100)",
    )
    bull_target_price: float = Field(
        ...,
        gt=0,
        description="红军多头提出的目标价 (必须 > 0)",
    )
    bear_target_price: float = Field(
        ...,
        gt=0,
        description="蓝军空头提出的目标价 (必须 > 0)",
    )
    bull_confidence: float = Field(
        ...,
        ge=0,
        le=100,
        description="红军信心分 (0-100)",
    )
    bear_confidence: float = Field(
        ...,
        ge=0,
        le=100,
        description="蓝军信心分 (0-100)",
    )
    final_target_price: float = Field(
        default=0.0,
        ge=0,
        description="加权计算后的最终目标价",
    )
    reasoning: str = Field(
        default="",
        min_length=10,
        description="详细判词",
    )

    @field_validator("bull_target_price", "bear_target_price", mode="before")
    @classmethod
    def _parse_price_string(cls, v: object) -> float:
        """允许 LLM 输出 "$200.00" 这样的字符串格式。"""
        if isinstance(v, str):
            cleaned = v.replace("$", "").replace(",", "").strip()
            return float(cleaned)
        return float(v)

    @field_validator("bull_total_score", "bear_total_score", "bull_confidence", "bear_confidence", mode="before")
    @classmethod
    def _parse_int_string(cls, v: object) -> int:
        """允许 LLM 输出 "85" 这样的字符串格式。"""
        if isinstance(v, str):
            return int(float(v))
        return int(v)

    def compute_final_target(self) -> "CIOVerdict":
        """根据加权公式自动计算最终目标价。

        formula: (bull_target × bull_conf + bear_target × bear_conf) / (bull_conf + bear_conf)
        """
        total_conf = self.bull_confidence + self.bear_confidence
        if total_conf > 0:
            self.final_target_price = round(
                (self.bull_target_price * self.bull_confidence
                 + self.bear_target_price * self.bear_confidence)
                / total_conf,
                2,
            )
        return self


# ── 系统提示词 ─────────────────────────────────────────────────

CIO_SYSTEM_PROMPT = """# 角色：首席投资官 (CIO) / 辩论裁判

你是一家顶级对冲基金的 CIO。你的职责不是参与辩论，而是**审判这场辩论**。

你需要像法官一样严格评估红蓝双方的论证质量。你没有任何预设的多空立场，
你的唯一标准是：**谁的数据引用更精准？谁的逻辑链条更严密？
谁在交叉质询中更有效地摧毁了对方的论点？**

---

## 反幻觉铁律 (Anti-Hallucination — 最高优先级)

1. **只评判引用数据的准确性**：检查双方的论证是否严格基于提供的财务数据。
   如果某一方引用了数据中不存在的数字，必须在评分中严厉扣分。
   如果数据中某字段为 N/A 而某一方假装它有值，直接判定该方在该项不得分。

2. **禁止在判词中编造数据**：你的 reasoning 中只能引用辩论记录中出现的数字
   或提供的财务数据中存在的数字。不得发明任何数据来支持你的评分判断。

3. **数据缺失的处理**：当双方都因数据缺失而无法充分论证某个维度时，
   该维度的评分应偏向中性，不得因数据缺失而惩罚任何一方。

---

## 评审维度 (每项满分 25 分，总分 100 分)

### 1. 论据相关性 (Relevance) — 0~25 分
是否严格基于给定的财务数据展开论证？有无脱离数据凭空臆造？
如发现捏造不存在的数据点，本项直接判 0 分。

### 2. 逻辑严密性 (Logic) — 0~25 分
论证链条是否完整？前提-推理-结论是否自洽？

### 3. 交叉质询杀伤力 (Cross-Examination Damage) — 0~25 分
是否有效拆解了对方的论点？是敷衍反驳还是精确打击？

### 4. 论据冲击力 (Conviction) — 0~25 分
论点是否有穿透力？是否抓住了最关键的矛盾点？

## 输出格式（严格执行——这是机器可读的 JSON Schema）

你必须输出**纯 JSON**，每行一个键值对。不得用 Markdown 代码块包裹。
不得添加任何前缀或后缀文字。

{
  "bull_total_score": 85,
  "bear_total_score": 72,
  "bull_target_price": 200.00,
  "bear_target_price": 150.00,
  "bull_confidence": 85,
  "bear_confidence": 72,
  "final_target_price": 176.67,
  "reasoning": "红军在逻辑严密性和交叉质询杀伤力两个维度..."
}

### 目标价加权公式
final_target_price = (bull_target_price × bull_confidence + bear_target_price × bear_confidence) / (bull_confidence + bear_confidence)

要求：不用 Markdown 代码块，直接输出纯 JSON。不能少逗号，不能多逗号。"""


# ── 四层 JSON 解析引擎 ─────────────────────────────────────────

def _try_json_loads(text: str) -> dict[str, Any] | None:
    """第 1 层：直接解析。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _try_strip_markdown(text: str) -> dict[str, Any] | None:
    """第 2 层：去除 Markdown 代码块后再解析。

    处理 LLM 最常见的"污染"——在 JSON 外包裹 ```json ... ```。
    """
    # 尝试提取 ```json ... ``` 块
    fence_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个 { 到最后一个 } 之间的内容
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _try_manual_repair(text: str) -> dict[str, Any] | None:
    """第 3 层：手动修复常见 JSON 语法错误。

    处理：
        - 尾随逗号 (trailing comma)
        - 单引号替代双引号
        - 缺失的闭合括号
        - 注释行 (// 或 #)
    """
    try:
        # 提取 JSON 对象主体
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if not brace_match:
            return None
        candidate = brace_match.group()

        # 去除注释行
        candidate = re.sub(r'^\s*//.*$', '', candidate, flags=re.MULTILINE)
        candidate = re.sub(r'^\s*#.*$', '', candidate, flags=re.MULTILINE)

        # 去除尾随逗号 (在 ] 或 } 之前)
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)

        # 单引号转双引号 (简单场景)
        # 检查是否大量使用单引号
        if candidate.count("'") > candidate.count('"'):
            # 将键的单引号替换为双引号
            candidate = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', candidate)
            # 将值的单引号替换为双引号
            candidate = re.sub(r":\s*'([^']*)'", r': "\1"', candidate)

        # 尝试解析修复后的文本
        return json.loads(candidate)

    except (json.JSONDecodeError, ValueError):
        return None


def _try_regex_fallback(text: str) -> dict[str, Any]:
    """第 4 层：正则表达式暴力提取关键字段。

    这是最后的兜底方案，不依赖 JSON 结构，直接从文本中抠数值。
    分两遍扫描：
        Pass 1: 尝试 JSON 格式 (带引号和冒号)
        Pass 2: 尝试完全非结构化的自然语言 (如 "bull_total_score 大概是 78 分")
    """

    def _extract_float(pattern: str, text: str, default: float = 0.0) -> float:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace("$", "").replace(",", ""))
        return default

    def _extract_int(pattern: str, text: str, default: int = 50) -> int:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(float(match.group(1)))
        return default

    # ── Pass 1: JSON 格式 ──
    bull_score = _extract_int(r'"bull_total_score"\s*:\s*(\d+)', text)
    bear_score = _extract_int(r'"bear_total_score"\s*:\s*(\d+)', text)
    bull_target = _extract_float(r'"bull_target_price"\s*:\s*([\d.]+)', text)
    bear_target = _extract_float(r'"bear_target_price"\s*:\s*([\d.]+)', text)
    bull_conf = _extract_float(r'"bull_confidence"\s*:\s*([\d.]+)', text, float(bull_score))
    bear_conf = _extract_float(r'"bear_confidence"\s*:\s*([\d.]+)', text, float(bear_score))
    final_price = _extract_float(r'"final_target_price"\s*:\s*([\d.]+)', text)

    # ── Pass 2: 非结构化自然语言 ──
    # 如果 Pass 1 全部没命中（说明不是 JSON 格式），用更宽松的模式
    if bull_score == 50 and bull_target == 0.0:
        bull_score = _extract_int(r'bull_total_score.*?(\d+)', text, default=0)
        bear_score = _extract_int(r'bear_total_score.*?(\d+)', text, default=0)
        bull_target = _extract_float(r'bull_target_price.*?(\d+\.?\d*)', text)
        bear_target = _extract_float(r'bear_target_price.*?(\d+\.?\d*)', text)
        bull_conf = _extract_float(r'bull_confidence.*?(\d+\.?\d*)', text, float(bull_score))
        bear_conf = _extract_float(r'bear_confidence.*?(\d+\.?\d*)', text, float(bear_score))
        final_price = _extract_float(r'final_target_price.*?(\d+\.?\d*)', text)

    # 提取 reasoning
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]{10,})"', text)
    if not reasoning_match:
        # 非结构化：取最后一个长句子
        reasoning_match = re.search(r'reasoning.*?(.{20,})', text, re.IGNORECASE)
    reasoning = reasoning_match.group(1) if reasoning_match else (text[:500] if text else "")

    # 如果没有提取到 final_target_price，用公式计算
    if final_price == 0.0 and (bull_conf + bear_conf) > 0:
        final_price = round(
            (bull_target * bull_conf + bear_target * bear_conf) / (bull_conf + bear_conf), 2
        )

    return {
        "bull_total_score": bull_score,
        "bear_total_score": bear_score,
        "bull_target_price": bull_target,
        "bear_target_price": bear_target,
        "bull_confidence": bull_conf,
        "bear_confidence": bear_conf,
        "final_target_price": final_price,
        "reasoning": reasoning,
    }


def _four_layer_parse(response: str) -> dict[str, Any]:
    """四层降级 JSON 解析器。

    按优先级依次尝试：
        1. 直接 json.loads
        2. 去 Markdown 包裹后 json.loads
        3. 手动修复常见语法错误后 json.loads
        4. 正则表达式暴力提取

    Args:
        response: LLM 原始响应。

    Returns:
        成功解析的字典。

    Raises:
        ValueError: 四层全部失败时抛出。
    """
    layers = [
        ("直接解析", _try_json_loads),
        ("去Markdown包裹", _try_strip_markdown),
        ("手动修复语法", _try_manual_repair),
        ("正则暴力提取", lambda t: _try_regex_fallback(t)),  # 这一层永不返回 None
    ]

    for layer_name, parser in layers:
        result = parser(response)
        if result is not None:
            logger.debug(f"CIO JSON 解析成功 (层级: {layer_name})")
            return result

    raise ValueError("CIO 响应解析失败：四层降级全部未能提取有效 JSON。")


# ── Pydantic 校验 + 自动修复 ───────────────────────────────────

def _validate_and_fix(data: dict[str, Any]) -> CIOVerdict:
    """用 Pydantic 校验并修复解析后的数据。

    对缺失字段填充默认值，对越界字段做裁剪。

    Args:
        data: 四层解析后的原始字典。

    Returns:
        校验通过的 CIOVerdict 实例。

    Raises:
        ValueError: 关键字段缺失且无法修复时抛出。
    """
    # 确保关键字段存在
    defaults = {
        "bull_total_score": 50,
        "bear_total_score": 50,
        "bull_target_price": 100.0,
        "bear_target_price": 100.0,
        "bull_confidence": 50.0,
        "bear_confidence": 50.0,
        "reasoning": "CIO 解析降级：无法提取完整判词。",
    }
    for key, default in defaults.items():
        data.setdefault(key, default)

    # Pydantic 校验
    try:
        verdict = CIOVerdict(**data)
    except Exception:
        # 如果 Pydantic 校验失败，尝试放宽约束重试
        cleaned = {k: v for k, v in data.items() if k in CIOVerdict.model_fields}
        verdict = CIOVerdict(**cleaned)

    # 自动计算目标价
    verdict.compute_final_target()
    return verdict


# ── 带 Retry 的主逻辑 ──────────────────────────────────────────

def run_cio_agent(
    state: DebateState,
    *,
    max_retries: int = 2,
) -> dict[str, object]:
    """执行 CIO 最终裁判（带解析失败自动重试）。

    流程：
        1. 构建 Prompt → 调用 LLM
        2. 四层 JSON 解析
        3. Pydantic 校验
        4. 若失败 → 将解析错误反馈给 LLM，重试

    这确保了即使 LLM 第 1 次输出格式不正确，
    也能通过错误反馈让 LLM 自我修正。

    Args:
        state:     辩论完成后的状态。
        max_retries: JSON 解析失败后的最大重试次数。

    Returns:
        包含 final_target_price, bull_confidence, bear_confidence,
        final_verdict 的更新字典。
    """
    # ── 构建 CIO 输入（使用 Context Slicer 读取 messages）──
    base_user_message: str = build_cio_context(state)

    # ── 带 Retry 的 LLM 调用 ──
    last_response: str = ""
    last_error: str = ""

    for attempt in range(max_retries + 1):
        # 构建消息（retry 时附加上次错误信息）
        if attempt == 0:
            user_message = base_user_message
        else:
            user_message = (
                base_user_message
                + f"\n\n⚠️ 你上一次的输出无法被解析。错误信息：\n{last_error}\n\n"
                + "请重新输出**纯 JSON**（不要用 ```json 包裹，不要有尾随逗号，不要有注释）。"
            )

        # 调用 LLM
        try:
            response = llm_client.chat(
                system_prompt=CIO_SYSTEM_PROMPT,
                user_message=user_message,
            )
            last_response = response
        except Exception as exc:
            logger.warning(f"CIO LLM 调用失败 (attempt {attempt + 1}): {exc}")
            if attempt < max_retries:
                last_error = str(exc)
                continue
            # 最终降级
            return _build_fallback_result(f"LLM 调用失败: {exc}")

        # 四层解析
        try:
            raw_data = _four_layer_parse(response)
        except ValueError as exc:
            logger.warning(f"CIO JSON 解析失败 (attempt {attempt + 1}): {exc}")
            last_error = str(exc)
            if attempt < max_retries:
                continue
            # 最终降级：用正则从原始响应中抠数据
            raw_data = _try_regex_fallback(response)

        # Pydantic 校验
        try:
            verdict = _validate_and_fix(raw_data)
            # 成功！返回
            return {
                "final_target_price": verdict.final_target_price,
                "bull_confidence": verdict.bull_confidence,
                "bear_confidence": verdict.bear_confidence,
                "final_verdict": verdict.reasoning,
            }
        except Exception as exc:
            logger.warning(f"CIO Pydantic 校验失败 (attempt {attempt + 1}): {exc}")
            last_error = str(exc)
            if attempt < max_retries:
                continue
            return _build_fallback_result(f"Pydantic 校验失败: {exc}")

    # 不应该到达这里
    return _build_fallback_result("未知错误")


def _build_fallback_result(reason: str) -> dict[str, object]:
    """构建安全的降级结果。

    Args:
        reason: 失败原因描述。

    Returns:
        包含安全默认值的字典。
    """
    logger.error(f"CIO 降级触发: {reason}")
    return {
        "final_target_price": 0.0,
        "bull_confidence": 50.0,
        "bear_confidence": 50.0,
        "final_verdict": (
            f"⚠️ CIO 裁判系统降级：自动评分模块因以下原因触发安全模式：\n"
            f"{reason}\n\n"
            f"建议：请人工审核辩论记录后手动给出目标价。"
        ),
    }
