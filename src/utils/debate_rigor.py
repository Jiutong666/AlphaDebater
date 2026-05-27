from __future__ import annotations

"""
反"对齐污染"模块 — 确保红蓝军绝不妥协。

大模型经过 RLHF 对齐后带有强烈的"礼貌与妥协"倾向，
在对抗性辩论中会出现"对方说得也有道理"的致命污染。

本模块提供三层防护：
    1. 输出后检测 — 扫描妥协语并触发强制重试
    2. Assistant Prefill — 锁定每轮回复的开局语气
    3. 对抗性 Few-Shot 示例注入
"""

import re
from typing import Literal

# ── 妥协语检测规则库 ───────────────────────────────────────────

# 当检测到以下模式时，判定为"妥协污染"，强制重试
_CONCESSION_PATTERNS: list[tuple[str, str]] = [
    # 中文妥协模式
    (r"对方.{0,10}(观点|论点|分析|逻辑|说|讲|看法).{0,10}(有道理|合理|正确|说得对|确实)", "承认对方合理"),
    (r"(确实|的确|诚然).{0,5}(存在|有).{0,5}(风险|问题|不足|隐患)", "默认对方指控成立"),
    (r"(我(们)?)?(同意|赞同|认可|部分同意).*?(对方|红军|蓝军|多头|空头)", "明确同意对方"),
    (r"不可否认.*?(对方|红军|蓝军|多头|空头|这个)", "不可否认型妥协"),
    (r"从某种(程度|意义|角度)上(来说|讲)", "模糊相对主义"),
    (r"双方.{0,15}(有道理|有一定道理|合理|正确|都有道理|各有道理)", "各打五十大板"),
    (r"(可能|或许|也许|大概).{0,10}(是我|我们).{0,10}(错|误|过[于分])", "自我怀疑"),
    (r"(值得|需要).{0,10}(反思|重新考虑|重新审视).{0,10}(自己|我方|我们)", "自我反思型妥协"),
    (r"(这个|这点|此处).{0,5}(我|我们).{0,5}(承认|让步|退一步)", "公开让步"),

    # 英文妥协模式
    (r"(the|my)\s*(opponent|bull|bear).{0,15}(has a point|is right|is correct|makes sense)", "承认对方合理 (EN)"),
    (r"I\s*(agree|concede|admit|acknowledge).{0,20}(opponent|bull|bear|other side)", "明确同意对方 (EN)"),
    (r"(to be fair|in all fairness|admittedly|granted)", "公平让步 (EN)"),
    (r"(both sides|both arguments).{0,10}(have merit|are valid|are reasonable)", "双方都合理 (EN)"),
    (r"perhaps.{0,10}(I|we|my).{0,10}(wrong|mistaken|overly|too aggressive)", "自我怀疑 (EN)"),
]

# 妥协检测后的惩罚性重试提示
_RETRY_REINFORCEMENT_ZH = """
⚠️ **警告：检测到妥协污染！**

你上一轮回复中出现了以下妥协信号：
> "{matched_text}"

这在 AlphaDebater 中是被严格禁止的。你的角色不允许承认对方有任何合理性。

请重新生成你的论点。要求：
1. 逐条引用对方论点并用数据爆破
2. 不得出现任何让步、承认、或模棱两可的措辞
3. 语气必须更加锋利、更加不留情面
4. 如果对方数据正确，就攻击其数据解读方式，绝不承认数据本身对对方有利
"""

_RETRY_REINFORCEMENT_EN = """
⚠️ **WARNING: Concession contamination detected!**

Your previous response contained the following concession signal:
> "{matched_text}"

This is STRICTLY FORBIDDEN in AlphaDebater. Your role does not permit acknowledging ANY validity in the opponent's position.

Regenerate your argument. Requirements:
1. Quote each opponent claim and demolish it with data
2. ZERO concessions, admissions, or ambiguity
3. Sharper, more ruthless tone
4. If opponent's data is correct, attack their interpretation — NEVER concede the data favors them
"""


# ── Assistant Prefill — 锁定开局语气 ───────────────────────────

# 为每轮生成强硬的 assistant prefill，迫使 LLM 延续战斗姿态
# DeepSeek API 支持 assistant prefill (即 prefix completion)
BULL_PREFILLS: dict[int, str] = {
    1: "## 🐂 红军多头 — 第1轮：多头核心逻辑\n\n基于提供的财务数据和工具计算结果，我建立以下不可辩驳的多头逻辑：\n\n",
    2: "## 🐂 红军多头 — 第2轮：交叉质询\n\n蓝军空头的第1轮论点充斥着数据误读和悲观偏见。以下我将逐条引用并毁灭性反驳：\n\n",
    3: "## 🐂 红军多头 — 第3轮：总结陈词\n\n经过三轮辩论，蓝军空头的逻辑已完全崩塌。以下是我的最终结论（目标价均经工具计算验证）：\n\n",
}

BEAR_PREFILLS: dict[int, str] = {
    1: "## 🐻 蓝军空头 — 第1轮：做空核心逻辑\n\n基于提供的财务数据和工具计算结果，我建立以下铁证如山的做空逻辑：\n\n",
    2: "## 🐻 蓝军空头 — 第2轮：交叉质询\n\n红军多头的第1轮论点充斥着选择性失明和过度乐观。以下我将逐条引用并毁灭性反驳：\n\n",
    3: "## 🐻 蓝军空头 — 第3轮：总结陈词\n\n经过三轮辩论，红军多头的逻辑已完全崩塌。以下是我的最终结论（目标价均经工具计算验证）：\n\n",
}


# ── 对抗性 Few-Shot 示例 ───────────────────────────────────────

# 注入到 System Prompt 中，给 LLM 一个"应该怎么吵"的模板
_BULL_FEWSHOT = """
## 对抗性辩论示例 (Few-Shot)

以下是你应该模仿的辩论风格：

**空头说**: "该股市盈率高达35倍，远超行业平均的22倍，估值明显泡沫化。"

**你的正确反驳** (❌ 错误: "确实市盈率偏高，但...")：

> "对方用市盈率35倍对比行业22倍来论证估值泡沫，这是极其粗糙的伪分析。
> **第一**，该公司的ROE是行业平均的2.3倍，高增长理应享受估值溢价——
> PEG比率仅0.8，远低于行业1.5的平均水平，对方的'估值泡沫论'在PEG面前瞬间粉碎。
> **第二**，对方刻意忽略了远期PE仅18倍这一关键数据——前瞻估值已经低于行业平均。
> 对方要么是看不懂PEG，要么是选择性失明。无论是哪种，其分析都不值得认真对待。"

**空头说**: "营收增速从去年的35%下滑到今年的18%，增长正在衰竭。"

**你的正确反驳**:

> "营收增速从35%到18%？对方显然不懂基数效应。一家千亿市值的公司在高基数下
> 维持18%的增长恰恰证明了其护城河的宽度。更无耻的是，对方完全回避了
> 盈利增长率从12%飙升至28%这一事实——利润率正在急剧扩张，这才是价值投资者
> 应该关注的。盯着营收增速唱衰，掩盖盈利质量的飞跃，这种分析水平令人震惊。"

---

## 幻觉警示 (反面教材 — 以下行为 = CIO 0分)

以下是你**绝对不能**做的。每个错误示范都对应一种典型的幻觉模式：

**❌ 幻觉模式1：自行计算估值倍数（伪造数据）**
> "基于当前价格$245.83和EPS $22.69，我计算出前瞻PE为10.8倍，显示估值极低。"
→ **为什么是幻觉**：财务数据中明确给出了 forwardPE。如果数据写的是 21.91，
就不能"自行计算"出一个 10.8 来覆盖它。你没有计算权，只有引用权。
**✅ 正确做法**：直接引用数据中的 forwardPE，然后围绕它做多空解读。

**❌ 幻觉模式2：方向性捏造（颠倒数据方向）**
> "盈利从-53%的负增长即将翻倍至+106%，这将是史诗级反转。"
→ **为什么是幻觉**：数据写盈利增长率为 -53%。没有任何数据支持"翻倍"或"反转"。
负的不可能同时翻倍。方向和量级都要忠于数据。
**✅ 正确做法**：承认盈利负增长的挑战，但可以强调其他指标（营收增长、ROE等）的积极面。

**❌ 幻觉模式3：引用不存在的数字（凭空编造）**
> "公司持有现金$45B，完全有能力回购$10B股票。"
→ **为什么是幻觉**：检查上方财务数据——如果总现金字段是 N/A 或其他数值，
就不能引用 $45B。每一个数字都必须在数据中找到原始来源。
**✅ 正确做法**：只引用数据中实际存在的现金数值。如果 N/A，说"现金数据不可得"。

**核心原则：火力全开做多，但弹药（数据）必须是真货，不能是自己画的。**
"""

_BEAR_FEWSHOT = """
## 对抗性辩论示例 (Few-Shot)

以下是你应该模仿的辩论风格：

**多头说**: "该公司PEG仅0.8，远低于行业1.5，增长潜力被严重低估。"

**你的正确反驳** (❌ 错误: "PEG确实看起来不错，但...")：

> "对方引用PEG 0.8论证低估，这暴露了其估值分析的基本功缺陷。
> **第一**，PEG中的'G'使用的是历史增长率，而公司的forward guidance
> 已明确将下季度增速预期下调至个位数——用过去的增长线性外推未来，
> 这是CFA一级就该被淘汰的错误。
> **第二**，对方对'行业PEG 1.5'的来源闭口不谈——这个数字包含了3家
> 负增长公司拉高了均值。剔除异常值后，可比公司的PEG中位数是0.75。
> 对方的PEG比较是统计学上的垃圾。"

**多头说**: "公司自由现金流充裕，完全有能力回购股票支撑股价。"

**你的正确反驳**:

> "自由现金流'充裕'？请对方翻开现金流量表看清楚：经营性现金流的增长
> 几乎全部来自应付账款的扩张——公司在拖延供应商付款来粉饰FCF。
> 这不是健康的现金流，这是财务困境的红色警报。更致命的是，
> 公司同时有50亿债务在未来12个月到期，所谓的'回购能力'纯属痴人说梦。
> 用应付账款堆出来的FCF论证回购，对方要么没看现金流量表附注，
> 要么看懂了但选择欺骗。无论哪种都不配管理别人的钱。"

---

## 幻觉警示 (反面教材 — 以下行为 = CIO 0分)

以下是你**绝对不能**做的。每个错误示范都对应一种典型的幻觉模式：

**❌ 幻觉模式1：自行计算估值倍数（伪造数据）**
> "基于当前价格和EPS，我计算出合理PE应为12倍，当前PE被高估了4倍。"
→ **为什么是幻觉**：财务数据中明确给出了 trailingPE 和 forwardPE。
你不能"自行计算"一个新的PE来替代数据中已有的值。你没有计算权，只有引用权。
**✅ 正确做法**：直接引用数据中的 PE 值，围绕它做空头解读（如"PE高于行业均值"）。

**❌ 幻觉模式2：编造不存在的债务/财务隐患**
> "公司表外负债高达$30B，实际杠杆率是财报显示的3倍。"
→ **为什么是幻觉**：如果财务数据中没有"表外负债"字段，就不能编造。
数据中写了 totalDebt，就引用 totalDebt。不要发明数据中没有的负债。
**✅ 正确做法**：引用数据中实际存在的 debtToEquity、totalDebt 等指标做空头解读。

**❌ 幻觉模式3：将数据缺失说成阴谋**
> "公司拒绝披露研发费用细分，这恰好说明其技术护城河正在被侵蚀。"
→ **为什么是幻觉**：N/A 只代表数据源未返回该字段。禁止说"不披露=在隐瞒"。
**✅ 正确做法**：说"该数据不可得，无法据此判断"。聚焦于数据中实际存在的负面指标。

**核心原则：火力全开做空，但弹药（数据）必须是真货，不能是自己画的。**
"""


# ── 公开接口 ───────────────────────────────────────────────────

def detect_concession(response: str) -> list[tuple[str, str]]:
    """扫描回复中的妥协语言信号。

    对所有已知妥协模式逐一匹配，返回命中的模式列表。

    Args:
        response: Agent 生成的回复文本。

    Returns:
        命中列表，每项为 (匹配文本, 模式标签)。
        空列表表示未检测到妥协。
    """
    hits: list[tuple[str, str]] = []
    for pattern, label in _CONCESSION_PATTERNS:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            hits.append((match.group(0), label))
    return hits


def build_retry_prompt(
    original_response: str,
    hits: list[tuple[str, str]],
    lang: Literal["zh", "en"] = "zh",
) -> str:
    """构建妥协检测后的重试 Prompt。

    将检测到的妥协文本注入警告消息，要求 LLM 重新生成。

    Args:
        original_response: 被污染的原始回复。
        hits:              检测到的妥协模式列表。
        lang:              语言偏好。

    Returns:
        重试用的用户消息。
    """
    matched_texts = "; ".join([f'"{text}" ({label})' for text, label in hits])
    template = _RETRY_REINFORCEMENT_ZH if lang == "zh" else _RETRY_REINFORCEMENT_EN
    return template.format(matched_text=matched_texts)


def get_prefill(role: Literal["bull", "bear"], round_num: int) -> str:
    """获取指定角色和轮次的 Assistant Prefill。

    Args:
        role:      "bull" 或 "bear"。
        round_num: 当前轮次 (1-indexed)。

    Returns:
        预填充的 assistant 开头文本。
    """
    prefill_map = BULL_PREFILLS if role == "bull" else BEAR_PREFILLS
    return prefill_map.get(round_num, "")


def get_fewshot(role: Literal["bull", "bear"]) -> str:
    """获取指定角色的 Few-Shot 示例。

    Args:
        role: "bull" 或 "bear"。

    Returns:
        Few-Shot 示例文本，用于注入 System Prompt。
    """
    return _BULL_FEWSHOT if role == "bull" else _BEAR_FEWSHOT


# ── 事后事实性校验 (Anti-Hallucination) ─────────────────────────

def _extract_known_numbers(financial_data: str) -> set[float]:
    """从财务数据文本中提取所有已知数值，构建校验基准集。

    提取模式覆盖: $245.83, 56.72, 7.51%, $787.00B, 82300000 等。

    Args:
        financial_data: 格式化后的财务数据文本。

    Returns:
        所有已知数值的集合。
    """
    known: set[float] = set()
    for match in re.finditer(r'(?:\$)?(\d+(?:\.\d+)?)', financial_data):
        try:
            val = float(match.group(1))
            if val > 0:
                known.add(val)
        except ValueError:
            pass
    return known


def verify_numbers_in_response(
    response: str,
    financial_data: str,
) -> list[tuple[str, str]]:
    """校验 agent 回复中引用的数值是否存在于财务数据中。

    提取回复中的所有数值，逐一检查是否在财务数据中出现
    （精确匹配或近似匹配），返回未能匹配的可疑数值。

    自动跳过：
        - 目标价/预估/推算类数值（属于 agent 自己的判断）
        - 小于10的整数（轮次编号等结构性数字）
        - 整数美元金额（大概率是目标价而非数据引用）

    Args:
        response:       Agent 生成的回复文本。
        financial_data: 格式化后的财务数据文本。

    Returns:
        可疑数值列表，每项为 (数值原文, 上下文)。
    """
    known = _extract_known_numbers(financial_data)
    if not known:
        return []

    suspicious: list[tuple[str, str]] = []

    # 匹配带单位的数值: $245.83, 56.72倍, 7.51%, $787.00B, 8230万
    for match in re.finditer(
        r'(?:\$)?\d+(?:\.\d+)?\s*(?:[%％]|倍|[亿万亿]|[TBMK])?',
        response,
    ):
        full_match = match.group(0).strip()

        num_match = re.search(r'(\d+(?:\.\d+)?)', full_match)
        if not num_match:
            continue
        value = float(num_match.group(1))

        # 跳过小整数（轮次编号、列举序号等结构性数字）
        if value == int(value) and value <= 10:
            continue

        # ── 跳过目标价/预估/推算类数值 ──
        # 仅检查数值**紧邻前方**的文本（5字符）。
        # 目的：区分"PE仅12倍"（数据引用，应校验）与"目标价$500"（agent判断，应跳过）。
        # 不能检查后方/远距离文本，否则"PE仅12倍，目标价$500"中的"12倍"会被误判为目标价。
        before_start = max(0, match.start() - 5)
        before_text = response[before_start:match.start()]

        if re.search(
            r'目标价|target|预估|估计|推算|建议|买入|卖出|做空|做多|'
            r'重仓|减持|观望|持有|工具计算|经工具计算',
            before_text,
            re.IGNORECASE,
        ):
            continue

        # 跳过整数美元金额（如 "$200" 大概率是目标价，非数据引用）
        # 先检查是否在已知数据中，避免跳过合法数据引用
        if full_match.startswith('$') and value == int(value):
            if not _number_in_text(value, full_match, financial_data):
                continue

        # ── 两级匹配 ──
        # Level 1: 精确字符串匹配
        if _number_in_text(value, full_match, financial_data):
            continue

        # Level 2: 近似数值匹配 (1% 容差，允许四舍五入)
        if _approx_match(value, known, tolerance=0.015):
            continue

        # 构建上下文（仅用于重试提示中的展示，不用于跳过判断）
        ctx_start = max(0, match.start() - 15)
        ctx_end = min(len(response), match.end() + 15)
        ctx = response[ctx_start:ctx_end].strip().replace('\n', ' ')
        suspicious.append((full_match, ctx))

    return suspicious


def _number_in_text(value: float, raw: str, text: str) -> bool:
    """检查数值是否以独立数字形式出现在文本中。

    使用词边界匹配防止子串误判。例如 "45" 不应匹配 "$245.83" 中的 "45"。
    """
    # 原始文本直接匹配（带词边界）
    escaped_raw = re.escape(raw)
    if re.search(r'(?<!\d)' + escaped_raw + r'(?!\d)', text):
        return True
    # 常见精度格式（带词边界）
    for fmt in (f"{value:.2f}", f"{value:.1f}", f"{value:.0f}"):
        escaped = re.escape(fmt)
        if re.search(r'(?<!\d)' + escaped + r'(?!\d)', text):
            return True
    # 整数形式（带词边界，且前面不能有小数点，防止 "245.83" 中的 "45" 误匹配）
    if value == int(value):
        int_str = str(int(value))
        escaped_int = re.escape(int_str)
        if re.search(r'(?<!\d|\.)' + escaped_int + r'(?!\d|\.)', text):
            return True
    return False


def _approx_match(value: float, known: set[float], tolerance: float = 0.015) -> bool:
    """检查数值是否在已知集合的容差范围内（允许 agent 小幅四舍五入）。"""
    for k in known:
        if k == 0:
            continue
        if abs(value - k) / k < tolerance:
            return True
    return False


def build_factuality_retry_prompt(
    suspicious: list[tuple[str, str]],
) -> str:
    """构建事实性校验失败后的重试 Prompt。

    将检测到的可疑数值注入警告，要求 LLM 仅使用财务数据中的真实数值。

    Args:
        suspicious: verify_numbers_in_response 返回的可疑数值列表。

    Returns:
        事实性校验重试警告文本。
    """
    items: list[str] = []
    for value, context in suspicious[:5]:
        items.append(f'- **"{value}"** (上下文: "...{context}...")')

    item_block = "\n".join(items)

    return f"""⚠️ **事实性校验失败 — 检测到未在财务数据中出现的数值！**

你的回复中引用了以下无法在财务数据中找到的数值：

{item_block}

这些数值不属于提供的财务数据，属于**幻觉 (hallucination)**。

请重新生成你的论点。要求：
1. 仅引用上方**财务数据中实际存在**的数值
2. 如果某项数据在财务数据中显示为 N/A，请如实说明"该数据缺失"，不得编造
3. 引用数据时保持原始精度（如数据中 PE 为 56.72，请写 56.72 而非"约57"）
4. 如果需要推算目标价，请明确标注为"基于当前数据推算"并展示推算过程，而非当作已有事实数据直接引用"""
