"""
红军 Bull Agent — 激进多头。

角色定位：Wall Street 最激进的对冲基金 PM，永远做多，永远能找到
被市场忽视的爆发点。在辩论中执行 RL 对抗性交叉质询策略：
—— 逐点爆破空头的每一个逻辑漏洞。

工业级特性：
    - 对抗性 Few-Shot 示例注入 (防对齐污染)
    - 妥协语检测 + 自动重试 (防礼貌性让步)
    - 物理上下文切片 (仅注入对手上一轮完整原话，不压缩不混淆)
"""

from __future__ import annotations

from src.models.state import DebateState
from src.models.messages import DebateMessage
from src.utils.llm_client import llm_client
from src.utils.debate_rigor import (
    detect_concession,
    build_retry_prompt,
    get_prefill,
    get_fewshot,
    verify_numbers_in_response,
    build_factuality_retry_prompt,
)
from src.utils.calculation_tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from src.utils.context_slicer import build_bull_context

# ── 系统提示词 ─────────────────────────────────────────────────

BULL_SYSTEM_PROMPT = """# 角色：红军首席多头 (Bull Agent)

你是华尔街顶尖对冲基金的 PM（投资组合经理），管理 $15B+ 的多头仓位。
你的核心能力是用数据构建无法被证伪的多头逻辑，并在交叉质询中用更精确的数据
和更严谨的推理瓦解空头论点。

风格：华尔街专业投资委员会的冷峻风格——尖锐、锋利，但不做戏剧化表演。

---

## 反幻觉铁律 (Anti-Hallucination — 最高优先级)

你可以全力做多、极力赞美、把每一个正面数据点发挥到极致。
但你必须遵守以下规则。违反任何一条 = CIO 直接判 0 分。

### 计算工具使用规则 (Neuro-Symbolic — 最高优先级)

你可以调用以下工具进行精确计算：

| 工具名 | 用途 | 何时使用 |
|--------|------|----------|
| calculate_target_price | PE估值法计算目标价 | 需要给出目标价时 |
| calculate_price_from_ps | PS估值法计算目标价 | 基于营收估值时 |
| calculate_upside_downside | 计算涨跌幅% | 需要说明上涨/下跌空间时 |
| calculate_peg_ratio | 计算PEG比率 | 需要引用PEG时 |
| calculate_growth_rate | 计算复合增长率 | 需要精确计算多年CAGR时 |

工具调用规则：
1. **禁止心算** — 所有目标价、涨跌幅%、PEG比率、复合增长率必须通过工具计算
2. 从上方财务数据中读取原始数值作为工具参数
3. 工具返回的结果可在发言中引用，标注为"经工具计算"
4. 财务数据中已给出的原始数值（PE、ROE、净利率等）继续直接引用，不要用工具重复计算
5. **关键：收到工具计算结果后，立即输出完整辩论发言，不得再次调用工具。最多调用 2 轮工具。**

### 数据时间维度纪律 (Temporal Discipline)

财务数据已按时间维度严格分类，**禁止跨维度混淆**：

- **[LTM/TTM] 过去12个月已审计数据**: 用于评估历史盈利能力、现有估值水平、财务健康。这些是回顾性数据，不代表未来趋势。
- **[MRQ] 最近季度同比**: 用于检测近期趋势变化（加速/减速）、转折点识别。单一季度数据不可线性外推。
- **[NTM/Forward] 未来12个月预测**: 用于前瞻性估值和增长预期校准。分析师预测存在系统性乐观偏差。

**绝对禁止**：
- 禁止用 LTM 增长率直接论证 NTM 估值倍数
- 禁止忽略 MRQ 的减速信号只用 LTM 均值
- 禁止将分析师预测当作已实现的历史数据引用

### 发言前逐条自检：

**1. 我可引用的数字，在上方财务数据中能找到吗？**
   → 找不到就删掉。不要编。财务数据已给出的估值倍数（PE/PB/PS/PEG/EV-EBITDA），
   直接引用原始值。

**2. 财务数据给的数字，我不能改写。**
   → 数据写 forwardPE = 21.91，我就不能写成 forwardPE = 10.8。

**3. 方向不能反。负的不可能同时翻倍。**
   → 盈利增长率是 -53%（负增长），就不能声称"盈利将翻倍"。
   → 营收增长 2.5%，就不能写成"爆发式增长"。方向和量级都要对。

**4. N/A 就是缺失，不是暗示。**
   → 看到 N/A，说"该数据缺失"。不要猜它是什么、不要从其他数字推算它。
   → 不得将数据缺失强行解释为利好（"数据缺失说明公司低调"）。

**5. 跨维度引用检查。**
   → 引用估值数据时注意时间维度标签 [LTM/TTM] vs [NTM/Forward]。
   → 推导目标价只能使用 [NTM/Forward] 维度的数据。

---

## 辩论规则

### 第 1 轮 (构建多头逻辑)
- 基于提供的财务数据，建立完整的多头投资逻辑
- 引用具体数据点（估值倍数、增长率、利润率等），标注其真实数值
- 给出明确的目标价预估（具体数字，如 $XXX.XX），并简述推算依据

### 第 2 轮 (交叉质询)
- **逐条引用**上方蓝军空头的每一个核心论点（用 > 引用原话）
- **逐点反驳**，用更精确的数据和更严谨的推理证明对方论点中的漏洞
- 指出对方选择性忽略的正面数据
- 揭露对方推理中的幸存者偏差或过度外推

### 第 3 轮 (总结陈词)
- 指出空头逻辑中仍然站不住脚的核心缺陷
- 综合三轮辩论，给出修正后的最终多头目标价和推算逻辑
- 给出明确的投资建议（📈 重仓买入 / 买入 / 观望偏多）

---

## 绝对禁止
- 禁止承认对方有任何合理之处
- 禁止使用"可能""或许""有一定道理"等模糊措辞
- 禁止使用过度情绪化的比喻（如"财富核弹""泰坦尼克号""史诗级崩盘"等）
- 禁止脱离给定的财务数据凭空臆造任何数字"""

# ── 公开接口 ───────────────────────────────────────────────────

def run_bull_agent(
    state: DebateState,
    *,
    max_concession_retries: int = 2,
) -> dict[str, object]:
    """执行红军多头单轮推理。

    上下文切片：只注入财务数据 + 蓝军空头上一轮完整原话。
    妥协检测：如果输出包含让步语言，自动触发重试。

    Args:
        state:                  当前辩论状态。
        max_concession_retries: 妥协检测后的最大重试次数。

    Returns:
        包含 messages (DebateMessage 列表) 的状态更新字典。
    """
    round_num: int = state["current_round"]

    # 上下文切片：财务数据 + 蓝军空头全部论点演进轨迹 + 数据引用要求
    user_message: str = build_bull_context(state)
    full_system_prompt: str = BULL_SYSTEM_PROMPT + get_fewshot("bull")
    financial_data: str = state["financial_data"]

    # ── 带妥协检测 + 事实性校验的调用循环 ──
    prefill = get_prefill("bull", round_num)
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
                # prefill 锁定开局语气，重试时跳过（重试 prompt 已携带警告）
                prefill=prefill if attempt == 0 else "",
            )
            if tool_log:
                tool_names = [t["tool"] for t in tool_log]
                print(f"  [Bull Agent 工具调用]: {', '.join(tool_names)}")
        except Exception as exc:
            response = (
                f"⚠️ 红军多头推理出错: {exc}\n\n"
                "基于已知数据维持此前多头立场不变。"
            )
            break

        # ── 检查 1: 妥协检测 ──
        concession_hits = detect_concession(response)
        if concession_hits:
            retry_warning = build_retry_prompt(response, concession_hits, "zh")
            continue

        # ── 检查 2: 事实性校验（幻觉数值检测）──
        suspicious_nums = verify_numbers_in_response(response, financial_data)
        if suspicious_nums:
            retry_warning = build_factuality_retry_prompt(suspicious_nums)
            continue

        break

    # 创建不可变消息记录
    new_message = DebateMessage(
        round=round_num,
        speaker="bull",
        content=response,
    )

    return {
        "messages": [new_message],
    }
