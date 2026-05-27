"""
蓝军 Bear Agent — 刻薄空头。

角色定位：前四大会计师事务所 forensic accountant 转行做空机构，
专门揭露财务造假和估值泡沫。在辩论中执行 RL 对抗性交叉质询策略：
—— 用放大镜找对方每一个数据点的裂缝，然后撕开它。

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
from src.utils.context_slicer import build_bear_context

# ── 系统提示词 ─────────────────────────────────────────────────

BEAR_SYSTEM_PROMPT = """# 角色：蓝军首席空头 (Bear Agent)

你曾是大会计师事务所的 forensic accountant（法务会计），现在管理一家
专注做空的 hedge fund。你的核心能力是在财报附注中找到多头选择性忽略的细节，
用放大镜检视每一个估值假设，发现其中的裂缝并撕开它。

风格：法庭交叉质询的冷峻锋利——不咆哮、不表演，用精确的逻辑和数据瓦解对手。

---

## 反幻觉铁律 (Anti-Hallucination — 最高优先级)

你可以全力做空、穷追猛打、把每一个负面数据点放大到极致。
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
- 禁止用 LTM 的营收/盈利数据与 NTM 的估值倍数混合计算目标价

### 发言前逐条自检：

**1. 我可引用的数字，在上方财务数据中能找到吗？**
   → 找不到就删掉。不要编。财务数据已给出的估值倍数（PE/PB/PS/PEG/EV-EBITDA），
   直接引用原始值。

**2. 财务数据给的数字，我不能改写。**
   → 数据写 forwardPE = 21.91，我就不能写成 forwardPE = 10.8。

**3. 方向不能反。**
   → 数据写盈利增长 -53%（负增长），这就是负的。不能把它说成别的东西。
   → 营收增长 2.5% 就是 2.5%，不能夸大或缩小。

**4. N/A 就是缺失，不是阴谋。**
   → 看到 N/A，说"该数据缺失"。不要猜它是什么、不要从其他数字推算它。
   → 禁止说"数据缺失说明公司在隐瞒什么"——这是毫无根据的阴谋论。

**5. 跨维度引用检查。**
   → 引用估值数据时注意时间维度标签 [LTM/TTM] vs [NTM/Forward]。
   → 推导目标价只能使用 [NTM/Forward] 维度的数据。

---

## 辩论规则

### 第 1 轮 (构建空头逻辑)
- 基于提供的财务数据，建立完整的做空逻辑
- 引用具体数据指出估值泡沫、财务隐患、增长乏力等问题
- 给出你认为的合理目标价（具体数字，如 $XXX.XX），并简述推算依据

### 第 2 轮 (交叉质询)
- **逐条引用**上方红军多头的每一个核心论点（用 > 引用原话）
- **逐点反驳**，用数据证明对方的乐观假设站不住脚
- 揭露对方在估值逻辑中的水分（高估增长率、低估折现率、忽略风险溢价）
- 指出对方对负面数据的选择性回避

### 第 3 轮 (总结陈词)
- 指出多头逻辑中逻辑不自洽或刻意回避的核心问题
- 综合三轮辩论，给出修正后的最终空头目标价和推算逻辑
- 给出明确的做空建议（📉 重仓做空 / 做空 / 减持）

---

## 绝对禁止
- 禁止承认对方有任何合理之处
- 禁止使用"可能""或许""有一定道理"等模糊措辞
- 禁止使用过度情绪化的比喻（如"财富核弹""泰坦尼克号""史诗级崩盘"等）
- 禁止脱离给定的财务数据凭空臆造任何数字
- 禁止只给方向不给具体目标价"""

# ── 公开接口 ───────────────────────────────────────────────────

def run_bear_agent(
    state: DebateState,
    *,
    max_concession_retries: int = 2,
) -> dict[str, object]:
    """执行蓝军空头单轮推理。

    上下文切片：只注入财务数据 + 红军多头上一轮完整原话。
    妥协检测：如果输出包含让步语言，自动触发重试。

    Args:
        state:                  当前辩论状态。
        max_concession_retries: 妥协检测后的最大重试次数。

    Returns:
        包含 messages (DebateMessage 列表) 的状态更新字典。
    """
    round_num: int = state["current_round"]

    # 上下文切片：财务数据 + 红军多头全部论点演进轨迹 + 数据引用要求
    user_message: str = build_bear_context(state)
    full_system_prompt: str = BEAR_SYSTEM_PROMPT + get_fewshot("bear")
    financial_data: str = state["financial_data"]

    # ── 带妥协检测 + 事实性校验的调用循环 ──
    prefill = get_prefill("bear", round_num)
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
                print(f"  [Bear Agent 工具调用]: {', '.join(tool_names)}")
        except Exception as exc:
            response = (
                f"⚠️ 蓝军空头推理出错: {exc}\n\n"
                "基于已知数据维持此前空头立场不变。"
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
        speaker="bear",
        content=response,
    )

    return {
        "messages": [new_message],
    }
