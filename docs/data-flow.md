# 数据流与状态管理

## LangGraph 辩论图生命周期

一次完整的辩论执行以下步骤：

```
时间线 ──────────────────────────────────────────────────────────────►

[用户]                [LangGraph 状态机]              [LLM / 外部]
  │                         │                              │
  │  python -m src.main     │                              │
  │  AAPL ─────────────────►│                              │
  │                         │                              │
  │                         │  创建初始状态                  │
  │                         │  ticker="AAPL"                │
  │                         │  current_round=1              │
  │                         │  messages=[]                  │
  │                         │                              │
  │                         │  fetch_data ─────────────────►│  FMP/yfinance
  │                         │◄─────────────────────────────│  返回财务数据
  │                         │                              │
  │                         │  financial_data="====\n..."   │
  │                         │                              │
  │                         │  bull_node ──────────────────►│  LLM (Bull)
  │                         │◄─────────────────────────────│  返回多头发言
  │                         │                              │
  │                         │  messages += [Bull R1]        │
  │                         │                              │
  │                         │  bear_node ──────────────────►│  LLM (Bear)
  │                         │◄─────────────────────────────│  返回空头发言
  │                         │                              │
  │                         │  messages += [Bear R1]        │
  │                         │                              │
  │                         │  increment_round              │
  │                         │  current_round = 2            │
  │                         │                              │
  │                         │  [条件: current_round <= max] │
  │                         │  → bull_node ────────────────►│  LLM (Bull R2)
  │                         │◄─────────────────────────────│
  │                         │  ...循环...                   │
  │                         │                              │
  │                         │  [条件: current_round > max]  │
  │                         │  → cio_node ─────────────────►│  LLM (CIO)
  │                         │◄─────────────────────────────│  返回裁判 JSON
  │                         │                              │
  │                         │  final_target_price = 176.67  │
  │                         │  bull_confidence = 85.0       │
  │                         │  bear_confidence = 72.0       │
  │                         │  final_verdict = "..."        │
  │                         │                              │
  │◄────────────────────────│  返回最终状态                  │
  │                         │                              │
  ▼  打印 CIO 裁决          │                              │
```

---

## 状态流转详解

### 初始状态

```python
DebateState(
    ticker="AAPL",
    financial_data="",
    structured_data={},
    current_round=1,
    messages=[],
    final_target_price=0.0,
    bull_confidence=0.0,
    bear_confidence=0.0,
    final_verdict="",
)
```

### fetch_data 后

```python
# financial_data: 带时间维度标签的文本
financial_data = """=====
股票名称: Apple Inc. (AAPL)
行业: Consumer Electronics | 板块: Technology
=====
[维度一: 基础与估值]
  [LTM/TTM — 过去12个月已审计/已报告数据]
  当前价格:              $195.87
  市盈率 (TTM):           30.12
  ...
  [NTM/Forward — 未来12个月预测数据]
  远期市盈率 (Forward):   28.57
  ...
=====
⚠️ 时间维度纪律 (Temporal Discipline):
  - [LTM/TTM] = 过去12个月已审计数据 → 用于历史估值
  - [MRQ]     = 最近季度同比 → 用于趋势检测
  - [NTM/Forward] = 未来12个月预测 → 仅此维度可用于推导目标价
  - 禁止跨维度混合计算
====="""

# structured_data: Pydantic 模型序列化结果
structured_data = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "current_price": 195.87,
    "ltm_ttm": {"trailing_pe": 30.12, "trailing_eps": 6.50, ...},
    "mrq": {"revenue_growth_yoy": 0.05, "earnings_growth_yoy": 0.081, ...},
    "ntm_forward": {"forward_pe": 28.57, "analyst_target_mean": 216.34, ...},
}
```
```

### 每轮 Bull → Bear 后

```python
# messages 字段通过 Reducer 追加:
messages = [
    DebateMessage(round=1, speaker="bull",   content="## 🐂 红军多头 — 第1轮..."),
    DebateMessage(round=1, speaker="bear",   content="## 🐻 蓝军空头 — 第1轮..."),
    DebateMessage(round=2, speaker="bull",   content="## 🐂 红军多头 — 第2轮..."),
    DebateMessage(round=2, speaker="bear",   content="## 🐻 蓝军空头 — 第2轮..."),
    DebateMessage(round=3, speaker="bull",   content="## 🐂 红军多头 — 第3轮..."),
    DebateMessage(round=3, speaker="bear",   content="## 🐻 蓝军空头 — 第3轮..."),
]
```

### CIO 执行后

```python
# 新增:
final_target_price = 176.67
bull_confidence = 85.0
bear_confidence = 72.0
final_verdict = "红军在逻辑严密性和交叉质询杀伤力两个维度..."
```

---

## 神经符号计算流 (Neuro-Symbolic Tool Calling)

### 概述

Bull/Bear Agent 调用 LLM 时携带 5 个计算工具。LLM 不能自行做算术——它必须调用 Python 工具执行计算。工具调用在 `LLMClient.chat_with_tools()` 内部自动处理：

```
run_bull_agent(state)
    │
    ├── build_bull_context(state) → user_message
    │
    └── llm_client.chat_with_tools(system_prompt, user_message, tools, registry, prefill)
        │
        │   messages = [system, user, (prefill as assistant)]
        │
        │   ┌─── API 调用 (带 tools 参数) ───┐
        │   │                                │
        │   │   LLM 返回 tool_calls?          │
        │   │   ├── YES: 执行 Python 工具    │
        │   │   │   → 结果回传 (role: tool)   │
        │   │   │   → 继续调用 API           │
        │   │   └── NO:  返回最终文本        │
        │   │                                │
        │   └────────────────────────────────┘
        │   (最多 max_tool_turns=5 轮)
        │
        └── 返回 (final_text, tool_call_log)
```

### 工具调用日志

`chat_with_tools()` 返回的 `tool_call_log` 记录每次工具调用：

```python
[
    {
        "tool": "calculate_target_price",
        "arguments": '{"eps": 4.33, "pe_multiple": 25.0}',
        "result": {"target_price": 108.25, "formula": "EPS $4.33 × PE 25.0x = $108.25"}
    },
    {
        "tool": "calculate_upside_downside",
        "arguments": '{"current_price": 245.83, "target_price": 108.25}',
        "result": {"percentage": -55.96, "direction": "下跌", "formula": "..."}
    },
]
```

### 工具计算结果与事实性校验

`verify_numbers_in_response()` 会自动跳过标注为"工具计算"/"经工具计算"的数值——这些是 Python 计算结果，不需要在财务数据原文中验证。

---

## 消息 Reducer 机制

### 设计动机

LangGraph 默认行为: 节点返回 `{"key": value}` 会**覆盖** State 中已有的 `key`。但辩论需要历史消息的**增量追加**而非覆盖。

### 解决方案

`DebateState` 的 `messages` 字段声明为 `MessageList`:

```python
MessageList = Annotated[
    list[DebateMessage],
    debate_message_reducer,  # ← 自定义 Reducer
]
```

`debate_message_reducer` 函数签名 `(existing, new) -> existing + new`，确保:
- 节点只需 `return {"messages": [msg]}`
- LangGraph 自动将 `msg` 追加到列表末尾
- 已有消息永不修改、永不删除

### Reducer 执行时机

```
State.messages = [m1, m2]
    │
    ├── bull_node 返回 {"messages": [m3]}
    │
    └── LangGraph 自动调用 reducer(messages, [m3])
        → 返回 [m1, m2, m3]
        → 更新 State.messages = [m1, m2, m3]
```

### 不变性保证

`DebateMessage` 是 Pydantic BaseModel，所有字段带类型校验。一旦创建，没有公开的修改接口。下游代码只能读取 `msg.content`，不能写入。这是"单次写入、多次读取"的不可变模式。

---

## 上下文切片机制

### 核心问题

LLM 上下文窗口有限。传统做法是用摘要压缩历史，但这对辩论是致命的——交叉质询需要精确引用对方的原话和原始数值，任何摘要压缩都会导致数据失真和"稻草人谬误"。

### 解决方案：物理切片而非语义压缩

不做语义摘要，只做**角色过滤**——注入对手的全部轮次发言，隔离自己的历史发言。

### Bull Agent 的 Prompt 结构

```
┌─────────────────────────────────────┐
│  ## 📊 财务数据（完整）              │  ← 始终完整，不被过滤
│  =====                              │
│  当前价格: $245.83                  │
│  市盈率: 56.72                      │
│  ...                                │
│  =====                              │
├─────────────────────────────────────┤
│  ## 🐻 蓝军空头论点演进轨迹           │  ← 对手全部轮次完整原文
│                                     │
│  ### 第 1 轮 — 初始立论              │
│  [Bear R1 完整发言]                  │
│                                     │
│  ### 第 2 轮 — 交叉质询              │
│  [Bear R2 完整发言]                  │
│                                     │
│  ⚠️ 跨轮追踪提示：                  │
│  1. 逐条回应每一轮的核心指控          │
│  2. 揭露论点演进中的矛盾或退让        │
│  3. 指出已被摧毁后不再提起的论点      │
├─────────────────────────────────────┤
│  ## 数据引用硬性要求                 │  ← 防空洞论证
│  必须引用至少3个来自财务数据的具体数值  │
├─────────────────────────────────────┤
│  ## ⚔️ 第 N 轮任务                  │  ← 动态注入
│  （根据轮次：立论/交叉质询/总结陈词）  │
└─────────────────────────────────────┘
```

### 关键实现函数

| 函数 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `build_bull_context(state)` | DebateState | Prompt 字符串 | Bull Agent 本轮上下文 |
| `build_bear_context(state)` | DebateState | Prompt 字符串 | Bear Agent 本轮上下文 |
| `build_cio_context(state)` | DebateState | Prompt 字符串 | CIO 裁判上下文（双方全部发言） |
| `build_opponent_timeline(messages, opponent)` | 消息列表 + 对手角色 | 对手全部轮次发言 | 跨轮记忆核心 |
| `slice_opponent_last_speech(messages, opponent)` | 消息列表 + 对手角色 | 对手最近一条发言 | 单轮记忆（备用） |

### 跨轮记忆设计

在第 3 轮时，agent 会看到对手在第 1 轮和第 2 轮的**全部原文**，而非摘要。这解决了传统 RAG 方案的两个致命问题：

1. **遗忘**: 第 3 轮 agent "忘记" 第 1 轮对手的核心论点
2. **失真**: 摘要可能扭曲对手论点的精确表述

跨轮追踪提示会明确引导 agent 利用完整历史：
```
⚠️ 跨轮追踪提示：上方是对方在所有轮次中的完整论点链。你必须：
1. 逐条回应对方每一轮的核心指控，尤其是前几轮中你尚未充分回应的论点
2. 揭露对方在论点演进中的矛盾或退让（如对方第1轮强调X，第2轮却回避X转而谈Y）
3. 如果对方某一论点被你摧毁后未再提起，明确指出其已默认放弃该论点
```

### CIO 的上下文差异

CIO 需要看到**全部双方发言**，因为其任务是审判整个辩论过程的质量。但同样不压缩——每条发言用 XML 标签物理隔离：

```
# 📜 红蓝军辩论完整记录（原文，无压缩）

## 红军多头 (Bull) 各轮论点
### 🐂 红军多头 — 第 1 轮
[Bull R1 全文]

### 🐂 红军多头 — 第 2 轮
[Bull R2 全文]

## 蓝军空头 (Bear) 各轮论点
### 🐻 蓝军空头 — 第 1 轮
[Bear R1 全文]

### 🐻 蓝军空头 — 第 2 轮
[Bear R2 全文]

## 🎯 裁判任务
请严格按系统提示中的评分标准和纯 JSON 格式进行裁判。
```

---

## 数据引用硬性要求

为防止 agent 做空洞论证（"估值偏高"、"盈利能力不错"），每个 agent 的 Prompt 中强制注入数据引用要求：

```
## 数据引用硬性要求（必须遵守）

你必须在发言中引用至少3个来自上方财务数据的具体数值。
未引用具体数据的论证将被 CIO 降权评分。

注意时间维度标注：数据已按 [LTM/TTM]、[MRQ]、[NTM/Forward] 分类。
不同时间维度的数据不可直接对比或混合计算。

正确引用示例：
- "当前PE为35.2倍（见上方[LTM/TTM 估值倍数]），但..."
- "财务数据显示ROE仅16.8%（见上方[LTM/TTM 盈利能力]），远低于行业水平"
- "分析师预估下一财年EPS为$6.50（见上方[NTM/Forward 预期差]）"

错误示例（将被扣分）：
- "估值偏高" — 未引用具体数值
- "盈利能力不错" — 模糊描述，无数据支撑
- 将 LTM 数据当作 NTM 数据引用 — 时间维度混淆
```

---

## 轮次计数逻辑

`increment_round_node` 在每轮 Bear 发言后执行，将 `current_round` 加 1：

```python
def increment_round_node(state):
    return {"current_round": state["current_round"] + 1}
```

条件边 `should_continue_debate` 在 Node 执行后评估：

```python
def should_continue_debate(state):
    if state["current_round"] <= settings.max_debate_rounds:
        return "bull"   # 下一轮开始
    return "cio"        # 进入最终裁判
```

**注意**: `current_round` 在 `increment_round_node` 中加 1 后才进入 `should_continue_debate` 的判断。例如 `max_debate_rounds=3`:

| bull_node 执行 | bear_node 执行 | increment 后 | should_continue 判断 |
|---------------|---------------|-------------|---------------------|
| round=1 | round=1 | round=2 | 2 <= 3 → "bull" |
| round=2 | round=2 | round=3 | 3 <= 3 → "bull" |
| round=3 | round=3 | round=4 | 4 > 3 → "cio" |

---

## 异常流与降级路径

### 数据源降级

```
FMP/YFinance 请求失败
    │
    ├── 限流/鉴权错误 → 自动切换 SampleSource
    │
    └── 其他错误 → ValueError(无效ticker) / RuntimeError(不可恢复)
```

### Agent 异常处理

```
chat_with_tools() 调用 (含工具调用循环)
    │
    ├── 成功 → 检查妥协检测
    │   ├── 通过 → 检查事实性校验
    │   │   ├── 通过 → 返回 DebateMessage
    │   │   └── 命中 → 重试 (最多 2 次，整个工具对话丢弃)
    │   └── 命中 → 重试 (最多 2 次，整个工具对话丢弃)
    │
    └── 失败 → 生成降级发言 (⚠️ 标记)
```

### CIO 解析降级

```
LLM 响应
    │
    ├── L1 json.loads → 成功 ✓
    ├── L2 去 Markdown → 成功 ✓
    ├── L3 手动修复 → 成功 ✓
    ├── L4 正则提取 → 成功 ✓
    │
    └── 全部失败 → Retry (最多 2 次)
        │
        ├── 成功 → Pydantic 校验 ✓
        └── 全部失败 → _build_fallback_result()
            → final_target_price=0.0
            → confidence=50.0
            → verdict="⚠️ CIO 裁判系统降级..."
```
