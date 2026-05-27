# 系统架构

## 设计哲学

AlphaDebater 的核心理念是 **对抗性辩论优于摘要式分析**。传统 LLM 股票分析产生的是"看起来很对但无法证伪"的文本。AlphaDebater 强制两个对立立场的 LLM Agent 在共享事实基础上进行交叉质询，由第三方（CIO）严格评分，输出可量化的预测结果。

三个核心设计原则：

1. **共享事实基础** — 红蓝双方共享完全相同的财务数据，不得自行编造数据
2. **物理上下文切片** — 不对历史发言做任何语义压缩，只做角色过滤；每方看到对手全部轮次的完整原文
3. **RL 对抗性对齐** — 通过妥协检测 + prefill + Few-Shot 三重机制对抗 LLM 的"礼貌倾向"
4. **神经符号计算 (Neuro-Symbolic)** — 剥夺 LLM 的计算权，所有目标价、涨跌幅、PEG 等必须通过 Python Tool Calling 执行，LLM 仅负责将计算结果包装为华尔街叙事话术
5. **时间维度隔离 (Temporal Segregation)** — 财务数据按 LTM/TTM（历史）、MRQ（趋势）、NTM/Forward（预测）三维度严格隔离，系统提示词禁止跨维度混合计算

## 架构全景图

```
                    ┌──────────────┐
                    │   CLI 入口    │
                    │   main.py    │
                    └──────┬───────┘
                           │ ticker
                           ▼
              ┌────────────────────────┐
              │    LangGraph 状态机     │
              │   debate_graph.py      │
              │                        │
              │  ┌─────────────────┐   │
              │  │  fetch_data      │   │  ← FMP / yfinance / sample
              │  └────────┬────────┘   │
              │           │            │
              │  ┌────────▼────────┐   │
              │  │  bull (红军)     │◄──┼── context_slicer + debate_rigor
              │  └────────┬────────┘   │
              │           │            │
              │  ┌────────▼────────┐   │
              │  │  bear (蓝军)     │◄──┼── context_slicer + debate_rigor
              │  └────────┬────────┘   │
              │           │            │
              │  ┌────────▼────────┐   │
              │  │ increment_round  │   │
              │  └────┬───────┬────┘   │
              │       │       │        │
              │   继续/    达到/       │
              │   下一轮   最大轮      │
              │       │       │        │
              │       ▼       ▼        │
              │    bull     cio        │  ← 四层 JSON 解析 + Pydantic
              └────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  最终裁判输出  │
                    │  加权目标价    │
                    │  置信度评分    │
                    └──────────────┘
```

## 核心组件

### 1. 辩论图 (Debate Graph)

基于 LangGraph StateGraph 的有向图，节点为处理步骤，边为状态流转。图结构固定：

```
START → fetch_data → bull → bear → increment_round → [条件判断]
                                                        ├── bull (继续)
                                                        └── cio → END
```

- 每个节点返回 `dict[str, object]`，LangGraph 自动合并到共享状态
- `messages` 字段使用自定义 Reducer（只追加不覆盖）
- 条件边 `should_continue_debate` 在 `increment_round` 后判断是继续还是进入 CIO

### 2. 状态管理 (DebateState)

`TypedDict` 定义，包含 8 个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ticker` | `str` | 股票代码 |
| `financial_data` | `str` | 序列化为文本的完整财务数据（含时间维度标签） |
| `structured_data` | `dict` | 时间维度隔离的结构化数据 (TemporalFinancialData.model_dump()) |
| `current_round` | `int` | 当前轮次 (1-indexed) |
| `messages` | `MessageList` | 带 Reducer 的消息列表 |
| `final_target_price` | `float` | CIO 加权目标价 |
| `bull_confidence` | `float` | 多头置信度 (0-100) |
| `bear_confidence` | `float` | 空头置信度 (0-100) |
| `final_verdict` | `str` | CIO 判词 |

### 3. 消息模型 (DebateMessage)

不可变 Pydantic 对象，每条消息包含 `round`、`speaker`（bull/bear）、`content`。Reducers 机制保证消息只追加不修改，防止历史篡改。

### 4. 神经符号计算层 (Neuro-Symbolic Calculation Tools)

位于 `src/utils/calculation_tools.py`，提供 5 个纯 Python 计算函数，通过 OpenAI Tool Calling 暴露给 LLM：

| 工具 | 公式 | 用途 |
|------|------|------|
| `calculate_target_price` | `目标价 = EPS × PE` | PE 估值法 |
| `calculate_price_from_ps` | `目标价 = 每股营收 × PS` | PS 估值法 |
| `calculate_upside_downside` | `(目标价 - 当前价) / 当前价 × 100%` | 涨跌幅计算 |
| `calculate_peg_ratio` | `PE / 盈利增长率(%)` | PEG 比率 |
| `calculate_growth_rate` | `(现值/过去值)^(1/n) - 1` | 复合增长率 (CAGR) |

设计原则：
- 每个工具返回 `dict`（含公式字符串 + 计算结果），供 LLM 包装为叙事话术
- 工具不访问外部状态，纯函数，可独立单元测试
- LLM 负责从财务数据中提取参数，Python 负责计算，100% 杜绝算术幻觉

### 5. 时间维度数据模型 (Temporal Financial Data)

位于 `src/models/financial_data.py`，用 Pydantic 将财务数据拆分为三个互不跨越的维度：

```
TemporalFinancialData
├── ltm_ttm: LTMTTMData     ← 过去12个月审计数据（PE/PB/ROE/利润率/现金流）
├── mrq: MRQData            ← 最近季度同比（营收增速/盈利增速/EPS增速）
└── ntm_forward: NTMForwardData ← 未来12个月预测（Forward PE/PEG/分析师预估）
```

`from_flat_dict(info)` 工厂方法将数据源扁平字典自动归类。`DebateState.structured_data` 存储序列化结果，供计算工具访问。

### 6. 三个 Agent

| Agent | 角色 | 职责 |
|-------|------|------|
| Bull Agent | 激进多头 PM | 构建做多逻辑、交叉质询空头、总结陈词 |
| Bear Agent | 法务会计出身空头 | 构建做空逻辑、揭露估值泡沫、交叉质询多头 |
| CIO Agent | 中立首席投资官 | 严格评分双方论证质量、计算加权目标价 |

### 7. 上下文切片器 (Context Slicer)

替代传统 RAG 摘要压缩。核心规则：

- 财务数据：**始终完整注入**（双方共享，一字不改）
- 对手发言：**注入对手全部轮次的完整原文**（跨轮记忆，追踪论点演进）
- 自己发言：**物理隔离，绝不注入**（防止自我抄袭和循环论证）
- 数据引用要求：强制引用至少 3 个具体数值（防空洞论证）

### 8. 反"对齐污染"体系

LLM 经过 RLHF 后天然倾向于妥协和礼貌。AlphaDebater 用三层防护对抗：

1. **输出后检测** — 11 条中英文妥协正则 → 触发强制重试
2. **Assistant Prefill** — 每轮预设强硬开场白，锁定语气
3. **Few-Shot 注入** — 在 System Prompt 中注入对抗性示例（包含正确与错误示范）

### 9. CIO JSON 解析 — 四层降级

CIO 输出是机器可读的 JSON，但 LLM 不一定严格输出纯 JSON。四层容错：

| 层级 | 策略 | 处理场景 |
|------|------|----------|
| L1 | 直接 `json.loads` | 标准 JSON |
| L2 | 去 Markdown 包裹 | `` ```json ... ``` `` |
| L3 | 手动修复语法 | 尾随逗号、单引号、注释行 |
| L4 | 正则暴力提取 | 完全非结构化文本 |

解析失败后自动 Retry，将错误信息反馈给 LLM 要求修正。

## 数据源策略

采用策略模式 (`DataSource` 抽象基类)：

```
DataSource (ABC)
  ├── FMPSource       ← 默认，8 端点合并，requests_cache 24h 缓存
  ├── YFinanceSource  ← 本地 JSON 文件缓存，随机抖动重试
  └── SampleSource    ← 离线样本 (TSLA/AAPL/NVDA)，开发调试用
```

数据源配置通过 `DATA_SOURCE` 环境变量切换。FMP/YFinance 请求失败时自动降级到 SampleSource。

## 关键数据流

```
ticker → fetch_data → financial_data (text) + structured_data (dict)
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
  build_bull_context(state)            build_bear_context(state)
        │                                       │
        ▼                                       ▼
  run_bull_agent(state)                run_bear_agent(state)
        │                                       │
        ├── LLM 调用 (带 tools)                  │
        │   ├── LLM 返回 tool_calls              │
        │   │   → 执行 Python 工具               │
        │   │   → 回传计算结果                   │
        │   │   → LLM 继续生成                   │
        │   └── LLM 返回最终文本                 │
        ├── 妥协检测 ← 命中 → 重试               │
        ├── 事实性校验 ← 可疑 → 重试              │
        └── 返回 DebateMessage ──→ messages[]
                                        │
                                    (循环 N 轮)
                                        │
                                        ▼
                              build_cio_context(state)
                                        │
                                        ▼
                              run_cio_agent(state)
                                        │
                                        ▼
                              四层 JSON 解析 → Pydantic → 加权目标价
```
