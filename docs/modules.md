# 模块参考

## `src/main.py` — CLI 入口

**职责**: 解析命令行参数，调用辩论图，异常处理。

```bash
python -m src.main AAPL
python -m src.main TSLA --rounds 5
python -m src.main NVDA --sample
```

参数:
- `ticker` (位置): 股票代码
- `--rounds` (int): 辩论轮数，默认 3，范围 1-5
- `--sample` (flag): 强制使用离线样本数据

异常处理三级：
- `KeyboardInterrupt` → 优雅退出
- `ValueError` → 格式化错误信息
- 通用 `Exception` → 兜底捕获

---

## `src/config/settings.py` — 全局配置

基于 `pydantic-settings`，从 `.env` 文件和环境变量加载。

**LLM 配置**:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容 API 地址 |
| `LLM_API_KEY` | (内置测试 key) | API 密钥 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |

**辩论配置**:

| 变量 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `MAX_DEBATE_ROUNDS` | 3 | 1-5 | 辩论轮数 |
| `TEMPERATURE` | 0.7 | 0.0-2.0 | 采样温度 |
| `MAX_TOKENS` | 8192 | 256-8192 | 单次调用最大 Token |

**数据源配置**:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATA_SOURCE` | `fmp` | 数据源: fmp / yfinance / sample |
| `FMP_API_KEY` | (内置 key) | FMP Stable API Key |

全局单例 `settings` 在模块导入时创建，可在运行时修改覆盖。

---

## `src/models/messages.py` — 消息模型

### `DebateMessage`

不可变 Pydantic 模型，表示单条辩论发言：

```python
class DebateMessage(BaseModel):
    round: int          # 辩论轮次 (1-10)
    speaker: Literal["bull", "bear"]  # 发言者
    content: str        # 完整发言原文
```

字段校验：
- `round`: `ge=1, le=10`
- `speaker`: 严格 `Literal["bull", "bear"]`
- `content`: `min_length=1`

### `debate_message_reducer`

LangGraph 原生 Reducer 函数。签名: `(existing, new) -> existing + new`。

关键行为：
- `existing` 为 `None` → 返回 `[]`
- `new` 为 `None` → 返回 `[]`
- 正常情况 → `existing + new`（追加，不覆盖）

绑定方式: `MessageList = Annotated[list[DebateMessage], debate_message_reducer]`

---

## `src/models/state.py` — 状态定义

### `DebateState`

LangGraph TypedDict，8 个字段：

```python
class DebateState(TypedDict):
    ticker: str
    financial_data: str
    structured_data: dict      # TemporalFinancialData.model_dump()
    current_round: int
    messages: MessageList       # 带 Reducer，只追加
    final_target_price: float
    bull_confidence: float
    bear_confidence: float
    final_verdict: str
```

### `create_initial_state(ticker) -> DebateState`

工厂函数，创建填充默认值的初始状态：
- `ticker` 转为大写
- `financial_data` 置空 (`""`)
- `structured_data` 置空 (`{}`)
- `current_round` 设为 `1`
- `messages` 空列表
- 所有 float 字段初始化为 `0.0`

---

## `src/models/financial_data.py` — 时间维度数据模型

将财务数据拆分为三个互不跨越的时间维度：

### `LTMTTMData`
过去12个月已审计/已报告数据。字段包括: `trailing_pe`, `trailing_eps`, `price_to_book`, `price_to_sales`, `return_on_equity`, `return_on_assets`, `gross_margin`, `net_profit_margin`, `debt_to_equity`, `current_ratio`, `quick_ratio`, `free_cashflow_per_share`, `total_cash`, `total_debt`, `dividend_yield`, `payout_ratio`。所有字段为 `Optional[float]`。

### `MRQData`
最近一个季度同比数据（趋势拐点检测）。字段: `revenue_growth_yoy`, `earnings_growth_yoy`, `earnings_quarterly_growth`。

### `NTMForwardData`
未来12个月预测数据（仅此维度可用于推导目标价）。字段: `forward_pe`, `peg_ratio`, `est_eps_next_year_avg/high/low`, `est_eps_next_quarter_avg`, `est_revenue_next_year_avg`, `analyst_target_mean/high/low`, `analyst_count`。

### `TemporalFinancialData`
顶层容器，包含 `ticker`, `company_name`, `industry`, `sector`, `current_price`, `market_cap`, `beta`, `data_source` 及三个子维度模型。

`from_flat_dict(info)` 工厂方法将数据源扁平字典自动归类到正确的维度。

---

## `src/agents/bull_agent.py` — 红军多头

### `run_bull_agent(state, max_concession_retries=2) -> dict`

执行流程：

1. **构建上下文**: 调用 `build_bull_context(state)` — 财务数据 + 空头全部轮次发言 + 数据引用要求 + 本轮任务指令
2. **组装 System Prompt**: `BULL_SYSTEM_PROMPT + get_fewshot("bull")`（含计算工具使用规则和时间维度纪律）
3. **调用 LLM (带 Tools)**: `llm_client.chat_with_tools()`，传入 5 个计算工具
   - LLM 可以调用 `calculate_target_price`、`calculate_upside_downside` 等工具
   - 工具由 Python 执行，结果回传给 LLM 包装为叙事
   - 首轮使用 prefill 锁定语气
4. **妥协检测**: `detect_concession(response)` → 命中则重试
5. **事实性校验**: `verify_numbers_in_response(response, financial_data)` → 可疑则重试（工具计算结果自动跳过）
6. **创建消息**: `DebateMessage(round, "bull", response)` → 返回 `{"messages": [msg]}`

System Prompt 核心要求：
- 神经符号计算规则：禁止心算，所有目标价/涨跌幅/PEG/增长率必须通过工具
- 时间维度纪律：LTM/MRQ/NTM 不可跨维度混合
- 每轮不同的任务描述（立论 / 交叉质询 / 总结陈词）
- 绝对禁止承认对方合理性、使用模糊措辞、情绪化比喻

异常处理: LLM 调用失败时生成带 `⚠️` 标记的降级发言，不中断辩论。

---

## `src/agents/bear_agent.py` — 蓝军空头

### `run_bear_agent(state, max_concession_retries=2) -> dict`

与 Bull Agent 完全对称的结构，差异点：

- 上下文: `build_bear_context(state)` — 注入多头全部轮次发言
- System Prompt: `BEAR_SYSTEM_PROMPT + get_fewshot("bear")`（含计算工具使用规则和时间维度纪律）
- 角色定位: 法务会计背景，专注揭露财务造假和估值泡沫
- 同样使用 `chat_with_tools()` 调用 5 个计算工具
- Prefill: 空头专用强硬开场白
- Few-Shot: 空头视角的对抗性示例

---

## `src/agents/cio_agent.py` — CIO 裁判

### `run_cio_agent(state, max_retries=2) -> dict`

最复杂的 Agent，包含：

1. **上下文构建**: `build_cio_context(state)` — 双方全部发言（按角色分组，不过滤）
2. **System Prompt**: 定义 4 个评分维度（论据相关性、逻辑严密性、交叉质询杀伤力、论据冲击力），每项 25 分
3. **LLM 调用**: 要求输出纯 JSON（不用 Markdown 包裹）
4. **四层 JSON 解析**: 见 `_four_layer_parse()`
5. **Pydantic 校验**: `CIOVerdict` 模型，自动计算加权目标价
6. **Retry 机制**: 解析失败将错误反馈给 LLM，要求重新输出

### `CIOVerdict` (Pydantic Model)

```python
class CIOVerdict(BaseModel):
    bull_total_score: int        # 0-100
    bear_total_score: int        # 0-100
    bull_target_price: float     # > 0
    bear_target_price: float     # > 0
    bull_confidence: float       # 0-100
    bear_confidence: float       # 0-100
    final_target_price: float    # >= 0
    reasoning: str               # min 10 chars
```

`compute_final_target()` 方法实现加权公式:
```
final = (bull_target * bull_conf + bear_target * bear_conf) / (bull_conf + bear_conf)
```

内置 `field_validator` 处理 `"$200.00"` 字符串格式的自动转换。

### 四层 JSON 解析

| 层 | 函数 | 策略 |
|----|------|------|
| L1 | `_try_json_loads` | `json.loads` 直接解析 |
| L2 | `_try_strip_markdown` | 正则提取 `` ```json ``` `` 块或 `{...}` 括号内容 |
| L3 | `_try_manual_repair` | 去注释行、去尾随逗号、单引号转双引号 |
| L4 | `_try_regex_fallback` | 正则从非结构化文本中提取数字字段 |

L4 分两遍扫描：
- Pass 1: JSON 格式正则 (`"key": value`)
- Pass 2: 自然语言正则 (`key 大概是 value`)

### 降级策略

全部重试失败后，`_build_fallback_result()` 返回安全默认值（目标价 0.0，置信度 50.0），并在 `final_verdict` 中附加失败原因。

---

## `src/data/base.py` — 数据源抽象

```python
class DataSource(ABC):
    @abstractmethod
    def fetch(self, ticker: str) -> dict[str, Any]: ...

    @abstractmethod
    def to_text(self, data: dict[str, Any]) -> str: ...
```

两个抽象方法定义标准接口。所有数据源返回 `{"info": {...}}` 格式的扁平化字典。

---

## `src/data/fmp_source.py` — FMP 机构级数据源

默认数据源，合并 **8 个 Stable API 端点**：

| 端点 | 维度 |
|------|------|
| `/profile` | 公司概况（价格、市值、行业） |
| `/ratios-ttm` | 估值倍数（PE/PB/PS/PEG） |
| `/key-metrics-ttm` | ROE/ROA/EV-EBITDA |
| `/financial-growth` | 营收/盈利增长率 |
| `/analyst-estimates` | 分析师 EPS 预估 |
| `/price-target-consensus` | 分析师目标价（高/低/共识） |
| `/insider-trading` | 内部人士买卖 |
| `/balance-sheet-statement` | 现金与债务 |

### 缓存机制

```python
_session = requests_cache.CachedSession(
    cache_name="fmp_cache",
    backend="sqlite",
    expire_after=86400,  # 24h
    allowable_codes=(200,),
)
```

同一 URL 24h 内直接从 `fmp_cache.sqlite` 读取，零网络请求。

### 重试策略

- 指数退避: `BASE_DELAY * 2^attempt`（BASE_DELAY=2s）
- 最多 3 次重试
- 区分 HTTP 状态码: 429 重试, 401/403 抛错提示, 402 静默返回空

### 数据处理

- `_parse_52w_range()`: 解析 `"138.80-488.54"` 格式
- `_summarize_insider_trading()`: 汇总买入/卖出/授予笔数和股数
- `_extract_analyst_estimates()`: 从年度预估列表提取 Forward EPS 信息
- `_build_info()`: 8 端点数据合并为扁平化字典（字段名兼容 yfinance 格式）。`forwardPE` 由 `currentPrice / estEpsNextY_avg` 计算得出（非直接复制 TTM PE）。

### `to_text()` 输出格式

按四大维度 + 市场数据 + 股息分组展示，数值自动格式化（货币、百分比、大数缩写 T/B/M）。每个维度的数据标注了时间标签 `[LTM/TTM]`、`[MRQ]`、`[NTM/Forward]`，并在末尾附有时间维度纪律警告。

---

## `src/data/yfinance_source.py` — Yahoo Finance 数据源

### 缓存策略

本地 JSON 文件缓存 (`.cache/{TICKER}.json`)，TTL 1 小时。缓存命中时零网络请求。

### 重试策略

- 指数退避 + 随机抖动: 避免雷同重试时间
- `_is_retryable()`: 判断是否限流/网络类错误
- 最多 3 次重试

### 设计要点

- 不劫持 yfinance 内部 HTTP session，兼容 `curl_cffi`
- 错误分类：`ValueError`（不可恢复）vs `RuntimeError`（可降级）

---

## `src/data/sample_source.py` — 离线样本数据源

预置 TSLA、AAPL、NVDA 三只股票的近似真实数据。毫秒级响应，完全无网络依赖。用于：
- 开发调试（yfinance 限流时）
- 演示与测试
- CI/CD 环境

---

## `src/graph/debate_graph.py` — 辩论状态机

### `build_debate_graph() -> StateGraph`

构建编译好的 LangGraph 有向图：

```python
graph = StateGraph(DebateState)
graph.add_node("fetch_data", fetch_data_node)
graph.add_node("bull", bull_node)
graph.add_node("bear", bear_node)
graph.add_node("increment_round", increment_round_node)
graph.add_node("cio", cio_node)
graph.set_entry_point("fetch_data")
graph.add_edge("fetch_data", "bull")
graph.add_edge("bull", "bear")
graph.add_edge("bear", "increment_round")
graph.add_conditional_edges("increment_round", should_continue_debate, {...})
graph.add_edge("cio", END)
return graph.compile()
```

### 数据源工厂 `_get_data_source()`

根据 `settings.data_source` 返回对应的 DataSource 实例。同时包含自动降级逻辑：如果 FMP/YFinance 因限流或鉴权失败，自动切换到 SampleSource。

### `run_debate(ticker) -> DebateState`

公开入口。创建初始状态 → 执行图 → 返回最终状态。

---

## `src/utils/llm_client.py` — LLM 客户端

### `LLMClient`

封装 OpenAI 兼容接口：

```python
class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=..., base_url=...)
        self.model = settings.llm_model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens

    def chat(system_prompt, user_message, prefill="", max_retries=3, retry_delay=2.0) -> str

    def chat_with_tools(system_prompt, user_message, tools, tool_registry, *,
                        prefill="", max_tool_turns=5) -> tuple[str, list[dict]]
```

### `chat_with_tools()` — 神经符号计算核心

带工具调用的聊天方法，内部处理完整的多轮工具调用循环：

1. 发送消息 + 工具定义 → LLM
2. 若 LLM 返回 `tool_calls` → 从 `tool_registry` 查找并执行工具 → 将结果作为 `tool` 角色消息回传
3. 若 LLM 返回纯文本 → 返回 `(文本, 工具调用日志)`
4. 最多 `max_tool_turns` 轮，防止无限循环

返回值 `tool_call_log` 记录每次工具调用的工具名、参数和返回值，可用于调试和透明度展示。

### Prefill 机制

DeepSeek / OpenAI 兼容 API 支持 `assistant` 角色的 `content` 作为生成前缀。AlphaDebater 利用此特性注入预设开场白，锁定语气。工具调用与 prefill 兼容：prefill 确立开局语调后，LLM 仍可在后续轮次中调用工具。

### 错误处理

指数退避重试（最多 3 次），全部失败后抛 `RuntimeError`。异常在 Agent 层被捕获，不会中断辩论流程。

---

## `src/utils/calculation_tools.py` — 神经符号计算工具

提供 5 个纯 Python 计算函数及 OpenAI Tool Calling Schema：

### 计算函数

| 函数 | 参数 | 返回值 |
|------|------|--------|
| `calculate_target_price` | `eps, pe_multiple` | `{target_price, formula}` |
| `calculate_price_from_ps` | `revenue_per_share, ps_multiple` | `{target_price, formula}` |
| `calculate_upside_downside` | `current_price, target_price` | `{percentage, direction, formula}` |
| `calculate_peg_ratio` | `pe_ratio, earnings_growth_pct` | `{peg_ratio, pe_used, growth_used, formula}` |
| `calculate_growth_rate` | `past_value, current_value, periods` | `{growth_rate_pct, past, current, periods, formula}` |

每个函数返回 `dict`（公式+结果），供 LLM 包装为华尔街叙事话术。异常输入（除零、正负号变化）返回含 `error` 字段的结果。

### 导出

- `TOOL_REGISTRY: dict[str, Callable]` — 工具名到函数的映射
- `TOOL_DEFINITIONS: list[dict]` — OpenAI 兼容的 JSON Schema 定义

---

## `src/utils/printer.py` — 终端输出

纯展示层，无业务逻辑。主要函数：

| 函数 | 用途 |
|------|------|
| `print_header(ticker)` | 程序横幅 |
| `print_section(emoji, title)` | 章节标题 |
| `print_data_summary(ticker, info)` | 数据抓取摘要（关键字段表） |
| `print_round_header(round_num)` | 辩论轮次分界 |
| `print_agent_message(name, emoji, content)` | Agent 发言（带缩进） |
| `print_cio_verdict(target, bull_conf, bear_conf, verdict)` | CIO 最终裁决 |
| `print_error(message)` | 错误信息 |

终端宽度自动检测 (`shutil.get_terminal_size()`)，分割线长度自适应。

---

## `src/utils/context_slicer.py` — 上下文切片器

详见 [data-flow.md](data-flow.md) 中的上下文切片章节。

## `src/utils/debate_rigor.py` — 辩论严格性保障

详见 [anti-hallucination.md](anti-hallucination.md)。

---

## `tests/test_debate.py` — 单元测试

使用 pytest，覆盖八大核心模块：

1. **TestDebateMessage** — 消息模型校验、Reducer 追加、None 处理
2. **TestDebateState** — 状态初始化、ticker 大写转换
3. **TestContextSlicer** — 切片过滤正确性、跨轮记忆、角色隔离
4. **TestConcessionDetection** — 中英文妥协检测、零误报、Prefill/FewShot 内容校验
5. **TestFactualityVerification** — 数值提取、幻觉检测、目标价豁免、小整数豁免、工具计算结果豁免
6. **TestCIOParsing** — 四层降级全场景（正常/Markdown包裹/尾随逗号/单引号/完全非结构化）
7. **TestFMPSource** — 字段映射正确性、缺失数据兜底、工厂方法
8. **TestTemporalFinancialData** — 时间维度字段归类、None 值处理、TSLA 真实数据校验
9. **TestCalculationTools** — 5 个工具函数正确性（含边界条件：除零、零股价、正负号变化）
10. **TestToolDefinitions** — Schema 有效性、注册表与定义一致性
11. **TestToolCallingLogic** — 工具注册表执行、错误处理、事实性校验豁免工具计算结果
