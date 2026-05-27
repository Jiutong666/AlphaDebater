<p align="center">
  <h1 align="center">⚔️ AlphaDebater</h1>
  <p align="center">
    <strong>对抗式多智能体股票估值系统 · 神经符号计算 · LangGraph 状态机</strong>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/框架-LangGraph-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/LLM-DeepSeek | GPT--4o-purple.svg" alt="DeepSeek">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
  &nbsp;
  <a href="README.md">🇬🇧 English</a>
</p>

---

## 这是什么？

AlphaDebater 把**华尔街投资委员会的辩论流程**塞进了一个 LangGraph 状态机。

传统 LLM 股票分析输出的是"看起来很有道理但谁也不敢信"的八股文。AlphaDebater 不干这个——它让**红军多头**和**蓝军空头**在完全相同的财务数据上展开 3 轮交叉质询，最后由 **CIO 裁判**严格评估双方论证质量，输出一个**数学加权**的目标价和置信度评分。

> 不妥协。不编数字。只认数据和逻辑。

## 解决了什么问题？

LLM 在金融分析中会犯五类致命错误：

| 问题 | 典型 LLM 的表现 | AlphaDebater 怎么破 |
|------|----------------|-------------------|
| **算术幻觉** | "目标价 $302，现价 $215，下跌 40%"（实际是 +40%） | 剥夺 LLM 计算权，5 个 Python 工具接管所有算术 |
| **对齐污染** | "空头的观点也有一定道理……" | 11 条正则检测让步语言，命中即强制重写 |
| **数值编造** | 引用一个财报里根本不存在的 PE | 事实性校验：回复中每个数字必须在数据原文中出现 |
| **时间维度混淆** | 用历史 P/S 乘远期市值，得出不存在的营收 | Pydantic 模型强制 LTM/MRQ/NTM 三维隔离，跨维度混合被 prompt 禁止 |
| **各打五十大板** | "综合考虑，双方都有合理之处" | CIO 评分的是论证质量，不是平衡感 |

## 架构总览

```
┌─────────────┐     ┌──────────────────────────────────┐     ┌──────────┐
│  FMP /      │────▶│       LangGraph 状态机             │────▶│ 终端输出  │
│  Yahoo Fin  │     │                                    │     └──────────┘
└─────────────┘     │  fetch_data → bull → bear → CIO   │
                    │       ▲           ▲                │
                    │       │   神经符号工具              │
                    │       │   (5 个 Python 计算器)     │
                    │       │   反幻觉体系 (6 层)         │
                    └───────────────────────────────────┘
```

**完整流程**: 股票代码 → 抓取财务数据（LTM/MRQ/NTM 三维隔离）→ 3 轮辩论 → CIO 裁决 → 加权目标价 + 多空信心分

## 核心机制

### 一、神经符号计算 (Neuro-Symbolic)

LLM **不配做算术**。5 个纯 Python 函数通过 OpenAI 兼容的 Tool Calling 暴露，LLM 负责从数据中提取参数，Python 执行确定性计算，LLM 将结果包装为华尔街叙事。

| 工具 | 公式 | 消灭的幻觉 |
|------|------|-----------|
| `calculate_target_price` | EPS × PE | PE 估值乘错 |
| `calculate_upside_downside` | (目标 − 现价) / 现价 × 100% | 涨跌方向反转 |
| `calculate_peg_ratio` | PE / 增长率% | 百分比与小数混淆 |
| `calculate_growth_rate` | (终值/初值)^(1/n) − 1 | 复合增长率算错 |
| `calculate_price_from_ps` | 每股营收 × PS | PS 估值乘错 |

LLM 的职责：从财务数据中提取参数 → 调用工具 → 将 Python 返回的 `{结果, 公式}` 包装为华尔街话术。

### 二、反幻觉体系 (6 层防御)

```
第 0 层: 神经符号工具        ← 剥夺计算权，只保留推理权
第 1 层: System Prompt 铁律  ← 时间维度纪律 + 5 条自检规则
第 2 层: Few-Shot 行为模板   ← 对抗性示例注入 + 反面教材
第 3 层: Prefill 语气锁定    ← 预设开场白，锁定战斗姿态
    │
    ▼ [LLM 生成]
    │
第 4 层: 妥协检测 → 重试     ← 11 条中英文正则，命中即重写
第 5 层: 事实性校验 → 重试   ← 交叉验证每个数字是否在源数据中存在
```

前 4 层是**事前防护**（影响生成过程），后 2 层是**事后校验**（检测并修正输出）。

**妥协检测的 11 条规则**覆盖中英文让步模式：承认对方合理、默认对方指控、明确同意对方、不可否认型、模糊相对主义、各打五十大板、自我怀疑、自我反思、公开让步、英文让步、双方合理。误报率通过**语义匹配**而非关键词匹配控制在极低水平。

**事实性校验**从回复中提取所有数值，执行 4 道过滤（小整数 ≤10、目标价上下文、整数美元金额、工具计算结果标注）后，通过精确匹配 + 1.5% 容差匹配两级校验，未命中即触发重试。

### 三、时间维度数据隔离

财务数据被 Pydantic 模型强制拆分为三个**互不跨越**的维度：

| 维度 | 标签 | 用途 | 典型字段 |
|------|------|------|---------|
| LTM/TTM | 过去 12 个月审计数据 | 历史估值、盈利能力、财务健康 | PE, EPS, ROE, 毛利率, D/E, FCF |
| MRQ | 最近季度同比 | 趋势检测、拐点识别 | 营收增长率, 盈利增长率 |
| NTM/Forward | 未来 12 个月预测 | **仅此维度可用于推导目标价** | Forward PE, PEG, 分析师预估 EPS, 目标价范围 |

System Prompt 明确禁止跨维度混合计算——比如用 LTM 的营收增长率论证 NTM 的估值倍数，或用 LTM EPS × NTM PE 得出一个在时间维度上不存在的数字。

### 四、物理上下文切片

传统 RAG 方案用摘要压缩历史，这对辩论是致命的——交叉质询需要精确引用对手的原话和原始数值，任何摘要压缩都会导致数据失真和"稻草人谬误"。

AlphaDebater 的做法：
- 每个 Agent 看到对手**全部轮次的完整原文**
- 自己的历史发言**物理隔离，绝不注入**（防止自我抄袭和循环论证）
- 财务数据**始终完整注入**，一字不改
- 跨轮追踪提示引导 Agent 揭露对手论点演进中的矛盾

CIO 裁判则看到双方全部发言（同样不压缩），用 XML 标签物理隔离。

### 五、CIO 四层 JSON 容错

CIO 必须输出结构化 JSON 供程序计算（评分维度、目标价、置信度），但 LLM 不一定老实输出纯 JSON：

| 层 | 策略 | 处理场景 |
|----|------|---------|
| L1 | `json.loads` 直接解析 | 标准 JSON |
| L2 | 正则提取，去 Markdown 包裹 | `` ```json ... ``` `` |
| L3 | 手动修复语法 | 尾随逗号、单引号、注释行 |
| L4 | 正则暴力提取 | 完全非结构化文本 |

四层全部失败 → 自动重试（将解析错误反馈给 LLM 要求修正）→ 仍失败 → `_build_fallback_result()` 返回安全默认值（目标价 0.0，置信度 50.0）。

### 六、三数据源 + 自动降级

| 数据源 | 机制 | 适用场景 |
|--------|------|---------|
| **FMP Stable API** | 8 端点合并，requests-cache 24h SQLite 缓存，指数退避重试 | 生产环境，机构级数据 |
| **Yahoo Finance** | 本地 JSON 缓存，1h TTL，随机抖动重试 | 备用数据源 |
| **Sample Data** | 离线预置数据 (TSLA/AAPL/NVDA)，毫秒级响应 | 开发调试、CI/CD、演示 |

FMP/YFinance 请求失败（限流/鉴权）时自动降级到 SampleSource，辩论不中断。

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/Jiutong666/AlphaDebater.git
cd AlphaDebater

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 填入:
#   LLM_API_KEY=sk-...        (DeepSeek 或任意 OpenAI 兼容 API)
#   FMP_API_KEY=...            (Financial Modeling Prep, 或改用 yfinance/sample)

# 4. 运行
python -m src.main AAPL                # 使用 FMP 实时数据
python -m src.main TSLA --rounds 5     # 5 轮交叉质询
python -m src.main NVDA --sample       # 离线模式（零 API 调用）
```

## CLI 参考

```
python -m src.main TICKER [选项]

参数:
  TICKER              股票代码 (如 AAPL, TSLA, NVDA)

选项:
  --rounds INT        辩论轮数 (1-5, 默认: 3)
  --sample            强制使用离线样本数据 (无需 API Key)
```

## 项目结构

```
AlphaDebater/
├── src/
│   ├── main.py                    # CLI 入口
│   ├── agents/                    # 三个 LLM Agent
│   │   ├── bull_agent.py          # 红军多头 — 激进 PM
│   │   ├── bear_agent.py          # 蓝军空头 — 法务会计出身
│   │   └── cio_agent.py           # CIO 裁判 — 中立评审判词
│   ├── config/settings.py         # 全局配置 (pydantic-settings, 从 .env 加载)
│   ├── data/                      # 金融数据源
│   │   ├── base.py                # DataSource 抽象基类
│   │   ├── fmp_source.py          # FMP Stable API (8 端点合并)
│   │   ├── yfinance_source.py     # Yahoo Finance
│   │   └── sample_source.py       # 离线样本数据
│   ├── graph/debate_graph.py      # LangGraph 辩论状态机
│   ├── models/
│   │   ├── financial_data.py      # 时间维度隔离数据模型 (Pydantic)
│   │   ├── messages.py            # DebateMessage (不可变消息 + Reducer)
│   │   └── state.py               # DebateState (TypedDict)
│   └── utils/
│       ├── calculation_tools.py   # 神经符号计算工具 (5 个纯函数)
│       ├── context_slicer.py      # 物理上下文切片 + 跨轮记忆
│       ├── debate_rigor.py        # 妥协检测 + 事实性校验 + Prefill
│       ├── llm_client.py          # LLM 客户端 (Tool Calling 循环)
│       └── printer.py             # 终端美化输出
├── tests/test_debate.py           # 10 个测试类，30+ 条用例
├── docs/                          # 开发文档 (中文)
│   ├── architecture.md            # 系统架构与设计哲学
│   ├── modules.md                 # 模块逐一详解
│   ├── data-flow.md               # 数据流与状态管理
│   ├── anti-hallucination.md      # 反幻觉体系设计
│   └── configuration.md           # 环境变量与参数调优
├── .env.example                   # 环境变量模板（无真实 Key！）
└── requirements.txt
```

## 配置参数

所有参数通过 `.env` 文件或环境变量设置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容 API 地址 |
| `LLM_API_KEY` | (必填) | API 密钥 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `MAX_DEBATE_ROUNDS` | `3` | 辩论轮数 (1-5) |
| `TEMPERATURE` | `0.7` | LLM 采样温度 (0.0-2.0) |
| `MAX_TOKENS` | `8192` | 单次调用最大 Token (256-8192) |
| `DATA_SOURCE` | `fmp` | 数据源: `fmp` / `yfinance` / `sample` |
| `FMP_API_KEY` | (选填) | FMP API Key（使用 FMP 时必填） |

详见 [docs/configuration.md](docs/configuration.md)

## 文档

| 文档 | 内容 |
|------|------|
| [系统架构](docs/architecture.md) | 设计哲学、核心组件、数据源策略 |
| [模块参考](docs/modules.md) | 每个模块的详细 API 文档 |
| [数据流](docs/data-flow.md) | 状态管理、上下文切片、跨轮记忆机制 |
| [反幻觉体系](docs/anti-hallucination.md) | 6 层防护的完整设计思路与实现细节 |
| [配置指南](docs/configuration.md) | 环境变量、数据源切换、参数调优 |

## 运行测试

```bash
pytest tests/test_debate.py -v
```

覆盖：消息模型校验、状态初始化、上下文切片正确性、中英文妥协检测、事实性校验（精确匹配/容差匹配/过滤逻辑）、CIO 四层 JSON 解析全场景、FMP 字段映射、时间维度隔离数据模型、计算工具边界条件、工具定义 Schema 有效性。

## License

MIT — 详见 [LICENSE](LICENSE)
