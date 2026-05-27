# AlphaDebater 开发文档

## 文档索引

| 文档 | 内容 |
|------|------|
| [architecture.md](architecture.md) | 系统架构总览、设计哲学、技术选型 |
| [modules.md](modules.md) | 模块逐一详解（agents / data / graph / models / utils / config） |
| [data-flow.md](data-flow.md) | 辩论状态流转、LangGraph 图编排、上下文切片机制 |
| [configuration.md](configuration.md) | 环境变量配置、数据源切换、参数调优 |
| [anti-hallucination.md](anti-hallucination.md) | 反幻觉五层防护体系详解 |

## 项目简介

AlphaDebater 是一个基于 LLM 的对抗式股票辩论预测系统。它不生成摘要式分析，而是模拟华尔街投资委员会的辩论流程：Bull Agent（多头）与 Bear Agent（空头）在相同财务数据上展开多轮交叉质询，最后由 CIO Agent 根据论证质量计算加权目标价。

## 技术栈

- **编排引擎**: LangGraph (StateGraph)
- **LLM 接口**: OpenAI 兼容 API (默认 DeepSeek)
- **数据源**: FMP Stable API / Yahoo Finance / 离线样本
- **数据校验**: Pydantic v2 + 四层降级 JSON 解析
- **缓存**: requests-cache (SQLite, 24h TTL) + 本地 JSON 文件缓存

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY 和 FMP_API_KEY

# 运行辩论
python -m src.main AAPL
python -m src.main TSLA --rounds 3
python -m src.main NVDA --sample    # 使用离线样本数据
```

## 项目结构

```
AlphaDebater/
├── src/
│   ├── main.py                  # CLI 入口
│   ├── agents/                  # 三个 LLM Agent
│   │   ├── bull_agent.py        # 红军多头
│   │   ├── bear_agent.py        # 蓝军空头
│   │   └── cio_agent.py         # CIO 裁判
│   ├── config/
│   │   └── settings.py          # 全局配置 (pydantic-settings)
│   ├── data/                    # 金融数据源
│   │   ├── base.py              # DataSource 抽象基类
│   │   ├── fmp_source.py        # FMP Stable API
│   │   ├── yfinance_source.py   # Yahoo Finance
│   │   └── sample_source.py     # 离线样本
│   ├── graph/
│   │   └── debate_graph.py      # LangGraph 辩论状态机
│   ├── models/
│   │   ├── messages.py          # DebateMessage (不可变消息)
│   │   └── state.py             # DebateState (状态定义)
│   └── utils/
│       ├── context_slicer.py    # 物理上下文切片
│       ├── debate_rigor.py      # 反"对齐污染" + 事实性校验
│       ├── llm_client.py        # LLM 客户端封装
│       └── printer.py           # 终端美化输出
├── tests/
│   └── test_debate.py           # 核心流程单元测试
├── docs/                        # 开发文档
├── .env.example                 # 环境变量模板
└── requirements.txt             # Python 依赖
```
