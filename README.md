<p align="center">
  <h1 align="center">⚔️ AlphaDebater</h1>
  <p align="center">
    <strong>Adversarial Multi-Agent Stock Valuation · Neuro-Symbolic · LangGraph</strong>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/framework-LangGraph-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
</p>

---

## What is AlphaDebater?

AlphaDebater simulates a **Wall Street investment committee** inside an LLM-powered state machine. Instead of producing a bland "balanced view" summary, it forces a **Bull agent** and a **Bear agent** through multiple rounds of adversarial cross-examination over real financial data. A **CIO agent** then scores both sides on reasoning quality and computes a mathematically-weighted target price.

> No polite hedging. No hallucinated numbers. Just data-backed, adversarial conviction.

## Why This Exists

LLMs produce confident-sounding financial analysis that is often **dangerously wrong**:

| Problem | Typical LLM | AlphaDebater |
|---------|-------------|--------------|
| Arithmetic hallucination | "Target $302 vs current $215 = 40% decline" (should be +40%) | Python tool-calling — LLM cannot do math |
| Alignment pollution | "The bear also has a point..." | 11 regex patterns detect + force retry on any concession |
| Data fabrication | Citing P/E ratios that don't exist in the data | Factuality verification against source data with retry |
| Temporal confusion | Mixing historical P/S with forward revenue | Data segregated into LTM/MRQ/NTM with prompt-enforced discipline |
| "Both sides" cop-out | Splits the difference | CIO scores reasoning quality, not balance |

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────┐     ┌──────────┐
│  FMP /      │────▶│       LangGraph State Machine      │────▶│ Terminal │
│  Yahoo Fin  │     │                                    │     │ Output   │
└─────────────┘     │  fetch_data → bull → bear → CIO   │     └──────────┘
                    │       ▲           ▲                │
                    │       │   Neuro-Symbolic Tools     │
                    │       │   (5 Python calculators)   │
                    │       │   Anti-Hallucination (6L)  │
                    └───────────────────────────────────┘
```

**Data flow**: ticker → financial data (LTM/MRQ/NTM segregated) → 3 rounds of debate → CIO verdict with weighted target price and confidence scores.

## Key Features

### Neuro-Symbolic Calculation
The LLM **cannot do arithmetic**. Five pure-Python tools are exposed via OpenAI-compatible tool calling — the LLM extracts parameters from data, Python computes the result, the LLM wraps it in narrative.

| Tool | Formula | Use Case |
|------|---------|----------|
| `calculate_target_price` | EPS × PE | PE-valuation target |
| `calculate_upside_downside` | (Target − Current) / Current × 100% | Return potential |
| `calculate_peg_ratio` | PE / Growth% | Growth-adjusted valuation |
| `calculate_growth_rate` | (FV/PV)^(1/n) − 1 | CAGR calculation |
| `calculate_price_from_ps` | Revenue/Share × PS | PS-valuation target |

### Anti-Hallucination (6-Layer Defense)

```
Layer 0: Neuro-Symbolic Tools     ← LLM loses math privileges
Layer 1: System Prompt Rules      ← Temporal discipline + 5 self-checks
Layer 2: Few-Shot Templates       ← Adversarial examples + hallucination warnings
Layer 3: Prefill Tone Lock        ← Assistant prefill forces combat stance
    │
    ▼ [LLM Generation]
    │
Layer 4: Concession Detection     ← 11 regex patterns (CN/EN) → retry
Layer 5: Factuality Verification  ← Cross-check every cited number against source data
```

### Temporal Data Segregation

Financial data split into three **non-crossable** dimensions:

| Dimension | Label | Use For |
|-----------|-------|---------|
| LTM/TTM | Past 12 months audited | Historical valuation, profitability |
| MRQ | Most recent quarter | Trend detection, inflection points |
| NTM/Forward | Next 12 months consensus | **Only this dimension for target prices** |

The system prompt prohibits cross-dimension mixing (e.g., LTM growth rate → NTM valuation multiple).

### Context Slicing (No Semantic Compression)

Each agent sees the opponent's **complete original speeches** across all rounds — never summarized, never compressed. Summarization would distort exact numbers and wording, enabling straw-man arguments. Physical slicing preserves adversarial precision.

### Robust JSON Parsing (CIO Output)

The CIO must output structured JSON for scoring. Four-layer fallback parsing handles every failure mode:

| Layer | Strategy | Handles |
|-------|----------|---------|
| L1 | `json.loads` directly | Clean JSON |
| L2 | Strip markdown fences | `` ```json ... ``` `` |
| L3 | Manual syntax repair | Trailing commas, single quotes, comment lines |
| L4 | Regex field extraction | Completely unstructured text |

### Real Financial Data

Three data source backends with automatic fallback:

- **FMP Stable API** — 8-endpoint merge (profile, ratios, growth, analyst estimates, price targets, insider trading, balance sheet). `requests-cache` with 24h SQLite TTL.
- **Yahoo Finance** — Local JSON cache with 1h TTL and jittered retry.
- **Sample Data** — Offline snapshot (TSLA/AAPL/NVDA) for dev and CI.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Jiutong666/AlphaDebater.git
cd AlphaDebater

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API keys:
#   LLM_API_KEY=sk-...        (DeepSeek or any OpenAI-compatible)
#   FMP_API_KEY=...            (Financial Modeling Prep, or switch to yfinance/sample)

# 4. Run
python -m src.main AAPL                # Live debate with FMP data
python -m src.main TSLA --rounds 5     # 5 rounds of cross-examination
python -m src.main NVDA --sample       # Offline mode (zero API calls)
```

## CLI

```
python -m src.main TICKER [OPTIONS]

Arguments:
  TICKER              Stock ticker symbol (e.g., AAPL, TSLA, NVDA)

Options:
  --rounds INT        Debate rounds (1-5, default: 3)
  --sample            Force offline sample data (no API keys needed)
```

## Project Structure

```
AlphaDebater/
├── src/
│   ├── main.py                    # CLI entry point
│   ├── agents/                    # Bull, Bear, CIO agents
│   ├── config/settings.py         # pydantic-settings (env-based)
│   ├── data/                      # FMP / Yahoo Finance / Sample sources
│   ├── graph/debate_graph.py      # LangGraph StateGraph definition
│   ├── models/                    # Pydantic models & state TypedDict
│   └── utils/
│       ├── calculation_tools.py   # Neuro-symbolic Python calculators
│       ├── context_slicer.py      # Physical context slicing
│       ├── debate_rigor.py        # Concession detection + factuality check
│       ├── llm_client.py          # OpenAI-compatible client with tool calling
│       └── printer.py             # Terminal output formatting
├── tests/test_debate.py           # 10 test classes, 30+ tests
├── docs/                          # Architecture, data flow, module reference
├── .env.example                   # Environment template (no real keys!)
└── requirements.txt
```

## Configuration

All settings via `.env` or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible API |
| `LLM_API_KEY` | (required) | Your API key |
| `LLM_MODEL` | `deepseek-chat` | Model name |
| `MAX_DEBATE_ROUNDS` | `3` | Debate rounds (1-5) |
| `TEMPERATURE` | `0.7` | LLM sampling temperature |
| `MAX_TOKENS` | `8192` | Max tokens per call |
| `DATA_SOURCE` | `fmp` | `fmp` / `yfinance` / `sample` |
| `FMP_API_KEY` | (optional) | FMP API key (if using FMP) |

See [docs/configuration.md](docs/configuration.md) for details.

## Documentation

| Document | Content |
|----------|---------|
| [Architecture](docs/architecture.md) | Design philosophy, component overview |
| [Modules](docs/modules.md) | Per-module reference (agents, data, graph, models, utils) |
| [Data Flow](docs/data-flow.md) | State management, context slicing, message reducer |
| [Anti-Hallucination](docs/anti-hallucination.md) | 6-layer defense system deep-dive |
| [Configuration](docs/configuration.md) | All env vars, data source switching, parameter tuning |

## Tests

```bash
pytest tests/test_debate.py -v
```

10 test classes covering: message models, state initialization, context slicing, concession detection (CN/EN), factuality verification, CIO JSON parsing (4-layer fallback), FMP field mapping, temporal data segregation, calculation tools (boundary conditions), and tool definitions.

## License

MIT — see [LICENSE](LICENSE) for details.
