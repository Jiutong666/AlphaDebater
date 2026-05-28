# 配置指南

## 环境变量配置

复制模板文件并编辑：

```bash
cp .env.example .env
```

`.env` 文件内容：

```env
# === LLM 配置 (DeepSeek / OpenAI 兼容) ===
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-your-api-key-here
LLM_MODEL=deepseek-chat

# === 辩论配置 ===
MAX_DEBATE_ROUNDS=3
TEMPERATURE=0.7
MAX_TOKENS=8192

# === 数据源 ===
# 可选: fmp (默认), yfinance, sample
DATA_SOURCE=fmp
FMP_API_KEY=your-fmp-api-key-here
```

所有环境变量也可通过系统环境变量设置（环境变量优先级高于 `.env` 文件）。

---

## 配置项详解

### LLM 配置

#### `LLM_BASE_URL`
OpenAI 兼容 API 的 Base URL。支持的提供商：
- **DeepSeek**: `https://api.deepseek.com/v1`
- **OpenAI**: `https://api.openai.com/v1`
- **其他兼容服务**: 任何实现 `/chat/completions` 端点的服务

#### `LLM_MODEL`
模型名称，取决于所用提供商：
- DeepSeek: `deepseek-chat`, `deepseek-reasoner`
- OpenAI: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`

#### `LLM_API_KEY`
API 密钥。对于 DeepSeek，从 [platform.deepseek.com](https://platform.deepseek.com) 获取。

### 辩论配置

#### `MAX_DEBATE_ROUNDS`
红蓝军辩论轮数，范围 1-5，默认 3。

| 轮数 | 适用场景 |
|------|----------|
| 1 | 快速预览，仅初始立论无交叉质询 |
| 2 | 立论 + 一轮交叉质询 |
| 3 | 完整辩论（立论 → 交叉质询 → 总结陈词） |
| 5 | 深度辩论，适合研究性分析 |

注意：每轮需要 2 次 LLM 调用（Bull + Bear），3 轮 = 6 次 Bull/Bear + 1 次 CIO = 7 次总计。

#### `TEMPERATURE`
LLM 采样温度，范围 0.0-2.0，默认 0.7。
- 较低 (0.0-0.3): 更确定性的输出，适合需要精确数据的辩论
- 较高 (0.7-1.0): 更有创造性的论证，但增加幻觉风险
- 默认 0.7 在创造性与一致性之间取得平衡

#### `MAX_TOKENS`
单次 LLM 调用最大输出 Token 数，范围 256-8192，默认 8192。
- 3 轮辩论的总结陈词可能较长，建议保持较高值
- 如果使用较小的模型或限制成本，可适当降低

### 数据源配置

#### `DATA_SOURCE`

| 值 | 数据源 | 特点 |
|----|--------|------|
| `fmp` | Financial Modeling Prep | 默认。五大维度数据，免费版 250 req/day |
| `yfinance` | Yahoo Finance | 免费，但可能限流 |
| `sample` | 离线样本 | 预置 TSLA/AAPL/NVDA 数据，零网络依赖 |

#### `FMP_API_KEY`
仅在 `DATA_SOURCE=fmp` 时需要。从 [financialmodelingprep.com](https://financialmodelingprep.com) 注册获取。免费版每日 250 次请求。免费版限制：
- `/insider-trading` 端点不可用（程序自动降级为显示"无数据"）
- 部分标的可能返回 402

---

## 数据源设置建议

### 场景 1: 日常使用

```env
DATA_SOURCE=fmp
FMP_API_KEY=sk-your-real-key
```

### 场景 2: 开发调试 / 被限流

```env
DATA_SOURCE=sample
```

或使用命令行参数：

```bash
python -m src.main AAPL --sample
```

### 场景 3: 备选数据源

```env
DATA_SOURCE=yfinance
```

---

## API 用量预估

### 单次辩论（3 轮）

| 组件 | 调用次数 |
|------|----------|
| FMP API（数据抓取） | 8 端点 × 1 次 = 8 次（首次，之后 24h 缓存） |
| LLM API（Bull Agent） | 3 次（可能因妥协/幻觉重试而增加） |
| LLM API（Bear Agent） | 3 次（可能因妥协/幻觉重试而增加） |
| LLM API（CIO Agent） | 1 次（可能因 JSON 解析失败重试而增加） |
| **LLM 总计** | **7+ 次** |

### Token 估算（单次辩论，3 轮）

| 组件 | 输入 Token (估算) | 输出 Token (估算) |
|------|-------------------|-------------------|
| System Prompt (Bull/Bear) | ~2,500 | — |
| 财务数据文本 | ~1,500 | — |
| 对手发言 (全部轮次) | ~200-2,000+（随轮次增长） | — |
| Agent 输出 | — | ~500-1,500 |
| **单次辩论总计** | **~20,000-30,000** | **~5,000-10,000** |

---

## 命令行参考

```
usage: alphadebater [-h] [--rounds ROUNDS] [--sample] ticker

🏦 AlphaDebater — 对抗式股票预测系统

positional arguments:
  ticker          股票代码 (例如: AAPL, TSLA, 600519.SS)

options:
  -h, --help      show this help message and exit
  --rounds ROUNDS  辩论轮数 (默认: 3)
  --sample         强制使用离线样本数据 (跳过 yfinance)
```

---

## 缓存管理

### FMP 缓存

FMP 数据源使用 `requests-cache` SQLite 缓存，文件位于项目根目录 `fmp_cache.sqlite`。

- **TTL**: 24 小时
- **手动清理**: 删除 `fmp_cache.sqlite` 文件即可
- **已被 `.gitignore` 忽略**: 不会提交到仓库

### yfinance 缓存

YFinance 数据源使用本地 JSON 文件缓存，文件位于 `.cache/{TICKER}.json`。

- **TTL**: 1 小时
- **手动清理**: 删除 `.cache/` 目录
- **已被 `.gitignore` 忽略**

---

## 错误排查

### `FMP API Key 无效`
```
RuntimeError: FMP API Key 无效 (HTTP 401).
请到 https://financialmodelingprep.com 注册获取 API Key.
```
→ 检查 `.env` 中的 `FMP_API_KEY` 是否正确。切换到 `DATA_SOURCE=sample` 作为临时方案。

### `yfinance 限流`
```
RuntimeError: 无法获取股票数据：已重试 3 次仍失败。
建议：等待 5-10 分钟后重试。
```
→ 等待几分钟后重试，或切换到 `DATA_SOURCE=fmp` 或 `DATA_SOURCE=sample`。

### `LLM API 超时或限流`
```
RuntimeError: LLM 调用失败，已重试 3 次。
```
→ 检查 `LLM_API_KEY` 和 `LLM_BASE_URL`。检查 API 提供商的用量配额。
