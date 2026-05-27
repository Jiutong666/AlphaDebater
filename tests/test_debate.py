"""
AlphaDebater 核心流程单元测试。

覆盖：
    - 状态初始化 (DebateState + DebateMessage)
    - 物理上下文切片 (Context Slicer)
    - 妥协检测 (反对齐污染)
    - CIO JSON 解析 (四层降级)
"""

import pytest

from src.models.state import create_initial_state, DebateState
from src.models.messages import DebateMessage, debate_message_reducer
from src.agents.cio_agent import _four_layer_parse, _validate_and_fix, CIOVerdict
from src.utils.debate_rigor import (
    detect_concession,
    get_prefill,
    get_fewshot,
    verify_numbers_in_response,
    build_factuality_retry_prompt,
    _extract_known_numbers,
)
from src.utils.context_slicer import (
    build_bull_context,
    build_bear_context,
    build_cio_context,
    slice_opponent_last_speech,
    build_opponent_timeline,
)


# ══════════════════════════════════════════════════════════════════
# 消息模型测试
# ══════════════════════════════════════════════════════════════════

class TestDebateMessage:
    """DebateMessage Pydantic 模型测试。"""

    def test_create_valid_message(self) -> None:
        msg = DebateMessage(round=1, speaker="bull", content="做多逻辑")
        assert msg.round == 1
        assert msg.speaker == "bull"
        assert msg.content == "做多逻辑"

    def test_invalid_speaker_rejected(self) -> None:
        with pytest.raises(Exception):
            DebateMessage(round=1, speaker="neutral", content="test")

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(Exception):
            DebateMessage(round=1, speaker="bull", content="")

    def test_reducer_append(self) -> None:
        """Reducer 应追加新消息，不修改已有消息。"""
        existing = [
            DebateMessage(round=1, speaker="bull", content="bull r1"),
        ]
        new = [
            DebateMessage(round=1, speaker="bear", content="bear r1"),
        ]
        result = debate_message_reducer(existing, new)
        assert len(result) == 2
        assert result[0].content == "bull r1"  # 未修改
        assert result[1].content == "bear r1"

    def test_reducer_handles_none(self) -> None:
        """Reducer 应处理 None 输入。"""
        result = debate_message_reducer(None, None)
        assert result == []


# ══════════════════════════════════════════════════════════════════
# 状态初始化测试
# ══════════════════════════════════════════════════════════════════

class TestDebateState:
    """DebateState 初始化测试。"""

    def test_create_initial_state(self) -> None:
        state: DebateState = create_initial_state("AAPL")
        assert state["ticker"] == "AAPL"
        assert state["financial_data"] == ""
        assert state["current_round"] == 1
        assert state["messages"] == []
        assert state["final_target_price"] == 0.0
        assert state["bull_confidence"] == 0.0
        assert state["bear_confidence"] == 0.0
        assert state["final_verdict"] == ""

    def test_ticker_uppercase(self) -> None:
        state: DebateState = create_initial_state("aapl")
        assert state["ticker"] == "AAPL"


# ══════════════════════════════════════════════════════════════════
# 物理上下文切片测试
# ══════════════════════════════════════════════════════════════════

class TestContextSlicer:
    """验证物理切片 — 只切对手上一轮完整原话，不压缩。"""

    def test_slice_bull_round1(self) -> None:
        """第1轮 Bull 上下文：只有财务数据和任务，无对手发言。"""
        state: DebateState = create_initial_state("AAPL")
        state["financial_data"] = "FINANCIAL DATA HERE"

        ctx = build_bull_context(state)

        assert "FINANCIAL DATA HERE" in ctx
        assert "第 1 轮任务" in ctx
        assert "尚未发言" in ctx  # Bear 尚未发言

    def test_slice_bull_round2(self) -> None:
        """第2轮 Bull 上下文：只包含 Bear 的上一轮完整原话。"""
        state: DebateState = create_initial_state("AAPL")
        state["financial_data"] = "FINANCIAL DATA"
        state["current_round"] = 2
        state["messages"] = [
            DebateMessage(round=1, speaker="bull", content="BULL_R1_SPEECH"),
            DebateMessage(round=1, speaker="bear", content="BEAR_R1_SPEECH"),
        ]

        ctx = build_bull_context(state)

        # 必须包含对手发言
        assert "BEAR_R1_SPEECH" in ctx
        # 不得包含自己的历史发言
        assert "BULL_R1_SPEECH" not in ctx
        # 必须是"交叉质询"
        assert "交叉质询" in ctx

    def test_slice_bear_round3(self) -> None:
        """第3轮 Bear 上下文：包含 Bull 的全部轮次发言（跨轮记忆）。"""
        state: DebateState = create_initial_state("AAPL")
        state["financial_data"] = "FINANCIAL DATA"
        state["current_round"] = 3
        state["messages"] = [
            DebateMessage(round=1, speaker="bull", content="BULL_R1"),
            DebateMessage(round=1, speaker="bear", content="BEAR_R1"),
            DebateMessage(round=2, speaker="bull", content="BULL_R2_REBUTTAL"),
            DebateMessage(round=2, speaker="bear", content="BEAR_R2_REBUTTAL"),
        ]

        ctx = build_bear_context(state)

        # 包含 Bull 全部轮次发言（跨轮记忆）
        assert "BULL_R1" in ctx
        assert "BULL_R2_REBUTTAL" in ctx
        assert "论点演进轨迹" in ctx  # 跨轮追踪标题
        # 不得包含自己过去的发言
        assert "BEAR_R2_REBUTTAL" not in ctx
        assert "BEAR_R1" not in ctx
        # 第3轮应该是"总结陈词"
        assert "总结陈词" in ctx

    def test_slice_no_cross_contamination(self) -> None:
        """Bull 上下文不得包含 Bear 标签外的任何 Bear 内容混入。"""
        state: DebateState = create_initial_state("AAPL")
        state["financial_data"] = "FINANCIAL DATA"
        state["current_round"] = 2
        state["messages"] = [
            DebateMessage(round=1, speaker="bull", content="EXCLUSIVE_BULL_CONTENT"),
            DebateMessage(round=1, speaker="bear", content="EXCLUSIVE_BEAR_CONTENT"),
        ]

        ctx = build_bull_context(state)

        assert "EXCLUSIVE_BEAR_CONTENT" in ctx   # 对手发言（用于爆破）
        assert "EXCLUSIVE_BULL_CONTENT" not in ctx  # 自己的历史被物理隔离

    def test_cio_context_includes_all(self) -> None:
        """CIO 上下文应包含全部 Bull + Bear 发言。"""
        state: DebateState = create_initial_state("AAPL")
        state["messages"] = [
            DebateMessage(round=1, speaker="bull", content="BULL_R1"),
            DebateMessage(round=1, speaker="bear", content="BEAR_R1"),
        ]

        ctx = build_cio_context(state)

        assert "BULL_R1" in ctx
        assert "BEAR_R1" in ctx
        assert "裁判任务" in ctx

    def test_slice_opponent_takes_last_only(self) -> None:
        """多轮后只取对手最近一条。"""
        msgs = [
            DebateMessage(round=1, speaker="bear", content="OLD_BEAR"),
            DebateMessage(round=2, speaker="bull", content="BULL_R2"),
            DebateMessage(round=2, speaker="bear", content="LATEST_BEAR"),
        ]
        result = slice_opponent_last_speech(msgs, opponent="bear")
        assert result == "LATEST_BEAR"
        assert "OLD_BEAR" not in result

    def test_opponent_timeline_shows_all_rounds(self) -> None:
        """跨轮记忆：对手全部轮次发言均应出现。"""
        msgs = [
            DebateMessage(round=1, speaker="bear", content="BEAR_R1_ARGUMENT"),
            DebateMessage(round=2, speaker="bull", content="BULL_R2"),
            DebateMessage(round=2, speaker="bear", content="BEAR_R2_REBUTTAL"),
        ]
        timeline = build_opponent_timeline(msgs, opponent="bear")
        assert "BEAR_R1_ARGUMENT" in timeline
        assert "BEAR_R2_REBUTTAL" in timeline
        assert "论点演进轨迹" in timeline
        # 不应混入其他角色的发言
        assert "BULL_R2" not in timeline

    def test_opponent_timeline_cross_round_hint(self) -> None:
        """多轮时应包含跨轮追踪提示。"""
        msgs = [
            DebateMessage(round=1, speaker="bull", content="BULL_R1"),
            DebateMessage(round=1, speaker="bear", content="BEAR_R1"),
            DebateMessage(round=2, speaker="bull", content="BULL_R2"),
        ]
        timeline = build_opponent_timeline(msgs, opponent="bull")
        assert "跨轮追踪提示" in timeline
        assert "论点演进中的矛盾或退让" in timeline

    def test_opponent_timeline_single_round_no_hint(self) -> None:
        """单轮时不需要跨轮追踪提示。"""
        msgs = [
            DebateMessage(round=1, speaker="bull", content="BULL_R1"),
        ]
        timeline = build_opponent_timeline(msgs, opponent="bull")
        assert "BULL_R1" in timeline
        assert "跨轮追踪提示" not in timeline  # 仅1轮，无需提示

    def test_bull_context_includes_data_citation(self) -> None:
        """Bull 上下文应包含数据引用硬性要求。"""
        state: DebateState = create_initial_state("AAPL")
        state["financial_data"] = "FINANCIAL DATA"
        ctx = build_bull_context(state)
        assert "数据引用硬性要求" in ctx
        assert "至少3个" in ctx


# ══════════════════════════════════════════════════════════════════
# 妥协检测测试 (反对齐污染)
# ══════════════════════════════════════════════════════════════════

class TestConcessionDetection:
    """验证妥协语言检测规则库。"""

    def test_detect_chinese_concession(self) -> None:
        hits = detect_concession("对方说的确实有道理，但我觉得还是可以买。")
        assert len(hits) > 0

    def test_detect_english_concession(self) -> None:
        hits = detect_concession("The bear has a point about the valuation.")
        assert len(hits) > 0

    def test_no_false_positive_on_aggressive(self) -> None:
        hits = detect_concession(
            "对方的分析完全是垃圾。数据被恶意曲解，逻辑漏洞百出。"
        )
        assert len(hits) == 0

    def test_no_false_positive_on_data_citation(self) -> None:
        hits = detect_concession(
            "对方声称PE为35倍，但这一数据忽略了一次性减值的影响。"
        )
        assert len(hits) == 0

    def test_both_sides_pattern(self) -> None:
        hits = detect_concession("双方的观点都有一定道理，应该综合来看。")
        assert len(hits) > 0

    def test_prefill_bull_round2(self) -> None:
        prefill = get_prefill("bull", 2)
        assert "交叉质询" in prefill
        assert "毁灭性反驳" in prefill

    def test_prefill_bear_round1(self) -> None:
        prefill = get_prefill("bear", 1)
        assert "做空" in prefill

    def test_fewshot_contains_examples(self) -> None:
        bull_shot = get_fewshot("bull")
        assert "PEG" in bull_shot
        assert "粉碎" in bull_shot

        bear_shot = get_fewshot("bear")
        assert "垃圾" in bear_shot


# ══════════════════════════════════════════════════════════════════
# 事后事实性校验测试 (Anti-Hallucination)
# ══════════════════════════════════════════════════════════════════

class TestFactualityVerification:
    """验证事后数值校验 — 检测 agent 是否引用了不存在的数值。"""

    SAMPLE_FINANCIAL_DATA = """
[估值指标]
  当前价格:              $245.83
  市盈率 (TTM):           56.72
  市净率:                 8.91

[盈利能力]
  ROE (净资产收益率):      16.80%
  毛利率:                 17.80%
  每股收益 (TTM):         $4.33

[增长指标]
  营收增长率 (YoY):       2.50%
  盈利增长率 (YoY):       -53.00%
"""

    def test_extract_known_numbers(self) -> None:
        known = _extract_known_numbers(self.SAMPLE_FINANCIAL_DATA)
        assert 245.83 in known
        assert 56.72 in known
        assert 8.91 in known
        assert 16.80 in known
        assert 4.33 in known
        assert 2.50 in known
        assert 53.00 in known

    def test_clean_response_no_suspicious(self) -> None:
        """正确引用财务数据中的数值，不应被标记。"""
        response = """
        当前PE为56.72倍，市净率8.91倍，ROE仅16.80%。
        营收增长率仅2.50%，盈利暴跌53.00%。
        当前股价$245.83对应EPS $4.33。
        """
        suspicious = verify_numbers_in_response(
            response, self.SAMPLE_FINANCIAL_DATA
        )
        assert len(suspicious) == 0

    def test_hallucinated_number_detected(self) -> None:
        """引用不存在于财务数据中的数值，应被标记。"""
        response = "公司营收增长率高达45%，远超行业平均水平。"
        suspicious = verify_numbers_in_response(
            response, self.SAMPLE_FINANCIAL_DATA
        )
        assert len(suspicious) > 0
        assert any("45" in s[0] for s in suspicious)

    def test_hallucinated_pe_detected(self) -> None:
        """引用错误的PE倍数应被标记。"""
        response = "当前市盈率仅15倍，显示出极高的投资价值。"
        suspicious = verify_numbers_in_response(
            response, self.SAMPLE_FINANCIAL_DATA
        )
        # 15 不在财务数据中 (实际PE为56.72)
        assert any("15" in s[0] for s in suspicious)

    def test_target_price_not_flagged(self) -> None:
        """目标价属于 agent 自身判断，不应标记。"""
        response = "综合以上分析，我给出目标价$300.00，建议买入。"
        suspicious = verify_numbers_in_response(
            response, self.SAMPLE_FINANCIAL_DATA
        )
        # $300 不应被标记（目标价上下文）
        assert not any("300" in s[0] for s in suspicious)

    def test_rounded_number_not_flagged(self) -> None:
        """合理四舍五入（如 56.72 → 56.7）不应标记。"""
        response = "当前PE约56.7倍，估值处于高位。"
        suspicious = verify_numbers_in_response(
            response, self.SAMPLE_FINANCIAL_DATA
        )
        assert len(suspicious) == 0

    def test_small_integers_not_flagged(self) -> None:
        """小整数（轮次编号等）不应标记。"""
        response = "综合以上3个维度分析，第2轮中对方回避了核心问题。"
        suspicious = verify_numbers_in_response(
            response, self.SAMPLE_FINANCIAL_DATA
        )
        # 3 和 2 是小整数，不应被标记
        assert len(suspicious) == 0

    def test_multiple_hallucinations_detected(self) -> None:
        """多个幻觉数值应全部被标记。"""
        response = "营收增长45%，ROE高达92%，PE仅12倍，目标价$500。"
        suspicious = verify_numbers_in_response(
            response, self.SAMPLE_FINANCIAL_DATA
        )
        # 45, 92, 12 是幻觉值，500 是目标价（不标记）
        flagged_values = [s[0] for s in suspicious]
        assert any("45" in v for v in flagged_values)
        assert any("92" in v for v in flagged_values)
        assert any("12" in v for v in flagged_values)
        assert not any("500" in v for v in flagged_values)

    def test_factuality_retry_prompt_contains_details(self) -> None:
        suspicious = [
            ("45%", "营收增长率高达45%，远超行业"),
            ("15倍", "市盈率仅15倍，显示出极高"),
        ]
        prompt = build_factuality_retry_prompt(suspicious)
        assert "45%" in prompt
        assert "15倍" in prompt
        assert "幻觉" in prompt
        assert "事实性校验失败" in prompt

    def test_empty_financial_data_no_false_positive(self) -> None:
        """空财务数据时不产生误报。"""
        suspicious = verify_numbers_in_response(
            "PE为35倍，营收增长20%", ""
        )
        assert len(suspicious) == 0


# ══════════════════════════════════════════════════════════════════
# CIO JSON 解析测试 (四层降级)
# ══════════════════════════════════════════════════════════════════

class TestCIOParsing:
    """验证 CIO JSON 响应的四层降级解析。"""

    # ── 第 1 层：正常 JSON ──

    def test_layer1_valid_json(self) -> None:
        response = """{
  "bull_total_score": 85,
  "bear_total_score": 72,
  "bull_target_price": 200.0,
  "bear_target_price": 150.0,
  "bull_confidence": 85,
  "bear_confidence": 72,
  "final_target_price": 177.07,
  "reasoning": "红军逻辑更严密，蓝军在第二轮出现了明显的回避行为。"
}"""
        data = _four_layer_parse(response)
        assert data["bull_total_score"] == 85
        assert data["final_target_price"] == 177.07

    # ── 第 2 层：Markdown 包裹 ──

    def test_layer2_markdown_fence(self) -> None:
        response = """以下是裁判结果：

```json
{
  "bull_total_score": 90,
  "bear_total_score": 65,
  "bull_target_price": 250.0,
  "bear_target_price": 180.0,
  "bull_confidence": 90,
  "bear_confidence": 65,
  "final_target_price": 220.97,
  "reasoning": "红军在交叉质询维度上完全碾压蓝军。"
}
```

以上是本次裁判结果。"""
        data = _four_layer_parse(response)
        assert data["bull_total_score"] == 90
        assert data["final_target_price"] == 220.97

    def test_layer2_no_json_tag(self) -> None:
        response = """```
{
  "bull_total_score": 75,
  "bear_total_score": 80,
  "bull_target_price": 180.0,
  "bear_target_price": 140.0,
  "bull_confidence": 75,
  "bear_confidence": 80,
  "final_target_price": 155.48,
  "reasoning": "蓝军逻辑链条更严密。"
}
```"""
        data = _four_layer_parse(response)
        assert data["bull_total_score"] == 75

    # ── 第 3 层：语法修复 ──

    def test_layer3_trailing_comma(self) -> None:
        response = """{
  "bull_total_score": 88,
  "bear_total_score": 70,
  "bull_target_price": 210.0,
  "bear_target_price": 160.0,
  "bull_confidence": 88,
  "bear_confidence": 70,
  "final_target_price": 187.85,
  "reasoning": "红军论证质量更高。",
}"""
        data = _four_layer_parse(response)
        assert data["bull_total_score"] == 88

    def test_layer3_single_quotes(self) -> None:
        response = """{
  'bull_total_score': 82,
  'bear_total_score': 75,
  'bull_target_price': 190.0,
  'bear_target_price': 155.0,
  'bull_confidence': 82,
  'bear_confidence': 75,
  'final_target_price': 172.3,
  'reasoning': '蓝军在逻辑上存在硬伤。'
}"""
        data = _four_layer_parse(response)
        assert data["bull_total_score"] == 82

    # ── 第 4 层：正则暴力提取 ──

    def test_layer4_broken_json(self) -> None:
        response = """
        我也不知道为什么输出成了这样...
        bull_total_score 大概是 78 分吧
        bear_total_score 给个 68
        目标价 bull_target_price 是 "195.50"
        bear那边 "bear_target_price" 是 145.00
        confidence 分别是 78 和 68
        final_target_price 按道理应该是 170 左右
        reasoning 我也不知道该写什么但必须写够十个字以上才行
        """
        data = _four_layer_parse(response)
        assert data["bull_total_score"] == 78
        assert data["bear_total_score"] == 68
        assert data["bull_target_price"] == 195.50
        assert data["final_target_price"] == 170.0

    def test_layer4_empty_response(self) -> None:
        data = _four_layer_parse("")
        assert data["bull_total_score"] == 0
        assert data["final_target_price"] == 0.0

    # ── Pydantic 校验 ──

    def test_pydantic_validation_success(self) -> None:
        raw = {
            "bull_total_score": 85,
            "bear_total_score": 72,
            "bull_target_price": 200.0,
            "bear_target_price": 150.0,
            "bull_confidence": 85,
            "bear_confidence": 72,
            "reasoning": "红军在逻辑维度表现更好的十个字以上理由",
        }
        verdict = _validate_and_fix(raw)
        assert isinstance(verdict, CIOVerdict)
        assert verdict.bull_total_score == 85
        assert verdict.final_target_price > 0

    def test_pydantic_auto_compute_target(self) -> None:
        raw = {
            "bull_total_score": 80,
            "bear_total_score": 70,
            "bull_target_price": 200.0,
            "bear_target_price": 150.0,
            "bull_confidence": 80,
            "bear_confidence": 70,
            "final_target_price": 0,
            "reasoning": "测试自动计算目标价的十个字以上理由",
        }
        verdict = _validate_and_fix(raw)
        assert verdict.final_target_price == 176.67

    def test_pydantic_defaults_for_missing(self) -> None:
        raw = {
            "bull_total_score": 70,
            "bear_total_score": 65,
            "reasoning": "缺了很多字段但至少reasoning够长",
        }
        verdict = _validate_and_fix(raw)
        assert verdict.bull_target_price == 100.0
        assert verdict.bull_confidence == 50.0


# ══════════════════════════════════════════════════════════════════
# FMP 数据源测试
# ══════════════════════════════════════════════════════════════════

class TestFMPSource:
    """验证 FMP 数据源的字段映射与缓存逻辑。"""

    def test_build_info_maps_all_fields(self) -> None:
        from src.data.fmp_source import _build_info

        profile = {
            "companyName": "Tesla Inc.",
            "symbol": "TSLA",
            "industry": "Auto",
            "sector": "Consumer Cyclical",
            "price": 245.83,
            "mktCap": 787000000000,
            "beta": 2.39,
            "volAvg": 82300000,
            "range": "138.80-488.54",
            "sharesOutstanding": 3200000000,
            "totalCash": 33400000000,
            "totalDebt": 13427000000,
        }
        key_metrics = {
            "peRatioTTM": 56.72,
            "priceToBookRatioTTM": 8.91,
            "priceToSalesRatioTTM": 7.51,
            "roeTTM": 0.168,
            "roaTTM": 0.074,
            "grossProfitMarginTTM": 0.178,
            "netProfitMarginTTM": 0.132,
            "netIncomePerShareTTM": 4.33,
            "revenueGrowthTTM": 0.025,
            "netIncomeGrowthTTM": -0.53,
            "debtToEquityTTM": 24.65,
            "currentRatioTTM": 1.86,
            "quickRatioTTM": 1.29,
            "freeCashFlowPerShareTTM": 1.085,
            "evToEbitdaTTM": 41.25,
        }
        ratios = {
            "pegRatioTTM": 2.83,
            "dividendYielTTM": None,
            "payoutRatioTTM": 0.0,
            "earningsGrowthTTM": -0.46,
        }

        info = _build_info("TSLA", profile, key_metrics, ratios)

        # 基本信息
        assert info["shortName"] == "Tesla Inc."
        assert info["symbol"] == "TSLA"

        # 估值
        assert info["currentPrice"] == 245.83
        assert info["trailingPE"] == 56.72
        assert info["priceToBook"] == 8.91
        assert info["pegRatio"] == 2.83

        # 盈利能力
        assert info["returnOnEquity"] == 0.168
        assert info["grossMargins"] == 0.178
        assert info["trailingEps"] == 4.33

        # 增长
        assert info["revenueGrowth"] == 0.025
        assert info["earningsGrowth"] == -0.53

        # 财务健康
        assert info["debtToEquity"] == 24.65
        assert info["currentRatio"] == 1.86
        assert info["quickRatio"] == 1.29

        # 52周高低
        assert info["fiftyTwoWeekLow"] == 138.80
        assert info["fiftyTwoWeekHigh"] == 488.54

        # 自由现金流 (per share × shares = total)
        assert info["freeCashflow"] == 1.085 * 3200000000

        # 市值
        assert info["marketCap"] == 787000000000
        assert info["beta"] == 2.39

        # 股息 (FMP typo: "dividendYielTTM")
        assert info["dividendYield"] is None
        assert info["payoutRatio"] == 0.0

    def test_build_info_handles_missing_data(self) -> None:
        from src.data.fmp_source import _build_info

        info = _build_info("UNKNOWN", {}, {}, {})
        assert info["shortName"] == "UNKNOWN"
        assert info["currentPrice"] is None
        assert info["trailingPE"] is None

    def test_data_source_factory_fmp(self) -> None:
        from src.data.fmp_source import FMPSource
        from src.graph.debate_graph import _get_data_source
        from src.config.settings import settings

        old = settings.data_source
        try:
            settings.data_source = "fmp"
            source = _get_data_source()
            assert isinstance(source, FMPSource)
        finally:
            settings.data_source = old


# ══════════════════════════════════════════════════════════════════
# 时间维度数据模型测试
# ══════════════════════════════════════════════════════════════════

class TestTemporalFinancialData:
    """TemporalFinancialData 模型测试。"""

    def test_from_flat_dict_classifies_ltm_fields(self) -> None:
        from src.models.financial_data import TemporalFinancialData

        info = {
            "symbol": "TEST",
            "shortName": "Test Corp",
            "currentPrice": 100.0,
            "trailingPE": 20.0,
            "trailingEps": 5.0,
            "priceToBook": 3.5,
            "priceToSalesTrailing12Months": 2.0,
            "returnOnEquity": 0.25,
            "returnOnAssets": 0.10,
            "grossMargins": 0.45,
            "debtToEquity": 50.0,
            "totalCash": 1e9,
            "totalDebt": 5e8,
        }
        result = TemporalFinancialData.from_flat_dict(info)

        assert result.ticker == "TEST"
        assert result.company_name == "Test Corp"
        assert result.current_price == 100.0
        assert result.ltm_ttm.trailing_pe == 20.0
        assert result.ltm_ttm.trailing_eps == 5.0
        assert result.ltm_ttm.price_to_book == 3.5
        assert result.ltm_ttm.price_to_sales == 2.0
        assert result.ltm_ttm.return_on_equity == 0.25
        assert result.ltm_ttm.return_on_assets == 0.10
        assert result.ltm_ttm.gross_margin == 0.45
        assert result.ltm_ttm.debt_to_equity == 50.0
        assert result.ltm_ttm.total_cash == 1e9
        assert result.ltm_ttm.total_debt == 5e8

    def test_from_flat_dict_classifies_mrq_fields(self) -> None:
        from src.models.financial_data import TemporalFinancialData

        info = {
            "symbol": "TEST",
            "shortName": "Test Corp",
            "revenueGrowth": 0.15,
            "earningsGrowth": -0.10,
            "earningsQuarterlyGrowth": 0.05,
        }
        result = TemporalFinancialData.from_flat_dict(info)

        assert result.mrq.revenue_growth_yoy == 0.15
        assert result.mrq.earnings_growth_yoy == -0.10
        assert result.mrq.earnings_quarterly_growth == 0.05

    def test_from_flat_dict_classifies_ntm_fields(self) -> None:
        from src.models.financial_data import TemporalFinancialData

        info = {
            "symbol": "TEST",
            "shortName": "Test Corp",
            "forwardPE": 18.0,
            "pegRatio": 1.5,
            "estEpsNextY_avg": 6.0,
            "estEpsNextY_high": 7.0,
            "estEpsNextY_low": 5.0,
            "targetMeanPrice": 120.0,
            "targetHighPrice": 150.0,
            "targetLowPrice": 90.0,
            "numberOfAnalystOpinions": 30,
        }
        result = TemporalFinancialData.from_flat_dict(info)

        assert result.ntm_forward.forward_pe == 18.0
        assert result.ntm_forward.peg_ratio == 1.5
        assert result.ntm_forward.est_eps_next_year_avg == 6.0
        assert result.ntm_forward.est_eps_next_year_high == 7.0
        assert result.ntm_forward.est_eps_next_year_low == 5.0
        assert result.ntm_forward.analyst_target_mean == 120.0
        assert result.ntm_forward.analyst_target_high == 150.0
        assert result.ntm_forward.analyst_target_low == 90.0
        assert result.ntm_forward.analyst_count == 30

    def test_from_flat_dict_none_values(self) -> None:
        from src.models.financial_data import TemporalFinancialData

        result = TemporalFinancialData.from_flat_dict({"symbol": "TEST", "shortName": "Test"})
        assert result.ltm_ttm.trailing_pe is None
        assert result.mrq.revenue_growth_yoy is None
        assert result.ntm_forward.forward_pe is None

    def test_from_flat_dict_with_tsla_sample(self) -> None:
        from src.models.financial_data import TemporalFinancialData
        from src.data.sample_source import _SAMPLE_DATA

        tsla = _SAMPLE_DATA["TSLA"]["info"]
        result = TemporalFinancialData.from_flat_dict(tsla)

        assert result.ticker == "TSLA"
        assert result.current_price == 245.83
        assert result.ltm_ttm.trailing_pe == 56.72
        assert result.mrq.earnings_growth_yoy == -0.53
        assert result.ntm_forward.forward_pe == 68.97


# ══════════════════════════════════════════════════════════════════
# 计算工具测试
# ══════════════════════════════════════════════════════════════════

class TestCalculationTools:
    """计算工具函数测试。"""

    def test_calculate_target_price(self) -> None:
        from src.utils.calculation_tools import calculate_target_price

        result = calculate_target_price(eps=5.0, pe_multiple=20.0)
        assert result["target_price"] == 100.0
        assert "5.00" in result["formula"]

    def test_calculate_target_price_fractional(self) -> None:
        from src.utils.calculation_tools import calculate_target_price

        result = calculate_target_price(eps=4.33, pe_multiple=25.0)
        assert result["target_price"] == 108.25

    def test_calculate_upside(self) -> None:
        from src.utils.calculation_tools import calculate_upside_downside

        result = calculate_upside_downside(current_price=100.0, target_price=150.0)
        assert result["percentage"] == 50.0
        assert result["direction"] == "上涨"

    def test_calculate_downside(self) -> None:
        from src.utils.calculation_tools import calculate_upside_downside

        result = calculate_upside_downside(current_price=100.0, target_price=75.0)
        assert result["percentage"] == -25.0
        assert result["direction"] == "下跌"

    def test_calculate_upside_zero_current_price(self) -> None:
        from src.utils.calculation_tools import calculate_upside_downside

        result = calculate_upside_downside(current_price=0.0, target_price=100.0)
        assert result["percentage"] is None
        assert "error" in result

    def test_calculate_peg_ratio(self) -> None:
        from src.utils.calculation_tools import calculate_peg_ratio

        result = calculate_peg_ratio(pe_ratio=30.0, earnings_growth_pct=20.0)
        assert result["peg_ratio"] == 1.5

    def test_calculate_peg_zero_growth(self) -> None:
        from src.utils.calculation_tools import calculate_peg_ratio

        result = calculate_peg_ratio(pe_ratio=20.0, earnings_growth_pct=0.0)
        assert result["peg_ratio"] is None
        assert "error" in result

    def test_calculate_growth_rate_simple(self) -> None:
        from src.utils.calculation_tools import calculate_growth_rate

        result = calculate_growth_rate(past_value=100.0, current_value=110.0, periods=1)
        assert result["growth_rate_pct"] == 10.0

    def test_calculate_growth_rate_cagr(self) -> None:
        from src.utils.calculation_tools import calculate_growth_rate

        result = calculate_growth_rate(past_value=100.0, current_value=133.1, periods=3)
        assert result["growth_rate_pct"] == pytest.approx(10.0, rel=0.01)

    def test_calculate_price_from_ps(self) -> None:
        from src.utils.calculation_tools import calculate_price_from_ps

        result = calculate_price_from_ps(revenue_per_share=50.0, ps_multiple=3.0)
        assert result["target_price"] == 150.0


# ══════════════════════════════════════════════════════════════════
# 工具 Schema 定义测试
# ══════════════════════════════════════════════════════════════════

class TestToolDefinitions:
    """工具定义有效性测试。"""

    def test_all_tools_have_required_fields(self) -> None:
        from src.utils.calculation_tools import TOOL_DEFINITIONS

        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params

    def test_tool_registry_matches_definitions(self) -> None:
        from src.utils.calculation_tools import TOOL_REGISTRY, TOOL_DEFINITIONS

        defined_names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        registered_names = set(TOOL_REGISTRY.keys())
        assert defined_names == registered_names

    def test_tool_functions_are_callable(self) -> None:
        from src.utils.calculation_tools import TOOL_REGISTRY

        for name, fn in TOOL_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"


# ══════════════════════════════════════════════════════════════════
# 工具调用逻辑测试 (无 LLM)
# ══════════════════════════════════════════════════════════════════

class TestToolCallingLogic:
    """工具调用逻辑单元测试 (不依赖 LLM API)。"""

    def test_tool_registry_execution(self) -> None:
        from src.utils.calculation_tools import TOOL_REGISTRY

        fn = TOOL_REGISTRY["calculate_target_price"]
        result = fn(eps=6.50, pe_multiple=28.0)
        assert result["target_price"] == 182.0

    def test_tool_registry_error_handling(self) -> None:
        from src.utils.calculation_tools import TOOL_REGISTRY

        fn = TOOL_REGISTRY["calculate_peg_ratio"]
        result = fn(pe_ratio=20.0, earnings_growth_pct=0)
        assert "error" in result

    def test_factuality_skip_tool_calculated(self) -> None:
        """验证事实性校验跳过工具计算结果。"""
        financial_data = "当前价格: $245.83\n每股收益 (TTM): $4.33\n"
        response = "经工具计算，目标价为$108.25，上涨空间为-55.96%。"
        suspicious = verify_numbers_in_response(response, financial_data)
        # 108.25 和 55.96 都应被跳过（工具计算结果）
        values = [v for v, _ in suspicious]
        assert 108.25 not in values
        assert 55.96 not in values
