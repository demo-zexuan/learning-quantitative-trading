"""
回测引擎测试

I. 测试重点

1. **会计恒等式**：equity_t = equity_{t-1} * (1 + gross_ret_t) - cost_t
2. **成交时序**：第 t 日权重必须在第 t+1 日开盘成交，不能提前
3. **零成本一致性**：零成本下净值必须严格等于毛收益累乘
4. **停牌处理**：停牌日不可交易，只能用前收盘价估值
5. **资金约束**：现金不足时不得产生负现金（隐性杠杆）
6. **整手约束**：A 股必须按 100 股整手成交

@module tests.test_engine
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlearn.backtest import BacktestEngine, CostModel
from qlearn.strategies import BuyAndHoldStrategy, create_strategy

# 浮点比较的绝对容差（净值量级为 1e6 元，1e-6 相当于分位）
_TOL = 1e-6


def test_zero_cost_accounting_identity(synthetic_panel, zero_cost_engine):
    """零成本下，净值必须严格等于毛收益的累乘，并且没有任何成本。"""
    strategy = create_strategy("ma_cross", fast=5, slow=20)
    result = zero_cost_engine.run(strategy.generate_weights(synthetic_panel), synthetic_panel)

    # I. 成本必须为 0
    assert np.allclose(result.costs.to_numpy(), 0.0, atol=_TOL)

    # II. 净收益 == 毛收益
    assert np.allclose(result.returns.to_numpy(), result.gross_returns.to_numpy(), atol=1e-12)

    # III. 净值 == 初始资金 * cumprod(1 + 收益)
    expected_equity = (1.0 + result.returns).cumprod() * zero_cost_engine.initial_capital
    assert np.allclose(result.equity.to_numpy(), expected_equity.to_numpy(), rtol=1e-10)


def test_gross_net_identity_with_costs(synthetic_panel, default_engine):
    """有成本时，必须满足 equity_t = equity_{t-1} * (1 + gross_ret_t) - cost_t。"""
    strategy = create_strategy("momentum", lookback=40, top_n=3, rebalance_days=20)
    result = default_engine.run(strategy.generate_weights(synthetic_panel), synthetic_panel)

    equity = result.equity.to_numpy()
    equity_prev = np.concatenate([[default_engine.initial_capital], equity[:-1]])
    gross = result.gross_returns.to_numpy()
    costs = result.costs.to_numpy()

    reconstructed = equity_prev * (1.0 + gross) - costs
    assert np.allclose(equity, reconstructed, rtol=1e-10)

    # 累计成本应为正，且毛收益累计应高于净收益
    assert result.costs.sum() > 0
    assert result.gross_returns.sum() > result.returns.sum()


def test_execution_uses_next_open(single_symbol_panel, zero_cost_engine):
    """成交必须发生在决策日的**次日开盘**，且按开盘价成交。"""
    strategy = BuyAndHoldStrategy(gross_exposure=1.0)
    weights = strategy.generate_weights(single_symbol_panel)
    result = zero_cost_engine.run(weights, single_symbol_panel)

    # I. 只在第 2 个交易日（索引 1）有一笔成交，价格为该日开盘价 10.5
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["date"] == single_symbol_panel.dates[1]
    assert trade["price"] == pytest.approx(10.5)
    assert trade["side"] == "buy"

    # II. 成交股数必须为 100 的整数倍（A 股整手约束）
    assert int(trade["shares"]) % 100 == 0
    assert trade["shares"] == pytest.approx(
        np.floor(zero_cost_engine.initial_capital / 10.5 / 100) * 100
    )


def test_buy_and_hold_equity_matches_manual(single_symbol_panel, zero_cost_engine):
    """买入持有的净值应能与手工计算完全对上。"""
    result = zero_cost_engine.run(
        BuyAndHoldStrategy().generate_weights(single_symbol_panel), single_symbol_panel
    )

    capital = zero_cost_engine.initial_capital
    shares = np.floor(capital / 10.5 / 100) * 100
    cash = capital - shares * 10.5
    close = single_symbol_panel.valued_close["600000"].to_numpy()

    # 第 0 日尚未建仓，权益必须等于初始资金；第 1 日起为「现金 + 持仓市值」
    expected = cash + shares * close
    expected[0] = capital
    assert np.allclose(result.equity.to_numpy(), expected, rtol=1e-10)

    # 持仓股数在第 2 日起恒定不变（真正的买入持有）
    positions = result.positions["600000"].to_numpy()
    assert positions[0] == 0
    assert np.allclose(positions[1:], shares)


def test_positions_unchanged_when_weights_are_nan(single_symbol_panel, zero_cost_engine):
    """NaN 权重必须表示"维持持仓"，不能触发调仓。"""
    weights = pd.DataFrame(
        np.nan,
        index=single_symbol_panel.dates,
        columns=single_symbol_panel.symbols,
    )
    weights.iloc[0] = 0.5  # 仅首日下达一次指令

    result = zero_cost_engine.run(weights, single_symbol_panel)
    positions = result.positions["600000"].to_numpy()

    assert len(result.trades) == 1
    assert np.allclose(positions[1:], positions[1])


def test_suspended_symbol_not_traded(panel_with_suspensions, default_engine):
    """停牌日（成交量或开盘价为空）不允许成交。"""
    strategy = create_strategy("ma_cross", fast=5, slow=20)
    result = default_engine.run(
        strategy.generate_weights(panel_with_suspensions), panel_with_suspensions
    )

    if result.trades.empty:
        pytest.skip("该随机种子下没有产生成交")

    # I. 每笔成交的日期，该标的当日必须是可交易的
    tradable = panel_with_suspensions.tradable
    for row in result.trades.itertuples():
        assert bool(tradable.loc[row.date, row.symbol]), (
            f"在停牌日成交: {row.symbol} @ {row.date}"
        )

    # II. 停牌日不应产生任何成本
    traded_dates = set(result.trades["date"])
    suspended_dates = [
        date for date in result.costs.index if not tradable.loc[date].any()
    ]
    assert not (set(suspended_dates) & traded_dates)


def test_cash_constraint_prevents_leverage(synthetic_panel):
    """请求 2 倍杠杆时，引擎必须压缩买单，实际权重不得超过 1（无隐性杠杆）。"""
    engine = BacktestEngine(initial_capital=1_000_000.0)
    # equal_active 模式下只要有一个信号，该行权重之和就等于 gross_exposure，
    # 因此必然触发杠杆警告与现金约束
    strategy = create_strategy(
        "ma_cross", fast=5, slow=20, gross_exposure=2.0, sizing="equal_active"
    )

    with pytest.warns(UserWarning, match="目标权重之和超过 1"):
        result = engine.run(strategy.generate_weights(synthetic_panel), synthetic_panel)

    # 实际持仓权重之和不得超过 1（留 2% 容差覆盖建仓日的成本与整手取整）
    gross = result.weights.sum(axis=1)
    assert gross.max() <= 1.02, f"实际总仓位达到 {gross.max():.4f}，出现了隐性杠杆"

    # 净值必须始终为正（未破产）
    assert (result.equity > 0).all()


def test_negative_weights_warn(synthetic_panel, default_engine):
    """负权重（做空）应触发警告，因为 A 股个股做空受限。"""
    weights = pd.DataFrame(
        0.0, index=synthetic_panel.dates, columns=synthetic_panel.symbols
    )
    weights.iloc[5:, 0] = -0.5
    weights.iloc[5:, 1] = 0.5

    with pytest.warns(UserWarning, match="负值"):
        default_engine.run(weights, synthetic_panel)


def test_engine_rejects_unknown_symbols(synthetic_panel, default_engine):
    """权重矩阵包含面板中不存在的标的时，必须直接报错而非静默忽略。"""
    weights = pd.DataFrame(
        0.5, index=synthetic_panel.dates, columns=[*synthetic_panel.symbols, "999999"]
    )
    with pytest.raises(ValueError, match="不存在的标的"):
        default_engine.run(weights, synthetic_panel)


def test_max_participation_limits_trade_size(synthetic_panel):
    """流动性约束生效时，单日成交股数不得超过当日成交量的一定比例。"""
    engine = BacktestEngine(
        initial_capital=1_000_000_000.0,  # 超大资金，强制触发流动性约束
        max_participation=0.01,
    )
    strategy = BuyAndHoldStrategy()
    result = engine.run(strategy.generate_weights(synthetic_panel), synthetic_panel)

    volume = synthetic_panel.volume
    for row in result.trades.itertuples():
        limit = 0.01 * float(volume.loc[row.date, row.symbol])
        assert row.shares <= limit + 100, (
            f"{row.symbol}@{row.date} 成交 {row.shares} 股，超过流动性上限 {limit:.0f}"
        )


def test_initial_capital_must_be_positive():
    """初始资金非法时必须立刻报错。"""
    with pytest.raises(ValueError, match="initial_capital"):
        BacktestEngine(initial_capital=0)


def test_cost_model_zero_and_default_rates():
    """成本模型的基本性质：卖出比买入贵（印花税），往返成本大于单边。"""
    model = CostModel()
    assert model.sell_rate > model.buy_rate
    assert model.cost(100_000, "sell") > model.cost(100_000, "buy")
    assert CostModel.zero().cost(1_000_000, "buy") == 0.0

    # 最低佣金生效：极小金额的佣金必须被抬到 5 元
    tiny = model.cost(1000.0, "buy")
    assert tiny >= model.min_commission


def test_result_reports_are_aligned_and_readable(synthetic_panel, default_engine):
    """结果概览与成本影响报告必须能渲染成多行文本且列对齐。"""
    from qlearn.utils.text import display_width

    strategy = create_strategy("ma_cross", fast=5, slow=20)
    result = default_engine.run(strategy.generate_weights(synthetic_panel), synthetic_panel)

    # I. 概览
    overview = result.describe()
    assert "回测结果概览" in overview
    assert "累计收益" in overview

    # II. 成本影响：中文按 2 列宽对齐，分隔符必须落在同一列
    report = result.describe_cost_impact()
    separators = {display_width(line.split(" : ")[0]) for line in report.split("\n")}
    assert len(separators) == 1, f"成本影响表未对齐: {separators}"

    # III. 成本影响数值必须自洽
    impact = result.cost_impact()
    assert impact["成本侵蚀收益"] == pytest.approx(
        impact["毛累计收益"] - impact["净累计收益"]
    )
    assert impact["累计成本(元)"] == pytest.approx(result.costs.sum())
