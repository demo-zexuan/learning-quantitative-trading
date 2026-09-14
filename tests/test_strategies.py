"""
策略层测试

I. 测试重点

1. **无前视偏差**：用截断到 t 日的数据算出的权重，必须与用完整数据算出的
   t 日之前的权重完全一致。这是量化里最容易犯、也最难发现的错误。
2. 权重区间合法：默认做多策略的权重必须落在 [0, 1] 内
3. 参数校验：非法参数必须尽早抛错，而不是产生错误的回测结果
4. 缺失值语义：NaN 表示维持持仓而非清仓

@module tests.test_strategies
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlearn.strategies import (
    MACrossStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    create_strategy,
    list_strategies,
    register_strategy,
)

# 参与无前视偏差检验的策略与其参数
_CAUSAL_CASES = [
    ("ma_cross", {"fast": 5, "slow": 20}),
    ("ma_cross", {"fast": 10, "slow": 60, "rebalance_daily": True}),
    ("momentum", {"lookback": 40, "top_n": 2, "rebalance_days": 15}),
    ("mean_reversion", {"window": 20, "entry_z": -1.5, "exit_z": -0.5}),
    ("buy_and_hold", {}),
]


@pytest.mark.parametrize("name,params", _CAUSAL_CASES)
def test_no_lookahead_bias(synthetic_panel, name, params):
    """无前视偏差：前 t 日的权重不得因看到未来数据而改变。

    I. 检验原理

    1. 用完整面板计算权重矩阵 W_full
    2. 把面板截断到第 300 个交易日，重新计算 W_trunc
    3. 若策略引用了未来数据（如 shift(-1)、bfill、全样本 z-score），
       W_full 与 W_trunc 在前 300 行必然不一致
    """
    cutoff = synthetic_panel.dates[300]
    strategy = create_strategy(name, **params)

    full_weights = strategy.generate_weights(synthetic_panel)
    truncated_weights = strategy.generate_weights(synthetic_panel.slice_dates(end=cutoff))

    pd.testing.assert_frame_equal(
        full_weights.loc[:cutoff],
        truncated_weights,
        check_exact=True,
        obj=f"策略 {name} 的前 {cutoff:%Y-%m-%d} 权重",
    )


@pytest.mark.parametrize("name,params", _CAUSAL_CASES)
def test_weights_within_bounds(synthetic_panel, name, params):
    """默认做多策略的权重必须落在 [0, 1]，且每行合计不超过 1。"""
    weights = create_strategy(name, **params).generate_weights(synthetic_panel)

    values = weights.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    assert (finite >= -1e-12).all(), f"{name} 产生了负权重"
    assert (finite <= 1.0 + 1e-9).all(), f"{name} 产生了超过 1 的权重"

    row_sums = weights.fillna(0.0).sum(axis=1)
    assert (row_sums <= 1.0 + 1e-9).all(), f"{name} 每行权重之和超过 1"


@pytest.mark.parametrize("name,params", _CAUSAL_CASES)
def test_weights_shape_matches_panel(synthetic_panel, name, params):
    """权重矩阵的索引与列必须与面板完全一致。"""
    weights = create_strategy(name, **params).generate_weights(synthetic_panel)

    assert weights.index.equals(synthetic_panel.dates)
    assert list(weights.columns) == list(synthetic_panel.symbols)


def test_ma_cross_rejects_invalid_windows():
    """慢线必须大于快线，否则策略逻辑自相矛盾。"""
    with pytest.raises(ValueError, match="必须大于"):
        MACrossStrategy(fast=20, slow=5)
    with pytest.raises(ValueError, match="至少为 2"):
        MACrossStrategy(fast=1, slow=10)


def test_mean_reversion_rejects_inverted_thresholds():
    """开仓阈值必须为负、平仓阈值必须大于开仓阈值，写反会变成追涨策略。"""
    with pytest.raises(ValueError, match="必须为负值"):
        MeanReversionStrategy(entry_z=1.5)
    with pytest.raises(ValueError, match="必须大于"):
        MeanReversionStrategy(entry_z=-0.5, exit_z=-1.5)


def test_mean_reversion_state_machine_is_causal():
    """均值回归的逐标的状态机也不得使用未来数据。"""
    panel = _momentum_panel()
    strategy = MeanReversionStrategy(window=10, entry_z=-1.0, exit_z=-0.2, max_holding_days=10)

    cutoff = panel.dates[80]
    full = strategy.generate_weights(panel)
    truncated = strategy.generate_weights(panel.slice_dates(end=cutoff))

    pd.testing.assert_frame_equal(full.loc[:cutoff], truncated, check_exact=True)


def _momentum_panel():
    from qlearn.data import make_synthetic_panel

    return make_synthetic_panel(seed=99, autocorrelation=-0.1)


def test_momentum_only_emits_on_rebalance_dates(synthetic_panel):
    """动量策略只应在调仓日下达指令，其余交易日的权重为 NaN（维持持仓）。"""
    rebalance_days = 20
    lookback = 40
    strategy = MomentumStrategy(
        lookback=lookback, top_n=2, rebalance_days=rebalance_days, require_positive=False
    )
    weights = strategy.generate_weights(synthetic_panel)

    instructed = weights.notna().any(axis=1)
    instructed_rows = np.nonzero(instructed.to_numpy())[0]
    assert len(instructed_rows) > 0

    # 相邻两次指令的间隔必须严格等于 rebalance_days
    assert np.all(np.diff(instructed_rows) == rebalance_days)
    assert instructed_rows[0] == lookback


def test_emit_on_change_suppresses_repeated_identical_weights(synthetic_panel):
    """日频再平衡开关必须显著改变换手相关的指令密度。

    每日再平衡会产生远多于"仅信号变化时下单"的指令行。
    """
    daily = MACrossStrategy(fast=5, slow=20, rebalance_daily=True).generate_weights(synthetic_panel)
    on_change = MACrossStrategy(fast=5, slow=20, rebalance_daily=False).generate_weights(
        synthetic_panel
    )

    assert daily.notna().sum().sum() > on_change.notna().sum().sum()


def test_create_strategy_rejects_unknown_name():
    """未注册的策略名必须给出可用列表。"""
    with pytest.raises(KeyError, match="未知策略"):
        create_strategy("not_a_strategy")


def test_create_strategy_rejects_unknown_params():
    """非法参数必须提前拦截，避免静默使用默认值。"""
    with pytest.raises(TypeError, match="不接受参数"):
        create_strategy("ma_cross", fast=5, slow=20, nonsense=1)


def test_registry_contains_builtin_strategies():
    """内置策略必须全部注册到位。"""
    names = list_strategies()
    for expected in ("buy_and_hold", "ma_cross", "mean_reversion", "momentum"):
        assert expected in names


def test_register_strategy_requires_name():
    """自定义策略必须声明唯一的 name，否则无法被 CLI 调用。"""
    from qlearn.strategies import Strategy

    with pytest.raises(ValueError, match="name"):

        @register_strategy
        class NamelessStrategy(Strategy):  # noqa: D101
            def generate_weights(self, panel):  # noqa: ANN001, ANN201
                return pd.DataFrame()


def test_param_names_excludes_variadic():
    """参数名解析必须排除 self 与 *args/**kwargs。"""
    assert MACrossStrategy.param_names() == [
        "fast", "slow", "gross_exposure", "sizing", "price_field", "rebalance_daily"
    ]
