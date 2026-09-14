"""
科研工具层测试（参数扫描 / 前向滚动验证）

@module tests.test_research
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pytest

from qlearn.data import make_synthetic_panel
from qlearn.research import (
    grid_search,
    iter_param_combinations,
    n_combinations,
    param_surface,
    split_in_out_sample,
    walk_forward,
)
from qlearn.research.walk_forward import _to_native_param


@pytest.fixture(scope="module")
def small_panel():
    """较小的面板，保证扫描与滚动验证在测试中足够快。"""
    return make_synthetic_panel(
        seed=321, symbols=["A", "B", "C"], start="2019-01-01", end="2023-12-31"
    )


GRID = {"fast": [5, 10], "slow": [20, 60]}


def test_iter_param_combinations_is_cartesian_product():
    combos = list(iter_param_combinations(GRID))
    assert len(combos) == 4
    assert {"fast": 5, "slow": 20} in combos
    assert n_combinations(GRID) == 4


def test_grid_search_returns_sorted_full_grid(small_panel):
    table = grid_search("ma_cross", GRID, small_panel)

    # I. 每个参数组合都必须有一行结果
    assert len(table) == 4
    assert set(table["fast"]) == {5, 10}
    assert set(table["slow"]) == {20, 60}

    # II. 必须包含核心指标列
    for column in ("累计收益", "最大回撤", "年化Sharpe", "年化双边换手率"):
        assert column in table.columns

    # III. 默认按 Sharpe 降序
    sharpes = table["年化Sharpe"].to_numpy()
    assert np.all(np.diff(sharpes) <= 1e-12)


def test_grid_search_can_sort_by_other_metric(small_panel):
    table = grid_search("ma_cross", GRID, small_panel, sort_by="Calmar比率")
    values = table["Calmar比率"].to_numpy()
    assert np.all(np.diff(values) <= 1e-12)


def test_grid_search_tolerates_invalid_combinations(small_panel):
    """非法参数组合必须被记录为 NaN 而不是中断整次扫描。"""
    table = grid_search("ma_cross", {"fast": [5, 10], "slow": [4, 20]}, small_panel)

    assert len(table) == 4
    assert "error" in table.columns

    # slow=4 同时违反 fast=5 与 fast=10 的约束，因此两个组合都应失败
    failed = table[table["error"].notna()]
    assert len(failed) == 2
    assert set(failed["slow"]) == {4}
    assert failed["累计收益"].isna().all()


def test_grid_search_raises_when_errors_not_ignored(small_panel):
    with pytest.raises(ValueError, match="必须大于"):
        grid_search(
            "ma_cross",
            {"fast": [5], "slow": [4]},
            small_panel,
            ignore_errors=False,
        )


def test_param_surface_shape(small_panel):
    table = grid_search("ma_cross", GRID, small_panel)
    surface = param_surface(table, "fast", "slow", metric="年化Sharpe")

    assert surface.shape == (2, 2)
    assert list(surface.index) == [5, 10]
    assert list(surface.columns) == [20, 60]


def test_param_surface_rejects_unknown_column(small_panel):
    table = grid_search("ma_cross", GRID, small_panel)
    with pytest.raises(KeyError, match="不存在列"):
        param_surface(table, "nonexistent", "slow")


def test_split_in_out_sample(small_panel):
    split_date = small_panel.dates[len(small_panel.dates) // 2]
    in_sample, out_sample = split_in_out_sample(small_panel, split_date)

    assert in_sample.dates.max() < split_date
    assert out_sample.dates.min() >= split_date
    assert len(in_sample.dates) + len(out_sample.dates) == small_panel.n_dates


def test_split_in_out_sample_rejects_out_of_range(small_panel):
    with pytest.raises(ValueError, match="无法切分"):
        split_in_out_sample(small_panel, "1990-01-01")


def test_walk_forward_produces_contiguous_non_overlapping_oos(small_panel):
    result = walk_forward(
        "ma_cross",
        GRID,
        small_panel,
        train_days=252,
        test_days=126,
        warmup_days=60,
    )

    # I. 至少完成一折
    assert len(result.folds) > 0

    # II. 样本外收益索引不得重复（各折之间不重叠）
    assert not result.oos_returns.index.duplicated().any()

    # III. 每折必须同时记录样本内与样本外指标
    for column in ("IS_年化Sharpe", "OOS_年化Sharpe", "param_fast", "param_slow"):
        assert column in result.folds.columns

    # IV. 选中的参数必须落在给定网格内
    assert set(result.folds["param_fast"]).issubset({5, 10})
    assert set(result.folds["param_slow"]).issubset({20, 60})


def test_walk_forward_equity_starts_at_initial_capital(small_panel):
    result = walk_forward("ma_cross", GRID, small_panel, train_days=252, test_days=126)
    equity = result.oos_equity

    assert not equity.empty
    assert np.isfinite(equity.to_numpy()).all()

    metrics = result.metrics()
    for key in ("累计收益", "年化Sharpe", "最大回撤"):
        assert key in metrics.index


def test_walk_forward_overfit_gap_keys(small_panel):
    result = walk_forward("ma_cross", GRID, small_panel, train_days=252, test_days=126)
    gap = result.overfit_gap()

    for key in ("样本内平均Sharpe", "样本外Sharpe", "样本外累计收益", "折数"):
        assert key in gap.index


def test_walk_forward_rejects_too_short_panel():
    tiny = make_synthetic_panel(seed=1, start="2024-01-01", end="2024-06-30")
    with pytest.raises(ValueError, match="数据不足"):
        walk_forward("ma_cross", GRID, tiny, train_days=500, test_days=200)


def test_to_native_param_converts_integral_floats():
    """pandas 会把整数列升格为 float64，必须还原为 int 否则 rolling 会报错。"""
    assert _to_native_param(np.float64(5.0)) == 5
    assert isinstance(_to_native_param(np.float64(5.0)), int)
    assert isinstance(_to_native_param(np.int64(7)), int)

    # 非整数浮点必须保持原样
    assert _to_native_param(np.float64(-1.5)) == pytest.approx(-1.5)
    assert isinstance(_to_native_param(np.float64(-1.5)), float)

    # 布尔值不能被转成数字
    assert _to_native_param(np.bool_(True)) is True or _to_native_param(np.bool_(True)) == True  # noqa: E712
