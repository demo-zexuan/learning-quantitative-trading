"""
数据层测试：面板一致性与校验

@module tests.test_panel
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlearn.data import PricePanel, make_synthetic_panel


def _frames() -> dict[str, pd.DataFrame]:
    """构造两个标的、日期不完全重合的行情表。"""
    idx_a = pd.bdate_range("2024-01-01", periods=5, name="date")
    idx_b = pd.bdate_range("2024-01-03", periods=5, name="date")

    def build(index: pd.DatetimeIndex, base: float) -> pd.DataFrame:
        values = np.arange(len(index), dtype=float) + base
        return pd.DataFrame(
            {
                "open": values,
                "high": values + 1.0,
                "low": values - 1.0,
                "close": values + 0.5,
                "volume": np.full(len(index), 1e6),
            },
            index=index,
        )

    return {"AAA": build(idx_a, 10.0), "BBB": build(idx_b, 20.0)}


def test_from_frames_uses_union_index():
    """日期索引必须取并集，未覆盖处填 NaN，而不是被最早上市标的截断。"""
    panel = PricePanel.from_frames(_frames())

    assert len(panel.dates) == 7  # 1/1..1/9 的工作日并集
    assert list(panel.symbols) == ["AAA", "BBB"]

    # AAA 在 1/8 之后没有数据，应为 NaN
    assert panel.close.loc["2024-01-09", "AAA"] != panel.close.loc["2024-01-09", "AAA"]  # NaN
    assert np.isfinite(panel.close.loc["2024-01-09", "BBB"])


def test_tradable_mask_requires_open_and_volume():
    """可交易必须同时满足开盘价有效且成交量有效。"""
    panel = make_synthetic_panel(seed=3, suspension_prob=0.05, symbols=["X", "Y"])
    tradable = panel.tradable

    suspended = panel.open.isna()
    assert not bool((tradable.to_numpy() & suspended.to_numpy()).any()), "停牌日被误判为可交易"
    assert bool((tradable.to_numpy() <= panel.volume.notna().to_numpy()).all())


def test_valued_close_only_fills_forward():
    """估值价只能前值填充，绝不能后向填充（后向填充会引入未来信息）。"""
    panel = make_synthetic_panel(seed=5, suspension_prob=0.05, symbols=["X"])
    close = panel.close
    valued = panel.valued_close

    # I. 有数据的日期必须完全一致（用 numpy 掩码比较，避免 NaN 干扰 allclose）
    mask = close.notna().to_numpy()
    assert np.allclose(
        valued.to_numpy()[mask], close.to_numpy()[mask], equal_nan=True
    )

    # II. 首个有效值之前必须仍为 NaN（若做了后向填充，这里会是非 NaN）
    first_valid = close.notna().idxmax().iloc[0]
    before_first_valid = valued.loc[:first_valid].iloc[:-1]
    assert bool(before_first_valid.isna().to_numpy().all())


def test_panel_rejects_non_datetime_index():
    """索引不是 DatetimeIndex 时必须报错。"""
    frame = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=[0],
    )
    with pytest.raises(TypeError, match="DatetimeIndex"):
        PricePanel.from_frames({"X": frame})


def test_panel_rejects_duplicated_dates():
    """重复日期会让 rolling 与 pct_change 语义失真，必须拒绝。"""
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    frame = pd.DataFrame(
        {"open": [1.0, 1.0, 1.0], "high": [1.0] * 3, "low": [1.0] * 3,
         "close": [1.0] * 3, "volume": [1.0] * 3},
        index=index,
    )
    with pytest.raises(ValueError, match="重复日期"):
        PricePanel.from_frames({"X": frame})


def test_panel_rejects_unsorted_index():
    """降序索引会让引擎的时间推进逻辑完全错乱，必须拒绝。

    注意：`from_frames` 会把各标的索引求并集后排序，因此这里直接构造
    PricePanel 来验证校验逻辑本身。
    """
    index = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
    columns = pd.Index(["X"], name="symbol")

    def frame(values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(values, index=index, columns=columns)

    fields = {
        "open": frame([1.0] * 3),
        "high": frame([1.0] * 3),
        "low": frame([1.0] * 3),
        "close": frame([1.0] * 3),
        "volume": frame([1.0] * 3),
    }
    with pytest.raises(ValueError, match="升序"):
        PricePanel(fields)


def test_from_frames_normalizes_index_order():
    """from_frames 必须把乱序的输入索引整理为升序。"""
    index = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
    frame = pd.DataFrame(
        {"open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3,
         "close": [1.0] * 3, "volume": [1.0] * 3},
        index=index,
    )
    panel = PricePanel.from_frames({"X": frame})
    assert panel.dates.is_monotonic_increasing


def test_select_and_slice_preserve_consistency():
    """筛选与切片后，各字段的索引与列必须仍然严格对齐。"""
    panel = make_synthetic_panel(seed=9, symbols=["A", "B", "C"])
    subset = panel.select(["C", "A"]).slice_dates(start=panel.dates[10], end=panel.dates[100])

    assert list(subset.symbols) == ["C", "A"]
    assert subset.n_dates == 91
    for name, frame in subset.fields.items():
        assert frame.index.equals(subset.dates), f"字段 {name} 索引未对齐"


def test_drop_inactive_symbols_filters_by_observation_count():
    """样本不足的标的（新股/长期停牌）应被剔除。"""
    panel = make_synthetic_panel(seed=11, symbols=["A", "B"], suspension_prob=0.5)
    counts = panel.close.notna().sum(axis=0)
    threshold = int(counts.max())

    filtered = panel.drop_inactive_symbols(min_observations=threshold)
    assert filtered.n_symbols >= 1
    assert (filtered.close.notna().sum(axis=0) >= threshold).all()


def test_synthetic_autocorrelation_changes_serial_dependence():
    """自相关旋钮必须真实改变收益序列的一阶自相关符号。"""
    returns_positive = make_synthetic_panel(
        seed=13, symbols=["X"], autocorrelation=0.2
    ).close.pct_change().dropna()["X"]
    returns_negative = make_synthetic_panel(
        seed=13, symbols=["X"], autocorrelation=-0.2
    ).close.pct_change().dropna()["X"]

    assert returns_positive.autocorr(1) > 0.05
    assert returns_negative.autocorr(1) < -0.05


def test_invalid_autocorrelation_rejected():
    """自相关系数越界必须报错。"""
    with pytest.raises(ValueError, match="autocorrelation"):
        make_synthetic_panel(autocorrelation=1.5)


def test_synthetic_high_low_bracket_open_close():
    """合成的 K 线必须合法：high ≥ max(open, close) 且 low ≤ min(open, close)。"""
    panel = make_synthetic_panel(seed=17)
    assert (panel.high >= panel.open).all().all()
    assert (panel.high >= panel.close).all().all()
    assert (panel.low <= panel.open).all().all()
    assert (panel.low <= panel.close).all().all()
