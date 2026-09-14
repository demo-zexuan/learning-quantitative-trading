"""
测试公共夹具

@module tests.conftest
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlearn.backtest import BacktestEngine, CostModel
from qlearn.data import PricePanel, make_synthetic_panel


@pytest.fixture(scope="session")
def synthetic_panel() -> PricePanel:
    """默认合成面板（5 标的 x 约 5 年，固定种子）。"""
    return make_synthetic_panel(seed=20240101)


@pytest.fixture(scope="session")
def panel_with_suspensions() -> PricePanel:
    """带随机停牌的合成面板，用于测试可交易掩码。"""
    return make_synthetic_panel(seed=7, suspension_prob=0.02)


@pytest.fixture
def zero_cost_engine() -> BacktestEngine:
    """零成本引擎：用于验证会计恒等式。"""
    return BacktestEngine(initial_capital=1_000_000.0, cost_model=CostModel.zero())


@pytest.fixture
def default_engine() -> BacktestEngine:
    """使用 A 股默认成本的引擎。"""
    return BacktestEngine(initial_capital=1_000_000.0)


@pytest.fixture
def single_symbol_panel() -> PricePanel:
    """单标的、价格可手算的面板。

    I. 设计说明

    1. 交易日只有 6 天，价格完全可控
    2. 第 3 个交易日故意设为停牌（open/close/volume 全为 NaN）
    3. 用于精确验证「成交价 = 次日开盘价」与「停牌不可交易」
    """
    dates = pd.bdate_range("2024-01-01", periods=6, name="date")
    close = np.array([10.0, 11.0, 12.0, np.nan, 13.0, 14.0])
    open_px = np.array([10.0, 10.5, 11.5, np.nan, 12.5, 13.5])
    high = np.fmax(open_px, close) + 0.2
    low = np.fmin(open_px, close) - 0.2
    volume = np.array([1e6, 1e6, 1e6, np.nan, 1e6, 1e6])

    columns = pd.Index(["600000"], name="symbol")
    return PricePanel(
        {
            "open": pd.DataFrame(open_px.reshape(-1, 1), index=dates, columns=columns),
            "high": pd.DataFrame(high.reshape(-1, 1), index=dates, columns=columns),
            "low": pd.DataFrame(low.reshape(-1, 1), index=dates, columns=columns),
            "close": pd.DataFrame(close.reshape(-1, 1), index=dates, columns=columns),
            "volume": pd.DataFrame(volume.reshape(-1, 1), index=dates, columns=columns),
        }
    )
