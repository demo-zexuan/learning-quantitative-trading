"""
买入持有基准策略

I. 用途

1. 作为任何策略的**最低及格线**：跑不赢买入持有，策略就没有存在价值
2. 演示引擎的 "NaN = 维持持仓" 语义：只在首日下达一次指令，之后不再调仓

II. 常见误区

很多人把"等权买入 N 只股票并每日再平衡"当作买入持有，这是错的。
每日再平衡本身就是一个高换手策略，会持续产生成本，
且在趋势市中表现优于真实的买入持有（因为不断减仓上涨的标的）。
本策略只发一次指令，之后持仓股数完全不变，是真正的 buy & hold。

@module qlearn.strategies.buy_and_hold
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlearn.data.panel import PricePanel
from qlearn.strategies.base import Strategy

__all__ = ["BuyAndHoldStrategy"]


class BuyAndHoldStrategy(Strategy):
    """等权买入并持有。

    Args:
        gross_exposure: 总目标仓位，1.0 表示满仓。
        start_position: 首次建仓的日期序号（从 0 开始），用于构造
            "从某个时点才开始"的基准。
    """

    name = "buy_and_hold"

    def __init__(self, gross_exposure: float = 1.0, start_position: int = 0) -> None:
        if gross_exposure <= 0:
            raise ValueError("gross_exposure 必须为正数")
        if start_position < 0:
            raise ValueError("start_position 不能为负")

        super().__init__(gross_exposure=gross_exposure, start_position=start_position)
        self.gross_exposure = gross_exposure
        self.start_position = start_position

    def generate_weights(self, panel: PricePanel) -> pd.DataFrame:
        """只在建仓日发出一次指令，其余交易日均返回 NaN（维持持仓）。"""
        weights = pd.DataFrame(np.nan, index=panel.dates, columns=panel.symbols)

        position = min(self.start_position, len(panel.dates) - 1)
        weights.iloc[position] = self.gross_exposure / panel.n_symbols
        return weights
