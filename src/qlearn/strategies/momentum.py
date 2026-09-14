"""
横截面动量策略

I. 策略逻辑（学术界最强的异象之一）

1. 计算每个标的过去 lookback 个交易日的累计收益（动量分数）
2. 按动量分数排序，选取排名前 top_n 的标的
3. 等权持有，每 rebalance_days 个交易日重新排序并调仓
4. 可选绝对动量过滤：动量分数为负的标的一律不买（"双动量"）

II. 为什么横截面动量比时序动量稳健

1. 时序动量（单标的择时）的收益高度依赖该标的本身的走势
2. 横截面动量做的是**相对强弱**，天然分散，回撤通常更浅
3. 但横截面动量对"调仓时点"与"交易成本"极其敏感，换手率远高于趋势策略

III. 实盘注意事项

1. 动量崩溃（momentum crash）：市场反转时前期赢家会集体暴跌，
   2009 年与 2020 年 3 月都是典型案例
2. A 股有涨跌停与 T+1，调仓日若撞上涨停可能根本买不进去
3. 需警惕"动量因子的拥挤度"，太多资金使用同一因子会使其失效

@module qlearn.strategies.momentum
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlearn.data.panel import PricePanel
from qlearn.strategies.base import Strategy

__all__ = ["MomentumStrategy"]


class MomentumStrategy(Strategy):
    """横截面动量策略。

    Args:
        lookback: 动量回看窗口（交易日）。
        top_n: 每期持有的标的数量。
        rebalance_days: 调仓间隔（交易日）。
        require_positive: 是否启用绝对动量过滤（动量分数 ≤ 0 不持有）。
        gross_exposure: 总目标仓位。
        min_momentum: 动量分数的绝对下限，None 表示不设限。
    """

    name = "momentum"

    def __init__(
        self,
        lookback: int = 60,
        top_n: int = 3,
        rebalance_days: int = 20,
        require_positive: bool = True,
        gross_exposure: float = 1.0,
        min_momentum: float | None = None,
    ) -> None:
        # I. 参数校验
        if lookback < 2:
            raise ValueError("lookback 至少为 2")
        if top_n < 1:
            raise ValueError("top_n 至少为 1")
        if rebalance_days < 1:
            raise ValueError("rebalance_days 至少为 1")

        super().__init__(
            lookback=lookback,
            top_n=top_n,
            rebalance_days=rebalance_days,
            require_positive=require_positive,
            gross_exposure=gross_exposure,
            min_momentum=min_momentum,
        )
        self.lookback = lookback
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.require_positive = require_positive
        self.gross_exposure = gross_exposure
        self.min_momentum = min_momentum

    def momentum_scores(self, panel: PricePanel) -> pd.DataFrame:
        """计算动量分数矩阵（过去 lookback 日累计收益率）。

        I. 计算方式

        1. score_t = close_t / close_{t-lookback} - 1
        2. 用累计收益而非滚动回归斜率，是为了让量纲统一、便于跨标的比较
        """
        return panel.close.pct_change(self.lookback)

    def generate_weights(self, panel: PricePanel) -> pd.DataFrame:
        """生成动量组合的目标权重。

        I. 计算步骤

        1. 计算动量分数矩阵
        2. 只在调仓日给出权重指令，其余交易日填 NaN（维持既有持仓）
           (1) 这样调仓日之间是"买入持有"，不会产生无谓的每日再平衡换手
        3. 调仓日选出 top_n 个标的，等权分配 gross_exposure
        4. 可选绝对动量过滤与下限过滤，过滤后若不足 top_n 则按实际数量等权

        Returns:
            目标权重矩阵，NaN 表示维持持仓。
        """
        close = panel.close
        scores = self.momentum_scores(panel)

        n_dates = len(close)
        rebalance_rows: list[pd.Series] = []
        rebalance_dates: list[pd.Timestamp] = []

        # I. 从 lookback 之后开始，每 rebalance_days 个交易日调仓一次
        for i in range(self.lookback, n_dates, self.rebalance_days):
            row_scores = scores.iloc[i].dropna()

            # II. 过滤条件
            # 1. 绝对动量过滤：只买上涨的标的
            if self.require_positive:
                row_scores = row_scores[row_scores > 0]
            # 2. 动量下限过滤
            if self.min_momentum is not None:
                row_scores = row_scores[row_scores >= self.min_momentum]

            # 3. 可选标的不足 top_n 时，按实际数量持有（不做补位）
            if row_scores.empty:
                continue

            selected = row_scores.nlargest(min(self.top_n, len(row_scores))).index

            # III. 等权分配
            target = pd.Series(0.0, index=close.columns, name=close.index[i])
            target.loc[selected] = self.gross_exposure / len(selected)

            rebalance_rows.append(target)
            rebalance_dates.append(close.index[i])

        # IV. 非调仓日保持 NaN（不产生指令）
        if not rebalance_rows:
            return pd.DataFrame(
                np.nan, index=close.index, columns=close.columns
            )

        rebalance_frame = pd.DataFrame(rebalance_rows, index=pd.DatetimeIndex(rebalance_dates))
        return rebalance_frame.reindex(index=close.index, columns=close.columns)
