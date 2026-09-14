"""
均值回归策略（Z-Score 版）

I. 策略逻辑

1. 对每个标的计算收盘价相对其滚动均值的 Z-Score
   z_t = (close_t - MA_window) / STD_window
2. 开仓条件：z ≤ entry_z（价格显著低于均值，"超卖"）
3. 平仓条件：z ≥ exit_z（价格回归均值）
4. 可选最长持有天数强制平仓，防止"越跌越拿"

II. 与动量的关系（重要认知）

1. 均值回归赚的是"震荡"的钱，趋势策略赚的是"延续"的钱
2. 两者在同一标的上通常是**负相关**的，组合起来能显著平滑净值
3. 因此不要问"哪个更好"，而要问"在什么市场状态下用哪个"

III. 致命风险

1. "接飞刀"：价格下跌往往有基本面原因，均值回归会不断买入下跌中的股票
2. 一旦遇到结构性变化（退市、财务造假、行业逻辑瓦解），亏损可能无法回归
3. 因此**必须**设置止损或最长持有期，并严格控制单标的仓位

@module qlearn.strategies.mean_reversion
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlearn.data.panel import PricePanel
from qlearn.strategies.base import Strategy, emit_on_change, rolling_zscore

__all__ = ["MeanReversionStrategy"]


class MeanReversionStrategy(Strategy):
    """Z-Score 均值回归策略（仅做多）。

    Args:
        window: 滚动均值与标准差的窗口长度。
        entry_z: 开仓阈值，必须为负值（如 -1.5）。
        exit_z: 平仓阈值，必须大于 entry_z（如 -0.5）。
        max_holding_days: 最长持有交易日数，None 表示不限制。
        gross_exposure: 总目标仓位。
        stop_loss: 单标的相对成本的止损比例（如 0.1 表示亏损 10% 止损），
            None 表示不止损。
        rebalance_daily: 是否每日再平衡。默认 False，即信号不变就一直持有，
            仅在开平仓时下单，避免无谓换手。
    """

    name = "mean_reversion"

    def __init__(
        self,
        window: int = 20,
        entry_z: float = -1.5,
        exit_z: float = -0.5,
        max_holding_days: int | None = 30,
        gross_exposure: float = 1.0,
        stop_loss: float | None = 0.10,
        rebalance_daily: bool = False,
    ) -> None:
        # I. 参数校验：阈值方向写反是最常见的错误
        if window < 2:
            raise ValueError("window 至少为 2")
        if entry_z >= 0:
            raise ValueError(f"entry_z({entry_z}) 必须为负值，否则会变成追涨策略")
        if exit_z <= entry_z:
            raise ValueError(f"exit_z({exit_z}) 必须大于 entry_z({entry_z})")
        if max_holding_days is not None and max_holding_days < 1:
            raise ValueError("max_holding_days 至少为 1")
        if stop_loss is not None and not (0 < stop_loss < 1):
            raise ValueError("stop_loss 必须落在 (0, 1) 区间")

        super().__init__(
            window=window,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding_days=max_holding_days,
            gross_exposure=gross_exposure,
            stop_loss=stop_loss,
            rebalance_daily=rebalance_daily,
        )
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_holding_days = max_holding_days
        self.gross_exposure = gross_exposure
        self.stop_loss = stop_loss
        self.rebalance_daily = rebalance_daily

    def zscores(self, panel: PricePanel) -> pd.DataFrame:
        """计算价格 Z-Score 矩阵（用于绘图与调试）。"""
        return rolling_zscore(panel.close, self.window)

    def generate_weights(self, panel: PricePanel) -> pd.DataFrame:
        """生成均值回归组合的目标权重。

        I. 逐标的状态机

        1. 空仓 -> 持仓：z ≤ entry_z（超卖开仓）
        2. 持仓 -> 空仓：满足以下任一条件
           (1) z ≥ exit_z（回归均值，正常止盈）
           (2) 持有天数 ≥ max_holding_days（时间止损）
           (3) 相对建仓价亏损 ≥ stop_loss（价格止损）
        3. Z-Score 为 NaN（预热期或停牌）时不改变现有状态

        II. 仓位分配

        每个标的固定分配 gross_exposure / 标的总数，不做动态再分配。
        这样单标的最大亏损上限是可控的，不会因信号数量变化而失控。

        III. 实现说明

        由于含状态机（路径依赖），无法完全向量化，此处使用显式循环。
        日频策略的标的数与交易日数量级都不大，性能完全可接受。
        """
        close = panel.close
        zscore = self.zscores(panel)

        n_dates = len(close)
        n_symbols = close.shape[1]
        per_symbol_weight = self.gross_exposure / n_symbols

        weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        stop_loss_enabled = self.stop_loss is not None

        for j in range(n_symbols):
            price = close.iloc[:, j].to_numpy(dtype=float)
            z_values = zscore.iloc[:, j].to_numpy(dtype=float)

            holding = False
            days_held = 0
            entry_price = np.nan

            for i in range(n_dates):
                zi = z_values[i]
                price_i = price[i]

                # III.1 数据缺失时保持既有状态不变
                if not np.isfinite(zi) or not np.isfinite(price_i):
                    weights.iloc[i, j] = per_symbol_weight if holding else 0.0
                    continue

                if holding:
                    days_held += 1

                    # (1) 正常止盈：价格回归均值
                    exit_now = zi >= self.exit_z

                    # (2) 时间止损
                    if self.max_holding_days is not None and days_held >= self.max_holding_days:
                        exit_now = True

                    # (3) 价格止损
                    if (
                        stop_loss_enabled
                        and np.isfinite(entry_price)
                        and price_i <= entry_price * (1.0 - float(self.stop_loss))
                    ):
                        exit_now = True

                    if exit_now:
                        holding = False
                        days_held = 0
                        entry_price = np.nan
                # (1) 开仓判断
                elif zi <= self.entry_z:
                    holding = True
                    days_held = 0
                    entry_price = price_i

                weights.iloc[i, j] = per_symbol_weight if holding else 0.0

        # 默认仅在开平仓时下达指令，避免每日再平衡带来的无效换手
        return weights if self.rebalance_daily else emit_on_change(weights)
