"""
双均线交叉策略（趋势跟随的最小实现）

I. 策略逻辑

1. 快线 = 收盘价的 fast 日简单移动平均
2. 慢线 = 收盘价的 slow 日简单移动平均
3. 快线在慢线**上方** -> 持有；下方 -> 空仓

II. 为什么用它入门

1. 逻辑透明，参数量少（只有 fast / slow），便于理解回测全流程
2. 是"参数敏感性"的绝佳教具：不同参数对结果的影响极大
   (1) 若最优参数是一个孤立的尖峰，几乎可以确定是过拟合
   (2) 若一片连续参数区间都能赚钱，说明存在某种真实的结构性特征
3. 也是理解"趋势跟随为什么要承担高换手成本"的最直观案例

III. 已知缺陷

1. 均线是滞后指标，在震荡市中会反复被打脸（"whipsaw"）
2. 单一标的的趋势策略分散度为零，净值波动完全取决于该标的
3. 未考虑成交量确认、波动率过滤等常用改进

@module qlearn.strategies.ma_cross
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import pandas as pd

from qlearn.data.panel import PricePanel
from qlearn.strategies.base import SizingMode, Strategy, allocate, emit_on_change

__all__ = ["MACrossStrategy"]


class MACrossStrategy(Strategy):
    """双均线交叉策略。

    Args:
        fast: 快线窗口长度。
        slow: 慢线窗口长度，必须大于 fast。
        gross_exposure: 总目标仓位。
        sizing: 仓位分配模式，见 `qlearn.strategies.base.allocate`。
        price_field: 使用的价格字段，默认收盘价。
        rebalance_daily: 是否每日再平衡。
            (1) False（默认）：信号不变就一直持有，权重自然漂移，
                换手率仅来自信号翻转——这是真实趋势策略的做法
            (2) True：每天把权重拉回目标值，换手率会飙升到 1000%+/年，
                可用于教学对比，直观展示"无谓调仓"的成本代价
    """

    name = "ma_cross"

    def __init__(
        self,
        fast: int = 5,
        slow: int = 20,
        gross_exposure: float = 1.0,
        sizing: SizingMode = "fixed",
        price_field: str = "close",
        rebalance_daily: bool = False,
    ) -> None:
        # I. 参数校验：快慢线顺序错误是最常见的低级 bug
        if fast < 2:
            raise ValueError("fast 至少为 2")
        if slow <= fast:
            raise ValueError(f"slow({slow}) 必须大于 fast({fast})")

        super().__init__(
            fast=fast,
            slow=slow,
            gross_exposure=gross_exposure,
            sizing=sizing,
            price_field=price_field,
            rebalance_daily=rebalance_daily,
        )
        self.fast = fast
        self.slow = slow
        self.gross_exposure = gross_exposure
        self.sizing = sizing
        self.price_field = price_field
        self.rebalance_daily = rebalance_daily

    def generate_weights(self, panel: PricePanel) -> pd.DataFrame:
        """生成双均线信号对应的目标权重。

        I. 计算步骤

        1. 取价格矩阵
        2. 分别计算快线与慢线
           (1) min_periods 设为窗口长度，保证指标在窗口填满前为 NaN
           (2) 这是为了避免"用 3 个点算出的 20 日均线"这类虚假信号
        3. 快线 > 慢线 -> 信号 1，否则 0
        4. 慢线尚未形成时（预热期）不下任何指令，保持空仓
        5. 按 sizing 模式转换为目标权重
        6. 默认压缩为「仅在信号变化时下达指令」，避免每日再平衡的无效换手
        """
        price = panel[self.price_field]

        # II. 移动平均线
        fast_ma = price.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = price.rolling(self.slow, min_periods=self.slow).mean()

        # III. 信号：1 = 持有，0 = 空仓
        signal = (fast_ma > slow_ma).astype(float)

        # IV. 预热期（慢线为 NaN）明确置 0，等价于空仓等待
        signal = signal.mask(slow_ma.isna(), 0.0)

        # V. 转换为目标权重
        weights = allocate(signal, self.gross_exposure, self.sizing)

        # VI. 仅在信号变化时下单
        return weights if self.rebalance_daily else emit_on_change(weights)

    def indicators(self, panel: PricePanel) -> pd.DataFrame:
        """返回快慢均线，用于绘图与人工核验信号。

        Returns:
            列名为 `MA_fast` / `MA_slow` 的 DataFrame。
        """
        price = panel[self.price_field]
        return pd.DataFrame(
            {
                f"MA{self.fast}": price.rolling(self.fast, min_periods=self.fast).mean().mean(
                    axis=1
                ),
                f"MA{self.slow}": price.rolling(self.slow, min_periods=self.slow).mean().mean(
                    axis=1
                ),
            }
        )
