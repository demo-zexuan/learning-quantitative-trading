"""
A 股交易成本模型

I. A 股实际交易成本构成

1. 佣金（双边）
   (1) 主流券商万分之 1 ~ 万分之 3，按成交金额收取
   (2) 通常设有**单笔最低 5 元**的门槛，小资金的高频策略会被此项严重侵蚀
2. 印花税（仅卖出）
   (1) 2023-08-28 起由千分之 1 减半为万分之 5
   (2) 只对卖方征收，这是 A 股"卖出更贵"的根源
3. 过户费（双边）
   (1) 沪深两市统一为成交金额的 0.001%（十万分之 1）
4. 滑点
   (1) 真实成交价与理论价之间的偏差，来源于买卖价差与冲击成本
   (2) 无法精确建模，只能用保守估计（如 2~10 个基点）

II. 为什么必须显式建模成本

一个年换手 500% 的策略，若双边总成本为 0.1%，每年仅成本就吃掉约 1% 的收益；
若换成日频高换手策略（年换手 5000%+），成本可达 5% 以上，足以让"回测赚钱"的策略变成实盘亏损。

@module qlearn.backtest.costs
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

__all__ = ["CostModel", "Side"]

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class CostModel:
    """A 股交易成本模型（不可变，可安全共享）。

    Attributes:
        commission_rate: 佣金费率，双边收取。
        min_commission: 单笔最低佣金（元）。
        stamp_duty_rate: 印花税率，仅卖出收取。
        transfer_fee_rate: 过户费率，双边收取。
        slippage_bps: 滑点，单位为基点（1 bps = 0.01%），双边收取。
        apply_min_commission: 是否启用单笔最低佣金。
    """

    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_bps: float = 2.0
    apply_min_commission: bool = True

    # ------------------------------------------------------------------
    # 派生费率
    # ------------------------------------------------------------------

    @property
    def slippage_rate(self) -> float:
        """滑点费率（由基点换算为小数）。"""
        return self.slippage_bps / 10_000.0

    @property
    def buy_rate(self) -> float:
        """买入方向的**比例型**综合费率（不含最低佣金约束）。"""
        return self.commission_rate + self.transfer_fee_rate + self.slippage_rate

    @property
    def sell_rate(self) -> float:
        """卖出方向的**比例型**综合费率（不含最低佣金约束）。"""
        return (
            self.commission_rate
            + self.transfer_fee_rate
            + self.stamp_duty_rate
            + self.slippage_rate
        )

    # ------------------------------------------------------------------
    # 成本计算
    # ------------------------------------------------------------------

    def cost(self, notional: float, side: Side) -> float:
        """计算单笔成交的总成本。

        I. 计算顺序

        1. 佣金 = max(成交金额 * 佣金费率, 最低佣金)
           (1) 最低佣金只对佣金本身生效，不影响印花税与过户费
        2. 过户费 = 成交金额 * 过户费率
        3. 印花税 = 成交金额 * 印花税率（仅卖出）
        4. 滑点成本 = 成交金额 * 滑点费率
        5. 合计 = 1 + 2 + 3 + 4

        Args:
            notional: 成交金额（元），应取绝对值。
            side: "buy" 或 "sell"。

        Returns:
            该笔成交的总成本（元）。成交金额为 0 或负数时返回 0。
        """
        # I. 边界处理
        # NaN 与 0 都不产生成本；注意 NaN <= 0 为 False，必须显式用 isfinite 判断
        if notional <= 0 or not math.isfinite(notional):
            return 0.0

        # II. 逐项计算
        # 1. 佣金（含最低佣金约束）
        commission = notional * self.commission_rate
        if self.apply_min_commission:
            commission = max(commission, self.min_commission)

        # 2. 过户费
        transfer_fee = notional * self.transfer_fee_rate

        # 3. 印花税（仅卖出）
        stamp_duty = notional * self.stamp_duty_rate if side == "sell" else 0.0

        # 4. 滑点
        slippage = notional * self.slippage_rate

        return float(commission + transfer_fee + stamp_duty + slippage)

    def rate(self, side: Side) -> float:
        """返回该方向的比例型费率（不含最低佣金），用于快速估算。"""
        return self.buy_rate if side == "buy" else self.sell_rate

    def describe(self) -> pd.Series:
        """返回人类可读的成本参数表。"""
        return pd.Series(
            {
                "佣金费率(双边)": f"{self.commission_rate:.4%}",
                "单笔最低佣金": f"{self.min_commission:.2f} 元"
                if self.apply_min_commission
                else "未启用",
                "印花税(仅卖出)": f"{self.stamp_duty_rate:.4%}",
                "过户费(双边)": f"{self.transfer_fee_rate:.5%}",
                "滑点(双边)": f"{self.slippage_bps:.1f} bps",
                "买入综合费率": f"{self.buy_rate:.4%}",
                "卖出综合费率": f"{self.sell_rate:.4%}",
                "单次往返成本(约)": f"{self.buy_rate + self.sell_rate:.4%}",
            },
            name="cost_model",
        )

    # ------------------------------------------------------------------
    # 构造器
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CostModel:
        """由配置字典构造，忽略未知字段。

        Args:
            data: 配置字典，如 config/backtest/costs。
        """
        if not data:
            return cls()
        valid = set(asdict(cls()).keys())
        filtered = {key: value for key, value in data.items() if key in valid}
        return cls(**filtered)

    @classmethod
    def zero(cls) -> CostModel:
        """零成本模型，用于对比"毛收益"与"净收益"。

        ⚠️ 只能用于学术对比，绝不可用它来评估策略是否可实盘。
        """
        return cls(
            commission_rate=0.0,
            min_commission=0.0,
            stamp_duty_rate=0.0,
            transfer_fee_rate=0.0,
            slippage_bps=0.0,
            apply_min_commission=False,
        )
