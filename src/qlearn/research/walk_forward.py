"""
前向滚动验证（Walk-Forward Analysis）

I. 它解决什么问题

样本内优化出的"最优参数"在样本外往往失效，这称为**数据窥探偏差**
（data snooping bias）。前向滚动验证通过"只用过去的数据选参数、
只在下一天/下一段交易"的方式，模拟真实的策略迭代流程。

II. 流程

```
|---- 训练窗口 1 ----|-- 测试 1 --|
              |---- 训练窗口 2 ----|-- 测试 2 --|
                            |---- 训练窗口 3 ----|-- 测试 3 --|
```

1. 在训练窗口上做参数扫描，选出最优参数
2. 把该参数**原封不动**地应用到紧随其后的测试窗口
3. 测试窗口向前滚动，重复上述过程
4. 把所有测试窗口的收益首尾相接，得到一条**近乎无偏**的样本外净值曲线

III. 关键实现细节

1. 每个测试窗口额外向前多取 `warmup_days` 天的数据用于指标预热，
   但只评估正式测试区间的收益
   (1) 若不预热，测试窗口开头会因均线未形成而空仓，人为拉低结果
   (2) 若不裁剪，预热区收益会被重复计入，人为抬高结果
2. 测试窗口之间不重叠，避免同一天的收益被计算多次
3. 本实现假设策略参数在测试窗口内保持不变（不重新优化）

@module qlearn.research.walk_forward
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qlearn.backtest.engine import BacktestEngine
from qlearn.data.panel import PricePanel
from qlearn.metrics import performance as perf
from qlearn.research.grid import grid_search
from qlearn.strategies import create_strategy

__all__ = ["WalkForwardResult", "walk_forward", "split_in_out_sample"]


def _to_native_param(value: Any) -> Any:
    """把从 DataFrame 行中取出的参数值还原为 Python 原生标量。

    I. 为什么需要它

    1. pandas 在对**混合 dtype** 的表取行时（`df.iloc[0]`），会把整数列
       统一升格为 float64，于是 `fast` 变成 `5.0`
    2. 而 `Series.rolling(min_periods=...)` 要求 `min_periods` 是整数，
       传入 `5.0` 会直接抛 `ValueError: min_periods must be an integer`
    3. 因此这里把"取值为整数的浮点数"还原为 int

    II. 已知副作用

    若某个策略参数本意是浮点数但恰好取整数值（如 `gross_exposure=1.0`），
    会被还原为 `1`。数值语义不变，仅影响 `repr` 展示，可接受。
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


@dataclass
class WalkForwardResult:
    """前向滚动验证的结果。

    Attributes:
        oos_returns: 拼接后的样本外日收益序列。
        folds: 每一折的明细（训练区间、测试区间、选中参数、样本内/外指标）。
        initial_capital: 用于还原净值曲线的初始资金。
    """

    oos_returns: pd.Series
    folds: pd.DataFrame
    initial_capital: float
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def oos_equity(self) -> pd.Series:
        """由样本外收益还原的净值曲线。"""
        return (1.0 + self.oos_returns).cumprod() * self.initial_capital

    def metrics(self, risk_free_rate: float = 0.0) -> pd.Series:
        """样本外绩效指标。

        ⚠️ 这才是评价策略的**正确口径**。样本内指标（`folds` 中的 IS 列）
        只能用于说明"优化过程是否过拟合"，不能作为业绩证明。
        """
        return perf.summary(
            self.oos_equity, returns=self.oos_returns, risk_free_rate=risk_free_rate
        )

    def overfit_gap(self) -> pd.Series:
        """过拟合程度诊断：样本内指标均值 与 样本外指标的差距。

        I. 判读标准

        1. IS 平均 Sharpe 与 OOS Sharpe 差距很大 -> 严重过拟合
        2. IS 平均 Sharpe 本身就很低 -> 策略没有优势，不必谈样本外
        3. 两者接近且都为正 -> 参数稳健，策略值得进一步研究
        """
        is_cols = self.folds[[c for c in self.folds.columns if c.startswith("IS_")]]
        oos = self.metrics()
        return pd.Series(
            {
                "样本内平均Sharpe": float(is_cols.get("IS_年化Sharpe", pd.Series([np.nan])).mean()),
                "样本内平均收益": float(is_cols.get("IS_累计收益", pd.Series([np.nan])).mean()),
                "样本外Sharpe": float(oos["年化Sharpe"]),
                "样本外累计收益": float(oos["累计收益"]),
                "样本外最大回撤": float(oos["最大回撤"]),
                "折数": float(len(self.folds)),
            }
        )

    def __repr__(self) -> str:
        metrics = self.metrics()
        return (
            f"WalkForwardResult(折数={len(self.folds)}, "
            f"样本外累计收益={metrics['累计收益']:.2%}, "
            f"样本外最大回撤={metrics['最大回撤']:.2%}, "
            f"样本外Sharpe={metrics['年化Sharpe']:.2f})"
        )


def split_in_out_sample(
    panel: PricePanel, split_date: str | pd.Timestamp
) -> tuple[PricePanel, PricePanel]:
    """按日期把面板切分为样本内 / 样本外两段。

    Args:
        panel: 完整行情面板。
        split_date: 切分点。该日期归入**样本外**。

    Returns:
        (样本内面板, 样本外面板)
    """
    split = pd.Timestamp(split_date)
    dates = panel.dates
    in_sample_dates = dates[dates < split]
    out_sample_dates = dates[dates >= split]

    if len(in_sample_dates) == 0 or len(out_sample_dates) == 0:
        raise ValueError(
            f"切分点 {split:%Y-%m-%d} 落在数据范围 "
            f"[{dates[0]:%Y-%m-%d}, {dates[-1]:%Y-%m-%d}] 之外，无法切分"
        )

    return (
        panel.slice_dates(end=in_sample_dates[-1]),
        panel.slice_dates(start=out_sample_dates[0]),
    )


def walk_forward(
    strategy_name: str,
    param_grid: dict[str, Sequence[Any]],
    panel: PricePanel,
    engine: BacktestEngine | None = None,
    train_days: int = 504,
    test_days: int = 126,
    warmup_days: int = 120,
    sort_by: str = "年化Sharpe",
    risk_free_rate: float = 0.0,
) -> WalkForwardResult:
    """执行前向滚动验证。

    Args:
        strategy_name: 策略名称。
        param_grid: 参数网格。
        panel: 完整行情面板。
        engine: 回测引擎。
        train_days: 每个训练窗口的长度（交易日）。
        test_days: 每个测试窗口的长度（交易日），也是滚动步长。
        warmup_days: 测试窗口额外向前延伸的天数，用于指标预热。
        sort_by: 训练窗口内选择参数所用的指标。
        risk_free_rate: 年化无风险利率。

    Returns:
        WalkForwardResult。

    Raises:
        ValueError: 数据长度不足以完成至少一折。
    """
    engine = engine or BacktestEngine()
    dates = panel.dates
    n_dates = len(dates)

    if n_dates < train_days + test_days:
        raise ValueError(
            f"数据不足：共 {n_dates} 个交易日，"
            f"至少需要 train_days({train_days}) + test_days({test_days}) = "
            f"{train_days + test_days} 个"
        )

    oos_parts: list[pd.Series] = []
    fold_records: list[dict[str, Any]] = []
    fold_id = 0
    start = 0

    # 逐折推进：每折测试窗口向前滚动 test_days 天
    while start + train_days + test_days <= n_dates:
        train_start = start
        train_end = start + train_days - 1
        test_start = train_end + 1
        test_end = test_start + test_days - 1

        train_panel = panel.slice_dates(dates[train_start], dates[train_end])

        # I. 样本内参数寻优
        grid_table = grid_search(
            strategy_name,
            param_grid,
            train_panel,
            engine=engine,
            sort_by=sort_by,
            risk_free_rate=risk_free_rate,
        )
        if grid_table.empty or sort_by not in grid_table.columns:
            raise RuntimeError(f"第 {fold_id} 折的参数扫描没有产生有效结果")

        best_row = grid_table.iloc[0]
        # 参数扫描返回的是 numpy 标量（且整数可能被升格为 float），需还原为原生类型
        best_params = {name: _to_native_param(best_row[name]) for name in param_grid}

        # II. 样本外评估
        # 额外向前取 warmup_days 用于预热，但只保留正式测试区间的收益
        extended_start = max(0, test_start - warmup_days)
        extended_panel = panel.slice_dates(dates[extended_start], dates[test_end])

        strategy = create_strategy(strategy_name, **best_params)
        result = engine.run(
            strategy.generate_weights(extended_panel),
            extended_panel,
            label=strategy.label,
        )
        oos_returns = result.returns.loc[dates[test_start] : dates[test_end]]
        oos_parts.append(oos_returns)

        # III. 记录本折明细
        oos_equity = (1.0 + oos_returns).cumprod()
        fold_records.append(
            {
                "fold": fold_id,
                "train_start": dates[train_start],
                "train_end": dates[train_end],
                "test_start": dates[test_start],
                "test_end": dates[test_end],
                **{f"param_{k}": v for k, v in best_params.items()},
                "IS_累计收益": float(best_row.get("累计收益", np.nan)),
                "IS_最大回撤": float(best_row.get("最大回撤", np.nan)),
                "IS_年化Sharpe": float(best_row.get("年化Sharpe", np.nan)),
                "OOS_累计收益": float(oos_equity.iloc[-1] - 1.0),
                "OOS_最大回撤": float(perf.max_drawdown(oos_equity)),
                "OOS_年化Sharpe": float(perf.sharpe_ratio(oos_returns, risk_free_rate)),
            }
        )

        fold_id += 1
        start += test_days

    oos_returns_all = pd.concat(oos_parts).sort_index()
    # 理论上各折不重叠，此处做一次保险去重
    oos_returns_all = oos_returns_all[~oos_returns_all.index.duplicated(keep="first")]
    oos_returns_all.name = "oos_returns"

    return WalkForwardResult(
        oos_returns=oos_returns_all,
        folds=pd.DataFrame(fold_records),
        initial_capital=engine.initial_capital,
        params={"train_days": train_days, "test_days": test_days, "sort_by": sort_by},
    )
