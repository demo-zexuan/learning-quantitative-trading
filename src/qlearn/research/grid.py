"""
参数网格扫描

I. 为什么必须做参数扫描

1. 单一参数组合的回测结果毫无意义——你无法知道它是"真实优势"还是"运气"
2. 扫描后要看的是**参数曲面形状**，而不是最优点本身
   (1) 平缓的高原：邻近参数都能赚钱，说明存在稳健的结构性特征
   (2) 孤立的尖峰：只有某个精确参数赚钱，几乎必然是过拟合
       i.  这类"最优点"在样本外几乎一定会失效
       ii. 常见成因是参数恰好对齐了某几次极端行情的日期
3. 因此本模块的产出重点是整张表与热力图，而非"最佳参数"

II. 选择评估指标的注意事项

1. 不要只看累计收益，它会被单次幸运交易主导
2. 推荐用 Sharpe 或 Calmar，它们同时惩罚了波动与回撤
3. 换手率必须一起看：高换手的高收益在扣费后往往不复存在

@module qlearn.research.grid
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from qlearn.backtest.engine import BacktestEngine
from qlearn.data.panel import PricePanel
from qlearn.strategies import create_strategy

__all__ = ["iter_param_combinations", "grid_search", "param_surface", "n_combinations"]

# 每次评估都固定输出的核心指标
_REPORTED_METRICS = (
    "累计收益",
    "年化收益(CAGR)",
    "年化波动率",
    "最大回撤",
    "年化Sharpe",
    "Calmar比率",
    "年化双边换手率",
    "累计交易成本(元)",
)


def n_combinations(param_grid: dict[str, Sequence[Any]]) -> int:
    """返回参数网格的组合总数，用于评估计算量。"""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def iter_param_combinations(
    param_grid: dict[str, Sequence[Any]],
) -> Iterator[dict[str, Any]]:
    """按笛卡尔积遍历参数网格。

    Args:
        param_grid: 形如 {"fast": [5, 10], "slow": [20, 60]} 的字典。

    Yields:
        单个参数组合，如 {"fast": 5, "slow": 20}。
    """
    keys = list(param_grid)
    for values in itertools.product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, values, strict=True))


def grid_search(
    strategy_name: str,
    param_grid: dict[str, Sequence[Any]],
    panel: PricePanel,
    engine: BacktestEngine | None = None,
    sort_by: str = "年化Sharpe",
    risk_free_rate: float = 0.0,
    ignore_errors: bool = True,
) -> pd.DataFrame:
    """遍历参数网格并回测，返回按指标降序排列的结果表。

    I. 输出列

    1. 参数网格中的每一列（作为索引维度）
    2. `_REPORTED_METRICS` 中的核心指标
    3. 失败组合附加 `error` 列（ignore_errors=True 时）

    II. 使用建议

    不要直接取第一行作为"最优参数"，请先看 `param_surface()` 输出的曲面形状。

    Args:
        strategy_name: 策略名称。
        param_grid: 参数网格。
        panel: 行情面板（通常只用**样本内**区间）。
        engine: 回测引擎，None 表示使用默认配置。
        sort_by: 排序指标名，需存在于指标表中。
        risk_free_rate: 年化无风险利率。
        ignore_errors: 是否忽略单个参数组合的异常。

    Returns:
        结果表，按 `sort_by` 降序排列。
    """
    engine = engine or BacktestEngine()
    rows: list[dict[str, Any]] = []

    for params in iter_param_combinations(param_grid):
        try:
            strategy = create_strategy(strategy_name, **params)
            result = engine.run(strategy.generate_weights(panel), panel, label=strategy.label)
            metrics = result.metrics(risk_free_rate=risk_free_rate)
            row: dict[str, Any] = dict(params)
            row.update({name: float(metrics[name]) for name in _REPORTED_METRICS})
        except Exception as exc:  # noqa: BLE001 - 扫描需要容忍个别非法组合
            if not ignore_errors:
                raise
            row = dict(params)
            row.update({name: np.nan for name in _REPORTED_METRICS})
            row["error"] = str(exc)
        rows.append(row)

    table = pd.DataFrame(rows)
    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False)
    return table.reset_index(drop=True)


def param_surface(
    grid_table: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "年化Sharpe",
) -> pd.DataFrame:
    """把扫描结果透视为二维参数曲面，用于绘制热力图。

    I. 阅读方法

    1. 高原（连续多个格子的值都为正）：参数稳健，可以放心使用
    2. 尖峰（只有一个格子突出，周围为负）：过拟合，样本外必失效
    3. 山谷/悬崖（相邻参数结果剧烈反转）：参数选择风险极高

    Args:
        grid_table: `grid_search()` 的返回值。
        x_param: 作为行索引的参数名。
        y_param: 作为列索引的参数名。
        metric: 需要透视的指标名。

    Returns:
        index 为 x_param 取值、columns 为 y_param 取值的透视表。
    """
    for name in (x_param, y_param, metric):
        if name not in grid_table.columns:
            raise KeyError(f"结果表中不存在列 '{name}'")

    return grid_table.pivot_table(index=x_param, columns=y_param, values=metric)
