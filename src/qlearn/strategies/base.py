"""
策略基类与仓位分配工具

I. 策略的唯一职责

把行情面板映射为**目标权重矩阵**。策略不关心资金、成本、成交时点——
这些都交给回测引擎。这种职责切分让策略代码保持纯粹，也便于单独测试。

II. 权重矩阵契约（务必牢记）

1. 索引是**决策日**：第 t 行权重由截止 t 日收盘的信息算出，引擎在 t+1 日开盘执行
2. 因此策略内部**不需要**手动 `shift(1)`，引擎已经处理了延迟
3. 若策略自己再 shift 一次，就会把一个真实的收益延迟成两个 bar 的延迟，
   回测结果会系统性失真（通常是变差，从而让你误以为策略无效）

III. 缺失值约定

1. 数值权重 -> 调仓到该权重
2. NaN      -> 维持当前持仓不动（用于低频率调仓策略）
3. 0        -> 清仓该标的

@module qlearn.strategies.base
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import pandas as pd

from qlearn.data.panel import PricePanel

__all__ = ["Strategy", "SizingMode", "allocate", "emit_on_change", "rolling_zscore"]

# 仓位分配模式
SizingMode = Literal["fixed", "equal_active"]


class Strategy(ABC):
    """策略抽象基类。

    I. 子类实现要求

    1. 必须定义类属性 `name`（用于注册表与结果标签）
    2. 必须实现 `generate_weights`

    II. 参数管理

    1. 所有超参数通过关键字参数传入并存入 `self.params`
    2. `self.params` 用于参数扫描、结果命名与实验复现
    """

    #: 策略标识，子类必须覆盖
    name: str = "strategy"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = params

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_weights(self, panel: PricePanel) -> pd.DataFrame:
        """生成目标权重矩阵。

        Args:
            panel: 行情面板。

        Returns:
            index 与 `panel.dates` 一致、columns 与 `panel.symbols` 一致的权重矩阵。
        """

    # ------------------------------------------------------------------
    # 通用能力
    # ------------------------------------------------------------------

    @classmethod
    def param_names(cls) -> list[str]:
        """返回构造函数的参数名列表（不含 self 与可变参数）。"""
        signature = inspect.signature(cls.__init__)
        names: list[str] = []
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            names.append(name)
        return names

    @property
    def label(self) -> str:
        """人类可读的策略标识，形如 `ma_cross(fast=5, slow=20)`。"""
        if not self.params:
            return self.name
        rendered = ", ".join(f"{key}={value}" for key, value in self.params.items())
        return f"{self.name}({rendered})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.label})"

    __str__ = __repr__


def allocate(
    signal: pd.DataFrame,
    gross_exposure: float = 1.0,
    sizing: SizingMode = "fixed",
) -> pd.DataFrame:
    """把 0/1 持仓信号转换为目标权重矩阵。

    I. 两种分配模式的差异

    1. "fixed"（默认）：每个标的固定分配 `gross_exposure / 标的总数`
       (1) 优点：单标的风险暴露恒定，不会因信号数量波动而突然加仓
       (2) 缺点：信号稀疏时资金利用率低（可能长期只有 20% 仓位）
    2. "equal_active"：在当日**有信号**的标的间等权分配 `gross_exposure`
       (1) 优点：资金利用率高，满仓运作
       (2) 缺点：信号的进入/退出会引发其他持仓的被动调仓，产生额外换手成本

    教学建议：先用 "fixed" 观察策略的真实暴露，再用 "equal_active" 对比换手成本差异。

    II. 缺失值处理

    signal 中的 NaN 一律按 0 处理（不下单方向视为空仓）。

    Args:
        signal: 0/1 信号矩阵（NaN 视为 0）。
        gross_exposure: 总目标仓位，1.0 表示满仓。
        sizing: 分配模式。

    Returns:
        目标权重矩阵。
    """
    if sizing not in ("fixed", "equal_active"):
        raise ValueError(f"未知的 sizing 模式: {sizing!r}，可选 'fixed' / 'equal_active'")
    if gross_exposure <= 0:
        raise ValueError("gross_exposure 必须为正数")

    exposure = signal.fillna(0.0).astype(float)
    n_symbols = exposure.shape[1]
    if n_symbols == 0:
        raise ValueError("信号矩阵没有任何标的")

    # I. 固定分配：不随信号数量变化
    if sizing == "fixed":
        return exposure * (gross_exposure / n_symbols)

    # II. 动态等权：在有信号的标的间平分目标仓位
    active_count = exposure.sum(axis=1)
    weights = exposure.div(active_count.replace(0.0, np.nan), axis=0) * gross_exposure
    return weights.fillna(0.0)


def emit_on_change(weights: pd.DataFrame) -> pd.DataFrame:
    """只在目标权重发生变化时下达指令，其余交易日记 NaN（维持既有持仓）。

    I. 为什么需要它（最重要的实盘细节之一）

    1. 若策略每天都输出同一个目标权重（例如"持有，权重 0.2"），
       引擎会每天把组合拉回 0.2，即**每日再平衡**
       (1) 标的上涨后会被减仓、下跌后会被加仓，换手率可高达 1000%+/年
       (2) 这部分换手不带来任何信号价值，纯粹是成本损耗
       (3) 实测中，仅此一项就可能让一个"回测赚钱"的策略由盈转亏
    2. 真实趋势策略的做法是「信号不变就一直拿着」，
       持仓股数恒定，权重随价格自然漂移
    3. 本函数把"每日重复的同一指令"压缩为"仅在变化时下单"，
       精确实现了上述语义

    II. 副作用说明

    1. 权重漂移意味着单标的风险暴露会随行情变化，需要配合风控规则
       （如单一标的权重上限）一起使用
    2. 若策略的意图就是每日再平衡（例如风险平价），**不要**调用本函数

    Args:
        weights: 原始目标权重矩阵（允许含 NaN）。

    Returns:
        仅在权重发生变化（含首行）处保留数值、其余为 NaN 的矩阵。
    """
    # 用 0 填充仅用于比较，避免 NaN != NaN 恒为 True 导致每行都被判定为"变化"
    filled = weights.fillna(0.0)
    changed = filled.ne(filled.shift(1))
    return weights.where(changed)


def rolling_zscore(
    frame: pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.DataFrame:
    """滚动 Z-Score 标准化。

    I. 公式

    z_t = (x_t - mean(x_{t-w..t})) / std(x_{t-w..t})

    使用 ddof=1 的样本标准差；标准差为 0 时结果记 NaN 而非 inf，
    以免除零产生的无穷大污染后续的信号判断。

    Args:
        frame: 待标准化数据。
        window: 滚动窗口长度。
        min_periods: 有效样本下限，默认等于窗口长度（不产生"半成品"指标）。

    Returns:
        与输入同形状的 Z-Score 矩阵。
    """
    if window < 2:
        raise ValueError("window 至少为 2")

    min_periods = min_periods or window
    mean = frame.rolling(window, min_periods=min_periods).mean()
    std = frame.rolling(window, min_periods=min_periods).std(ddof=1)

    # 标准差为 0（价格完全不动）时置 NaN，避免产生 ±inf 信号
    std = std.where(std > 0)
    return (frame - mean) / std
