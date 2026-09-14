"""
策略层统一出口与注册表

I. 使用方式

```python
from qlearn.strategies import create_strategy, list_strategies

print(list_strategies())                       # 查看已注册策略
strategy = create_strategy("ma_cross", fast=5, slow=20)
weights = strategy.generate_weights(panel)
```

@module qlearn.strategies
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

from typing import Any

from qlearn.strategies.base import SizingMode, Strategy, allocate, rolling_zscore
from qlearn.strategies.buy_and_hold import BuyAndHoldStrategy
from qlearn.strategies.ma_cross import MACrossStrategy
from qlearn.strategies.mean_reversion import MeanReversionStrategy
from qlearn.strategies.momentum import MomentumStrategy

__all__ = [
    "STRATEGIES",
    "BuyAndHoldStrategy",
    "MACrossStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "SizingMode",
    "Strategy",
    "allocate",
    "create_strategy",
    "list_strategies",
    "rolling_zscore",
]

#: 策略注册表：名称 -> 策略类
STRATEGIES: dict[str, type[Strategy]] = {
    MACrossStrategy.name: MACrossStrategy,
    MomentumStrategy.name: MomentumStrategy,
    MeanReversionStrategy.name: MeanReversionStrategy,
    BuyAndHoldStrategy.name: BuyAndHoldStrategy,
}


def list_strategies() -> list[str]:
    """返回所有已注册策略的名称（按字母序）。"""
    return sorted(STRATEGIES)


def create_strategy(name: str, **params: Any) -> Strategy:
    """按名称创建策略实例。

    Args:
        name: 策略名称，见 `list_strategies()`。
        **params: 策略构造参数。

    Returns:
        策略实例。

    Raises:
        KeyError: 策略名称未注册。
        TypeError: 传入的参数不被该策略接受。
    """
    if name not in STRATEGIES:
        available = ", ".join(list_strategies())
        raise KeyError(f"未知策略 '{name}'，可用策略: {available}")

    strategy_cls = STRATEGIES[name]

    # 提前过滤非法参数，给出比 TypeError 更有用的提示
    valid = set(strategy_cls.param_names())
    unknown = set(params) - valid
    if unknown:
        raise TypeError(
            f"策略 '{name}' 不接受参数 {sorted(unknown)}，可接受的参数为 {sorted(valid)}"
        )

    return strategy_cls(**params)


def register_strategy(strategy_cls: type[Strategy]) -> type[Strategy]:
    """注册自定义策略（可用作装饰器）。

    ```python
    @register_strategy
    class MyStrategy(Strategy):
        name = "my_strategy"
        ...
    ```
    """
    if not getattr(strategy_cls, "name", None) or strategy_cls.name == "strategy":
        raise ValueError("自定义策略必须定义唯一的类属性 name")
    STRATEGIES[strategy_cls.name] = strategy_cls
    return strategy_cls
