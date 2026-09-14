"""
qlearn — A 股量化交易学习工具包

提供数据获取、向量化回测、绩效分析与策略模板，用于学习量化交易的完整链路。

I. 模块划分

1. data: A 股行情与指数成分数据获取、本地缓存
2. backtest: 向量化回测引擎与交易成本模型
3. metrics: 绩效与风险指标计算
4. strategies: 策略基类与经典策略实现
5. utils: 绘图与通用工具

II. 设计原则

1. 显式优于隐式：所有成本、滑点、延迟均需显式配置，不做魔法假设
2. 可复现：数据落盘缓存，随机过程固定种子
3. 教学优先：代码结构清晰，关键逻辑配详尽注释

@module qlearn
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
