"""
科研工具层统一出口

本层回答"策略到底能不能用"的问题，是连接"回测"与"实盘"之间最重要的一环。

@module qlearn.research
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

from qlearn.research.grid import (
    grid_search,
    iter_param_combinations,
    n_combinations,
    param_surface,
)
from qlearn.research.walk_forward import WalkForwardResult, split_in_out_sample, walk_forward

__all__ = [
    "WalkForwardResult",
    "grid_search",
    "iter_param_combinations",
    "n_combinations",
    "param_surface",
    "split_in_out_sample",
    "walk_forward",
]
