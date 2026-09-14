"""
绩效分析统一出口

@module qlearn.metrics
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

from qlearn.metrics.performance import (
    TRADING_DAYS_PER_YEAR,
    alpha_beta,
    annualized_volatility,
    cagr,
    calmar_ratio,
    conditional_value_at_risk,
    drawdown_series,
    format_summary,
    information_ratio,
    max_drawdown,
    monthly_returns_table,
    profit_factor,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    summary,
    total_return,
    value_at_risk,
    win_rate,
    yearly_returns,
)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "alpha_beta",
    "annualized_volatility",
    "cagr",
    "calmar_ratio",
    "conditional_value_at_risk",
    "drawdown_series",
    "format_summary",
    "information_ratio",
    "max_drawdown",
    "monthly_returns_table",
    "profit_factor",
    "rolling_sharpe",
    "sharpe_ratio",
    "sortino_ratio",
    "summary",
    "total_return",
    "value_at_risk",
    "win_rate",
    "yearly_returns",
]
