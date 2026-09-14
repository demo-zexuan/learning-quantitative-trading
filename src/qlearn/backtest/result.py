"""
回测结果容器

统一封装一次回测产出的全部时间序列与明细，并提供绩效计算、导出与概览能力。

I. 核心字段语义

1. equity:        逐日净值（期末权益）
2. returns:       逐日**净**收益率（已扣除全部交易成本）
3. gross_returns: 逐日**毛**收益率（假设零成本），用于量化成本拖累
4. positions:     逐日收盘持仓股数
5. weights:       逐日收盘持仓权重（市值 / 净值）
6. turnover:      逐日双边换手率（成交额 / 期初权益）
7. costs:         逐日交易成本金额（元）
8. trades:        成交明细（日期、标的、方向、股数、价格、金额、成本）

@module qlearn.backtest.result
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qlearn.backtest.costs import CostModel
from qlearn.metrics import performance as perf

__all__ = ["BacktestResult"]


@dataclass
class BacktestResult:
    """一次回测的完整结果。"""

    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    positions: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    trades: pd.DataFrame
    initial_capital: float
    cost_model: CostModel
    benchmark: pd.Series | None = None
    label: str = "strategy"
    meta: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 绩效
    # ------------------------------------------------------------------

    def metrics(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = perf.TRADING_DAYS_PER_YEAR,
        formatted: bool = False,
    ) -> pd.Series:
        """计算绩效指标汇总。

        Args:
            risk_free_rate: 年化无风险利率。
            periods_per_year: 年交易日数。
            formatted: True 时返回格式化后的字符串，便于直接展示。

        Returns:
            指标名 -> 指标值 的 Series。
        """
        table = perf.summary(
            self.equity,
            returns=self.returns,
            benchmark=self.benchmark,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

        # 补充回测独有的交易特征指标
        table["年化双边换手率"] = float(
            self.turnover.mean() * periods_per_year
        )
        table["平均持仓标的数"] = float((self.positions.abs() > 0).sum(axis=1).mean())
        total_cost = float(self.costs.sum())
        table["累计交易成本(元)"] = total_cost
        table["成本占初始资金"] = total_cost / self.initial_capital

        if formatted:
            return perf.format_summary(table)
        return table

    def gross_metrics(
        self, risk_free_rate: float = 0.0, periods_per_year: int = perf.TRADING_DAYS_PER_YEAR
    ) -> pd.Series:
        """基于毛收益（零成本）计算指标，用于衡量成本拖累。

        对比 `metrics()` 与 `gross_metrics()` 可以回答：
        「这个策略赚的钱，有多少被交易成本吃掉了？」
        """
        gross_equity = (1.0 + self.gross_returns).cumprod() * self.initial_capital
        return perf.summary(
            gross_equity, returns=self.gross_returns, risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    def cost_impact(self) -> pd.Series:
        """成本拖累分析：对比毛/净口径下的关键指标差异。"""
        net = self.metrics()
        gross = self.gross_metrics()
        return pd.Series(
            {
                "毛累计收益": gross["累计收益"],
                "净累计收益": net["累计收益"],
                "成本侵蚀收益": gross["累计收益"] - net["累计收益"],
                "毛Sharpe": gross["年化Sharpe"],
                "净Sharpe": net["年化Sharpe"],
                "累计成本(元)": float(self.costs.sum()),
            },
            name=self.label,
        )

    def describe_cost_impact(self) -> str:
        """把成本影响表渲染为对齐的多行文本（中文按 2 列宽度对齐）。"""
        from qlearn.utils.text import align_table

        ratio_like = {"毛累计收益", "净累计收益", "成本侵蚀收益"}
        rows: list[tuple[str, str]] = []
        for name, value in self.cost_impact().items():
            if not np.isfinite(value):
                rows.append((name, "N/A"))
            elif name in ratio_like:
                rows.append((name, f"{value:+.2%}"))
            elif name.endswith("(元)"):
                rows.append((name, f"{value:,.2f} 元"))
            else:
                rows.append((name, f"{value:+.4f}"))
        return align_table(rows)

    def drawdown(self) -> pd.Series:
        """回撤序列。"""
        return perf.drawdown_series(self.equity)
    def monthly_table(self) -> pd.DataFrame:
        """月度收益透视表。"""
        return perf.monthly_returns_table(self.returns)

    # ------------------------------------------------------------------
    # 展示与导出
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """返回人类可读的概览文本。"""
        from qlearn.utils.text import align_table

        table = self.metrics(formatted=True)
        header = [
            f"回测结果概览 [{self.label}]",
            f"区间: {self.equity.index[0]:%Y-%m-%d} ~ {self.equity.index[-1]:%Y-%m-%d}"
            f"（{len(self.equity)} 个交易日）",
            f"初始资金: {self.initial_capital:,.0f} 元  |  "
            f"期末权益: {self.equity.iloc[-1]:,.0f} 元",
            "-" * 46,
        ]
        # 中文占 2 列，必须按显示宽度对齐而非 len()
        return "\n".join([*header, align_table(list(table.items()))])

    def save(self, directory: str | Path, prefix: str | None = None) -> list[Path]:
        """将结果导出到指定目录。

        I. 产出文件

        1. {prefix}_equity.csv  净值曲线（含基准）
        2. {prefix}_metrics.csv 绩效指标表
        3. {prefix}_trades.csv  成交明细
        4. {prefix}_monthly.csv 月度收益表

        Returns:
            实际写出的文件路径列表。
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = prefix or self.label

        written: list[Path] = []

        # 净值与基准合并导出
        curves = {"equity": self.equity}
        if self.benchmark is not None:
            curves["benchmark"] = self.benchmark
        path = out_dir / f"{stem}_equity.csv"
        pd.DataFrame(curves).to_csv(path, encoding="utf-8-sig")
        written.append(path)

        # 指标表
        path = out_dir / f"{stem}_metrics.csv"
        self.metrics().rename("value").to_frame().to_csv(path, encoding="utf-8-sig")
        written.append(path)

        # 成交明细
        path = out_dir / f"{stem}_trades.csv"
        self.trades.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)

        # 月度收益
        path = out_dir / f"{stem}_monthly.csv"
        self.monthly_table().to_csv(path, encoding="utf-8-sig")
        written.append(path)

        return written

    def __repr__(self) -> str:
        net = self.metrics()
        return (
            f"BacktestResult(label={self.label!r}, "
            f"累计收益={net['累计收益']:.2%}, "
            f"最大回撤={net['最大回撤']:.2%}, "
            f"年化Sharpe={net['年化Sharpe']:.2f}, "
            f"换手率(年化)={net['年化双边换手率']:.1%})"
        )
