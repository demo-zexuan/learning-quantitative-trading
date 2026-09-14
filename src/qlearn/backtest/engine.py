"""
事件式回测引擎

I. 执行时序（核心，决定了回测是否可信）

1. 第 t 日**收盘后**：策略基于截止 t 日（含）的信息计算目标权重 w_t
2. 第 t+1 日**开盘**：按 w_t 以开盘价成交，扣除交易成本
3. 第 t+1 日**收盘**：按收盘价对持仓重新估值，得到净值 V_{t+1}

这个「收盘决策 -> 次日开盘成交」的时序，是避免**前视偏差**的关键。
注意第 3 步用的是**收盘价估值**而非开盘价，符合真实券商账户的每日结算方式。

II. 会计恒等式（引擎正确性的自检标准）

1. equity_t = cash_t + Σ(shares_i,t × close_i,t)
2. equity_t = equity_{t-1} × (1 + gross_ret_t) - cost_t
3. 若零成本模型，则 equity 必须等于 (1 + gross_returns).cumprod() × 初始资金

III. 已建模的真实约束

1. 交易成本：佣金（含单笔最低 5 元）、印花税（仅卖出）、过户费、滑点
2. 停牌不可交易：开盘价或成交量为空时，该标的当日无法调仓，只能持有
3. 资金约束：现金不足时按比例缩减买单，不引入隐性杠杆
4. T+1 制度：引擎按日频运行，天然满足 T+1（当日买入次日才可卖）
5. 可选流动性约束：单日成交不超过当日成交量的给定比例

IV. 尚未建模（实盘会显著影响结果）

1. 涨跌停无法成交（一字板买不进也卖不出）
2. 集合竞价与盘中滑点分布
3. 分红送股的现金流与税费细节
4. 融资融券的成本与强平机制

@module qlearn.backtest.engine
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from qlearn.backtest.costs import CostModel
from qlearn.backtest.result import BacktestResult
from qlearn.data.panel import PricePanel

__all__ = ["BacktestEngine"]


# 权重之和超过该阈值即视为使用了杠杆，触发警告
LEVERAGE_WARN_THRESHOLD = 1.0 + 1e-9

# 资金不足时压缩买单的最大迭代次数
_MAX_SCALE_ITERATIONS = 20


class BacktestEngine:
    """多标的事件式回测引擎。

    Args:
        initial_capital: 初始资金（元）。
        cost_model: 交易成本模型，None 表示使用 A 股默认参数。
        max_participation: 单个标的单日成交股数占当日成交量的上限比例，
            None 表示不限制。日频策略通常无需限制，小盘股策略必须限制。
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        cost_model: CostModel | None = None,
        max_participation: float | None = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital 必须为正数")
        if max_participation is not None and not (0 < max_participation <= 1):
            raise ValueError("max_participation 必须落在 (0, 1] 区间")

        self.initial_capital = float(initial_capital)
        self.cost_model = cost_model or CostModel()
        self.max_participation = max_participation

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(
        self,
        target_weights: pd.DataFrame,
        panel: PricePanel,
        benchmark: pd.Series | None = None,
        label: str = "strategy",
    ) -> BacktestResult:
        """执行回测。

        I. 权重语义

        1. 索引为「决策日」：该行权重由当日收盘信息决定，次日开盘执行
        2. 数值含义：目标市值占**当时总权益**的比例
           (1) 0 表示清仓
           (2) 正数表示做多（A 股个股做空受限，请勿使用负权重）
           (3) NaN 表示「维持当前持仓不变」，用于调仓频率低于日频的策略
        3. 若某行权重之和超过 1，引擎会发出杠杆警告

        Args:
            target_weights: 目标权重矩阵（index=日期, columns=标的代码）。
            panel: 行情面板。
            benchmark: 基准净值序列（可选），必须是价格序列而非收益率。
            label: 结果标签，用于展示与导出命名。

        Returns:
            BacktestResult。
        """
        # ------------------------------------------------------------------
        # I. 输入校验与对齐
        # ------------------------------------------------------------------
        weights = self._align_weights(target_weights, panel)

        # 杠杆检查：A 股普通账户无法使用杠杆，权重之和应 ≤ 1
        row_sums = weights.sum(axis=1, min_count=1)
        leveraged = row_sums[row_sums > LEVERAGE_WARN_THRESHOLD]
        if len(leveraged) > 0:
            warnings.warn(
                f"有 {len(leveraged)} 个交易日的目标权重之和超过 1"
                f"（最大 {leveraged.max():.2f}）。引擎不会强制归一化，"
                "但结果隐含杠杆，实盘需融资融券支持。",
                stacklevel=2,
            )
        if (weights < -1e-9).to_numpy().any():
            warnings.warn(
                "目标权重中存在负值（做空）。A 股个股做空受限，结果可能不可实现。",
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # II. 预取 NumPy 数组（循环内避免 pandas 开销）
        # ------------------------------------------------------------------
        dates = panel.dates
        symbols = panel.symbols
        n_dates = len(dates)
        n_symbols = len(symbols)

        open_arr = panel.open.to_numpy(dtype=float)
        # 估值用收盘价：前值填充后仍为空（从未有数据）的置 0，避免 NaN 污染净值
        valued_close_arr = panel.close.ffill().fillna(0.0).to_numpy(dtype=float)
        tradable_arr = panel.tradable.to_numpy()
        volume_arr = panel.volume.to_numpy(dtype=float)
        weight_arr = weights.to_numpy(dtype=float)

        # ------------------------------------------------------------------
        # III. 状态初始化
        # ------------------------------------------------------------------
        cash = self.initial_capital
        shares = np.zeros(n_symbols, dtype=float)

        equity_hist = np.full(n_dates, np.nan, dtype=float)
        net_ret_hist = np.zeros(n_dates, dtype=float)
        gross_ret_hist = np.zeros(n_dates, dtype=float)
        cost_hist = np.zeros(n_dates, dtype=float)
        turnover_hist = np.zeros(n_dates, dtype=float)
        position_hist = np.zeros((n_dates, n_symbols), dtype=float)
        weight_hist = np.zeros((n_dates, n_symbols), dtype=float)
        trade_records: list[dict[str, object]] = []

        equity_prev = self.initial_capital

        # ------------------------------------------------------------------
        # IV. 逐日推进
        # ------------------------------------------------------------------
        for i in range(n_dates):
            # ------------------------------------------------------------------
            # 1. 开盘再平衡（使用 i-1 日的决策权重）
            # ------------------------------------------------------------------
            if i > 0:
                w_target = weight_arr[i - 1]
                open_px = open_arr[i]
                tradable = tradable_arr[i]

                # (1) 开盘时点权益
                # 停牌标的用前一日收盘价估值，否则会用陈旧的开盘价高估/低估组合
                open_valuation_px = np.where(tradable, open_px, valued_close_arr[i - 1])
                equity_open = cash + float(np.sum(shares * open_valuation_px))

                # (2) 计算目标股数：NaN 权重表示维持现状
                desired = shares.copy()
                actionable = tradable & np.isfinite(w_target) & (open_px > 0)
                if equity_open > 0:
                    desired[actionable] = (
                        w_target[actionable] * equity_open / open_px[actionable]
                    )

                # (3) 流动性约束：限制单日成交股数
                if self.max_participation is not None:
                    max_shares = self.max_participation * volume_arr[i]
                    delta = desired - shares
                    delta = np.clip(delta, -np.abs(max_shares), np.abs(max_shares))
                    desired = shares + delta

                # (4) A 股按 100 股整手交易，向下取整到整手
                desired = np.floor(desired / 100.0) * 100.0
                desired[~tradable] = shares[~tradable]

                # (5) 资金约束：现金不足时等比例压缩买单
                desired, buy_cost, sell_cost = self._enforce_cash_constraint(
                    desired, shares, open_px, tradable, cash
                )

                # (6) 结算：先卖后买（A 股卖出所得资金当日可用于买入）
                delta_final = desired - shares
                buy_notional = float(
                    np.sum(np.where(delta_final > 0, delta_final * open_px, 0.0))
                )
                sell_notional = float(
                    np.sum(np.where(delta_final < 0, -delta_final * open_px, 0.0))
                )
                cash += sell_notional - sell_cost - buy_notional - buy_cost

                # (7) 记录成交明细
                for j in np.nonzero(np.abs(delta_final) > 1e-9)[0]:
                    notional = abs(float(delta_final[j])) * float(open_px[j])
                    side = "buy" if delta_final[j] > 0 else "sell"
                    trade_records.append(
                        {
                            "date": dates[i],
                            "symbol": symbols[j],
                            "side": side,
                            "shares": abs(float(delta_final[j])),
                            "price": float(open_px[j]),
                            "notional": notional,
                            "cost": self.cost_model.cost(notional, side),
                        }
                    )

                shares = desired
                cost_hist[i] = buy_cost + sell_cost

                # (8) 换手率：双边成交额 / 期初权益
                if equity_open > 0:
                    turnover_hist[i] = (buy_notional + sell_notional) / equity_open

            # ------------------------------------------------------------------
            # 2. 收盘估值
            # ------------------------------------------------------------------
            market_value = float(np.sum(shares * valued_close_arr[i]))
            equity = cash + market_value
            equity_hist[i] = equity

            # 3. 收益分解：equity_t = equity_{t-1} * (1 + gross_ret_t) - cost_t
            if i > 0 and equity_prev > 0:
                net_ret_hist[i] = equity / equity_prev - 1.0
                gross_ret_hist[i] = (equity + cost_hist[i]) / equity_prev - 1.0

            # 4. 记录持仓快照
            position_hist[i] = shares
            if equity > 0:
                weight_hist[i] = shares * valued_close_arr[i] / equity

            equity_prev = equity

        # ------------------------------------------------------------------
        # V. 组装结果
        # ------------------------------------------------------------------
        equity_series = pd.Series(equity_hist, index=dates, name="equity")
        returns_series = pd.Series(net_ret_hist, index=dates, name="returns")
        gross_series = pd.Series(gross_ret_hist, index=dates, name="gross_returns")

        self._verify_accounting(equity_series, gross_series)

        benchmark_curve = self._normalize_benchmark(benchmark, dates)

        return BacktestResult(
            equity=equity_series,
            returns=returns_series,
            gross_returns=gross_series,
            positions=pd.DataFrame(position_hist, index=dates, columns=symbols),
            weights=pd.DataFrame(weight_hist, index=dates, columns=symbols),
            turnover=pd.Series(turnover_hist, index=dates, name="turnover"),
            costs=pd.Series(cost_hist, index=dates, name="costs"),
            trades=pd.DataFrame(
                trade_records,
                columns=["date", "symbol", "side", "shares", "price", "notional", "cost"],
            ),
            initial_capital=self.initial_capital,
            cost_model=self.cost_model,
            benchmark=benchmark_curve,
            label=label,
            meta={
                "n_symbols": n_symbols,
                "n_dates": n_dates,
                "max_participation": self.max_participation,
            },
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _align_weights(target_weights: pd.DataFrame, panel: PricePanel) -> pd.DataFrame:
        """将权重矩阵对齐到面板的日期与标的，并做基础校验。"""
        if not isinstance(target_weights, pd.DataFrame):
            raise TypeError("target_weights 必须是 pandas.DataFrame")

        # 未知标的直接报错，避免静默丢弃导致回测结果与预期不符
        unknown = [code for code in target_weights.columns if code not in panel.symbols]
        if unknown:
            raise ValueError(f"权重矩阵包含面板中不存在的标的: {unknown}")

        weights = target_weights.reindex(index=panel.dates, columns=panel.symbols)
        # 非数值内容会在 to_numpy(dtype=float) 时抛错，这里提前给出更清晰的提示
        try:
            return weights.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"权重矩阵必须全部为数值: {exc}") from exc

    @staticmethod
    def _normalize_benchmark(
        benchmark: pd.Series | None, dates: pd.DatetimeIndex
    ) -> pd.Series | None:
        """将基准价格序列对齐到回测时间轴并归一化为净值曲线。"""
        if benchmark is None:
            return None
        if not isinstance(benchmark, pd.Series):
            raise TypeError("benchmark 必须是价格序列（pd.Series）")

        aligned = benchmark.reindex(dates).ffill().bfill()
        if aligned.isna().all():
            warnings.warn("基准序列与回测区间没有重叠，已忽略基准。", stacklevel=3)
            return None
        return (aligned / aligned.iloc[0]).rename("benchmark")

    def _enforce_cash_constraint(
        self,
        desired: np.ndarray,
        shares: np.ndarray,
        open_px: np.ndarray,
        tradable: np.ndarray,
        cash: float,
    ) -> tuple[np.ndarray, float, float]:
        """在市场开盘价上施加现金约束，返回（调整后目标股数, 买入成本, 卖出成本）。

        I. 处理逻辑

        1. 卖出不占用现金，反而补充现金，因此只需约束买入
        2. 若「买入金额 + 买入成本」超过可用现金，则等比例缩减买入股数
        3. 由于佣金存在最低 5 元的**非线性**门槛，一次缩放未必足够，
           因此迭代若干次直至满足约束或彻底放弃买入

        Returns:
            (adjusted_desired, buy_cost, sell_cost)
        """
        # I. 初始成本
        delta = desired - shares
        buy_mask = (delta > 0) & tradable
        sell_mask = (delta < 0) & tradable

        buy_notional = float(np.sum(delta[buy_mask] * open_px[buy_mask]))
        buy_cost = self._sum_cost(delta, open_px, buy_mask, "buy")
        sell_cost = self._sum_cost(-delta, open_px, sell_mask, "sell")

        # II. 现金充足，无需调整
        if buy_notional + buy_cost <= cash + 1e-9:
            return desired, buy_cost, sell_cost

        # III. 迭代缩减压缩买单
        adjusted = desired.copy()
        scale = 1.0
        for _ in range(_MAX_SCALE_ITERATIONS):
            # 可用现金需先扣除买入成本，按比例估算缩放系数
            headroom = cash - buy_cost
            if headroom <= 0 or buy_notional <= 1e-12:
                scale = 0.0
            else:
                scale = min(scale, max(0.0, headroom / buy_notional))

            adjusted[buy_mask] = shares[buy_mask] + delta[buy_mask] * scale
            adjusted = np.floor(adjusted / 100.0) * 100.0
            adjusted[~tradable] = shares[~tradable]

            new_delta = adjusted - shares
            new_buy_mask = (new_delta > 0) & tradable
            buy_notional = float(np.sum(new_delta[new_buy_mask] * open_px[new_buy_mask]))
            buy_cost = self._sum_cost(new_delta, open_px, new_buy_mask, "buy")

            if buy_notional + buy_cost <= cash + 1e-9:
                break
            scale *= 0.90  # 保守回退，应对最低佣金的台阶效应

        return adjusted, buy_cost, sell_cost

    def _sum_cost(
        self,
        delta: np.ndarray,
        open_px: np.ndarray,
        mask: np.ndarray,
        side: str,
    ) -> float:
        """对满足掩码的成交逐笔计算成本并求和。

        逐笔计算是必要的：最低佣金是**每笔订单**的门槛，无法用总量费率替代。
        """
        total = 0.0
        for j in np.nonzero(mask)[0]:
            notional = float(delta[j]) * float(open_px[j])
            total += self.cost_model.cost(notional, side)  # type: ignore[arg-type]
        return total

    @staticmethod
    def _verify_accounting(equity: pd.Series, gross_returns: pd.Series) -> None:
        """会计恒等式自检，用于及早暴露引擎实现错误。

        I. 校验项

        1. 净值序列无 NaN 且全部为正
        2. 净值末值 > 0（若为 0 说明资金被完全亏光，属于极端情况）

        注意：此处只做低成本的一致性检查。完整的恒等式验证由单元测试覆盖。
        """
        if equity.isna().any():
            raise AssertionError("回测产生了 NaN 净值，说明成本或市值计算存在缺陷")
        if (equity <= 0).any():
            raise AssertionError("回测产生了非正净值，组合已破产，请检查杠杆与权重设置")
        if len(equity) != len(gross_returns):
            raise AssertionError("净值与毛收益序列长度不一致")
