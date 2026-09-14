"""
稳健性检验：把「参数曲面 / 时段稳定性 / 仓位对标」三件事一次做完。

I. 为什么需要这组检验

1. 参数曲面：判断 10/30 是「高原上的一点」还是「孤立的尖峰」
2. 时段稳定性：把 6.7 年切成 4 段，看参数排序是否稳定——不稳定的排序=噪声拟合
3. 仓位对标：策略平均只有 45% 仓位，必须与「恒定的 45% 仓位」对比，
   否则会把「仓位不足」误判为「择时能力差」

@module notebooks/analysis_robustness
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlearn.backtest import BacktestEngine, CostModel  # noqa: E402
from qlearn.data import fetch_index_daily, load_panel  # noqa: E402
from qlearn.metrics import performance as perf  # noqa: E402
from qlearn.research import grid_search, param_surface  # noqa: E402
from qlearn.strategies import create_strategy  # noqa: E402

SYMBOLS = ["600519", "000858", "601318", "600036", "000001"]
OUT = Path(__file__).resolve().parent / "analysis_out"
OUT.mkdir(exist_ok=True)

FAST = [3, 5, 10, 15, 20, 30, 40, 60]
SLOW = [20, 30, 40, 60, 90, 120]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def rule(t: str) -> None:
    print(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


def main() -> None:
    panel = load_panel(SYMBOLS, start="2020-01-01", end=None, adjust="qfq", use_cache=True)
    close = panel.close  # 已确认无 NaN
    mkt = close.pct_change().mean(axis=1).fillna(0.0)

    # ------------------------------------------------------------------
    # I. 参数曲面（样本内，含成本）
    # ------------------------------------------------------------------
    rule("I. 参数曲面：年化 Sharpe / 累计收益（含成本，全样本 2020-2026）")
    eng = BacktestEngine(initial_capital=1_000_000.0)
    grid = grid_search("ma_cross", {"fast": FAST, "slow": SLOW}, panel, engine=eng)
    grid["年化双边换手率"] = grid["年化双边换手率"].round(2)
    surf_sharpe = param_surface(grid, "fast", "slow", "年化Sharpe")
    surf_ret = param_surface(grid, "fast", "slow", "累计收益")
    surf_turn = param_surface(grid, "fast", "slow", "年化双边换手率")

    print("\n-- 年化 Sharpe --")
    print((surf_sharpe * 100).round(1).to_string())
    print("\n-- 累计收益 % --")
    print((surf_ret * 100).round(1).to_string())
    print("\n-- 年化双边换手率（倍） --")
    print(surf_turn.round(1).to_string())
    print(f"\nSharpe > 0 的格子数: {(surf_sharpe > 0).sum().sum()} / {surf_sharpe.size}")
    print(f"累计收益 > 0 的格子数: {(surf_ret > 0).sum().sum()} / {surf_ret.size}")
    print(f"Sharpe 中位数: {np.nanmedian(surf_sharpe.to_numpy()):.3f}  最大: {np.nanmax(surf_sharpe.to_numpy()):.3f}")
    print("\n判读：曲面上每个格子都是同一条逻辑在不同参数下的表现。")
    print(f"      正负参半（{(surf_sharpe > 0).sum().sum()}/{surf_sharpe.size} 为正）说明")
    print("      「哪个参数赚钱」主要由行情时段决定，而不是由参数本身决定。")
    grid.to_csv(OUT / "10_参数网格全表.csv", index=False, encoding="utf-8-sig")
    surf_sharpe.to_csv(OUT / "11_参数曲面_sharpe.csv", encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # II. 时段稳定性：同一参数在不同子区间的表现
    # ------------------------------------------------------------------
    rule("II. 时段稳定性：10/30 在 4 个子区间（各约 1.7 年）")
    segs = [
        ("2020-01-02", "2021-08-31"),
        ("2021-09-01", "2023-04-30"),
        ("2023-05-01", "2024-12-31"),
        ("2025-01-01", "2026-09-14"),
    ]
    s = create_strategy("ma_cross", fast=10, slow=30)
    rows = []
    for a, b in segs:
        sub = panel.slice_dates(a, b)
        r = eng.run(s.generate_weights(sub), sub, label="seg")
        sub_mkt = sub.close.pct_change().mean(axis=1).fillna(0.0)
        pos = r.weights.sum(axis=1)
        rows.append(
            {
                "区间": f"{a}~{b}",
                "策略": perf.total_return(r.equity),
                "等权持有": (1 + sub_mkt).prod() - 1,
                "超额": perf.total_return(r.equity) - ((1 + sub_mkt).prod() - 1),
                "平均仓位": pos.mean(),
                "成本(元)": r.costs.sum(),
                "成交笔数": len(r.trades),
                "策略MDD": perf.max_drawdown(r.equity),
                "等权MDD": perf.max_drawdown((1 + sub_mkt).cumprod()),
            }
        )
    seg = pd.DataFrame(rows)
    print(seg.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    seg.to_csv(OUT / "12_时段稳定性.csv", index=False, encoding="utf-8-sig")

    # 参数排序在不同时段是否稳定
    rule("II-b. 参数排序稳定性：各子区间的最优 fast/slow")
    rank_rows = []
    for a, b in segs:
        sub = panel.slice_dates(a, b)
        g = grid_search("ma_cross", {"fast": [5, 10, 20, 40], "slow": [30, 60, 120]},
                        sub, engine=eng)
        best = g.iloc[0]
        rank_rows.append(
            {
                "区间": f"{a}~{b}",
                "最优参数": f"{int(best['fast'])}/{int(best['slow'])}",
                "最优Sharpe": best["年化Sharpe"],
                "最优累计": best["累计收益"],
                "10/30排名": int(g.index[(g["fast"] == 10) & (g["slow"] == 30)][0]) + 1,
                "10/30累计": float(g[(g["fast"] == 10) & (g["slow"] == 30)]["累计收益"].iloc[0]),
                "正收益格子": int((g["累计收益"] > 0).sum()),
                "总格子": len(g),
            }
        )
    rk = pd.DataFrame(rank_rows)
    print(rk.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\n判读：若最优参数在四段之间来回跳、且「正收益格子数」剧烈变化，")
    print("      说明参数有效性完全取决于行情时段，属于噪声拟合。")
    rk.to_csv(OUT / "13_参数排序稳定性.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # III. 仓位对标：把「仓位不足」与「择时亏损」分开
    # ------------------------------------------------------------------
    rule("III. 仓位对标：恒定 45% 仓位 vs 策略的动态 45% 仓位")
    r10 = eng.run(s.generate_weights(panel), panel, label="10/30")
    pos = r10.weights.sum(axis=1)
    avg_pos = float(pos.mean())

    # 恒定仓位：每天持有 avg_pos 的等权篮子，零成本（这是择时能力为 0 的基准）
    const_ret = mkt * avg_pos
    const_eq = (1 + const_ret).cumprod() * 1_000_000

    # 完美择时上界：在事前知道明天涨跌的情况下切换仓位（仅供参照）
    rows = [
        {"方案": "策略 10/30（实际）", "累计": perf.total_return(r10.equity),
         "年化": perf.cagr(r10.equity), "波动": perf.annualized_volatility(r10.returns),
         "MDD": perf.max_drawdown(r10.equity), "平均仓位": avg_pos},
        {"方案": f"恒定 {avg_pos:.1%} 仓位等权篮子（零成本）", "累计": perf.total_return(const_eq),
         "年化": perf.cagr(const_eq), "波动": perf.annualized_volatility(const_ret),
         "MDD": perf.max_drawdown(const_eq), "平均仓位": avg_pos},
        {"方案": "策略零成本（毛口径）", "累计": perf.total_return(
            (1 + r10.gross_returns).cumprod() * 1_000_000),
         "年化": perf.cagr((1 + r10.gross_returns).cumprod() * 1_000_000),
         "波动": perf.annualized_volatility(r10.gross_returns),
         "MDD": perf.max_drawdown((1 + r10.gross_returns).cumprod() * 1_000_000),
         "平均仓位": avg_pos},
    ]
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    timing_gap = perf.cagr(r10.equity) - perf.cagr(const_eq)
    print(f"\n策略年化 - 恒定仓位年化 = {timing_gap:.4f}  (这就是择时部分的净贡献)")
    print(f"其中成本贡献≈ {-r10.costs.sum() / len(r10.equity) * 252 / r10.equity.mean() / 1:.4f}"
          f"  (年化成本率 {(r10.costs.sum() / (len(r10.equity) / 252)) / r10.equity.mean():.2%})")
    cmp.to_csv(OUT / "14_仓位对标.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # IV. 每只标的的信号 vs 买入持有（用净值复利口径，避免口径错误）
    # ------------------------------------------------------------------
    rule("IV. 逐标的：信号驱动的持仓收益 vs 同期买入持有")
    sig = (close.rolling(10, min_periods=10).mean() > close.rolling(30, min_periods=30).mean())
    sig = sig.astype(float).mask(close.rolling(30, min_periods=30).mean().isna(), 0.0)
    rows = []
    for sym in SYMBOLS:
        ret = close[sym].pct_change().fillna(0.0)
        # 第 t 日生效的仓位由 t-1 日信号决定（与引擎时序一致）
        hold = sig[sym].shift(1).fillna(0.0)
        strat_ret = ret * hold
        net = (1 + strat_ret).prod() - 1
        bh = (1 + ret).prod() - 1
        rows.append(
            {
                "标的": sym,
                "信号驱动(零成本)": net,
                "买入持有": bh,
                "差": net - bh,
                "持有天数占比": hold.mean(),
                "区间涨幅": close[sym].iloc[-1] / close[sym].iloc[0] - 1,
            }
        )
    sr = pd.DataFrame(rows)
    print(sr.to_string(index=False, float_format=lambda v: f"{v:,.2%}"))
    print("\n判读：把每只标的单独拿出来看，信号驱动几乎都不如「一直拿着」。")
    sr.to_csv(OUT / "15_逐标的信号vs持有.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # V. 换手成本敏感性
    # ------------------------------------------------------------------
    rule("V. 成本敏感性：换手率不变时，不同滑点/佣金假设下的结果")
    rows = []
    for bps in (0, 2, 5, 10, 20):
        for comm in (0.0, 0.00025, 0.0005):
            cm = CostModel(commission_rate=comm, min_commission=5.0, slippage_bps=float(bps))
            rr = BacktestEngine(initial_capital=1_000_000.0, cost_model=cm).run(
                s.generate_weights(panel), panel, label="x")
            rows.append(
                {"滑点(bps)": bps, "佣金": f"{comm:.4%}",
                 "累计收益": perf.total_return(rr.equity),
                 "年化": perf.cagr(rr.equity),
                 "Sharpe": perf.sharpe_ratio(rr.returns, 0.02),
                 "成本(元)": rr.costs.sum()}
            )
    sens = pd.DataFrame(rows)
    print(sens.pivot_table(index="滑点(bps)", columns="佣金", values="累计收益")
          .to_string(float_format=lambda v: f"{v:,.2%}"))
    print("\n(表内为累计收益；纵轴滑点、横轴佣金)")
    sens.to_csv(OUT / "16_成本敏感性.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # VI. 起始日敏感性（很多趋势策略的结论由起点决定）
    # ------------------------------------------------------------------
    rule("VI. 起始日敏感性：从不同起点开始跑同一组参数")
    rows = []
    for start in ("2020-01-02", "2020-07-01", "2021-01-04", "2021-07-01",
                  "2022-01-04", "2022-07-01", "2023-01-03", "2024-01-02"):
        sub = panel.slice_dates(start, None)
        rr = eng.run(s.generate_weights(sub), sub, label="x")
        sub_mkt = sub.close.pct_change().mean(axis=1).fillna(0.0)
        rows.append(
            {
                "起点": start,
                "年数": (sub.dates[-1] - sub.dates[0]).days / 365.25,
                "策略年化": perf.cagr(rr.equity),
                "等权年化": perf.cagr((1 + sub_mkt).cumprod() * 1_000_000),
                "超额年化": perf.cagr(rr.equity) - perf.cagr((1 + sub_mkt).cumprod() * 1_000_000),
                "策略Sharpe": perf.sharpe_ratio(rr.returns, 0.02),
            }
        )
    st = pd.DataFrame(rows)
    print(st.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n判读：起点从 2020-01 挪到 2020-07，结论就从亏损变成接近打平——")
    print("      说明这个结果对起点高度敏感，统计上不稳健。")
    st.to_csv(OUT / "17_起点敏感性.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
