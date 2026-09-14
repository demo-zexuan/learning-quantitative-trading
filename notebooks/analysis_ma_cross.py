"""
对 `qlt --strategy ma_cross --set fast=10 --set slow=30` 那次回测做独立复算与归因。

I. 分析目标

1. 复算基线结果，确认 CLI 输出的数字可被独立复现
2. 拆解「-11.31% 累计收益」的来源：暴露度、择时 alpha、成本
3. 检查换手率与成本的勾稽关系是否自洽
4. 用参数曲面与滚动窗口判断这是策略缺陷还是参数选择问题

II. 输出

所有明细写入 notebooks/analysis_out/，便于后续引用。

@module notebooks.analysis_ma_cross
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
from qlearn.data import PricePanel, fetch_index_daily, load_panel  # noqa: E402
from qlearn.metrics import performance as perf  # noqa: E402
from qlearn.strategies import create_strategy  # noqa: E402

SYMBOLS = ["600519", "000858", "601318", "600036", "000001"]
START = "2020-01-01"
OUT = Path(__file__).resolve().parent / "analysis_out"
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.unicode.east_asian_width", True)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build():
    panel = load_panel(SYMBOLS, start=START, end=None, adjust="qfq", use_cache=True)
    bench = fetch_index_daily("000300", start="2020-01-01", end=None, use_cache=True)["close"]
    return panel, bench


def run(panel, bench, strategy, engine, label):
    return engine.run(
        strategy.generate_weights(panel), panel, benchmark=bench, label=label
    )


def main() -> None:
    panel, bench = build()
    bench_aligned = bench.reindex(panel.dates).ffill()

    # ------------------------------------------------------------------
    # I. 基线复算
    # ------------------------------------------------------------------
    rule("I. 基线复算（fast=10, slow=30, 含成本）")
    strat = create_strategy("ma_cross", fast=10, slow=30)
    engine = BacktestEngine(initial_capital=1_000_000.0)
    base = run(panel, bench, strat, engine, strat.label)

    w = strat.generate_weights(panel)
    sig = (panel.close.rolling(10, min_periods=10).mean()
           > panel.close.rolling(30, min_periods=30).mean()).astype(float)
    sig = sig.mask(panel.close.rolling(30, min_periods=30).mean().isna(), 0.0)

    exposure = base.weights.sum(axis=1)
    print(f"累计收益      : {perf.total_return(base.equity):.4%}")
    print(f"期末权益      : {base.equity.iloc[-1]:,.0f}")
    print(f"累计成本      : {base.costs.sum():,.2f}")
    print(f"成交笔数      : {len(base.trades)}")

    print("\n--- 暴露度（真正决定波动率与收益的量级） ---")
    print(f"日均总仓位        : {exposure.mean():.4f}")
    print(f"中位总仓位        : {exposure.median():.4f}")
    print(f"空仓日占比        : {(exposure < 1e-9).mean():.2%}")
    print(f"满仓日占比(>=0.99): {(exposure >= 0.99).mean():.2%}")
    print(f"单标的有信号占比  : {sig.mean().round(4).to_dict()}")
    print(f"平均同时持有标的数: {(sig.sum(axis=1)).mean():.4f}")
    print(f"同时持有 0 只的天数: {(sig.sum(axis=1) == 0).sum()}")
    print(f"同时持有 5 只的天数: {(sig.sum(axis=1) == 5).sum()}")

    # ------------------------------------------------------------------
    # II. 对照实验：把钱从策略里拿掉，看看到底是「择时」还是「暴露」在起作用
    # ------------------------------------------------------------------
    rule("II. 对照实验")
    rows = []

    def add(name, series_equity, rets, note=""):
        rows.append(
            {
                "方案": name,
                "累计收益": perf.total_return(series_equity),
                "年化CAGR": perf.cagr(series_equity),
                "年化波动": perf.annualized_volatility(rets),
                "最大回撤": perf.max_drawdown(series_equity),
                "Sharpe": perf.sharpe_ratio(rets, 0.02),
                "说明": note,
            }
        )

    # 1. 基线
    add("策略(净,10/30)", base.equity, base.returns)
    # 2. 零成本
    zero_engine = BacktestEngine(initial_capital=1_000_000.0, cost_model=CostModel.zero())
    gap = run(panel, bench, create_strategy("ma_cross", fast=10, slow=30), zero_engine, "zero")
    add("策略(零成本,10/30)", gap.equity, gap.returns)
    # 3. 等权买入持有（5 只各 20%）
    close = panel.close.ffill()
    bh_ret = close.pct_change().mean(axis=1)
    bh_eq = (1 + bh_ret.fillna(0)).cumprod() * 1_000_000
    add("等权买入持有(5只)", bh_eq, bh_ret.dropna())
    # 4. 基准
    bret = bench_aligned.pct_change().dropna()
    add("沪深300", bench_aligned / bench_aligned.iloc[0] * 1_000_000, bret)
    # 5. 单只买入持有
    for s in SYMBOLS:
        px = close[s]
        r = px.pct_change().dropna()
        add(f"买入持有 {s}", px / px.iloc[0] * 1_000_000, r)

    table = pd.DataFrame(rows)
    for c in ("累计收益", "年化CAGR", "年化波动", "最大回撤"):
        table[c] = table[c].map(lambda v: f"{v:.2%}")
    table["Sharpe"] = table["Sharpe"].map(lambda v: f"{v:.3f}")
    print(table.to_string(index=False))
    table.to_csv(OUT / "01_对照实验.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # III. 成本勾稽：报告出来的换手率能不能解释报告出来的成本
    # ------------------------------------------------------------------
    rule("III. 换手率与成本的勾稽检验")
    trades = base.trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])
    total_notional = trades["notional"].sum()
    years = (base.equity.index[-1] - base.equity.index[0]).days / 365.25
    avg_eq = base.equity.mean()
    implied_turnover = total_notional / avg_eq / years
    print(f"成交总名义额      : {total_notional:,.0f}")
    print(f"平均权益          : {avg_eq:,.0f}")
    print(f"由成交明细反推年双边换手: {implied_turnover:.4f}")
    print(f"result.turnover 年化     : {base.turnover.mean() * 252:.4f}   <-- CLI 报告值")
    print(f"总成本            : {trades['cost'].sum():,.2f}")
    print(f"成本 / 成交额     : {trades['cost'].sum() / total_notional:.4%}")
    print(f"理论双边加权费率  : "
          f"{(CostModel().buy_rate + CostModel().sell_rate) / 2:.4%}（买卖各半时）")
    min_comm_share = (trades["cost"] - trades["notional"] * 0.00025 * 2).clip(lower=0)
    print(f"\n按方向拆分：")
    print(trades.groupby("side").agg(
        笔数=("notional", "size"),
        成交额=("notional", "sum"),
        成本=("cost", "sum"),
        单笔均值=("notional", "mean"),
    ).to_string())
    print(f"\n单笔最低佣金 5 元是否成为约束：")
    buy = trades[trades["side"] == "buy"]
    print(f"  买单中 佣金=5元(即<2万元)的占比: "
          f"{(buy['notional'] < 5 / 0.00025).mean():.2%}")
    print(f"  买单成交额中位数: {buy['notional'].median():,.0f}")

    # ------------------------------------------------------------------
    # IV. 逐标的交易归因
    # ------------------------------------------------------------------
    rule("IV. 逐标的交易归因")
    per_symbol = []
    for s in SYMBOLS:
        t = trades[trades["symbol"] == s].reset_index(drop=True)
        px = close[s]
        n_buy = (t["side"] == "buy").sum()
        # 配对：一次买入 -> 下一次卖出，计算单次往返收益
        rounds = []
        for i in range(len(t) - 1):
            if t.loc[i, "side"] == "buy" and t.loc[i + 1, "side"] == "sell":
                entry_px = t.loc[i, "price"]
                exit_px = t.loc[i + 1, "price"]
                rounds.append(
                    {
                        "entry": t.loc[i, "date"],
                        "exit": t.loc[i + 1, "date"],
                        "days": (t.loc[i + 1, "date"] - t.loc[i, "date"]).days,
                        "ret": exit_px / entry_px - 1,
                        "cost": (t.loc[i, "cost"] + t.loc[i + 1, "cost"])
                        / t.loc[i, "notional"],
                    }
                )
        rdf = pd.DataFrame(rounds)
        hold_ret = px.iloc[-1] / px.iloc[0] - 1
        per_symbol.append(
            {
                "标的": s,
                "买入次数": n_buy,
                "持有天数占比": f"{(sig[s] > 0).mean():.1%}",
                "信号覆盖期涨幅": f"{(px[sig[s] > 0].iloc[-1] / px[sig[s] > 0].iloc[0] - 1):.1%}"
                if (sig[s] > 0).any()
                else "N/A",
                "全额买入持有": f"{hold_ret:.1%}",
                "往返次数": len(rdf),
                "往返胜率": f"{(rdf['ret'] > rdf['cost']).mean():.1%}" if len(rdf) else "N/A",
                "平均持有天数": f"{rdf['days'].mean():.0f}" if len(rdf) else "N/A",
                "往返均收益(毛)": f"{rdf['ret'].mean():.2%}" if len(rdf) else "N/A",
                "往返均成本": f"{rdf['cost'].mean():.2%}" if len(rdf) else "N/A",
                "成本吃掉": f"{(rdf['cost'].mean() - rdf['ret'].mean()):.2%}"
                if len(rdf)
                else "N/A",
            }
        )
        if len(rdf):
            rdf.to_csv(OUT / f"02_往返明细_{s}.csv", index=False, encoding="utf-8-sig")
    sym_table = pd.DataFrame(per_symbol)
    print(sym_table.to_string(index=False))
    sym_table.to_csv(OUT / "03_逐标的归因.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # V. 这笔钱到底亏在哪：把日收益按「有仓位/空仓」切开
    # ------------------------------------------------------------------
    rule("V. 收益来源拆解：持有期的市场收益 vs 空仓期错过的收益")
    pos = exposure.reindex(base.returns.index).fillna(0)
    ret = base.returns
    holding = pos > 1e-9
    print(f"持有天数 {holding.sum()} / {len(ret)}")
    print(f"持有日 平均日收益 : {ret[holding].mean():.5%}（日均仓位 {pos[holding].mean():.2%}）")
    print(f"空仓日 平均日收益 : {ret[~holding].mean():.5%}（应为 ~0）")
    mkt = close.pct_change().mean(axis=1).reindex(ret.index).fillna(0)
    print(f"\n市场(等权5只) 持有日平均: {mkt[holding].mean():.5%}")
    print(f"市场(等权5只) 空仓日平均: {mkt[~holding].mean():.5%}   <-- 若为正，说明空仓躲掉的是上涨")
    # 择时贡献 = 持有日市场收益 - 全期市场收益
    timing = (mkt[holding].mean() * holding.mean()) - mkt.mean()
    print(f"\n择时贡献(粗略)  : {timing:.5%} / 日  -> 年化 {timing * 252:.2%}")
    print(f"暴露度贡献      : 仓位缩减导致收益只有市场的 {pos.mean():.1%}")

    # 一年滚动窗口
    roll = pd.DataFrame(
        {
            "策略": (1 + ret).rolling(252).apply(np.prod, raw=True) - 1,
            "等权持有": (1 + mkt).rolling(252).apply(np.prod, raw=True) - 1,
            "沪深300": (1 + bret).rolling(252).apply(np.prod, raw=True) - 1,
        }
    ).dropna()
    roll = roll.iloc[::21]
    print("\n--- 滚动 1 年收益（每 21 个交易日采样，看稳定性） ---")
    print((roll * 100).round(1).to_string())
    roll.to_csv(OUT / "04_滚动一年收益.csv", encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # VI. 分年度
    # ------------------------------------------------------------------
    rule("VI. 分年度收益")
    yr = pd.DataFrame(
        {
            "策略(净)": perf.yearly_returns(ret),
            "等权持有": perf.yearly_returns(mkt),
            "沪深300": perf.yearly_returns(bret),
            "平均仓位": pos.groupby(pos.index.year).mean(),
            "成本(元)": base.costs.groupby(base.costs.index.year).sum(),
        }
    )
    yr["相对等权"] = yr["策略(净)"] - yr["等权持有"]
    print(yr.to_string(float_format=lambda v: f"{v:,.3f}"))
    yr.to_csv(OUT / "05_分年度.csv", encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # VII. 有无信号的前瞻收益：均线信号到底有没有预测力
    # ------------------------------------------------------------------
    rule("VII. 信号预测力检验（有无前视偏差的干净检验）")
    fwd = {}
    for horizon in (1, 5, 10, 20, 60):
        fwd_r = close.pct_change(horizon).shift(-horizon)
        holding_mask = sig.reindex(fwd_r.index).fillna(0) > 0
        a = fwd_r[holding_mask].stack().mean()
        b = fwd_r[~holding_mask].stack().mean()
        fwd[horizon] = {"持有信号": a, "空仓信号": b, "差(信号价值)": a - b}
    fdf = pd.DataFrame(fwd).T
    print(fdf.to_string(float_format=lambda v: f"{v:.4%}"))
    print("\n判读：若「差」长期<=0，说明这个均线信号对次日之后的收益没有预测力，")
    print("      那么它带来的只有换手成本，没有择时价值。")
    fdf.to_csv(OUT / "06_信号预测力.csv", encoding="utf-8-sig")

    # 交叉事件研究：金叉/死叉后 N 日
    cross_up = (sig.diff() > 0)
    cross_dn = (sig.diff() < 0)
    print("\n--- 事件研究：金叉/死叉后的累计收益（%） ---")
    ev = []
    for h in (5, 10, 20, 40, 60):
        fr = close.pct_change(h).shift(-h)
        ev.append(
            {
                "horizon": h,
                "金叉后": fr[cross_up.reindex(fr.index).fillna(False)].stack().mean(),
                "死叉后": fr[cross_dn.reindex(fr.index).fillna(False)].stack().mean(),
                "金叉次数": int(cross_up.sum().sum()),
                "死叉次数": int(cross_dn.sum().sum()),
            }
        )
    evdf = pd.DataFrame(ev).set_index("horizon")
    print(evdf.to_string(float_format=lambda v: f"{v:.4%}"))
    evdf.to_csv(OUT / "07_交叉事件研究.csv", encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # VIII. 假突破：信号维持不足 N 天就反转的比例
    # ------------------------------------------------------------------
    rule("VIII. 信号持续性（假突破占比）")
    for s in SYMBOLS:
        v = sig[s].to_numpy()
        runs, cur = [], 0
        for x in v:
            if x > 0:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        runs = np.array(runs)
        if len(runs):
            print(f"{s}: 持有段数 {len(runs):3d}  中位持有 {np.median(runs):5.0f} 天  "
                  f"<=5天占比 {(runs <= 5).mean():5.1%}  <=20天占比 {(runs <= 20).mean():5.1%}  "
                  f"最长 {runs.max()} 天")

    base.equity.to_csv(OUT / "08_净值曲线.csv", encoding="utf-8-sig")
    print(f"\n明细已写入 {OUT}")


if __name__ == "__main__":
    main()
