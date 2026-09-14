"""
可视化工具

I. 中文字体处理

matplotlib 默认字体不含中文字形，中文标题会显示为方框。本模块在导入时
自动探测系统中可用的中文字体（优先 macOS 的 PingFang SC），并全局设置。

II. 配色约定

1. 策略净值：深蓝
2. 基准净值：灰色虚线
3. 亏损/回撤：红色
4. 盈利：绿色

@module qlearn.utils.plotting
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "setup_chinese_font",
    "plot_equity",
    "plot_drawdown",
    "plot_monthly_heatmap",
    "plot_returns_distribution",
    "plot_rolling_sharpe",
    "plot_param_surface",
    "plot_price_with_trades",
]

# 数值型文本在单元格内的对齐色板
_COLOR_STRATEGY = "#1f4e79"
_COLOR_BENCHMARK = "#8c8c8c"
_COLOR_LOSS = "#c0392b"
_COLOR_GAIN = "#1e8449"


def setup_chinese_font() -> str | None:
    """探测并设置可用的中文字体。

    Returns:
        实际启用的字体名称；若系统无可用中文字体则返回 None
        （此时中文标签会显示为方框，但不影响绘图逻辑）。
    """
    from matplotlib import font_manager, rcParams

    candidates = [
        "PingFang SC",       # macOS 默认中文
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "Hiragino Sans GB",
        "Microsoft YaHei",   # Windows
        "SimHei",
        "Noto Sans CJK SC",  # Linux
        "WenQuanYi Zen Hei",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    rcParams["axes.unicode_minus"] = False  # 负号显示为方块是常见坑，必须关闭

    for name in candidates:
        if name in installed:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            return name
    return None


# 导入即生效：visualization 是"用了就要好看"的模块，无需调用者手动设置
setup_chinese_font()


def _finalize(axes, title: str | None, grid_axis: str = "y"):
    """统一图表收尾样式：标题、网格、紧凑布局。"""
    if title:
        axes.set_title(title, fontsize=12, pad=10)
    axes.grid(True, alpha=0.3, linestyle="--", linewidth=0.7, axis=grid_axis)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)


def plot_equity(result, benchmark_label: str = "基准", figsize: tuple = (12, 6), ax=None):
    """绘制净值曲线与回撤。

    I. 图形结构

    1. 上panel：策略与基准的净值曲线（均归一化到 1.0 起）
    2. 下panel：策略回撤区间填充

    Args:
        result: BacktestResult。
        benchmark_label: 基准图例名称。
        figsize: 图像尺寸。
        ax: 未使用，保留以兼容习惯用法（本函数总是返回新建的 Figure）。

    Returns:
        (Figure, ndarray of Axes)
    """
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

    # I. 净值曲线
    equity = result.equity / result.equity.iloc[0]
    ax_top.plot(equity.index, equity, color=_COLOR_STRATEGY, linewidth=1.6, label=result.label)

    if result.benchmark is not None:
        benchmark = result.benchmark / result.benchmark.iloc[0]
        ax_top.plot(
            benchmark.index,
            benchmark,
            color=_COLOR_BENCHMARK,
            linewidth=1.2,
            linestyle="--",
            label=benchmark_label,
        )

    ax_top.axhline(1.0, color="#555555", linewidth=0.8, linestyle=":")
    ax_top.set_ylabel("净值（初始 = 1）")
    ax_top.legend(loc="upper left", frameon=False)
    _finalize(ax_top, f"净值曲线 — {result.label}")

    # II. 回撤
    drawdown = result.drawdown()
    ax_bottom.fill_between(
        drawdown.index, drawdown.to_numpy(), 0.0, color=_COLOR_LOSS, alpha=0.35, linewidth=0
    )
    ax_bottom.plot(drawdown.index, drawdown, color=_COLOR_LOSS, linewidth=0.9)
    ax_bottom.set_ylabel("回撤")
    ax_bottom.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    _finalize(ax_bottom, None)

    fig.tight_layout()
    return fig, (ax_top, ax_bottom)


def plot_drawdown(result, figsize: tuple = (12, 3.5)):
    """单独绘制回撤曲线。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    drawdown = result.drawdown()
    ax.fill_between(
        drawdown.index, drawdown.to_numpy(), 0.0, color=_COLOR_LOSS, alpha=0.35, linewidth=0
    )
    ax.plot(drawdown.index, drawdown, color=_COLOR_LOSS, linewidth=1.0)
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    ax.set_ylabel("回撤")
    _finalize(ax, f"回撤曲线 — {result.label}")
    fig.tight_layout()
    return fig, ax


def plot_monthly_heatmap(result, figsize: tuple = (12, 5)):
    """绘制年度 x 月份收益率热力图。

    可以直观识别策略是否有明显的季节性，或者亏损是否集中在某几段行情。
    """
    import matplotlib.pyplot as plt

    table = result.monthly_table()
    if table.empty:
        raise ValueError("收益序列为空，无法绘制月度热力图")

    months = table.drop(columns=["YTD"], errors="ignore")
    values = months.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 0.05
    image = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(months.shape[1]))
    ax.set_xticklabels(months.columns)
    ax.set_yticks(range(months.shape[0]))
    ax.set_yticklabels([str(year) for year in months.index])

    # 单元格数值标注
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if not np.isfinite(value):
                continue
            ax.text(
                col, row, f"{value:.1%}", ha="center", va="center",
                fontsize=8, color="#222222",
            )

    fig.colorbar(image, ax=ax, shrink=0.8, format=lambda v, _pos: f"{v:.0%}")
    ax.set_title(f"月度收益热力图 — {result.label}", fontsize=12, pad=10)
    fig.tight_layout()
    return fig, ax


def plot_returns_distribution(result, figsize: tuple = (12, 4)):
    """绘制日收益分布直方图，并标注均值与 95% VaR。"""
    import matplotlib.pyplot as plt

    returns = result.returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns[returns != 0.0]
    if returns.empty:
        raise ValueError("收益序列为空，无法绘制分布图")

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(returns.to_numpy(), bins=60, color=_COLOR_STRATEGY, alpha=0.75, edgecolor="white")

    mean = returns.mean()
    var95 = returns.quantile(0.05)
    ax.axvline(mean, color=_COLOR_GAIN, linewidth=1.4, linestyle="-", label=f"均值 {mean:.3%}")
    ax.axvline(var95, color=_COLOR_LOSS, linewidth=1.4, linestyle="--",
               label=f"95% VaR {var95:.3%}")

    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:.1%}")
    ax.set_xlabel("日收益率")
    ax.set_ylabel("频数")
    ax.legend(frameon=False)
    _finalize(ax, f"日收益分布 — {result.label}")

    # 标注偏度与峰度：负偏 + 高峰度意味着"平时小赚、偶发巨亏"
    ax.text(
        0.99, 0.95,
        f"偏度 {returns.skew():.2f}  |  峰度 {returns.kurt():.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555555",
    )
    fig.tight_layout()
    return fig, ax


def plot_rolling_sharpe(result, window: int = 126, figsize: tuple = (12, 3.5)):
    """绘制滚动夏普比率，检验策略优势是否随时间衰减。"""
    import matplotlib.pyplot as plt

    from qlearn.metrics import rolling_sharpe

    series = rolling_sharpe(result.returns, window=window).dropna()
    if series.empty:
        raise ValueError(f"数据长度不足 {window} 个交易日，无法计算滚动夏普")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series, color=_COLOR_STRATEGY, linewidth=1.2)
    ax.axhline(0.0, color=_COLOR_LOSS, linewidth=1.0, linestyle="--")
    ax.fill_between(
        series.index, series.to_numpy(), 0.0,
        where=(series.to_numpy() >= 0), color=_COLOR_GAIN, alpha=0.15, linewidth=0,
    )
    ax.fill_between(
        series.index, series.to_numpy(), 0.0,
        where=(series.to_numpy() < 0), color=_COLOR_LOSS, alpha=0.15, linewidth=0,
    )
    ax.set_ylabel(f"滚动 Sharpe（{window} 日）")
    _finalize(ax, f"滚动夏普比率 — {result.label}")

    # 正负切换的频率越高，说明策略优势越不稳定
    sign_flips = int((np.diff(np.sign(series.to_numpy())) != 0).sum())
    ax.text(
        0.99, 0.95, f"正负切换次数: {sign_flips}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555555",
    )
    fig.tight_layout()
    return fig, ax


def plot_param_surface(
    grid_table: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "年化Sharpe",
    figsize: tuple = (9, 6),
):
    """绘制二维参数热力图，用于识别"高原"与"尖峰"。

    I. 判读方式

    1. 连成片的正值区域（高原）-> 参数稳健
    2. 孤立的单点高值（尖峰）-> 过拟合
    3. 相邻格差异巨大（悬崖）-> 参数风险高

    Args:
        grid_table: `grid_search()` 返回的结果表。
        x_param: 行索引维度参数名。
        y_param: 列索引维度参数名。
        metric: 评估指标。
    """
    import matplotlib.pyplot as plt

    from qlearn.research.grid import param_surface

    surface = param_surface(grid_table, x_param, y_param, metric)
    values = surface.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    image = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(surface.shape[1]))
    ax.set_xticklabels(surface.columns)
    ax.set_yticks(range(surface.shape[0]))
    ax.set_yticklabels(surface.index)
    ax.set_xlabel(y_param)
    ax.set_ylabel(x_param)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if not np.isfinite(value):
                continue
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=ax, shrink=0.85)
    ax.set_title(f"参数曲面 — {metric}", fontsize=12, pad=10)
    fig.tight_layout()
    return fig, ax


def plot_price_with_trades(
    panel,
    symbol: str,
    result,
    ma_windows: tuple[int, ...] = (),
    figsize: tuple = (13, 5),
):
    """绘制单标的收盘价、移动均线与买卖点。

    仅适用于单标的回测结果；多标的组合请使用净值曲线进行观察。

    Args:
        panel: 行情面板。
        symbol: 标的代码。
        result: 该标的对应的回测结果。
        ma_windows: 需要叠加的均线窗口，如 (5, 20)。
    """
    import matplotlib.pyplot as plt

    if symbol not in panel.symbols:
        raise ValueError(f"面板中不存在标的 {symbol}")

    price = panel.close[symbol]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(price.index, price, color="#34495e", linewidth=1.0, label=f"{symbol} 收盘价")

    for window in ma_windows:
        ax.plot(
            price.index,
            price.rolling(window, min_periods=window).mean(),
            linewidth=1.0,
            alpha=0.85,
            label=f"MA{window}",
        )

    # 买卖点：直接从成交明细取，保证与原回测逻辑完全一致
    trades = result.trades
    if not trades.empty:
        for side, color, marker in (("buy", _COLOR_LOSS, "^"), ("sell", _COLOR_GAIN, "v")):
            subset = trades[(trades["symbol"] == symbol) & (trades["side"] == side)]
            if subset.empty:
                continue
            prices = price.reindex(subset["date"]).to_numpy(dtype=float)
            ax.scatter(
                subset["date"], prices, marker=marker, s=55, color=color,
                zorder=5, label=f"{'买入' if side == 'buy' else '卖出'}（{len(subset)} 次）",
            )

    ax.set_ylabel("价格（元）")
    ax.legend(loc="best", frameon=False, fontsize=9, ncol=2)
    _finalize(ax, f"{symbol} 价格与买卖点 — {result.label}")
    fig.tight_layout()
    return fig, ax
