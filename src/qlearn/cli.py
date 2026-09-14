"""
命令行入口

I. 常用示例

```bash
# 用默认配置跑一次回测（真实 A 股数据）
uv run qlt

# 指定策略与参数
uv run qlt --strategy momentum --set lookback=60 --set top_n=3

# 使用合成数据，完全离线运行
uv run qlt --synthetic --autocorr 0.08

# 参数扫描
uv run qlt --grid "fast=5,10,20" --grid "slow=20,60,120"

# 前向滚动验证（回答"样本外到底行不行"）
uv run qlt --walk-forward --grid "fast=5,10,20" --grid "slow=20,60,120"

# 用沪深 300 成分股做股票池
uv run qlt --universe 000300 --strategy ma_cross
```

@module qlearn.cli
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from qlearn import __version__
from qlearn.backtest import BacktestEngine, CostModel
from qlearn.config import (
    CONFIG_DIR,
    DEFAULT_CONFIG,
    RESULTS_DIR,
    deep_merge,
    ensure_dirs,
    load_yaml,
)
from qlearn.data import (
    PricePanel,
    fetch_index_daily,
    load_panel,
    load_universe,
    make_synthetic_panel,
)
from qlearn.metrics import format_summary
from qlearn.research import grid_search, walk_forward
from qlearn.strategies import create_strategy, list_strategies

__all__ = ["main", "build_parser"]

_EPILOG = """\
提示：
  1. 请先看样本外结果（--walk-forward），再看样本内指标
  2. 参数扫描后请检查参数曲面是否为"高原"而非"尖峰"
  3. 任何回测结论都必须扣除交易成本，本工具默认已建模 A 股实际成本
"""


# ----------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="qlt",
        description="A 股量化交易回测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )

    # I. 基础
    parser.add_argument("--version", action="version", version=f"qlearn {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "default.yaml",
        help="YAML 配置文件路径（默认 config/default.yaml）",
    )

    # II. 数据
    data_group = parser.add_argument_group("数据")
    data_group.add_argument("--universe", help="标的池：内置池名 / 指数代码")
    data_group.add_argument("--symbols", help="直接指定标的代码，逗号分隔，如 600519,000858")
    data_group.add_argument("--benchmark", help="基准指数代码，如 000300")
    data_group.add_argument("--start", help="起始日期 YYYY-MM-DD")
    data_group.add_argument("--end", help="结束日期 YYYY-MM-DD")
    data_group.add_argument(
        "--adjust", choices=["qfq", "hfq", "raw"], help="复权方式（raw 表示不复权）"
    )
    data_group.add_argument("--no-cache", action="store_true", help="忽略本地行情缓存")
    data_group.add_argument(
        "--synthetic", action="store_true", help="使用合成数据（离线，仅用于验证流程）"
    )
    data_group.add_argument("--seed", type=int, default=42, help="合成数据随机种子")
    data_group.add_argument(
        "--autocorr",
        type=float,
        default=0.0,
        help="合成数据的收益自相关：>0 趋势市场，<0 反转市场，0 随机游走",
    )

    # III. 回测
    bt_group = parser.add_argument_group("回测")
    bt_group.add_argument("--capital", type=float, help="初始资金（元）")
    bt_group.add_argument("--rf", type=float, default=0.02, help="年化无风险利率，默认 0.02")
    bt_group.add_argument("--slippage", type=float, help="滑点（基点），如 5")
    bt_group.add_argument("--no-cost", action="store_true", help="使用零成本模型（仅用于对比）")

    # IV. 策略
    st_group = parser.add_argument_group("策略")
    st_group.add_argument(
        "--list-strategies", action="store_true", help="列出可用策略及其参数后退出"
    )
    st_group.add_argument("--strategy", help="策略名称")
    st_group.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="设置策略参数，可重复，如 --set fast=10",
    )
    st_group.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2,...",
        help="参数扫描维度，可重复。指定后启用参数扫描模式",
    )
    st_group.add_argument(
        "--walk-forward",
        action="store_true",
        help="启用前向滚动验证（需同时指定 --grid）",
    )
    st_group.add_argument("--train-days", type=int, default=504, help="前向验证训练窗口天数")
    st_group.add_argument("--test-days", type=int, default=126, help="前向验证测试窗口天数")

    # V. 输出
    out_group = parser.add_argument_group("输出")
    out_group.add_argument("--out", type=Path, default=RESULTS_DIR, help="结果输出目录")
    out_group.add_argument("--no-save", action="store_true", help="不保存结果文件")
    out_group.add_argument("--no-plot", action="store_true", help="不生成图表")

    return parser


def _parse_key_value(pairs: list[str], *, allow_list: bool = False) -> dict[str, Any]:
    """解析 KEY=VALUE 形式的参数。

    I. 类型推断规则

    1. 能转为 int 的按 int 处理
    2. 否则能转为 float 的按 float 处理
    3. true/false 转为 bool
    4. allow_list=True 时，逗号分隔的多个值解析为列表（用于 --grid）
    5. 其余按字符串处理

    Args:
        pairs: 形如 ["fast=5", "slow=20,60"] 的字符串列表。
        allow_list: 是否允许解析为列表。
    """
    result: dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"参数格式错误: {item!r}，应为 KEY=VALUE")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()

        if allow_list and "," in raw:
            result[key] = [_coerce_scalar(part.strip()) for part in raw.split(",")]
        else:
            result[key] = _coerce_scalar(raw)
    return result


def _coerce_scalar(text: str) -> Any:
    """把字符串推断为合适的 Python 标量类型。"""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", ""}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


# ----------------------------------------------------------------------
# 数据准备
# ----------------------------------------------------------------------


def _prepare_panel(args: argparse.Namespace, config: dict[str, Any]) -> PricePanel:
    """根据命令行与配置准备行情面板。"""
    data_config = config["data"]

    if args.synthetic:
        print("[数据] 使用合成数据（离线模式，结果不具备任何现实意义）")
        return make_synthetic_panel(
            start=data_config.get("start") or "2020-01-01",
            end=data_config.get("end") or "2024-12-31",
            seed=args.seed,
            autocorrelation=args.autocorr,
        )

    # I. 解析标的池
    if args.symbols:
        symbols = [code.strip() for code in args.symbols.split(",") if code.strip()]
    else:
        universe = args.universe or data_config.get("universe") or "demo_large_cap"
        symbols = load_universe(universe)

    adjust = "raw" if args.adjust == "raw" else (args.adjust or data_config.get("adjust", "qfq"))
    end_text = args.end or data_config.get("end") or "今天"
    print(
        f"[数据] 标的池 {len(symbols)} 只，复权方式 {adjust}，"
        f"区间 {args.start or data_config.get('start')} ~ {end_text}"
    )

    panel = load_panel(
        symbols,
        start=args.start or data_config.get("start", "2020-01-01"),
        end=args.end or data_config.get("end"),
        adjust=adjust,
        use_cache=not args.no_cache and bool(data_config.get("use_cache", True)),
        min_observations=int(data_config.get("min_observations", 0)),
    )
    print(f"[数据] 面板构建完成：{panel.n_symbols} 个标的 x {panel.n_dates} 个交易日")
    return panel


def _prepare_benchmark(args: argparse.Namespace, config: dict[str, Any], panel: PricePanel):
    """获取基准价格序列，失败时返回 None（不阻塞主流程）。"""
    code = args.benchmark or config["data"].get("benchmark")
    if not code or args.synthetic:
        return None
    try:
        benchmark = fetch_index_daily(
            code,
            start=config["data"].get("start", "2020-01-01"),
            end=args.end or config["data"].get("end"),
            use_cache=not args.no_cache,
        )
        print(f"[数据] 基准 {code} 加载完成")
        return benchmark["close"]
    except Exception as exc:  # noqa: BLE001 - 基准缺失不应中断回测
        print(f"[警告] 基准 {code} 加载失败，跳过相对指标: {exc}")
        return None


def _build_engine(args: argparse.Namespace, config: dict[str, Any]) -> BacktestEngine:
    """构造回测引擎。"""
    bt_config = config["backtest"]
    cost_model = CostModel.zero() if args.no_cost else CostModel.from_dict(bt_config.get("costs"))
    if args.slippage is not None:
        cost_model = CostModel(**{**cost_model.__dict__, "slippage_bps": args.slippage})

    if args.no_cost:
        print("[警告] 已启用零成本模型，结果不代表可实盘收益")

    return BacktestEngine(
        initial_capital=args.capital or float(bt_config.get("initial_capital", 1_000_000.0)),
        cost_model=cost_model,
        max_participation=bt_config.get("max_participation"),
    )


# ----------------------------------------------------------------------
# 各运行模式
# ----------------------------------------------------------------------


def _print_cost_model(engine: BacktestEngine) -> None:
    """打印成本模型参数，让用户清楚扣了什么钱。"""
    from qlearn.utils.text import align_table

    print("\n[成本模型]")
    print(align_table(list(engine.cost_model.describe().items()), separator=" : "))


def _run_single(
    strategy_name: str,
    params: dict[str, Any],
    panel: PricePanel,
    engine: BacktestEngine,
    benchmark,
    args: argparse.Namespace,
):
    """单策略回测模式。"""
    strategy = create_strategy(strategy_name, **params)
    print(f"\n[策略] {strategy.label}")

    result = engine.run(
        strategy.generate_weights(panel), panel, benchmark=benchmark, label=strategy.label
    )

    print()
    print(result.describe())
    print("\n[成本影响]")
    print(result.describe_cost_impact())

    # 样本内/外切分提示
    _warn_if_too_short(result)
    return result


def _warn_if_too_short(result) -> None:
    """对样本期过短或交易过少的情况给出提示。"""
    n_days = len(result.equity)
    n_trades = len(result.trades)
    if n_days < 252:
        print(f"\n[提示] 样本仅 {n_days} 个交易日（不足 1 年），统计量极不稳定。")
    if n_trades < 30:
        print(f"[提示] 全程仅 {n_trades} 笔成交，结果高度依赖个别交易，不具备统计意义。")


def _run_grid(
    strategy_name: str,
    param_grid: dict[str, list[Any]],
    panel: PricePanel,
    engine: BacktestEngine,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """参数扫描模式。"""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    print(f"\n[参数扫描] {total} 个参数组合")

    table = grid_search(strategy_name, param_grid, panel, engine=engine, risk_free_rate=args.rf)
    display = table.head(20).copy()
    for column in ("累计收益", "最大回撤"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.2%}")
    print(display.to_string(index=False))

    if total >= 4 and len(param_grid) >= 2:
        print(
            "\n[提示] 请绘制参数曲面（使用 qlearn.utils.plot_param_surface）检查：\n"
            "        连成一片的高原 = 稳健；孤立的尖峰 = 过拟合。"
        )
    return table


def _run_walk_forward(
    strategy_name: str,
    param_grid: dict[str, list[Any]],
    panel: PricePanel,
    engine: BacktestEngine,
    args: argparse.Namespace,
):
    """前向滚动验证模式。"""
    print(
        f"\n[前向滚动验证] 训练窗口 {args.train_days} 天 / 测试窗口 {args.test_days} 天"
    )
    result = walk_forward(
        strategy_name,
        param_grid,
        panel,
        engine=engine,
        train_days=args.train_days,
        test_days=args.test_days,
        risk_free_rate=args.rf,
    )

    print()
    print("各折明细（IS = 样本内，OOS = 样本外）:")
    print(result.folds.to_string(index=False))

    print("\n过拟合诊断:")
    for name, value in result.overfit_gap().items():
        print(f"  {name:<16}: {value:,.4f}")

    print("\n样本外绩效（评价策略的正确口径）:")
    print(format_summary(result.metrics(risk_free_rate=args.rf)).to_string())
    return result


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """命令行主入口。

    Returns:
        进程退出码，0 表示成功。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # I. 列出策略
    if args.list_strategies:
        print("可用策略：")
        for name in list_strategies():
            print(f"  {name}")

        hint = (
            "\n查看某策略的完整参数名单：\n"
            "  uv run python -c \"from qlearn.strategies import STRATEGIES;"
            " print(STRATEGIES['ma_cross'].param_names())\""
        )
        print(hint)
        return 0

    ensure_dirs()

    # II. 配置合并：文件 -> 命令行
    config = deep_merge(DEFAULT_CONFIG, load_yaml(args.config) if args.config.exists() else {})
    if not args.config.exists():
        print(f"[警告] 配置文件 {args.config} 不存在，使用内置默认配置")

    strategy_name = args.strategy or config["strategy"]["name"]
    params = dict(config["strategy"].get("params", {}))
    params.update(_parse_key_value(args.set))

    param_grid = {
        key: (value if isinstance(value, list) else [value])
        for key, value in _parse_key_value(args.grid, allow_list=True).items()
    }

    # III. 联网前置检查
    if not args.synthetic and not args.no_cache:
        pass  # 缓存与网络错误由 loader 统一处理

    try:
        panel = _prepare_panel(args, config)
    except Exception as exc:  # noqa: BLE001 - 顶层统一报错，避免堆栈吓人
        print(f"\n[错误] 数据准备失败: {exc}", file=sys.stderr)
        print("提示：若当前无网络，可加 --synthetic 使用合成数据验证流程。", file=sys.stderr)
        return 1

    engine = _build_engine(args, config)
    _print_cost_model(engine)

    # IV. 按模式执行
    result = None
    grid_table = None

    if args.walk_forward:
        if not param_grid:
            print("\n[错误] --walk-forward 必须配合 --grid 使用", file=sys.stderr)
            return 1
        result = _run_walk_forward(strategy_name, param_grid, panel, engine, args)
        artifact_name = f"walk_forward_{strategy_name}"
    elif param_grid:
        grid_table = _run_grid(strategy_name, param_grid, panel, engine, args)
        artifact_name = f"grid_{strategy_name}"
    else:
        benchmark = _prepare_benchmark(args, config, panel)
        result = _run_single(strategy_name, params, panel, engine, benchmark, args)
        artifact_name = f"backtest_{strategy_name}"

    # V. 保存与绘图
    if not args.no_save:
        args.out.mkdir(parents=True, exist_ok=True)
        if grid_table is not None:
            path = args.out / f"{artifact_name}.csv"
            grid_table.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"\n[输出] 参数扫描结果已保存: {path}")
        elif result is not None and hasattr(result, "save"):
            written = result.save(args.out, prefix=artifact_name)
            print(f"\n[输出] 结果已保存: {', '.join(str(p.name) for p in written)}")

    if not args.no_plot and result is not None and hasattr(result, "equity"):
        try:
            from qlearn.utils import plot_drawdown, plot_equity, plot_monthly_heatmap

            args.out.mkdir(parents=True, exist_ok=True)
            for name, plotter in (
                ("equity", plot_equity),
                ("drawdown", plot_drawdown),
                ("monthly", plot_monthly_heatmap),
            ):
                figure, _ = plotter(result)
                path = args.out / f"{artifact_name}_{name}.png"
                figure.savefig(path, dpi=130, bbox_inches="tight")
                print(f"[输出] 图表已保存: {path.name}")
        except Exception as exc:  # noqa: BLE001 - 绘图失败不应影响回测结果
            print(f"[警告] 绘图失败（可能是缺少中文字体）: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
