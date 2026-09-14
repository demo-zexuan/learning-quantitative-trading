# A 股量化交易学习脚手架

一个**从零到可实盘研究**的 A 股量化学习项目：真实数据、可审计的回测引擎、完备的绩效分析，
以及最重要的——识别自己回测结果里假象的工具链。

> 本项目的核心主张：**量化研究 90% 的工作是证伪，而不是发现。**

---

## I. 快速开始

```bash
# 1. 安装依赖（uv 会自动创建 .venv）
uv sync

# 2. 跑一次真实 A 股回测
uv run qlt

# 3. 打开教程笔记本，按顺序学习
#    notebooks/01_数据基础与探索.ipynb
#    notebooks/02_回测引擎与交易成本.ipynb
#    notebooks/03_策略研究与过拟合.ipynb
```

**没有网络？** 加 `--synthetic` 用合成数据跑通全流程（笔记本也会自动降级）：

```bash
uv run qlt --synthetic --autocorr 0.08
```

---

## II. 目录结构

```
.
├── config/default.yaml      # 所有可调参数的默认值
├── notebooks/               # 三个循序渐进的教程笔记本
├── src/qlearn/
│   ├── config.py            # 路径与配置合并
│   ├── cli.py               # 命令行入口（uv run qlt）
│   ├── data/                # 数据层
│   │   ├── panel.py         #   PricePanel：字段 × 宽表
│   │   ├── loader.py        #   akshare 拉取 + 缓存 + 重试 + 代理绕过
│   │   ├── universe.py      #   股票池（内置池 / 指数成分股）
│   │   └── synthetic.py     #   合成数据（含自相关旋钮，用于对照实验）
│   ├── backtest/            # 回测层
│   │   ├── engine.py        #   事件式引擎：收盘决策 → 次日开盘成交
│   │   ├── costs.py         #   A 股成本模型（佣金/印花税/过户费/滑点）
│   │   └── result.py        #   结果容器 + 绩效 + 导出
│   ├── metrics/             # 绩效与风险指标
│   ├── strategies/          # 策略层（注册表 + 4 个示例策略）
│   ├── research/            # 科研层：参数扫描 + 前向滚动验证
│   └── utils/               # 可视化（含中文字体自动配置）
└── tests/                   # 110 项测试，含无前视偏差检验
```

---

## III. 学习路线

建议按「**先动手 → 再看书 → 再证伪**」的顺序推进。

### 阶段 0 · 跑通链路（1 天）

```bash
uv run qlt --synthetic                       # 先离线跑通
uv run qlt --symbols 600519,000858 --start 2020-01-01   # 再来真实数据
```

读完 `notebooks/01_数据基础与探索.ipynb`，重点是搞清**复权、停牌、对齐**三个坑。

### 阶段 1 · 理解回测为什么可能是假的（1 周）

读完 `notebooks/02_回测引擎与交易成本.ipynb`。这一节的产出不是「策略赚了多少」，
而是「这个回测结果有多少水分」。必须掌握：

| 概念 | 为什么致命 |
|---|---|
| 收盘决策 / 次日开盘成交 | 顺序错了就是前视偏差，结果全不可信 |
| 单笔最低佣金 5 元 | 小资金高频策略的实际费率可达名义费率的 4 倍 |
| 印花税仅卖出 | A 股「卖出更贵」，往返成本天然不对称 |
| 每日再平衡 vs 信号驱动 | 前者能白送 1000%+ 年换手的手续费 |
| 停牌不可交易 | 忽略它就会在买不到的价位「买入」 |

### 阶段 2 · 学会怀疑自己的结果（1 周）

读完 `notebooks/03_策略研究与过拟合.ipynb`。核心是两件事：

1. **参数曲面**：你要的是「高原」（邻近参数都能赚），不是「尖峰」（只有某个精确参数能赚）
2. **前向滚动验证**：样本内 Sharpe 0.60 → 样本外 −0.28 是常态，不是意外

### 阶段 3 · 自己写策略（持续）

在 `src/qlearn/strategies/` 下新建文件，继承 `Strategy` 并实现一个方法：

```python
from qlearn.strategies import Strategy, register_strategy
from qlearn.data import PricePanel
import pandas as pd

@register_strategy
class MyStrategy(Strategy):
    """一句话说清这个策略赚的是谁的钱。"""

    name = "my_strategy"

    def __init__(self, window: int = 20) -> None:
        super().__init__(window=window)
        self.window = window

    def generate_weights(self, panel: PricePanel) -> pd.DataFrame:
        # 返回目标权重矩阵：index = 决策日，columns = 标的
        # NaN 表示维持当前持仓；0 表示清仓
        ...
```

然后 `uv run qlt --strategy my_strategy` 即可直接用上全部工具链。

### 阶段 4 · 走向实盘（长期）

| 主题 | 起点 |
|---|---|
| 风控与仓位管理 | 单笔风险上限、组合相关性约束、最大回撤熔断 |
| 组合构建 | 等权 → 风险平价 → 均值方差优化 |
| 交易执行 | 拆单、TWAP/VWAP、冲击成本估计 |
| 实盘监控 | 每日盈亏归因、信号衰减监控、异常告警 |

---

## IV. 命令行速查

```bash
uv run qlt                                  # 默认配置跑一次回测
uv run qlt --strategy momentum --set lookback=60 --set top_n=3
uv run qlt --symbols 600519,000858 --start 2020-01-01 --end 2024-12-31
uv run qlt --universe 000300                # 用沪深 300 成分股做股票池
uv run qlt --no-cost                        # 零成本对比（看成本拖累）
uv run qlt --slippage 20                    # 提高滑点压力测试
uv run qlt --list-strategies                # 列出可用策略

# 参数扫描：产出参数曲面，识别高原 vs 尖峰
uv run qlt --grid "fast=5,10,20" --grid "slow=30,60,120"

# 前向滚动验证：唯一值得相信的业绩口径
uv run qlt --walk-forward --grid "fast=5,10,20" --grid "slow=30,60,120" \
           --train-days 504 --test-days 126
```

---

## V. 核心设计决策（以及为什么）

| 设计 | 原因 |
|---|---|
| 策略只负责「产出目标权重」 | 资金、成本、成交时点全部交给引擎，保证策略可单独测试 |
| 用「次日开盘价」成交 | 收盘才能拿到收盘价，当天开盘成交必然包含前视偏差 |
| `NaN` 权重 = 维持持仓 | 让低频调仓策略（动量、均值回归）不必写无谓的再平衡 |
| 成本逐笔计算 | 最低佣金是**每笔订单**的门槛，无法用总量费率替代 |
| 停牌标的用前收盘价估值 | 否则组合净值会因停牌出现虚假跳变 |
| 整手（100 股）向下取整 | A 股实际规则，否则回测资金利用率会被高估 |
| 现金不足时压缩买单 | 绝不引入隐性杠杆，保证结果可实盘复现 |
| 零成本模式可切换 | 让「成本侵蚀了多少收益」成为一个可量化的数字 |

---

## VI. 已知边界

诚实地列出**没有**建模的东西，这些在实盘都会让结果变差：

1. **涨跌停无法成交** —— 一字板买不进也卖不出，对动量策略影响最大
2. **盘中止损/止盈** —— 引擎只支持日频决策，无法模拟盘中触发
3. **分红送股现金流的精确处理** —— 用前复权价格已大致抵消，但不完全等价
4. **融资融券成本与强平** —— 策略若隐含杠杆则完全未考虑
5. **冲击成本随规模上升** —— 大资金的实际容量远小于回测假设
6. **指数成分股的历史快照** —— 免费的 akshare 只提供当前成分股，存在**幸存者偏差**

> 经验法则：实盘能拿到回测 Sharpe 的 50%~70% 已属不错。

---

## VII. 常见问题

### 拉数据报 `ProxyError`

系统代理（如 Clash 监听 `127.0.0.1:7897`）通常无法访问国内行情接口，
而直连反而正常。本项目**默认对国内数据源绕过代理**，无需额外配置。
若需要改回默认行为：

```python
from qlearn.data import set_proxy_bypass
set_proxy_bypass(False)
```

### 数据源不稳定 / 返回空表

`loader` 已内置：主源（东方财富）→ 备用源（新浪）→ 本地缓存降级，三者依次兜底。
首次拉取成功后数据会缓存到 `data/raw/`，之后离线也能复现结果。
清缓存：`from qlearn.data import clear_cache; clear_cache()`。

### 图表中文显示为方框

`qlearn.utils.plotting` 在导入时会自动探测系统中文字体（macOS 优先 PingFang SC）。
若仍异常，手动指定：

```python
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["你的中文字体名"]
```

### 测试

```bash
uv run pytest -q          # 110 项测试
uv run ruff check .       # 静态检查
```

测试里包含一项**无前视偏差检验**：用截断到第 t 日的数据重新计算权重，
必须与用完整数据算出的前 t 日权重逐位相同。这是量化里最容易犯也最难发现的错误。

---

## VIII. 推荐延伸阅读

- **书**：《主动投资组合管理》(Grinold & Kahn)、《Advances in Financial Machine Learning》(López de Prado)、《量化交易》(Ernie Chan)
- **论文**：SSRN、arXiv `q-fin`、AQR 研究库
- **社区**：Quantitative Finance StackExchange、r/algotrading
- **下一步方向**：因子模型（Fama-French 三/五因子）→ 统计套利（协整）→ 组合优化 → 执行算法

---

## IX. 免责声明

本项目仅用于**学习与研究**。

1. 内置策略是教学示例，**不是**可直接盈利的策略
2. 合成数据产生的任何结果都**不具备**现实意义
3. 回测业绩不预示未来收益，实盘表现通常显著更差
4. 任何实盘决策及其后果由使用者自行承担

---

**@author zexuan.peng**
