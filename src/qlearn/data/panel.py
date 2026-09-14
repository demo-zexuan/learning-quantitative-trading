"""
价格面板（PricePanel）容器

将多个标的的行情数据组织为「字段名 -> 宽表」结构，作为策略与回测引擎之间的统一数据契约。

I. 数据结构

1. 每个字段（open / high / low / close / volume）都是一个 DataFrame
   (1) 行索引为交易日，必须是升序、无重复的 DatetimeIndex
   (2) 列名为标的代码，所有字段的索引与列必须严格一致
2. 所有字段共享同一套 index / columns，避免对齐错误导致的隐性 bug

II. 缺失值语义（关键）

1. 价格 NaN 表示该标的当日无成交（停牌 / 未上市 / 已退市），不可交易
2. 禁止在信号计算时用前值填充价格，否则会凭空造出"可交易"的假象
3. 仅允许在「估值」场景使用前值填充，见 `valued_close`，且必须在文档中显式说明

III. 主要接口

1. from_frames: 由 {标的代码: 单标的行情表} 组装
2. from_dict:   由 {字段名: 宽表} 组装
3. slice_dates: 按日期区间切片
4. select:      按标的代码筛选

@module qlearn.data.panel
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 必需字段：缺失任一字段则无法驱动回测引擎
PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# 可选字段：仅在全部标的都具备时才纳入面板
OPTIONAL_FIELDS: tuple[str, ...] = ("amount", "pct_chg", "turnover")

__all__ = ["PricePanel", "PRICE_FIELDS", "OPTIONAL_FIELDS"]


@dataclass
class PricePanel:
    """多标的行情面板。

    Attributes:
        fields: 字段名到宽表的映射。
    """

    fields: dict[str, pd.DataFrame]

    # ------------------------------------------------------------------
    # 构造与校验
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """校验面板内部一致性。

        I. 字段完整性
        II. 索引合法性
        III. 字段间对齐
        """
        # I. 字段完整性
        # 1. 面板不能为空
        if not self.fields:
            raise ValueError("PricePanel 不能为空，至少需要一个字段")

        # 2. 必需字段必须齐备
        missing = [name for name in PRICE_FIELDS if name not in self.fields]
        if missing:
            raise ValueError(f"PricePanel 缺少必需字段: {missing}")

        # II. 索引合法性（以 close 为基准）
        reference = self.fields["close"]

        # 1. 必须是 DatetimeIndex
        if not isinstance(reference.index, pd.DatetimeIndex):
            raise TypeError("PricePanel 的索引必须是 DatetimeIndex")

        # 2. 不允许重复日期，否则 rolling 与 pct_change 语义会失真
        if reference.index.has_duplicates:
            duplicate_dates = reference.index[reference.index.duplicated()].unique()
            raise ValueError(f"PricePanel 索引存在重复日期: {list(duplicate_dates[:5])}")

        # 3. 必须按日期升序，回测引擎依赖此顺序做时间推进
        if not reference.index.is_monotonic_increasing:
            raise ValueError("PricePanel 的索引必须按日期升序排列")

        # 4. 至少需要一个标的
        if reference.shape[1] == 0:
            raise ValueError("PricePanel 至少需要一个标的（列）")

        # III. 字段间对齐
        # 1. 所有字段的索引与列顺序必须与 close 完全一致
        for name, frame in self.fields.items():
            if not frame.index.equals(reference.index):
                raise ValueError(f"字段 '{name}' 的索引与 'close' 不一致")
            if list(frame.columns) != list(reference.columns):
                raise ValueError(f"字段 '{name}' 的列与 'close' 不一致")

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------

    @property
    def open(self) -> pd.DataFrame:
        """开盘价宽表。"""
        return self.fields["open"]

    @property
    def high(self) -> pd.DataFrame:
        """最高价宽表。"""
        return self.fields["high"]

    @property
    def low(self) -> pd.DataFrame:
        """最低价宽表。"""
        return self.fields["low"]

    @property
    def close(self) -> pd.DataFrame:
        """收盘价宽表。"""
        return self.fields["close"]

    @property
    def volume(self) -> pd.DataFrame:
        """成交量宽表。"""
        return self.fields["volume"]

    @property
    def amount(self) -> pd.DataFrame | None:
        """成交额宽表，若数据源未提供则为 None。"""
        return self.fields.get("amount")

    @property
    def dates(self) -> pd.DatetimeIndex:
        """交易日索引。"""
        return self.close.index

    @property
    def symbols(self) -> pd.Index:
        """标的代码。"""
        return self.close.columns

    @property
    def n_symbols(self) -> int:
        """标的数量。"""
        return int(self.close.shape[1])

    @property
    def n_dates(self) -> int:
        """交易日数量。"""
        return int(self.close.shape[0])

    @property
    def valued_close(self) -> pd.DataFrame:
        """仅用于估值的前值填充收盘价。

        ⚠️ 该数据**只能**用于持仓市值计算，绝不能用于信号生成或成交判定。
        停牌股需要用它延续上次收盘价来估值，否则组合净值会出现跳变。
        """
        return self.close.ffill()

    @property
    def tradable(self) -> pd.DataFrame:
        """可交易掩码。

        判定条件（同时满足）：
        1. 开盘价存在且为正
        2. 成交量存在且为正
        """
        return self.open.notna() & (self.open > 0) & self.volume.notna() & (self.volume > 0)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.fields[name]

    def __contains__(self, name: object) -> bool:
        return name in self.fields

    def keys(self) -> Iterable[str]:
        """返回全部字段名。"""
        return self.fields.keys()

    # ------------------------------------------------------------------
    # 变换
    # ------------------------------------------------------------------

    def select(self, symbols: Iterable[str]) -> PricePanel:
        """按标的代码筛选，保持传入顺序。

        Args:
            symbols: 目标标的代码序列。

        Returns:
            新的 PricePanel；若代码不存在则抛出 ValueError。
        """
        wanted = list(symbols)
        unknown = [code for code in wanted if code not in self.symbols]
        if unknown:
            raise ValueError(f"以下标的不在面板中: {unknown}")
        return PricePanel({name: frame.loc[:, wanted] for name, frame in self.fields.items()})

    def slice_dates(self, start: str | None = None, end: str | None = None) -> PricePanel:
        """按日期区间切片（闭区间）。

        Args:
            start: 起始日期，None 表示从头开始。
            end: 结束日期，None 表示到末尾结束。
        """
        sliced = {
            name: frame.loc[start:end] for name, frame in self.fields.items()
        }
        if sliced["close"].shape[0] == 0:
            raise ValueError(f"日期区间 [{start}, {end}] 内没有任何数据")
        return PricePanel(sliced)

    def drop_inactive_symbols(self, min_observations: int = 60) -> PricePanel:
        """剔除有效收盘价样本过少的标的。

        用于排除新股、长期停牌股、退市股，避免污染横截面统计。

        Args:
            min_observations: 保留标的最少需要的有效收盘价数量。
        """
        counts = self.close.notna().sum(axis=0)
        keep = counts[counts >= min_observations].index
        if len(keep) == 0:
            raise ValueError(f"没有任何标的满足 min_observations={min_observations}")
        return self.select(keep)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """还原为 {标的代码: 单标的行情表} 结构。"""
        out: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            columns = {
                name: frame[symbol] for name, frame in self.fields.items() if symbol in frame
            }
            out[symbol] = pd.DataFrame(columns, index=self.dates)
        return out

    # ------------------------------------------------------------------
    # 构造器
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, pd.DataFrame]) -> PricePanel:
        """由 {字段名: 宽表} 构造面板。"""
        return cls(dict(data))

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame]) -> PricePanel:
        """由 {标的代码: 单标的行情表} 构造面板。

        I. 对齐策略

        1. 取所有标的日期索引的并集作为统一时间轴
           (1) 并集而非交集，保证早期上市的标的不会被后期上市的标的截断
           (2) 未覆盖的日期填 NaN，由 tradable 掩码负责屏蔽交易
        2. 可选字段仅在**所有**标的都提供时纳入，避免出现整体为空的数据列

        Args:
            frames: 标的代码到行情表的映射，行情表索引须为 DatetimeIndex。
        """
        if not frames:
            raise ValueError("frames 不能为空")

        # I. 构建统一时间轴
        index: pd.DatetimeIndex | None = None
        for frame in frames.values():
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise TypeError("每个行情表的索引必须是 DatetimeIndex")
            index = frame.index if index is None else index.union(frame.index)
        assert index is not None
        index = index.sort_values()

        # II. 组装字段宽表
        def _extract(frame: pd.DataFrame, name: str) -> pd.Series:
            """提取单个字段并对齐到统一时间轴。"""
            if name in frame.columns:
                return frame[name].reindex(index)
            return pd.Series(np.nan, index=index, name=name)

        fields: dict[str, pd.DataFrame] = {}
        for name in PRICE_FIELDS:
            fields[name] = pd.DataFrame(
                {symbol: _extract(frame, name) for symbol, frame in frames.items()},
                index=index,
            )

        # III. 可选字段：全部标的自带时才纳入
        for name in OPTIONAL_FIELDS:
            if all(name in frame.columns for frame in frames.values()):
                fields[name] = pd.DataFrame(
                    {symbol: frame[name] for symbol, frame in frames.items()},
                    index=index,
                )

        return cls(fields)
