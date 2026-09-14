"""
全局配置与路径管理

I. 职责

1. 集中定义项目路径常量，避免各处硬编码相对路径
2. 提供 YAML 配置加载与合并能力
3. 提供目录自动创建工具

II. 路径约定

1. data/raw      原始行情缓存（parquet），可随时删除后重新拉取
2. data/processed 清洗后的中间数据
3. results       回测结果（净值曲线、绩效表、图）
4. notebooks     教学笔记本
5. config        配置模板

@module qlearn.config
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "RESULTS_DIR",
    "ensure_dirs",
    "load_yaml",
    "deep_merge",
    "DEFAULT_CONFIG",
]

# src/qlearn/config.py -> parents[0]=qlearn, [1]=src, [2]=项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = PROJECT_ROOT / "results"


def ensure_dirs() -> None:
    """创建项目运行所需的全部目录（幂等）。"""
    for directory in (CONFIG_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# 默认配置：所有可调参数集中于此，便于 notebook 与 CLI 复用
DEFAULT_CONFIG: dict[str, Any] = {
    # I. 数据配置
    "data": {
        "universe": "demo_large_cap",
        "benchmark": "000300",
        "start": "2020-01-01",
        "end": None,
        "adjust": "qfq",
        "use_cache": True,
        "min_observations": 250,
    },
    # II. 回测配置
    "backtest": {
        "initial_capital": 1_000_000.0,
        "max_participation": None,
        "costs": {
            "commission_rate": 0.00025,  # 佣金：万分之 2.5，双边
            "min_commission": 5.0,       # 单笔最低佣金 5 元
            "stamp_duty_rate": 0.0005,   # 印花税：万分之 5，仅卖出
            "transfer_fee_rate": 0.00001,  # 过户费：十万分之 1，双边
            "slippage_bps": 2.0,         # 滑点：2 个基点，双边
            "apply_min_commission": True,
        },
    },
    # III. 策略配置
    "strategy": {
        "name": "ma_cross",
        "params": {"fast": 5, "slow": 20},
    },
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件。

    Args:
        path: YAML 文件路径。

    Returns:
        解析后的字典；空文件返回空字典。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，override 优先。

    I. 语义

    1. 叶子节点：override 直接覆盖 base
    2. 嵌套字典：递归合并而非整体替换
       (1) 这样配置文件只需写需要改动的字段
    3. base 不会被修改（深拷贝返回）

    Args:
        base: 基础配置。
        override: 覆盖配置。

    Returns:
        合并后的新字典。
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
