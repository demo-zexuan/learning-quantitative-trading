"""
终端文本排版工具

I. 为什么需要它

Python 的 `len()` 把汉字按 1 计，但汉字在等宽终端里占 **2 列**。
直接用 f-string 的 `:<10` 对齐中英文混排的表格，一定会错位。

II. 实现原理

用 `unicodedata.east_asian_width` 判定字符宽度类别：

1. `W`（Wide）/ `F`（Fullwidth）-> 占 2 列
2. 其余（含 `Na` Narrow、`A` Ambiguous）-> 占 1 列

@module qlearn.utils.text
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import unicodedata

__all__ = ["display_width", "pad_display", "align_table"]

# 占用 2 列宽度的东亚字符类别
_WIDE_CATEGORIES = frozenset({"W", "F"})


def display_width(text: str) -> int:
    """计算字符串在等宽终端中的显示宽度。

    Args:
        text: 待测量文本。

    Returns:
        以「列」为单位的显示宽度。
    """
    return sum(2 if unicodedata.east_asian_width(char) in _WIDE_CATEGORIES else 1 for char in text)


def pad_display(text: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格，使中英文混排也能对齐。

    Args:
        text: 待补齐文本。
        width: 目标显示宽度（列数）。
        align: "left" 左对齐补在右侧，"right" 右对齐补在左侧。

    Returns:
        补齐后的字符串；若本身已超过宽度则不截断，保持原文可读。
    """
    padding = " " * max(0, width - display_width(text))
    if align == "right":
        return padding + text
    if align == "left":
        return text + padding
    raise ValueError(f"未知的对齐方式: {align!r}，可选 'left' / 'right'")


def align_table(rows: list[tuple[str, str]], separator: str = " : ") -> str:
    """把「键 -> 值」两列对齐渲染为多行文本。

    Args:
        rows: (键, 值) 序列。
        separator: 键与值之间的分隔符。

    Returns:
        已完成列对齐的多行字符串。
    """
    if not rows:
        return ""
    width = max(display_width(str(key)) for key, _ in rows)
    return "\n".join(
        f"{pad_display(str(key), width)}{separator}{value}" for key, value in rows
    )
