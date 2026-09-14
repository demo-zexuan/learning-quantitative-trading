"""
终端排版工具测试

@module tests.test_text
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import pytest

from qlearn.utils.text import align_table, display_width, pad_display


def test_display_width_counts_cjk_as_two_columns():
    assert display_width("abc") == 3
    assert display_width("中文") == 4
    assert display_width("毛利 123") == 4 + 1 + 3
    assert display_width("") == 0


def test_pad_display_aligns_mixed_width_text():
    """中英文混排补齐后，显示宽度必须一致。"""
    lines = [pad_display("毛利", 10), pad_display("return", 10)]

    assert len({display_width(line) for line in lines}) == 1
    assert lines[0].startswith("毛利")


def test_pad_display_right_alignment():
    line = pad_display("毛利", 8, align="right")
    assert display_width(line) == 8
    assert line.endswith("毛利")


def test_pad_display_does_not_truncate():
    """超过目标宽度时保持原文，不截断（可读性优先）。"""
    text = "这是一个很长的中文标题"
    assert pad_display(text, 4) == text


def test_pad_display_rejects_unknown_alignment():
    with pytest.raises(ValueError, match="未知的对齐方式"):
        pad_display("x", 5, align="center")


def test_align_table_produces_consistent_widths():
    text = align_table([("累计收益", "12.34%"), ("Sharpe", "0.85"), ("最大回撤", "-20.00%")])
    lines = text.split("\n")

    assert len(lines) == 3
    # 所有冒号必须落在同一列
    colon_columns = {display_width(line.split(" : ")[0]) for line in lines}
    assert len(colon_columns) == 1


def test_align_table_handles_empty_input():
    assert align_table([]) == ""
