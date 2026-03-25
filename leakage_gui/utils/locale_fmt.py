"""Форматирование чисел с запятой в качестве десятичного разделителя."""

from __future__ import annotations

from matplotlib.ticker import FuncFormatter


def fmt_comma(value: float, decimals: int = 4) -> str:
    """Форматировать число с запятой: 0.138 → '0,138'."""
    s = f"{value:.{decimals}f}"
    return s.replace(".", ",")


def fmt_comma_auto(value: float) -> str:
    """Форматировать число, убирая незначащие нули."""
    s = f"{value:g}"
    return s.replace(".", ",")


def comma_formatter(decimals: int | None = None) -> FuncFormatter:
    """Matplotlib FuncFormatter, заменяющий точку на запятую."""
    def _fmt(x: float, _pos: int | None = None) -> str:
        if decimals is not None:
            s = f"{x:.{decimals}f}"
        else:
            s = f"{x:g}"
        return s.replace(".", ",")
    return FuncFormatter(_fmt)


def apply_comma_format(ax, x_decimals: int | None = None,
                       y_decimals: int | None = None) -> None:
    """Применить запятую к осям matplotlib."""
    ax.xaxis.set_major_formatter(comma_formatter(x_decimals))
    ax.yaxis.set_major_formatter(comma_formatter(y_decimals))


def parse_float_comma(text: str) -> float:
    """Разобрать число с запятой или точкой: '0,138' → 0.138."""
    return float(text.strip().replace(",", "."))
