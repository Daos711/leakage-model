"""Единые настройки стиля графиков.

Импортировать в каждом plotting-модуле:
    from ..core.plot_style import setup_matplotlib, apply_comma_ticks

- setup_matplotlib() — вызвать один раз на уровне модуля
- apply_comma_ticks(fig) — вызвать перед savefig
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.use("Agg")


def _comma_formatter(x, _pos):
    """Форматтер чисел с десятичной запятой вместо точки."""
    s = f"{x:g}"
    return s.replace(".", ",")


COMMA_FMT = mticker.FuncFormatter(_comma_formatter)


def setup_matplotlib():
    """Применить единый стиль: шрифт, сетка."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def apply_comma_ticks(fig):
    """Установить десятичную запятую на всех осях фигуры."""
    for ax in fig.get_axes():
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_comma_formatter))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_comma_formatter))
