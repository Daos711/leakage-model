"""Единые настройки стиля графиков.

Импортировать в каждом plotting-модуле:
    from ..core.plot_style import setup_matplotlib, apply_comma_ticks

- setup_matplotlib() — вызвать один раз на уровне модуля
- apply_comma_ticks(fig) — вызвать перед savefig
"""

import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.use("Agg")


def _comma_formatter(x, _pos):
    """Форматтер чисел с десятичной запятой вместо точки."""
    s = f"{x:g}"
    return s.replace(".", ",")


def setup_matplotlib():
    """Применить единый стиль: шрифт, сетка."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _rescale_axis(axis, label_setter, current_label):
    """Если значения на оси > 10 000, вынести множитель ×10ⁿ в подпись оси.

    Вместо тиков 1e+06, 2e+06, ... → тики 1, 2, ... и подпись «Re, ×10⁶».
    """
    # Принудительно обновить тики
    fig = axis.get_figure()
    fig.canvas.draw_idle()

    locs = axis.get_major_locator()()
    if len(locs) == 0:
        return False

    max_abs = max(abs(v) for v in locs if np.isfinite(v))
    if max_abs < 10_000:
        return False

    # Определить порядок (ближайший целый)
    power = int(math.floor(math.log10(max_abs)))

    divisor = 10 ** power

    def scaled_formatter(x, _pos):
        val = x / divisor
        s = f"{val:g}"
        return s.replace(".", ",")

    axis.set_major_formatter(mticker.FuncFormatter(scaled_formatter))

    # Добавить множитель к подписи оси
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp_str = str(power).translate(superscripts)
    new_label = f"{current_label}, ×10{exp_str}"
    label_setter(new_label)
    return True


def apply_comma_ticks(fig):
    """Установить десятичную запятую на всех осях фигуры.

    Для осей с большими числами (>10 000) автоматически выносит
    множитель ×10ⁿ в подпись оси.
    """
    fig.canvas.draw_idle()  # обновить тики перед анализом

    for ax in fig.get_axes():
        # X-ось
        xlabel = ax.get_xlabel()
        rescaled_x = _rescale_axis(ax.xaxis, ax.set_xlabel, xlabel)
        if not rescaled_x:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_comma_formatter))

        # Y-ось
        ylabel = ax.get_ylabel()
        rescaled_y = _rescale_axis(ax.yaxis, ax.set_ylabel, ylabel)
        if not rescaled_y:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_comma_formatter))
