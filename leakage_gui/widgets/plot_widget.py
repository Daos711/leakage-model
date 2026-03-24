"""Переиспользуемый виджет matplotlib с контекстным меню экспорта."""

from __future__ import annotations

from typing import Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from ..utils.export import (
    figure_to_clipboard_bytes,
    save_figure_pdf,
    save_figure_png,
    save_figure_svg,
)
from ..utils.locale_fmt import apply_comma_format


class PlotWidget(QWidget):
    """Matplotlib canvas с контекстным меню экспорта."""

    def __init__(self, parent: QWidget | None = None,
                 figsize: tuple[float, float] = (6, 4)) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)

        self._setup_font()

    def _setup_font(self) -> None:
        """Настройка шрифта для графиков."""
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        })

    def plot(self, plot_func: Callable, clear: bool = True) -> None:
        """Вызвать plot_func(fig, ax) и обновить canvas.

        Args:
            plot_func: функция (fig, ax) -> None для одного axes,
                       или (fig) -> None если нужен свой layout.
            clear: очистить фигуру перед построением.
        """
        if clear:
            self.figure.clear()

        import inspect
        sig = inspect.signature(plot_func)
        n_params = len(sig.parameters)
        if n_params >= 2:
            ax = self.figure.add_subplot(111)
            plot_func(self.figure, ax)
        else:
            plot_func(self.figure)

        for ax in self.figure.get_axes():
            apply_comma_format(ax)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_multi(self, plot_func: Callable) -> None:
        """Вызвать plot_func(fig) для нескольких subplots."""
        self.figure.clear()
        plot_func(self.figure)
        for ax in self.figure.get_axes():
            apply_comma_format(ax)
        self.figure.tight_layout()
        self.canvas.draw()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)

        act_png = QAction("Сохранить как PNG", self)
        act_png.triggered.connect(self._save_png)
        menu.addAction(act_png)

        act_svg = QAction("Сохранить как SVG", self)
        act_svg.triggered.connect(self._save_svg)
        menu.addAction(act_svg)

        act_pdf = QAction("Сохранить как PDF", self)
        act_pdf.triggered.connect(self._save_pdf)
        menu.addAction(act_pdf)

        menu.addSeparator()

        act_copy = QAction("Копировать в буфер обмена", self)
        act_copy.triggered.connect(self._copy_to_clipboard)
        menu.addAction(act_copy)

        menu.exec(self.canvas.mapToGlobal(pos))

    def _save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить PNG", "", "PNG (*.png)")
        if path:
            save_figure_png(self.figure, path, dpi=300)

    def _save_svg(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить SVG", "", "SVG (*.svg)")
        if path:
            save_figure_svg(self.figure, path)

    def _save_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить PDF", "", "PDF (*.pdf)")
        if path:
            save_figure_pdf(self.figure, path)

    def _copy_to_clipboard(self) -> None:
        data = figure_to_clipboard_bytes(self.figure, dpi=150)
        img = QImage.fromData(data)
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setImage(img)
