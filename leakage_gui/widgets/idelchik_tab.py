"""Вкладка 4: Справочник Идельчика (этап 2)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..utils.locale_fmt import fmt_comma
from .base_tab import BaseTab
from .plot_widget import PlotWidget

logger = logging.getLogger("leakage_gui")

_VARIANT_COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


class IdelchikTab(BaseTab):
    """Вкладка справочника Идельчика."""

    tab_name = "Справочник Идельчика"

    def __init__(self, app_state: dict[str, Any], parent=None) -> None:
        super().__init__(app_state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        left = QVBoxLayout()
        ctrl = QGroupBox("Варианты расчёта")
        ctrl_layout = QVBoxLayout(ctrl)
        self._checks: dict[str, QCheckBox] = {}
        for key, label in [("A", "Вариант A"),
                            ("B", "Вариант B"),
                            ("C", "Вариант C")]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            self._checks[key] = cb
            ctrl_layout.addWidget(cb)
        left.addWidget(ctrl)

        self.btn_calc = QPushButton("Рассчитать")
        self.btn_calc.clicked.connect(self._run_calc)
        left.addWidget(self.btn_calc)

        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet("color: #666; font-style: italic;")
        left.addWidget(self.lbl_note)
        left.addStretch()

        right = QVBoxLayout()
        self.plot_water = PlotWidget(figsize=(8, 4))
        self.plot_air = PlotWidget(figsize=(8, 4))
        right.addWidget(self.plot_water)
        right.addWidget(self.plot_air)

        splitter = QSplitter()
        left_w = QWidget()
        left_w.setLayout(left)
        right_w = QWidget()
        right_w.setLayout(right)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([200, 780])
        main_layout.addWidget(splitter)

    def _run_calc(self) -> None:
        df_cal = self.app_state.get("cal_df")
        df_val = self.app_state.get("val_df")
        if df_cal is None:
            QMessageBox.warning(self, "Нет данных",
                                "Загрузите калибровочные данные.")
            return

        variants = [k for k, cb in self._checks.items() if cb.isChecked()]
        if not variants:
            return

        try:
            from leakage_model.core.config import GEOM_AIR, GEOM_WATER
            from leakage_model.stage2_idelchik.model import run_variant

            results_water = {}
            results_air = {}

            for v in variants:
                u1_w = df_cal["u1"].values
                res_w = run_variant(v, u1_w, GEOM_WATER)
                results_water[v] = res_w

                if df_val is not None and "u1" in df_val.columns:
                    u1_a = df_val["u1"].values
                    res_a = run_variant(v, u1_a, GEOM_AIR)
                    results_air[v] = res_a

            self._plot_model(self.plot_water, "Водяная модель (A_ок = 12 м²)",
                             results_water, df_cal)
            if results_air:
                self._plot_model(self.plot_air, "Воздушная модель (A_ок = 20 м²)",
                                 results_air, df_val)

            self.lbl_note.setText(
                "Примечание: формулы Идельчика дают r ≈ const, "
                "не воспроизводя зависимость r(u₁). "
                "Это связано с тем, что F_б/F_c > 1, "
                "выходя за рамки табличных данных.")

        except Exception as e:
            logger.error(f"Ошибка Идельчика: {e}")
            QMessageBox.warning(self, "Ошибка", str(e))

    def _plot_model(self, plot_widget, title, results, df_exp) -> None:
        def plot_func(fig, ax):
            if df_exp is not None and "r" in df_exp.columns:
                ax.plot(df_exp["u1"], df_exp["r"], "ko", ms=6,
                        label="Эксперимент")
            for v, res in results.items():
                color = _VARIANT_COLORS.get(v, "gray")
                mask = res.converged
                ax.plot(res.u1[mask], res.r_pred[mask], "s-",
                        color=color, ms=5, lw=1.5,
                        label=f"Вариант {v}")
            ax.set_xlabel("u₁, м/с")
            ax.set_ylabel("r")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title(title)
        plot_widget.plot(plot_func)
