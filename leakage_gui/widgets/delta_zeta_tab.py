"""Вкладка 2: Модель Δζ (этап 1)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
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
from .results_table import ResultsTable

logger = logging.getLogger("leakage_gui")


class DeltaZetaTab(BaseTab):
    """Вкладка модели Δζ."""

    tab_name = "Модель Δζ"

    def __init__(self, app_state: dict[str, Any], parent=None) -> None:
        super().__init__(app_state, parent)
        self._fit_result = None
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        left = QVBoxLayout()
        ctrl = QGroupBox("Аппроксимация")
        ctrl_layout = QFormLayout(ctrl)
        self.combo_variant = QComboBox()
        self.combo_variant.addItems(["Степенной закон", "Асимптотический"])
        ctrl_layout.addRow("Вариант:", self.combo_variant)
        left.addWidget(ctrl)

        self.btn_calc = QPushButton("Рассчитать")
        self.btn_calc.clicked.connect(self._run_calc)
        left.addWidget(self.btn_calc)

        metrics_group = QGroupBox("Метрики")
        metrics_layout = QFormLayout(metrics_group)
        self._metric_labels: dict[str, QLabel] = {}
        for key, label in [("r2_fit", "R² (аппрокс.)"),
                            ("rmse_cal", "RMSE (калибр.)"),
                            ("rmse_val", "RMSE (валид.)")]:
            lbl = QLabel("—")
            self._metric_labels[key] = lbl
            metrics_layout.addRow(label + ":", lbl)
        left.addWidget(metrics_group)
        left.addStretch()

        right = QVBoxLayout()
        top = QHBoxLayout()
        self.plot_dz = PlotWidget(figsize=(5, 4))
        self.plot_r = PlotWidget(figsize=(5, 4))
        top.addWidget(self.plot_dz)
        top.addWidget(self.plot_r)
        right.addLayout(top)

        self.plot_diag = PlotWidget(figsize=(10, 3))
        right.addWidget(self.plot_diag)

        self.table_metrics = ResultsTable()
        right.addWidget(self.table_metrics)

        splitter = QSplitter()
        left_w = QWidget()
        left_w.setLayout(left)
        right_w = QWidget()
        right_w.setLayout(right)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([220, 750])
        main_layout.addWidget(splitter)

    def _run_calc(self) -> None:
        df_cal = self.app_state.get("cal_df")
        df_val = self.app_state.get("val_df")
        if df_cal is None:
            QMessageBox.warning(self, "Нет данных",
                                "Загрузите калибровочные данные.")
            return

        try:
            from leakage_model.core.config import GEOM_AIR, GEOM_WATER
            from leakage_model.core.validation import compute_metrics
            from leakage_model.stage1_energy.calibration import (
                fit_asymptotic,
                fit_power_law,
            )
            from leakage_model.stage1_energy.model import (
                calc_delta_zeta,
                calc_r_explicit,
            )

            geom_w = GEOM_WATER
            u1_cal = df_cal["u1"].values
            r_cal = df_cal["r"].values
            Re_cal = u1_cal * geom_w["D_h"] / geom_w["nu"]
            dz_cal = calc_delta_zeta(r_cal, geom_w["A_ok"], geom_w["A_s"])

            variant = self.combo_variant.currentIndex()
            if variant == 0:
                fit = fit_power_law(Re_cal, dz_cal)
            else:
                fit = fit_asymptotic(Re_cal, dz_cal)
            self._fit_result = fit

            self._metric_labels["r2_fit"].setText(fmt_comma(fit.R2, 4))

            # Предсказание r
            r_pred_cal = calc_r_explicit(u1_cal, geom_w, fit.dz_func)
            m_cal = compute_metrics(r_cal, r_pred_cal)
            self._metric_labels["rmse_cal"].setText(fmt_comma(m_cal.RMSE, 4))

            # Валидация
            geom_a = GEOM_AIR
            rmse_val_text = "—"
            r_pred_val = None
            if df_val is not None and "r" in df_val.columns:
                u1_val = df_val["u1"].values
                r_val = df_val["r"].values
                r_pred_val = calc_r_explicit(u1_val, geom_a, fit.dz_func)
                m_val = compute_metrics(r_val, r_pred_val)
                rmse_val_text = fmt_comma(m_val.RMSE, 4)
            self._metric_labels["rmse_val"].setText(rmse_val_text)

            # Графики
            self._plot_dz(Re_cal, dz_cal, fit)
            self._plot_r(df_cal, df_val, geom_w, geom_a, fit)
            self._plot_diagnostic(df_cal, df_val, geom_w, geom_a)

        except Exception as e:
            logger.error(f"Ошибка Δζ: {e}")
            QMessageBox.warning(self, "Ошибка", str(e))

    def _plot_dz(self, Re_cal, dz_cal, fit) -> None:
        def plot_func(fig, ax):
            ax.plot(Re_cal, dz_cal, "ko", ms=6, label="Эксперимент")
            Re_fine = np.linspace(Re_cal.min() * 0.8, Re_cal.max() * 1.2, 200)
            dz_fine = fit.dz_func(Re_fine)
            ax.plot(Re_fine, dz_fine, "r-", lw=2,
                    label=f"{fit.name} (R²={fit.R2:.3f})".replace(".", ","))
            ax.set_xlabel("Re")
            ax.set_ylabel("Δζ")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("Δζ(Re)")
        self.plot_dz.plot(plot_func)

    def _plot_r(self, df_cal, df_val, geom_w, geom_a, fit) -> None:
        from leakage_model.stage1_energy.model import calc_r_explicit

        def plot_func(fig, ax):
            u1_fine = np.linspace(2, 20, 100)
            r_fine_w = calc_r_explicit(u1_fine, geom_w, fit.dz_func)
            ax.plot(u1_fine, r_fine_w, "b-", lw=2, label="Расчёт (вода)")
            ax.plot(df_cal["u1"], df_cal["r"], "bo", ms=6,
                    label="Эксперимент (вода)")

            if df_val is not None and "r" in df_val.columns:
                u1_fine_a = np.linspace(2, 25, 100)
                r_fine_a = calc_r_explicit(u1_fine_a, geom_a, fit.dz_func)
                ax.plot(u1_fine_a, r_fine_a, "r--", lw=2,
                        label="Расчёт (воздух)")
                ax.plot(df_val["u1"], df_val["r"], "rs", ms=6,
                        label="Эксперимент (воздух)")

            ax.set_xlabel("u₁, м/с")
            ax.set_ylabel("r")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("r(u₁)")
        self.plot_r.plot(plot_func)

    def _plot_diagnostic(self, df_cal, df_val, geom_w, geom_a) -> None:
        from leakage_model.stage1_energy.model import calc_delta_zeta

        def plot_func(fig, ax):
            Re_w = df_cal["u1"].values * geom_w["D_h"] / geom_w["nu"]
            dz_w = calc_delta_zeta(df_cal["r"].values,
                                    geom_w["A_ok"], geom_w["A_s"])
            ax.plot(Re_w, dz_w, "bo", ms=6, label="Вода (σ=0,24)")

            if df_val is not None and "r" in df_val.columns:
                Re_a = df_val["u1"].values * geom_a["D_h"] / geom_a["nu"]
                dz_a = calc_delta_zeta(df_val["r"].values,
                                        geom_a["A_ok"], geom_a["A_s"])
                ax.plot(Re_a, dz_a, "rs", ms=6, label="Воздух (σ=0,40)")

            ax.set_xlabel("Re")
            ax.set_ylabel("Δζ")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("Диагностика: Δζ для обоих наборов")
        self.plot_diag.plot(plot_func)
