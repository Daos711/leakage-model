"""Вкладка 3: Импульсная модель C_M (этап 1.1)."""

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


class MomentumTab(BaseTab):
    """Вкладка импульсной модели C_M."""

    tab_name = "Импульсная модель C_M"

    def __init__(self, app_state: dict[str, Any], parent=None) -> None:
        super().__init__(app_state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        left = QVBoxLayout()

        ctrl = QGroupBox("Аппроксимация C_M(Re)")
        ctrl_layout = QFormLayout(ctrl)
        self.combo_variant = QComboBox()
        self.combo_variant.addItems(["Степенной закон", "Асимптотический"])
        ctrl_layout.addRow("Вариант:", self.combo_variant)
        left.addWidget(ctrl)

        self.btn_calc = QPushButton("Рассчитать")
        self.btn_calc.clicked.connect(self._run_calc)
        left.addWidget(self.btn_calc)

        metrics_group = QGroupBox("Статистика C_M")
        metrics_layout = QFormLayout(metrics_group)
        self._metric_labels: dict[str, QLabel] = {}
        for key, label in [("cm_mean", "Среднее C_M"),
                            ("cm_cv", "CV, %"),
                            ("r2_fit", "R² (аппрокс.)"),
                            ("rmse_cal", "RMSE (калибр.)"),
                            ("rmse_val", "RMSE (валид.)")]:
            lbl = QLabel("—")
            self._metric_labels[key] = lbl
            metrics_layout.addRow(label + ":", lbl)
        left.addWidget(metrics_group)
        left.addStretch()

        right = QVBoxLayout()
        top = QHBoxLayout()
        self.plot_cm = PlotWidget(figsize=(5, 4))
        self.plot_r = PlotWidget(figsize=(5, 4))
        top.addWidget(self.plot_cm)
        top.addWidget(self.plot_r)
        right.addLayout(top)
        self.cm_table = ResultsTable()
        right.addWidget(self.cm_table)

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
            from leakage_model.core.config import GEOM_AIR, GEOM_WATER, RHO
            from leakage_model.core.validation import compute_metrics
            from leakage_model.stage1_momentum.calibration import (
                compute_C_M_table,
                fit_C_M_asymptotic,
                fit_C_M_power,
            )
            from leakage_model.stage1_momentum.solver import solve_r_brent

            geom_w = GEOM_WATER
            cm_df = compute_C_M_table(df_cal, geom_w)

            Re_arr = cm_df["Re"].values
            CM_arr = cm_df["C_M"].values

            cm_mean = CM_arr.mean()
            cm_cv = (CM_arr.std() / cm_mean * 100) if cm_mean != 0 else 0
            self._metric_labels["cm_mean"].setText(fmt_comma(cm_mean, 3))
            self._metric_labels["cm_cv"].setText(fmt_comma(cm_cv, 1))

            variant = self.combo_variant.currentIndex()
            if variant == 0:
                fit = fit_C_M_power(Re_arr, CM_arr)
            else:
                fit = fit_C_M_asymptotic(Re_arr, CM_arr)

            self._metric_labels["r2_fit"].setText(fmt_comma(fit.R2, 4))

            # Предсказание r
            r_pred_cal = []
            for _, row in df_cal.iterrows():
                Q1 = row["Q"] if "Q" in df_cal.columns else row["u1"] * geom_w["A_ok"]
                cm_val = fit.cm_func(row["u1"] * geom_w["D_h"] / geom_w["nu"])
                r = solve_r_brent(Q1, geom_w, RHO, cm_val)
                r_pred_cal.append(r)
            r_pred_cal = np.array(r_pred_cal)
            m_cal = compute_metrics(df_cal["r"].values, r_pred_cal)
            self._metric_labels["rmse_cal"].setText(fmt_comma(m_cal.RMSE, 4))

            # Валидация
            geom_a = GEOM_AIR
            if df_val is not None and "r" in df_val.columns:
                r_pred_val = []
                for _, row in df_val.iterrows():
                    Q1 = row.get("Q", row["u1"] * geom_a["A_ok"])
                    cm_val = fit.cm_func(row["u1"] * geom_a["D_h"] / geom_a["nu"])
                    r = solve_r_brent(Q1, geom_a, RHO, cm_val)
                    r_pred_val.append(r)
                r_pred_val = np.array(r_pred_val)
                m_val = compute_metrics(df_val["r"].values, r_pred_val)
                self._metric_labels["rmse_val"].setText(
                    fmt_comma(m_val.RMSE, 4))

            # Графики
            self._plot_cm(Re_arr, CM_arr, cm_mean, fit)
            self._plot_r(df_cal, df_val, geom_w, geom_a, fit)

            # Таблица
            headers = list(cm_df.columns)
            rows = cm_df.values.tolist()
            self.cm_table.set_data(headers, rows)

        except Exception as e:
            logger.error(f"Ошибка C_M: {e}")
            QMessageBox.warning(self, "Ошибка", str(e))

    def _plot_cm(self, Re, CM, cm_mean, fit) -> None:
        def plot_func(fig, ax):
            ax.plot(Re, CM, "ko", ms=6, label="C_M (обратный расчёт)")
            ax.axhline(cm_mean, ls="--", color="blue", alpha=0.7,
                       label=f"Среднее = {cm_mean:.3f}".replace(".", ","))
            Re_fine = np.linspace(Re.min() * 0.8, Re.max() * 1.2, 200)
            CM_fine = fit.cm_func(Re_fine)
            ax.plot(Re_fine, CM_fine, "r-", lw=2,
                    label=f"{fit.name} (R²={fit.R2:.3f})".replace(".", ","))
            ax.set_xlabel("Re")
            ax.set_ylabel("C_M")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("C_M(Re)")
        self.plot_cm.plot(plot_func)

    def _plot_r(self, df_cal, df_val, geom_w, geom_a, fit) -> None:
        from leakage_model.core.config import RHO
        from leakage_model.stage1_momentum.solver import solve_r_brent

        def plot_func(fig, ax):
            u1_fine = np.linspace(2, 20, 50)
            r_fine = []
            for u1 in u1_fine:
                Q1 = u1 * geom_w["A_ok"]
                cm = fit.cm_func(u1 * geom_w["D_h"] / geom_w["nu"])
                try:
                    r = solve_r_brent(Q1, geom_w, RHO, cm)
                except Exception:
                    r = np.nan
                r_fine.append(r)
            ax.plot(u1_fine, r_fine, "b-", lw=2, label="Расчёт (вода)")
            ax.plot(df_cal["u1"], df_cal["r"], "bo", ms=6,
                    label="Эксперимент (вода)")

            if df_val is not None and "r" in df_val.columns:
                ax.plot(df_val["u1"], df_val["r"], "rs", ms=6,
                        label="Эксперимент (воздух)")

            ax.set_xlabel("u₁, м/с")
            ax.set_ylabel("r")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("r(u₁)")
        self.plot_r.plot(plot_func)
