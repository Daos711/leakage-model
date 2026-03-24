"""Вкладка 6: Модель с пластинами."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..utils.locale_fmt import fmt_comma
from .base_tab import BaseTab
from .parameter_panel import CommaDoubleSpinBox, ParameterPanel
from .plot_widget import PlotWidget
from .results_table import ResultsTable

logger = logging.getLogger("leakage_gui")


class _CalibrateAllSignals(QObject):
    progress = pyqtSignal(int)
    result = pyqtSignal(object)
    error = pyqtSignal(str)


class _CalibrateAllWorker(QRunnable):
    """Калибровка всех вставок в фоновом потоке."""

    def __init__(self, plates_df, geom, base_params, beta, L, eps):
        super().__init__()
        self.signals = _CalibrateAllSignals()
        self.plates_df = plates_df
        self.geom = geom
        self.base_params = base_params
        self.beta = beta
        self.L = L
        self.eps = eps

    def run(self):
        try:
            from leakage_model.stage4_plates.calibration import calibrate_all
            results = calibrate_all(
                self.plates_df, self.geom, self.base_params,
                self.beta, self.L, self.eps,
            )
            self.signals.result.emit(results)
        except Exception as e:
            self.signals.error.emit(str(e))


class PlatesTab(BaseTab):
    """Вкладка модели с направляющими пластинами."""

    tab_name = "Модель с пластинами"

    def __init__(self, app_state: dict[str, Any], parent=None) -> None:
        super().__init__(app_state, parent)
        self._cal_results: list[dict] = []
        self._plates_df: pd.DataFrame | None = None
        self._surrogates: dict = {}
        self._build_ui()
        self._load_plates_data()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        # Левая панель управления
        left = QVBoxLayout()

        ctrl_group = QGroupBox("Управление")
        ctrl_layout = QVBoxLayout(ctrl_group)

        form = QFormLayout()
        self.combo_series = QComboBox()
        self.combo_series.addItems(["Все", "Серия 1", "Серия 2",
                                     "Серия 3", "Серия 4"])
        form.addRow("Серия:", self.combo_series)

        self.combo_model = QComboBox()
        self.combo_model.addItems(["M3 (ζ_пл + Δc₀)", "M1 (ζ_пл)", "M2 (Δc₀)"])
        form.addRow("Модель:", self.combo_model)
        ctrl_layout.addLayout(form)

        self.btn_calibrate = QPushButton("Калибровать все вставки")
        self.btn_calibrate.clicked.connect(self._run_calibration)
        ctrl_layout.addWidget(self.btn_calibrate)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ctrl_layout.addWidget(self.progress)

        self.btn_surrogates = QPushButton("Построить суррогаты")
        self.btn_surrogates.clicked.connect(self._build_surrogates)
        self.btn_surrogates.setEnabled(False)
        ctrl_layout.addWidget(self.btn_surrogates)

        left.addWidget(ctrl_group)

        # Расчёт в точке
        point_group = QGroupBox("Расчёт в точке")
        point_layout = QFormLayout(point_group)

        self.spin_alpha = CommaDoubleSpinBox()
        self.spin_alpha.setRange(5, 85)
        self.spin_alpha.setValue(45)
        self.spin_alpha.setSingleStep(1)
        point_layout.addRow("α, °:", self.spin_alpha)

        self.spin_b = CommaDoubleSpinBox()
        self.spin_b.setRange(50, 2000)
        self.spin_b.setValue(1000)
        self.spin_b.setSingleStep(10)
        self.spin_b.setDecimals(0)
        point_layout.addRow("b, мм:", self.spin_b)

        self.spin_u1_pt = CommaDoubleSpinBox()
        self.spin_u1_pt.setRange(0.1, 50)
        self.spin_u1_pt.setValue(12.5)
        self.spin_u1_pt.setSingleStep(0.1)
        self.spin_u1_pt.setDecimals(1)
        point_layout.addRow("u₁, м/с:", self.spin_u1_pt)

        btn_calc_pt = QPushButton("Рассчитать")
        btn_calc_pt.clicked.connect(self._calc_point)
        point_layout.addRow(btn_calc_pt)

        self._pt_labels: dict[str, QLabel] = {}
        for key, label in [("zeta_pl", "ζ_пл"), ("delta_c0", "Δc₀"),
                            ("r", "r"), ("k_ut", "k_ут")]:
            lbl = QLabel("—")
            self._pt_labels[key] = lbl
            point_layout.addRow(label + ":", lbl)

        left.addWidget(point_group)
        left.addStretch()

        # Правая панель: подвкладки
        right = QVBoxLayout()
        self.sub_tabs = QTabWidget()

        # Подвкладка: Сравнение M1/M2/M3
        self.comparison_widget = QWidget()
        comp_layout = QVBoxLayout(self.comparison_widget)
        self.plot_rmse_bars = PlotWidget(figsize=(8, 4))
        comp_layout.addWidget(self.plot_rmse_bars)
        self.metrics_table = ResultsTable()
        comp_layout.addWidget(self.metrics_table)
        self.sub_tabs.addTab(self.comparison_widget, "Сравнение M1/M2/M3")

        # Подвкладка: Параметры по вставкам
        self.params_widget = QWidget()
        params_layout = QVBoxLayout(self.params_widget)
        self.plot_zeta_inserts = PlotWidget(figsize=(8, 3))
        self.plot_dc0_inserts = PlotWidget(figsize=(8, 3))
        params_layout.addWidget(self.plot_zeta_inserts)
        params_layout.addWidget(self.plot_dc0_inserts)
        self.sub_tabs.addTab(self.params_widget, "Параметры по вставкам")

        # Подвкладка: Серия 3 (угол)
        self.series3_widget = QWidget()
        s3_layout = QVBoxLayout(self.series3_widget)
        self.plot_series3 = PlotWidget(figsize=(8, 5))
        s3_layout.addWidget(self.plot_series3)
        self.table_series3 = ResultsTable()
        s3_layout.addWidget(self.table_series3)
        self.sub_tabs.addTab(self.series3_widget, "Серия 3 (угол)")

        # Подвкладка: Серия 4 (ширина)
        self.series4_widget = QWidget()
        s4_layout = QVBoxLayout(self.series4_widget)
        self.plot_series4 = PlotWidget(figsize=(8, 5))
        s4_layout.addWidget(self.plot_series4)
        self.table_series4 = ResultsTable()
        s4_layout.addWidget(self.table_series4)
        self.sub_tabs.addTab(self.series4_widget, "Серия 4 (ширина)")

        right.addWidget(self.sub_tabs)

        splitter = QSplitter()
        left_w = QWidget()
        left_w.setLayout(left)
        right_w = QWidget()
        right_w.setLayout(right)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([280, 700])
        main_layout.addWidget(splitter)

    def _load_plates_data(self) -> None:
        """Загрузка экспериментальных данных по пластинам."""
        try:
            from leakage_model.stage4_plates.data import load_plates_with_geometry
            self._plates_df = load_plates_with_geometry()
            self.app_state["plates_df"] = self._plates_df
            logger.info(f"Загружены данные по пластинам: "
                        f"{len(self._plates_df)} строк")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных по пластинам: {e}")

    def _get_base_params(self) -> tuple[float, float, float]:
        cal = self.app_state.get("calibration", {})
        return (
            cal.get("a_xi", 38.51),
            cal.get("b_xi", -2.664),
            cal.get("c0", 1.983),
        )

    def _get_geom(self) -> dict:
        geo_tab = self.app_state.get("geometry_tab")
        if geo_tab:
            return geo_tab.get_geometry_dict()
        from leakage_model.core.config import GEOM_WATER
        return dict(GEOM_WATER)

    def _get_beta_L_eps(self) -> tuple[float, float, float]:
        geo_tab = self.app_state.get("geometry_tab")
        if geo_tab:
            return geo_tab.get_beta_rad(), geo_tab.get_L_up(), geo_tab.get_eps()
        import math
        return math.radians(45), 111.5, 0.002

    def _run_calibration(self) -> None:
        if self._plates_df is None:
            QMessageBox.warning(self, "Нет данных",
                                "Данные по пластинам не загружены.")
            return

        base_params = self._get_base_params()
        geom = self._get_geom()
        beta, L, eps = self._get_beta_L_eps()

        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setText("Калибровка...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate

        worker = _CalibrateAllWorker(
            self._plates_df, geom, base_params, beta, L, eps)
        worker.signals.result.connect(self._on_calibration_done)
        worker.signals.error.connect(self._on_calibration_error)
        QThreadPool.globalInstance().start(worker)

    def _on_calibration_done(self, results: list[dict]) -> None:
        self._cal_results = results
        self.app_state["plates_cal_results"] = results
        self.btn_calibrate.setEnabled(True)
        self.btn_calibrate.setText("Калибровать все вставки")
        self.progress.setVisible(False)
        self.btn_surrogates.setEnabled(True)

        logger.info(f"Калибровка пластин: {len(results)} вставок")
        self._update_comparison_plots()
        self._update_params_plots()

    def _on_calibration_error(self, msg: str) -> None:
        self.btn_calibrate.setEnabled(True)
        self.btn_calibrate.setText("Калибровать все вставки")
        self.progress.setVisible(False)
        logger.error(f"Ошибка калибровки пластин: {msg}")
        QMessageBox.warning(self, "Ошибка", f"Калибровка не удалась:\n{msg}")

    def _update_comparison_plots(self) -> None:
        """Столбчатая диаграмма RMSE и таблица метрик."""
        if not self._cal_results:
            return

        df = pd.DataFrame(self._cal_results)

        # Столбчатая диаграмма RMSE
        def plot_bars(fig, ax):
            x = np.arange(len(df))
            w = 0.25
            ax.bar(x - w, df["RMSE_M1"], w, label="M1", color="#4C72B0")
            ax.bar(x, df["RMSE_M2"], w, label="M2", color="#DD8452")
            ax.bar(x + w, df["RMSE_M3"], w, label="M3", color="#55A868")
            ax.set_xticks(x)
            ax.set_xticklabels(df["insert_id"].astype(str), rotation=90,
                               fontsize=7)
            ax.set_xlabel("Вставка")
            ax.set_ylabel("RMSE")
            ax.legend()
            ax.set_title("RMSE по вставкам")
            ax.grid(axis="y", alpha=0.3)

        self.plot_rmse_bars.plot(plot_bars)

        # Таблица
        headers = ["Вставка", "Название", "ζ_пл (M3)", "Δc₀ (M3)",
                    "RMSE M1", "RMSE M2", "RMSE M3"]
        rows = []
        for r in self._cal_results:
            rows.append([
                r["insert_id"], r["insert_name"],
                r.get("zeta_pl_M3", 0), r.get("delta_c0_M3", 0),
                r.get("RMSE_M1", 0), r.get("RMSE_M2", 0),
                r.get("RMSE_M3", 0),
            ])
        self.metrics_table.set_data(headers, rows)

    def _update_params_plots(self) -> None:
        """Графики ζ_пл и Δc₀ по вставкам."""
        if not self._cal_results:
            return
        df = pd.DataFrame(self._cal_results)

        def plot_zeta(fig, ax):
            ax.bar(range(len(df)), df["zeta_pl_M3"], color="#4C72B0")
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df["insert_id"].astype(str), rotation=90,
                               fontsize=7)
            ax.set_ylabel("ζ_пл")
            ax.set_title("Сопротивление пластин ζ_пл (M3)")
            ax.grid(axis="y", alpha=0.3)

        self.plot_zeta_inserts.plot(plot_zeta)

        def plot_dc0(fig, ax):
            colors = ["#DD3333" if v < 0 else "#4C72B0"
                      for v in df["delta_c0_M3"]]
            ax.bar(range(len(df)), df["delta_c0_M3"], color=colors)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df["insert_id"].astype(str), rotation=90,
                               fontsize=7)
            ax.set_ylabel("Δc₀")
            ax.set_title("Коррекция асимметрии Δc₀ (M3)")
            ax.axhline(0, color="black", lw=0.5)
            ax.grid(axis="y", alpha=0.3)

        self.plot_dc0_inserts.plot(plot_dc0)

    def _build_surrogates(self) -> None:
        """Построить суррогатные модели по сериям 3 и 4."""
        if not self._cal_results or self._plates_df is None:
            return

        try:
            from leakage_model.stage4_plates.surrogates import (
                fit_series3_angle,
                fit_series4_width,
            )

            df_res = pd.DataFrame(self._cal_results)
            df_merged = self._plates_df.drop_duplicates("insert_id").merge(
                df_res, on="insert_id", how="inner",
                suffixes=("", "_cal"))

            # Серия 3
            s3 = df_merged[df_merged["series_id"] == 3].copy()
            if len(s3) >= 3:
                surr3 = fit_series3_angle(
                    s3["angle_deg"].values,
                    s3["zeta_pl_M3"].values,
                    s3["delta_c0_M3"].values,
                )
                self._surrogates[3] = surr3
                self._plot_series3(s3, surr3)

            # Серия 4
            s4 = df_merged[df_merged["series_id"] == 4].copy()
            if len(s4) >= 2:
                surr4 = fit_series4_width(
                    s4["width_mm"].values,
                    s4["zeta_pl_M3"].values,
                    s4["delta_c0_M3"].values,
                )
                self._surrogates[4] = surr4
                self._plot_series4(s4, surr4)

            self.app_state["surrogates"] = self._surrogates
            logger.info("Суррогатные модели построены")

        except Exception as e:
            logger.error(f"Ошибка построения суррогатов: {e}")
            QMessageBox.warning(self, "Ошибка",
                                f"Не удалось построить суррогаты:\n{e}")

    def _plot_series3(self, s3_df, surr3) -> None:
        from leakage_model.stage4_plates.surrogates import predict_series3

        def plot_s3(fig, ax):
            angles = s3_df["angle_deg"].values
            ax.plot(angles, s3_df["zeta_pl_M3"], "bo", ms=8,
                    label="ζ_пл (данные)")
            a_fine = np.linspace(20, 65, 100)
            z_fine = [predict_series3(a, surr3)[0] for a in a_fine]
            ax.plot(a_fine, z_fine, "b-", lw=2, label="ζ_пл (регрессия)")

            ax2 = ax.twinx()
            ax2.plot(angles, s3_df["delta_c0_M3"], "rs", ms=8,
                     label="Δc₀ (данные)")
            dc_fine = [predict_series3(a, surr3)[1] for a in a_fine]
            ax2.plot(a_fine, dc_fine, "r--", lw=2, label="Δc₀ (регрессия)")
            ax2.set_ylabel("Δc₀")
            from ..utils.locale_fmt import apply_comma_format
            apply_comma_format(ax2)

            ax.set_xlabel("Угол α, °")
            ax.set_ylabel("ζ_пл")
            ax.set_title("Серия 3: зависимость от угла")

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)

        self.plot_series3.plot(plot_s3)

        # Таблица
        headers = ["α, °", "ζ_пл", "Δc₀", "RMSE M3"]
        rows = [[r["angle_deg"], r["zeta_pl_M3"], r["delta_c0_M3"],
                 r["RMSE_M3"]]
                for _, r in s3_df.iterrows()]
        self.table_series3.set_data(headers, rows)

    def _plot_series4(self, s4_df, surr4) -> None:
        from leakage_model.stage4_plates.surrogates import predict_series4

        def plot_s4(fig, ax):
            widths = s4_df["width_mm"].values
            ax.plot(widths, s4_df["zeta_pl_M3"], "bo", ms=8,
                    label="ζ_пл (данные)")
            b_fine = np.linspace(100, 1500, 100)
            z_fine = [predict_series4(b, surr4)[0] for b in b_fine]
            ax.plot(b_fine, z_fine, "b-", lw=2, label="ζ_пл (регрессия)")

            ax2 = ax.twinx()
            ax2.plot(widths, s4_df["delta_c0_M3"], "rs", ms=8,
                     label="Δc₀ (данные)")
            dc_fine = [predict_series4(b, surr4)[1] for b in b_fine]
            ax2.plot(b_fine, dc_fine, "r--", lw=2, label="Δc₀ (регрессия)")
            ax2.set_ylabel("Δc₀")
            from ..utils.locale_fmt import apply_comma_format
            apply_comma_format(ax2)

            ax.set_xlabel("Ширина b, мм")
            ax.set_ylabel("ζ_пл")
            ax.set_title("Серия 4: зависимость от ширины")

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)

        self.plot_series4.plot(plot_s4)

        headers = ["b, мм", "ζ_пл", "Δc₀", "RMSE M3"]
        rows = [[r["width_mm"], r["zeta_pl_M3"], r["delta_c0_M3"],
                 r["RMSE_M3"]]
                for _, r in s4_df.iterrows()]
        self.table_series4.set_data(headers, rows)

    def _calc_point(self) -> None:
        """Расчёт в одной точке с суррогатами."""
        alpha = self.spin_alpha.value()
        width = self.spin_b.value()
        u1 = self.spin_u1_pt.value()

        surr3 = self._surrogates.get(3)
        if surr3 is None:
            QMessageBox.warning(self, "Нет суррогатов",
                                "Сначала постройте суррогатные модели.")
            return

        try:
            from leakage_model.stage4_plates.surrogates import predict_series3
            from leakage_model.stage4_plates.model import solve_r_plates

            zeta_pl, delta_c0 = predict_series3(alpha, surr3)
            base_params = self._get_base_params()
            geom = self._get_geom()
            beta, L, eps = self._get_beta_L_eps()

            r, converged, _ = solve_r_plates(
                u1, geom, *base_params, beta,
                L_upper=L, eps=eps,
                zeta_pl=zeta_pl, delta_c0=delta_c0,
            )

            if converged:
                k_ut = r / (1 - r) if r < 1 else float("inf")
                self._pt_labels["zeta_pl"].setText(fmt_comma(zeta_pl, 4))
                self._pt_labels["delta_c0"].setText(fmt_comma(delta_c0, 4))
                self._pt_labels["r"].setText(fmt_comma(r, 4))
                self._pt_labels["k_ut"].setText(fmt_comma(k_ut, 3))
            else:
                self._pt_labels["r"].setText("не сошёлся")

        except Exception as e:
            logger.error(f"Ошибка расчёта в точке: {e}")
            QMessageBox.warning(self, "Ошибка", str(e))

    def get_state(self) -> dict[str, Any]:
        return {
            "cal_results": self._cal_results,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if "cal_results" in state and state["cal_results"]:
            self._cal_results = state["cal_results"]
            self._update_comparison_plots()
            self._update_params_plots()
