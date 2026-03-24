"""Вкладка 5: Физическая модель ξ + C_β."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from PyQt6.QtCore import QRunnable, QObject, QThreadPool, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..utils.locale_fmt import fmt_comma, fmt_comma_auto
from .base_tab import BaseTab
from .parameter_panel import CommaDoubleSpinBox, ParameterPanel
from .plot_widget import PlotWidget
from .results_table import ResultsTable

logger = logging.getLogger("leakage_gui")


class _CalibrateSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str)


class _CalibrateWorker(QRunnable):
    """Калибровка в фоновом потоке."""

    def __init__(self, u1_cal, r_cal, geom, beta, L, eps,
                 u1_val=None, r_val=None, geom_val=None):
        super().__init__()
        self.signals = _CalibrateSignals()
        self.u1_cal = u1_cal
        self.r_cal = r_cal
        self.geom = geom
        self.beta = beta
        self.L = L
        self.eps = eps
        self.u1_val = u1_val
        self.r_val = r_val
        self.geom_val = geom_val

    def run(self):
        try:
            from leakage_model.stage3_physics.calibration import calibrate
            result = calibrate(
                self.u1_cal, self.r_cal, self.geom, self.beta,
                L_upper=self.L, eps=self.eps,
            )
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))


class PhysicsTab(BaseTab):
    """Вкладка физической модели ξ + C_β."""

    tab_name = "Физическая модель ξ + C_β"

    def __init__(self, app_state: dict[str, Any], parent=None) -> None:
        super().__init__(app_state, parent)
        self._build_ui()
        self._update_point_calc()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        # Левая панель
        left = QVBoxLayout()

        # Калибровочные параметры
        self.param_panel = ParameterPanel()
        self.param_panel.add_group("Калибровочные параметры", [
            {"key": "a_xi", "label": "a_ξ", "default": 38.51,
             "min": -100, "max": 100, "step": 0.01, "decimals": 3},
            {"key": "b_xi", "label": "b_ξ", "default": -2.664,
             "min": -10, "max": 10, "step": 0.001, "decimals": 4},
            {"key": "c0", "label": "c₀", "default": 1.983,
             "min": 0, "max": 20, "step": 0.01, "decimals": 4},
        ])
        left.addWidget(self.param_panel)

        btn_layout = QVBoxLayout()
        self.btn_calibrate_water = QPushButton("Калибровать (вода)")
        self.btn_calibrate_water.clicked.connect(self._calibrate_water)
        btn_layout.addWidget(self.btn_calibrate_water)

        self.btn_calibrate_joint = QPushButton("Калибровать (совместная)")
        self.btn_calibrate_joint.clicked.connect(self._calibrate_joint)
        btn_layout.addWidget(self.btn_calibrate_joint)

        btn_reset_w = QPushButton("Сбросить (водяная)")
        btn_reset_w.clicked.connect(lambda: self._reset_params("water"))
        btn_layout.addWidget(btn_reset_w)

        btn_reset_j = QPushButton("Сбросить (совместная)")
        btn_reset_j.clicked.connect(lambda: self._reset_params("joint"))
        btn_layout.addWidget(btn_reset_j)

        left.addLayout(btn_layout)

        # Расчёт в точке
        point_group = QGroupBox("Расчёт в точке")
        point_layout = QVBoxLayout(point_group)

        # Ползунок u1
        u1_row = QHBoxLayout()
        u1_row.addWidget(QLabel("u₁, м/с:"))
        self.u1_slider = QSlider(Qt.Orientation.Horizontal)
        self.u1_slider.setMinimum(1)
        self.u1_slider.setMaximum(500)
        self.u1_slider.setValue(125)
        self.u1_slider.valueChanged.connect(self._on_slider_changed)
        u1_row.addWidget(self.u1_slider)

        self.u1_spin = CommaDoubleSpinBox()
        self.u1_spin.setMinimum(0.1)
        self.u1_spin.setMaximum(50.0)
        self.u1_spin.setSingleStep(0.1)
        self.u1_spin.setDecimals(1)
        self.u1_spin.setValue(12.5)
        self.u1_spin.valueChanged.connect(self._on_spin_changed)
        u1_row.addWidget(self.u1_spin)
        point_layout.addLayout(u1_row)

        # Результаты в точке
        self._point_labels: dict[str, QLabel] = {}
        point_form = QFormLayout()
        for key, label in [
            ("Re", "Re"),
            ("xi", "ξ"),
            ("phi2", "φ₂"),
            ("phi3", "φ₃"),
            ("C_beta", "C_β"),
            ("r", "r"),
            ("k_ut", "k_ут"),
            ("Q2", "Q₂, м³/с"),
            ("Q3", "Q₃, м³/с"),
        ]:
            lbl = QLabel("—")
            lbl.setStyleSheet("font-weight: bold;")
            self._point_labels[key] = lbl
            point_form.addRow(label + ":", lbl)
        point_layout.addLayout(point_form)
        left.addWidget(point_group)

        # Метрики
        metrics_group = QGroupBox("Метрики")
        metrics_layout = QFormLayout(metrics_group)
        self._metric_labels: dict[str, QLabel] = {}
        for key, label in [
            ("rmse_cal", "RMSE (калибр.)"),
            ("r2_cal", "R² (калибр.)"),
            ("rmse_val", "RMSE (валид.)"),
            ("r2_val", "R² (валид.)"),
        ]:
            lbl = QLabel("—")
            self._metric_labels[key] = lbl
            metrics_layout.addRow(label + ":", lbl)
        left.addWidget(metrics_group)
        left.addStretch()

        # Правая панель: графики 2×2
        right = QVBoxLayout()
        self.plot_r = PlotWidget(figsize=(5, 3.5))
        self.plot_xi = PlotWidget(figsize=(5, 3.5))
        self.plot_phi = PlotWidget(figsize=(5, 3.5))
        self.plot_cb = PlotWidget(figsize=(5, 3.5))

        top_plots = QHBoxLayout()
        top_plots.addWidget(self.plot_r)
        top_plots.addWidget(self.plot_xi)
        bottom_plots = QHBoxLayout()
        bottom_plots.addWidget(self.plot_phi)
        bottom_plots.addWidget(self.plot_cb)
        right.addLayout(top_plots)
        right.addLayout(bottom_plots)

        # Таблица снизу
        self.detail_table = ResultsTable()
        right.addWidget(self.detail_table)

        splitter = QSplitter()
        left_w = QWidget()
        left_w.setLayout(left)
        right_w = QWidget()
        right_w.setLayout(right)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([280, 700])
        main_layout.addWidget(splitter)

    def _get_params(self) -> tuple[float, float, float]:
        return (
            self.param_panel.get_value("a_xi"),
            self.param_panel.get_value("b_xi"),
            self.param_panel.get_value("c0"),
        )

    def _get_geom(self) -> dict:
        geo_tab = self._get_geometry_tab()
        if geo_tab:
            return geo_tab.get_geometry_dict()
        from leakage_model.core.config import GEOM_WATER
        return dict(GEOM_WATER)

    def _get_geom_val(self) -> dict:
        from leakage_model.core.config import GEOM_AIR
        return dict(GEOM_AIR)

    def _get_beta(self) -> float:
        geo_tab = self._get_geometry_tab()
        if geo_tab:
            return geo_tab.get_beta_rad()
        from leakage_model.core.config import BETA_RAD
        return BETA_RAD

    def _get_L_eps(self) -> tuple[float, float]:
        geo_tab = self._get_geometry_tab()
        if geo_tab:
            return geo_tab.get_L_up(), geo_tab.get_eps()
        return 111.5, 0.002

    def _get_geometry_tab(self):
        return self.app_state.get("geometry_tab")

    def _on_slider_changed(self, val: int) -> None:
        u1 = val / 10.0
        self.u1_spin.blockSignals(True)
        self.u1_spin.setValue(u1)
        self.u1_spin.blockSignals(False)
        self._update_point_calc()

    def _on_spin_changed(self, val: float) -> None:
        self.u1_slider.blockSignals(True)
        self.u1_slider.setValue(int(val * 10))
        self.u1_slider.blockSignals(False)
        self._update_point_calc()

    def _update_point_calc(self) -> None:
        """Пересчёт в одной точке (мгновенный)."""
        u1 = self.u1_spin.value()
        a_xi, b_xi, c0 = self._get_params()
        geom = self._get_geom()
        beta = self._get_beta()
        L, eps = self._get_L_eps()

        try:
            from leakage_model.stage3_physics.model import solve_r
            r, phi2, phi3, xi, cb, converged, _ = solve_r(
                u1, geom, a_xi, b_xi, c0, beta,
                L_upper=L, eps=eps,
            )
            if not converged:
                self._set_point_label("r", "не сошёлся")
                return

            Re = u1 * geom["D_h"] / geom["nu"]
            k_ut = r / (1 - r) if r < 1 else float("inf")
            Q1 = u1 * geom["A_ok"]
            Q2 = r * Q1
            Q3 = (1 - r) * Q1

            self._set_point_label("Re", f"{Re:,.0f}".replace(",", " "))
            self._set_point_label("xi", fmt_comma(xi, 3))
            self._set_point_label("phi2", fmt_comma(phi2, 3))
            self._set_point_label("phi3", fmt_comma(phi3, 3))
            self._set_point_label("C_beta", fmt_comma(cb, 3))
            self._set_point_label("r", fmt_comma(r, 4))
            self._set_point_label("k_ut", fmt_comma(k_ut, 3))
            self._set_point_label("Q2", fmt_comma(Q2, 2))
            self._set_point_label("Q3", fmt_comma(Q3, 2))
        except Exception as e:
            logger.error(f"Ошибка расчёта в точке: {e}")
            self._set_point_label("r", "ошибка")

    def _set_point_label(self, key: str, text: str) -> None:
        if key in self._point_labels:
            self._point_labels[key].setText(text)

    def update_plots(self) -> None:
        """Обновить все 4 графика."""
        a_xi, b_xi, c0 = self._get_params()
        geom_cal = self._get_geom()
        geom_val = self._get_geom_val()
        beta = self._get_beta()
        L, eps = self._get_L_eps()

        df_cal = self.app_state.get("cal_df")
        df_val = self.app_state.get("val_df")

        try:
            from leakage_model.stage3_physics.model import solve_all
            from leakage_model.core.validation import compute_metrics

            u1_fine_cal = np.linspace(2, 20, 100)
            res_cal = solve_all(u1_fine_cal, geom_cal, a_xi, b_xi, c0, beta,
                                L_upper=L, eps=eps)
            u1_fine_val = np.linspace(2, 25, 100)
            res_val = solve_all(u1_fine_val, geom_val, a_xi, b_xi, c0, beta,
                                L_upper=L, eps=eps)

            # Метрики
            if df_cal is not None and "r" in df_cal.columns:
                res_pts_cal = solve_all(
                    df_cal["u1"].values, geom_cal, a_xi, b_xi, c0, beta,
                    L_upper=L, eps=eps)
                m_cal = compute_metrics(df_cal["r"].values, res_pts_cal.r_pred)
                self._metric_labels["rmse_cal"].setText(fmt_comma(m_cal.RMSE, 4))
                self._metric_labels["r2_cal"].setText(fmt_comma(m_cal.R2, 4))

            if df_val is not None and "r" in df_val.columns:
                res_pts_val = solve_all(
                    df_val["u1"].values, geom_val, a_xi, b_xi, c0, beta,
                    L_upper=L, eps=eps)
                m_val = compute_metrics(df_val["r"].values, res_pts_val.r_pred)
                self._metric_labels["rmse_val"].setText(fmt_comma(m_val.RMSE, 4))
                self._metric_labels["r2_val"].setText(fmt_comma(m_val.R2, 4))

        except Exception as e:
            logger.error(f"Ошибка расчёта: {e}")
            return

        # График 1: r(u₁)
        def plot_r(fig, ax):
            ax.plot(res_cal.u1, res_cal.r_pred, "b-", lw=2,
                    label="Расчёт (A_ок=12)")
            ax.plot(res_val.u1, res_val.r_pred, "r--", lw=2,
                    label="Расчёт (A_ок=20)")
            if df_cal is not None:
                ax.plot(df_cal["u1"], df_cal["r"], "bo", ms=6,
                        label="Эксперимент (вода)")
            if df_val is not None:
                ax.plot(df_val["u1"], df_val["r"], "rs", ms=6,
                        label="Эксперимент (воздух)")
            ax.set_xlabel("u₁, м/с")
            ax.set_ylabel("r")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("Доля утечек r(u₁)")

        self.plot_r.plot(plot_r)

        # График 2: ξ(u₁)
        def plot_xi(fig, ax):
            ax.plot(res_cal.u1, res_cal.xi, "b-", lw=2, label="σ = 0,24")
            ax.plot(res_val.u1, res_val.xi, "r--", lw=2, label="σ = 0,40")
            ax.set_xlabel("u₁, м/с")
            ax.set_ylabel("ξ")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("Параметр блокировки ξ(u₁)")

        self.plot_xi.plot(plot_xi)

        # График 3: φ₂, φ₃
        def plot_phi(fig, ax):
            ax.plot(res_cal.u1, res_cal.phi_up, "b-", lw=2,
                    label="φ₂ (σ=0,24)")
            ax.plot(res_cal.u1, res_cal.phi_down, "b--", lw=2,
                    label="φ₃ (σ=0,24)")
            ax.plot(res_val.u1, res_val.phi_up, "r-", lw=1.5,
                    label="φ₂ (σ=0,40)")
            ax.plot(res_val.u1, res_val.phi_down, "r--", lw=1.5,
                    label="φ₃ (σ=0,40)")
            ax.set_xlabel("u₁, м/с")
            ax.set_ylabel("φ")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("Коэффициенты сужения φ₂, φ₃")

        self.plot_phi.plot(plot_phi)

        # График 4: C_β(u₁)
        def plot_cb(fig, ax):
            ax.plot(res_cal.u1, res_cal.C_beta, "b-", lw=2,
                    label="σ = 0,24")
            ax.plot(res_val.u1, res_val.C_beta, "r--", lw=2,
                    label="σ = 0,40")
            ax.set_xlabel("u₁, м/с")
            ax.set_ylabel("C_β")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("Асимметричный член C_β(u₁)")

        self.plot_cb.plot(plot_cb)

        # Таблица
        self._update_detail_table(geom_cal, a_xi, b_xi, c0, beta, L, eps)

    def _update_detail_table(self, geom, a_xi, b_xi, c0, beta, L, eps) -> None:
        """Таблица детальных результатов."""
        df_cal = self.app_state.get("cal_df")
        if df_cal is None:
            return
        try:
            from leakage_model.stage3_physics.model import solve_all
            u1_arr = df_cal["u1"].values
            res = solve_all(u1_arr, geom, a_xi, b_xi, c0, beta,
                            L_upper=L, eps=eps)
            headers = ["u₁", "Re", "ξ", "φ₂", "φ₃", "C_β",
                       "r_эксп", "r_расч"]
            rows = []
            for i, u1 in enumerate(u1_arr):
                Re = u1 * geom["D_h"] / geom["nu"]
                r_exp = df_cal["r"].values[i] if "r" in df_cal.columns else 0
                rows.append([
                    u1, Re, res.xi[i], res.phi_up[i], res.phi_down[i],
                    res.C_beta[i], r_exp, res.r_pred[i],
                ])
            self.detail_table.set_data(headers, rows)
        except Exception as e:
            logger.error(f"Ошибка таблицы: {e}")

    def _calibrate_water(self) -> None:
        self._run_calibration(joint=False)

    def _calibrate_joint(self) -> None:
        self._run_calibration(joint=True)

    def _run_calibration(self, joint: bool = False) -> None:
        df_cal = self.app_state.get("cal_df")
        if df_cal is None or "u1" not in df_cal.columns:
            QMessageBox.warning(self, "Нет данных",
                                "Загрузите калибровочные данные.")
            return

        u1_cal = df_cal["u1"].values
        r_cal = df_cal["r"].values
        geom = self._get_geom()
        beta = self._get_beta()
        L, eps = self._get_L_eps()

        if joint:
            df_val = self.app_state.get("val_df")
            if df_val is not None and "r" in df_val.columns:
                geom_val = self._get_geom_val()
                u1_cal = np.concatenate([u1_cal, df_val["u1"].values])
                r_cal = np.concatenate([r_cal, df_val["r"].values])
                # Для совместной нужен единый geom — используем водяную
                # (backend calibrate принимает один geom)

        self.btn_calibrate_water.setEnabled(False)
        self.btn_calibrate_joint.setEnabled(False)
        self.btn_calibrate_water.setText("Калибровка...")

        worker = _CalibrateWorker(u1_cal, r_cal, geom, beta, L, eps)
        worker.signals.result.connect(self._on_calibration_done)
        worker.signals.error.connect(self._on_calibration_error)
        QThreadPool.globalInstance().start(worker)

    def _on_calibration_done(self, result) -> None:
        a_xi, b_xi, c0, metrics, phys_result = result
        self.param_panel.set_value("a_xi", a_xi)
        self.param_panel.set_value("b_xi", b_xi)
        self.param_panel.set_value("c0", c0)

        self.app_state["calibration"] = {
            "a_xi": a_xi, "b_xi": b_xi, "c0": c0,
        }

        self.btn_calibrate_water.setEnabled(True)
        self.btn_calibrate_joint.setEnabled(True)
        self.btn_calibrate_water.setText("Калибровать (вода)")

        logger.info(f"Калибровка: a_xi={a_xi:.3f}, b_xi={b_xi:.4f}, "
                     f"c0={c0:.4f}, RMSE={metrics.RMSE:.4f}")
        self.update_plots()
        self._update_point_calc()

    def _on_calibration_error(self, msg: str) -> None:
        self.btn_calibrate_water.setEnabled(True)
        self.btn_calibrate_joint.setEnabled(True)
        self.btn_calibrate_water.setText("Калибровать (вода)")
        logger.error(f"Ошибка калибровки: {msg}")
        QMessageBox.warning(self, "Ошибка калибровки",
                            f"Калибровка не сошлась:\n{msg}")

    def _reset_params(self, variant: str) -> None:
        if variant == "water":
            self.param_panel.set_value("a_xi", 38.51)
            self.param_panel.set_value("b_xi", -2.664)
            self.param_panel.set_value("c0", 1.983)
        else:
            self.param_panel.set_value("a_xi", 26.14)
            self.param_panel.set_value("b_xi", -1.803)
            self.param_panel.set_value("c0", 2.192)
        self.app_state["calibration"] = {
            "a_xi": self.param_panel.get_value("a_xi"),
            "b_xi": self.param_panel.get_value("b_xi"),
            "c0": self.param_panel.get_value("c0"),
        }
        self._update_point_calc()

    def get_state(self) -> dict[str, Any]:
        return {
            "calibration_params": {
                "a_xi": self.param_panel.get_value("a_xi"),
                "b_xi": self.param_panel.get_value("b_xi"),
                "c0": self.param_panel.get_value("c0"),
            },
            "u1": self.u1_spin.value(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if "calibration_params" in state:
            self.param_panel.set_all_values(state["calibration_params"])
        if "u1" in state:
            self.u1_spin.setValue(state["u1"])
