"""Вкладка 7: Оптимизация пластин."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..utils.locale_fmt import fmt_comma
from .base_tab import BaseTab
from .parameter_panel import CommaDoubleSpinBox
from .plot_widget import PlotWidget
from .results_table import ResultsTable

logger = logging.getLogger("leakage_gui")


class _OptSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str)


class _OptWorker(QRunnable):
    def __init__(self, opt_type, surrogates, geom, base_params,
                 beta, L, eps, u1=12.5, alpha_fixed=45.0,
                 width_fixed=1000):
        super().__init__()
        self.signals = _OptSignals()
        self.opt_type = opt_type
        self.surrogates = surrogates
        self.geom = geom
        self.base_params = base_params
        self.beta = beta
        self.L = L
        self.eps = eps
        self.u1 = u1
        self.alpha_fixed = alpha_fixed
        self.width_fixed = width_fixed

    def run(self):
        try:
            from leakage_model.stage4_plates.optimization import (
                optimize_angle,
                optimize_joint,
                optimize_multi_speed,
                optimize_width,
            )

            if self.opt_type == "angle":
                result = optimize_angle(
                    self.surrogates, self.u1, self.geom,
                    self.base_params, self.beta, self.L, self.eps)
                self.signals.result.emit(("angle", result))
            elif self.opt_type == "width":
                result = optimize_width(
                    self.surrogates, self.u1, self.geom,
                    self.base_params, self.beta, self.L, self.eps)
                self.signals.result.emit(("width", result))
            elif self.opt_type == "joint":
                result = optimize_joint(
                    self.surrogates, self.u1, self.geom,
                    self.base_params, self.beta, self.L, self.eps)
                self.signals.result.emit(("joint", result))
            elif self.opt_type == "multi":
                result = optimize_multi_speed(
                    self.surrogates, self.geom,
                    self.base_params, self.beta, self.L, self.eps)
                self.signals.result.emit(("multi", result))
        except Exception as e:
            self.signals.error.emit(str(e))


class OptimizationTab(BaseTab):
    """Вкладка оптимизации пластин."""

    tab_name = "Оптимизация пластин"

    def __init__(self, app_state: dict[str, Any], parent=None) -> None:
        super().__init__(app_state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        # Левая панель
        left = QVBoxLayout()

        # Параметры оптимизации
        opt_group = QGroupBox("Параметры оптимизации")
        opt_layout = QFormLayout(opt_group)

        self.combo_type = QComboBox()
        self.combo_type.addItems([
            "По углу α", "По ширине b",
            "Совместная (α, b)", "Мультискоростная",
        ])
        self.combo_type.currentIndexChanged.connect(self._on_type_changed)
        opt_layout.addRow("Тип:", self.combo_type)

        self.spin_u1 = CommaDoubleSpinBox()
        self.spin_u1.setRange(0.1, 50)
        self.spin_u1.setValue(12.5)
        self.spin_u1.setDecimals(1)
        opt_layout.addRow("u₁, м/с:", self.spin_u1)

        self.spin_alpha_fix = CommaDoubleSpinBox()
        self.spin_alpha_fix.setRange(5, 85)
        self.spin_alpha_fix.setValue(45)
        self.spin_alpha_fix.setDecimals(1)
        opt_layout.addRow("α фикс., °:", self.spin_alpha_fix)

        self.spin_b_fix = CommaDoubleSpinBox()
        self.spin_b_fix.setRange(50, 2000)
        self.spin_b_fix.setValue(1000)
        self.spin_b_fix.setDecimals(0)
        opt_layout.addRow("b фикс., мм:", self.spin_b_fix)

        # Калибровка
        cal_group = QGroupBox("Калибровка")
        cal_layout = QVBoxLayout(cal_group)
        self.radio_water = QRadioButton("Водяная")
        self.radio_water.setChecked(True)
        self.radio_joint = QRadioButton("Совместная")
        cal_layout.addWidget(self.radio_water)
        cal_layout.addWidget(self.radio_joint)
        self.radio_water.toggled.connect(self._on_calibration_changed)
        opt_layout.addRow(cal_group)

        left.addWidget(opt_group)

        self.btn_optimize = QPushButton("Оптимизировать")
        self.btn_optimize.clicked.connect(self._run_optimization)
        left.addWidget(self.btn_optimize)

        # Результаты
        res_group = QGroupBox("Результат")
        res_layout = QFormLayout(res_group)
        self._res_labels: dict[str, QLabel] = {}
        for key, label in [
            ("alpha_opt", "α*, °"),
            ("b_opt", "b*, мм"),
            ("r_opt", "r*"),
            ("r_base", "r без пластин"),
            ("reduction", "Снижение утечек, %"),
        ]:
            lbl = QLabel("—")
            lbl.setStyleSheet("font-weight: bold;")
            self._res_labels[key] = lbl
            res_layout.addRow(label + ":", lbl)
        left.addWidget(res_group)
        left.addStretch()

        # Правая панель: графики
        right = QVBoxLayout()
        self.plot_main = PlotWidget(figsize=(8, 5))
        right.addWidget(self.plot_main)
        self.result_table = ResultsTable()
        right.addWidget(self.result_table)

        splitter = QSplitter()
        left_w = QWidget()
        left_w.setLayout(left)
        right_w = QWidget()
        right_w.setLayout(right)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([280, 700])
        main_layout.addWidget(splitter)

        self._on_type_changed(0)

    def _on_type_changed(self, idx: int) -> None:
        is_angle = idx == 0
        is_width = idx == 1
        is_multi = idx == 3
        self.spin_u1.setEnabled(not is_multi)
        self.spin_alpha_fix.setEnabled(is_width)
        self.spin_b_fix.setEnabled(is_angle)

    def _on_calibration_changed(self) -> None:
        if self.radio_water.isChecked():
            self.app_state.setdefault("calibration", {}).update({
                "a_xi": 38.51, "b_xi": -2.664, "c0": 1.983,
            })
        else:
            self.app_state.setdefault("calibration", {}).update({
                "a_xi": 26.14, "b_xi": -1.803, "c0": 2.192,
            })

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

    def _run_optimization(self) -> None:
        surrogates = self.app_state.get("surrogates", {})
        if not surrogates or 3 not in surrogates:
            QMessageBox.warning(
                self, "Нет суррогатов",
                "Сначала постройте суррогатные модели на вкладке «Пластины».")
            return

        opt_types = ["angle", "width", "joint", "multi"]
        opt_type = opt_types[self.combo_type.currentIndex()]

        geom = self._get_geom()
        base_params = self._get_base_params()
        beta, L, eps = self._get_beta_L_eps()

        self.btn_optimize.setEnabled(False)
        self.btn_optimize.setText("Оптимизация...")

        worker = _OptWorker(
            opt_type, surrogates, geom, base_params,
            beta, L, eps,
            u1=self.spin_u1.value(),
            alpha_fixed=self.spin_alpha_fix.value(),
            width_fixed=self.spin_b_fix.value(),
        )
        worker.signals.result.connect(self._on_opt_done)
        worker.signals.error.connect(self._on_opt_error)
        QThreadPool.globalInstance().start(worker)

    def _on_opt_done(self, result_tuple) -> None:
        self.btn_optimize.setEnabled(True)
        self.btn_optimize.setText("Оптимизировать")

        opt_type, result = result_tuple

        if opt_type == "angle":
            alpha_opt, r_opt, details = result
            self._show_angle_result(alpha_opt, r_opt, details)
        elif opt_type == "width":
            width_opt, r_opt, details = result
            self._show_width_result(width_opt, r_opt, details)
        elif opt_type == "joint":
            alpha_opt, width_opt, r_opt, details = result
            self._show_joint_result(alpha_opt, width_opt, r_opt, details)
        elif opt_type == "multi":
            self._show_multi_result(result)

        self.app_state["optimization_results"] = {
            "type": opt_type,
            "result": result,
        }

    def _on_opt_error(self, msg: str) -> None:
        self.btn_optimize.setEnabled(True)
        self.btn_optimize.setText("Оптимизировать")
        logger.error(f"Ошибка оптимизации: {msg}")
        QMessageBox.warning(self, "Ошибка", f"Оптимизация не удалась:\n{msg}")

    def _compute_r_base(self) -> float:
        """r без пластин при текущем u₁."""
        try:
            from leakage_model.stage3_physics.model import solve_r
            geom = self._get_geom()
            base_params = self._get_base_params()
            beta, L, eps = self._get_beta_L_eps()
            r, *_ = solve_r(
                self.spin_u1.value(), geom, *base_params, beta,
                L_upper=L, eps=eps)
            return r
        except Exception:
            return 0.138

    def _show_angle_result(self, alpha_opt, r_opt, details) -> None:
        r_base = self._compute_r_base()
        reduction = (1 - r_opt / r_base) * 100 if r_base > 0 else 0

        self._res_labels["alpha_opt"].setText(fmt_comma(alpha_opt, 1))
        self._res_labels["b_opt"].setText("—")
        self._res_labels["r_opt"].setText(fmt_comma(r_opt, 4))
        self._res_labels["r_base"].setText(fmt_comma(r_base, 4))
        self._res_labels["reduction"].setText(fmt_comma(reduction, 1))

        def plot_angle(fig, ax):
            alphas = np.array(details["alpha_range"])
            r_profile = np.array(details["r_profile"])
            ax.plot(alphas, r_profile, "b-", lw=2,
                    label=f"r(α), u₁={details['u1']} м/с".replace(".", ","))
            ax.axvline(alpha_opt, ls="--", color="red", alpha=0.7)
            ax.plot(alpha_opt, r_opt, "r*", ms=15, zorder=5,
                    label=f"α* = {alpha_opt:.1f}°, r* = {r_opt:.4f}".replace(".", ","))
            ax.axhline(r_base, ls=":", color="gray", lw=1.5,
                       label=f"без пластин (r = {fmt_comma(r_base, 3)})")
            ax.set_xlabel("Угол α, °")
            ax.set_ylabel("Доля утечек r")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_title("Оптимизация по углу наклона")

        self.plot_main.plot(plot_angle)
        logger.info(f"Оптимизация по углу: α*={alpha_opt:.1f}°, r*={r_opt:.4f}")

    def _show_width_result(self, width_opt, r_opt, details) -> None:
        r_base = self._compute_r_base()
        reduction = (1 - r_opt / r_base) * 100 if r_base > 0 else 0

        self._res_labels["alpha_opt"].setText("—")
        self._res_labels["b_opt"].setText(fmt_comma(width_opt, 0))
        self._res_labels["r_opt"].setText(fmt_comma(r_opt, 4))
        self._res_labels["r_base"].setText(fmt_comma(r_base, 4))
        self._res_labels["reduction"].setText(fmt_comma(reduction, 1))

        def plot_width(fig, ax):
            widths = np.array(details["width_range"])
            r_profile = np.array(details["r_profile"])
            ax.plot(widths, r_profile, "b-", lw=2,
                    label=f"r(b), u₁={details['u1']} м/с".replace(".", ","))
            ax.axvline(width_opt, ls="--", color="red", alpha=0.7)
            ax.plot(width_opt, r_opt, "r*", ms=15, zorder=5,
                    label=f"b* = {width_opt:.0f} мм, r* = {r_opt:.4f}".replace(".", ","))
            ax.axhline(r_base, ls=":", color="gray", lw=1.5,
                       label=f"без пластин (r = {fmt_comma(r_base, 3)})")
            ax.set_xlabel("Ширина b, мм")
            ax.set_ylabel("Доля утечек r")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_title("Оптимизация по ширине пластин")

        self.plot_main.plot(plot_width)
        logger.info(f"Оптимизация по ширине: b*={width_opt:.0f} мм, r*={r_opt:.4f}")

    def _show_joint_result(self, alpha_opt, width_opt, r_opt, details) -> None:
        r_base = self._compute_r_base()
        reduction = (1 - r_opt / r_base) * 100 if r_base > 0 else 0

        self._res_labels["alpha_opt"].setText(fmt_comma(alpha_opt, 1))
        self._res_labels["b_opt"].setText(fmt_comma(width_opt, 0))
        self._res_labels["r_opt"].setText(fmt_comma(r_opt, 4))
        self._res_labels["r_base"].setText(fmt_comma(r_base, 4))
        self._res_labels["reduction"].setText(fmt_comma(reduction, 1))

        logger.info(f"Совместная оптимизация: α*={alpha_opt:.1f}°, "
                     f"b*={width_opt:.0f} мм, r*={r_opt:.4f}")

    def _show_multi_result(self, result) -> None:
        per_speed = result["per_speed"]
        mean = result["mean"]

        self._res_labels["alpha_opt"].setText(
            fmt_comma(mean["alpha_opt"], 1))
        self._res_labels["b_opt"].setText("—")
        self._res_labels["r_opt"].setText(
            fmt_comma(mean["r_mean_opt"], 4))
        self._res_labels["r_base"].setText("—")
        self._res_labels["reduction"].setText("—")

        # Таблица
        headers = ["u₁, м/с", "α*, °", "r*"]
        rows = [[p["u1"], p["alpha_opt"], p["r_opt"]]
                for p in per_speed]
        rows.append(["Среднее", mean["alpha_opt"], mean["r_mean_opt"]])
        self.result_table.set_data(headers, rows)

        # График
        def plot_multi(fig, ax):
            u1s = [p["u1"] for p in per_speed]
            alphas = [p["alpha_opt"] for p in per_speed]
            ax.plot(u1s, alphas, "bo-", ms=8, lw=2)
            ax.axhline(mean["alpha_opt"], ls="--", color="red",
                       label=f"α*_средн = {mean['alpha_opt']:.1f}°".replace(".", ","))
            ax.set_xlabel("u₁, м/с")
            ax.set_ylabel("α*, °")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_title("Мультискоростная оптимизация")

        self.plot_main.plot(plot_multi)
        logger.info(f"Мультискоростная: α*_средн={mean['alpha_opt']:.1f}°")

    def get_state(self) -> dict[str, Any]:
        return {
            "opt_type": self.combo_type.currentIndex(),
            "u1": self.spin_u1.value(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if "opt_type" in state:
            self.combo_type.setCurrentIndex(state["opt_type"])
        if "u1" in state:
            self.spin_u1.setValue(state["u1"])
