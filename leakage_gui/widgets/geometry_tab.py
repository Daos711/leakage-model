"""Вкладка 1: Геометрия и данные."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
)

from ..utils.locale_fmt import fmt_comma, fmt_comma_auto
from .base_tab import BaseTab
from .parameter_panel import ParameterPanel
from .results_table import ResultsTable

logger = logging.getLogger("leakage_gui")

_GEOM_PARAMS = [
    {"key": "D", "label": "Диаметр ствола D, м",
     "default": 8.0, "min": 0.1, "max": 100, "step": 0.1, "decimals": 1},
    {"key": "b_ok", "label": "Ширина окна b_ок, м",
     "default": 3.0, "min": 0.1, "max": 50, "step": 0.1, "decimals": 1},
    {"key": "h_ok", "label": "Высота окна h_ок, м",
     "default": 4.0, "min": 0.1, "max": 50, "step": 0.1, "decimals": 1},
    {"key": "beta_deg", "label": "Угол β, °",
     "default": 45.0, "min": 1, "max": 89, "step": 1, "decimals": 1},
    {"key": "L_up", "label": "Длина L_верх, м",
     "default": 111.5, "min": 1, "max": 1000, "step": 0.5, "decimals": 1},
    {"key": "eps", "label": "Шероховатость ε, м",
     "default": 0.002, "min": 1e-6, "max": 0.1, "step": 0.001, "decimals": 4},
    {"key": "nu", "label": "Вязкость ν, м²/с",
     "default": 1.5e-5, "min": 1e-7, "max": 1e-3, "step": 1e-6, "decimals": 7},
    {"key": "rho", "label": "Плотность ρ, кг/м³",
     "default": 1.2, "min": 0.1, "max": 2000, "step": 0.1, "decimals": 2},
]

_COMPUTED_PARAMS = [
    {"key": "A_ok", "label": "Площадь окна A_ок, м²",
     "default": 12.0, "decimals": 2, "readonly": True},
    {"key": "A_s", "label": "Площадь ствола A_с, м²",
     "default": 50.27, "decimals": 2, "readonly": True},
    {"key": "sigma", "label": "Отношение σ = A_ок/A_с",
     "default": 0.2387, "decimals": 4, "readonly": True},
    {"key": "D_h", "label": "Гидравл. диаметр D_h, м",
     "default": 3.43, "decimals": 2, "readonly": True},
]


class GeometryTab(BaseTab):
    """Вкладка ввода геометрии и экспериментальных данных."""

    tab_name = "Геометрия и данные"

    def __init__(self, app_state: dict[str, Any],
                 parent=None) -> None:
        super().__init__(app_state, parent)
        self._build_ui()
        self._load_default_data()
        self._update_computed()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        # Левая панель: параметры
        left = QVBoxLayout()
        self.param_panel = ParameterPanel()
        self.param_panel.add_group("Геометрия ствола и окна", _GEOM_PARAMS)
        self.param_panel.add_group("Вычисляемые параметры", _COMPUTED_PARAMS)
        self.param_panel.value_changed.connect(self._update_computed)
        left.addWidget(self.param_panel)

        btn_layout = QHBoxLayout()
        btn_water = QPushButton("Сбросить (вода)")
        btn_water.clicked.connect(self._reset_water)
        btn_air = QPushButton("Сбросить (воздух)")
        btn_air.clicked.connect(self._reset_air)
        btn_layout.addWidget(btn_water)
        btn_layout.addWidget(btn_air)
        left.addLayout(btn_layout)
        left.addStretch()

        # Правая панель: таблицы данных
        right = QVBoxLayout()
        self.data_tabs = QTabWidget()

        self.cal_table = ResultsTable(editable=True)
        self.val_table = ResultsTable(editable=True)
        self.data_tabs.addTab(self.cal_table, "Калибровочные данные")
        self.data_tabs.addTab(self.val_table, "Валидационные данные")

        right.addWidget(self.data_tabs)

        btn_data = QHBoxLayout()
        btn_load_csv = QPushButton("Загрузить из CSV")
        btn_load_csv.clicked.connect(self._load_csv)
        btn_default = QPushButton("Загрузить по умолчанию")
        btn_default.clicked.connect(self._load_default_data)
        btn_data.addWidget(btn_load_csv)
        btn_data.addWidget(btn_default)
        right.addLayout(btn_data)

        splitter = QSplitter()
        left_w = self._wrap_layout(left)
        right_w = self._wrap_layout(right)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([300, 500])
        main_layout.addWidget(splitter)

    def _wrap_layout(self, layout):
        from PyQt6.QtWidgets import QWidget
        w = QWidget()
        w.setLayout(layout)
        return w

    def _update_computed(self) -> None:
        """Пересчитать вычисляемые параметры."""
        D = self.param_panel.get_value("D")
        b = self.param_panel.get_value("b_ok")
        h = self.param_panel.get_value("h_ok")

        A_ok = b * h
        A_s = math.pi * D ** 2 / 4
        sigma = A_ok / A_s if A_s > 0 else 0
        D_h = 2 * b * h / (b + h) if (b + h) > 0 else 0

        self.param_panel.set_value("A_ok", A_ok)
        self.param_panel.set_value("A_s", A_s)
        self.param_panel.set_value("sigma", sigma)
        self.param_panel.set_value("D_h", D_h)

        # Обновить app_state
        self.app_state["geometry"] = self.get_geometry_dict()

    def get_geometry_dict(self) -> dict[str, float]:
        """Получить словарь геометрии в формате backend."""
        v = self.param_panel.get_all_values()
        return {
            "D": v["D"],
            "A_s": v["A_s"],
            "b_ok": v["b_ok"],
            "h_ok": v["h_ok"],
            "A_ok": v["A_ok"],
            "D_h": v["D_h"],
            "nu": v["nu"],
            "L_up": v["L_up"],
            "beta": math.radians(v["beta_deg"]),
        }

    def get_rho(self) -> float:
        return self.param_panel.get_value("rho")

    def get_eps(self) -> float:
        return self.param_panel.get_value("eps")

    def get_beta_rad(self) -> float:
        return math.radians(self.param_panel.get_value("beta_deg"))

    def get_L_up(self) -> float:
        return self.param_panel.get_value("L_up")

    def _reset_water(self) -> None:
        defaults = {"D": 8.0, "b_ok": 3.0, "h_ok": 4.0, "beta_deg": 45.0,
                     "L_up": 111.5, "eps": 0.002, "nu": 1.5e-5, "rho": 1.2}
        self.param_panel.set_all_values(defaults)
        self._update_computed()

    def _reset_air(self) -> None:
        defaults = {"D": 8.0, "b_ok": 4.0, "h_ok": 5.0, "beta_deg": 45.0,
                     "L_up": 111.5, "eps": 0.002, "nu": 1.5e-5, "rho": 1.2}
        self.param_panel.set_all_values(defaults)
        self._update_computed()

    def _load_default_data(self) -> None:
        """Загрузить данные по умолчанию из backend."""
        try:
            from leakage_model.core.data import (
                load_calibration_data,
                load_validation_data,
            )
            df_cal = load_calibration_data()
            df_val = load_validation_data()
            self._set_cal_data(df_cal)
            self._set_val_data(df_val)
            logger.info("Загружены данные по умолчанию")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные:\n{e}")

    def _set_cal_data(self, df: pd.DataFrame) -> None:
        cols = list(df.columns)
        rows = df.values.tolist()
        self.cal_table.set_data(cols, rows)
        self.app_state["cal_df"] = df

    def _set_val_data(self, df: pd.DataFrame) -> None:
        cols = list(df.columns)
        rows = df.values.tolist()
        self.val_table.set_data(cols, rows)
        self.app_state["val_df"] = df

    def _load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить CSV", "", "CSV (*.csv);;Все файлы (*)")
        if not path:
            return
        try:
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(path, sep=sep, decimal=",",
                                     comment="#", encoding="utf-8-sig")
                    if len(df.columns) >= 2:
                        break
                except Exception:
                    continue
            else:
                raise ValueError("Не удалось определить формат CSV")

            # Попытка с точкой как десятичный разделитель
            if df.select_dtypes(include="number").empty:
                for sep in [";", ",", "\t"]:
                    try:
                        df = pd.read_csv(path, sep=sep, decimal=".",
                                         comment="#", encoding="utf-8-sig")
                        if len(df.columns) >= 2:
                            break
                    except Exception:
                        continue

            df.dropna(how="all", inplace=True)

            # Проверка обязательных столбцов
            renames = {"Q1": "Q", "velocity": "u1", "leakage_ratio": "r"}
            df.rename(columns=renames, inplace=True)

            required = {"u1", "r"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"Отсутствуют столбцы: {missing}. "
                    f"Ожидаются: Q (опц.), u1, r")

            idx = self.data_tabs.currentIndex()
            if idx == 0:
                self._set_cal_data(df)
            else:
                self._set_val_data(df)
            logger.info(f"Загружен CSV: {path}, {len(df)} строк")

        except Exception as e:
            logger.error(f"Ошибка загрузки CSV: {e}")
            QMessageBox.critical(self, "Ошибка CSV",
                                 f"Не удалось загрузить файл:\n{e}")

    def get_state(self) -> dict[str, Any]:
        return {
            "geometry": self.param_panel.get_all_values(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if "geometry" in state:
            self.param_panel.set_all_values(state["geometry"])
            self._update_computed()
