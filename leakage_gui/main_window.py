"""Главное окно приложения LeakageModel."""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from .utils.session import default_state, load_session, save_session
from .widgets.base_tab import BaseTab
from .widgets.delta_zeta_tab import DeltaZetaTab
from .widgets.geometry_tab import GeometryTab
from .widgets.idelchik_tab import IdelchikTab
from .widgets.momentum_tab import MomentumTab
from .widgets.optimization_tab import OptimizationTab
from .widgets.physics_tab import PhysicsTab
from .widgets.plates_tab import PlatesTab

logger = logging.getLogger("leakage_gui")


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "LeakageModel — Расчёт утечек воздуха через устье шахтного ствола")
        self.resize(1200, 800)

        self.app_state: dict[str, Any] = default_state()
        self._tabs: list[BaseTab] = []

        self._build_status_bar()
        self._build_ui()
        self._build_menu()
        self._build_hotkeys()

        # Начальные графики
        self._update_current_tab()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter()

        # Боковая панель навигации
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(230)
        self.nav_list.setSpacing(2)

        # Стек вкладок
        self.stack = QStackedWidget()

        # Создание вкладок
        tab_classes = [
            GeometryTab,
            DeltaZetaTab,
            MomentumTab,
            IdelchikTab,
            PhysicsTab,
            PlatesTab,
            OptimizationTab,
        ]

        for cls in tab_classes:
            tab = cls(self.app_state)
            self._tabs.append(tab)
            self.stack.addWidget(tab)
            item = QListWidgetItem(f"  {tab.tab_name}")
            self.nav_list.addItem(item)

        # Сохранить geometry_tab для доступа из других вкладок
        self.app_state["geometry_tab"] = self._tabs[0]

        self.nav_list.currentRowChanged.connect(self._on_tab_changed)
        self.nav_list.setCurrentRow(0)

        splitter.addWidget(self.nav_list)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        # Файл
        file_menu = menubar.addMenu("Файл")

        act_save = QAction("Сохранить проект", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_open = QAction("Открыть проект", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_load_cal = QAction("Загрузить калибровочные данные", self)
        act_load_cal.triggered.connect(self._load_cal_csv)
        file_menu.addAction(act_load_cal)

        file_menu.addSeparator()

        act_exit = QAction("Выход", self)
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Настройки
        settings_menu = menubar.addMenu("Настройки")

        act_water = QAction("Водяная калибровка (по умолчанию)", self)
        act_water.triggered.connect(
            lambda: self._set_calibration("water"))
        settings_menu.addAction(act_water)

        act_joint = QAction("Совместная калибровка", self)
        act_joint.triggered.connect(
            lambda: self._set_calibration("joint"))
        settings_menu.addAction(act_joint)

        # Справка
        help_menu = menubar.addMenu("Справка")
        act_about = QAction("О программе", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _build_hotkeys(self) -> None:
        from PyQt6.QtGui import QShortcut
        for i in range(min(7, len(self._tabs))):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            shortcut.activated.connect(
                lambda idx=i: self.nav_list.setCurrentRow(idx))

        shortcut_f5 = QShortcut(QKeySequence("F5"), self)
        shortcut_f5.activated.connect(self._update_current_tab)

        shortcut_f11 = QShortcut(QKeySequence("F11"), self)
        shortcut_f11.activated.connect(self._toggle_fullscreen)

        shortcut_calc = QShortcut(QKeySequence("Ctrl+R"), self)
        shortcut_calc.activated.connect(self._update_current_tab)

    def _build_status_bar(self) -> None:
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _on_tab_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self._update_status()
        self._update_current_tab()

    def _update_current_tab(self) -> None:
        idx = self.stack.currentIndex()
        if 0 <= idx < len(self._tabs):
            try:
                self._tabs[idx].update_plots()
            except Exception as e:
                logger.error(f"Ошибка обновления вкладки: {e}")

    def _update_status(self) -> None:
        if not hasattr(self, "stack"):
            return
        idx = self.stack.currentIndex()
        cal = self.app_state.get("calibration", {})
        source = cal.get("source", "water")
        tab_name = self._tabs[idx].tab_name if idx < len(self._tabs) else ""
        self.status_bar.showMessage(
            f"Вкладка: {tab_name}  |  "
            f"Калибровка: {source}  |  "
            f"a_ξ={cal.get('a_xi', 38.51):.2f}, "
            f"b_ξ={cal.get('b_xi', -2.664):.3f}, "
            f"c₀={cal.get('c0', 1.983):.3f}")

    def _save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект", "",
            "JSON (*.json)")
        if not path:
            return

        state = dict(self.app_state)
        # Убираем несериализуемые объекты
        serializable = {}
        for tab in self._tabs:
            serializable[tab.tab_name] = tab.get_state()
        state["tabs"] = serializable
        # Убрать ссылки на виджеты и DataFrame
        for key in ["geometry_tab", "cal_df", "val_df", "plates_df",
                     "plates_cal_results", "surrogates"]:
            state.pop(key, None)

        try:
            save_session(path, state)
            self.status_bar.showMessage(f"Проект сохранён: {path}", 5000)
            logger.info(f"Проект сохранён: {path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть проект", "",
            "JSON (*.json)")
        if not path:
            return

        try:
            data = load_session(path)
            # Восстановить геометрию
            if "geometry" in data:
                self.app_state["geometry"] = data["geometry"]
            if "calibration" in data:
                self.app_state["calibration"] = data["calibration"]
            if "settings" in data:
                self.app_state["settings"] = data["settings"]

            # Восстановить вкладки
            tabs_state = data.get("tabs", {})
            for tab in self._tabs:
                if tab.tab_name in tabs_state:
                    tab.set_state(tabs_state[tab.tab_name])

            self.status_bar.showMessage(f"Проект загружен: {path}", 5000)
            logger.info(f"Проект загружен: {path}")
            self._update_current_tab()

        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            QMessageBox.critical(self, "Ошибка",
                                 f"Не удалось загрузить проект:\n{e}")

    def _load_cal_csv(self) -> None:
        geo_tab = self._tabs[0]
        if hasattr(geo_tab, "_load_csv"):
            geo_tab._load_csv()

    def _set_calibration(self, variant: str) -> None:
        if variant == "water":
            self.app_state["calibration"] = {
                "source": "water",
                "a_xi": 38.51, "b_xi": -2.664, "c0": 1.983,
            }
        else:
            self.app_state["calibration"] = {
                "source": "joint",
                "a_xi": 26.14, "b_xi": -1.803, "c0": 2.192,
            }
        # Обновить physics tab
        physics_tab = self._tabs[4]
        if isinstance(physics_tab, PhysicsTab):
            cal = self.app_state["calibration"]
            physics_tab.param_panel.set_value("a_xi", cal["a_xi"])
            physics_tab.param_panel.set_value("b_xi", cal["b_xi"])
            physics_tab.param_panel.set_value("c0", cal["c0"])
        self._update_status()

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "О программе",
            "LeakageModel v1.0\n\n"
            "Расчёт утечек воздуха через устье шахтного ствола\n"
            "на основе полуэмпирической модели ξ(Re) + C_β\n"
            "с учётом направляющих пластин.\n\n"
            "Python + PyQt6 + matplotlib")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
