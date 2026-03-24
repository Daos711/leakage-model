"""Базовый класс для всех вкладок."""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtWidgets import QWidget

logger = logging.getLogger("leakage_gui")


class BaseTab(QWidget):
    """Базовый класс вкладок с общим интерфейсом."""

    tab_name: str = ""

    def __init__(self, app_state: dict[str, Any],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_state = app_state

    def update_plots(self) -> None:
        """Обновить все графики на вкладке."""

    def get_state(self) -> dict[str, Any]:
        """Получить состояние вкладки для сериализации."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Восстановить состояние вкладки."""

    def on_geometry_changed(self) -> None:
        """Вызывается при изменении геометрии в Tab 1."""
