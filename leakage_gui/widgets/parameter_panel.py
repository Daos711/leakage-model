"""Переиспользуемая панель ввода параметров с QDoubleSpinBox."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)


class CommaDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox, показывающий запятую и принимающий и точку, и запятую."""

    def textFromValue(self, value: float) -> str:
        text = super().textFromValue(value)
        return text.replace(".", ",")

    def valueFromText(self, text: str) -> float:
        return float(text.replace(",", "."))

    def validate(self, text: str, pos: int):
        text_dot = text.replace(",", ".")
        return super().validate(text_dot, pos)


class ParameterPanel(QWidget):
    """Панель с группами параметров в QFormLayout."""

    value_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._spinboxes: dict[str, CommaDoubleSpinBox] = {}

    def add_group(self, title: str,
                  params: list[dict[str, Any]]) -> QGroupBox:
        """Добавить группу параметров.

        Args:
            title: заголовок группы
            params: список словарей с ключами:
                key, label, default, min, max, step, decimals, suffix,
                scientific (bool), readonly (bool)
        """
        group = QGroupBox(title)
        form = QFormLayout(group)

        for p in params:
            spin = CommaDoubleSpinBox()
            spin.setMinimum(p.get("min", -1e9))
            spin.setMaximum(p.get("max", 1e9))
            spin.setDecimals(p.get("decimals", 4))
            spin.setSingleStep(p.get("step", 0.1))
            spin.setValue(p.get("default", 0.0))
            if p.get("suffix"):
                spin.setSuffix(f" {p['suffix']}")
            if p.get("readonly"):
                spin.setReadOnly(True)
                spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                spin.setProperty("readOnlyField", True)

            spin.valueChanged.connect(self.value_changed.emit)
            form.addRow(p["label"], spin)
            self._spinboxes[p["key"]] = spin

        self._layout.addWidget(group)
        return group

    def get_value(self, key: str) -> float:
        """Получить значение параметра по ключу."""
        return self._spinboxes[key].value()

    def set_value(self, key: str, value: float) -> None:
        """Установить значение параметра."""
        spin = self._spinboxes[key]
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def get_all_values(self) -> dict[str, float]:
        """Получить все значения как словарь."""
        return {k: s.value() for k, s in self._spinboxes.items()}

    def set_all_values(self, values: dict[str, float]) -> None:
        """Установить все значения из словаря."""
        for k, v in values.items():
            if k in self._spinboxes:
                self.set_value(k, v)

    def get_spinbox(self, key: str) -> CommaDoubleSpinBox:
        """Получить QDoubleSpinBox по ключу."""
        return self._spinboxes[key]
