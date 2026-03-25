"""Переиспользуемый виджет таблицы результатов с экспортом."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..utils.export import table_to_clipboard_text, table_to_csv
from ..utils.locale_fmt import fmt_comma


class ResultsTable(QWidget):
    """QTableWidget с кнопками экспорта."""

    def __init__(self, parent: QWidget | None = None,
                 editable: bool = False) -> None:
        super().__init__(parent)
        self._editable = editable

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        if not editable:
            self.table.setEditTriggers(
                QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_csv = QPushButton("Экспорт CSV")
        btn_csv.clicked.connect(self._export_csv)
        btn_copy = QPushButton("Копировать")
        btn_copy.clicked.connect(self._copy_clipboard)
        btn_layout.addWidget(btn_csv)
        btn_layout.addWidget(btn_copy)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def set_data(self, headers: list[str], rows: list[list],
                 fmt_decimals: int = 4) -> None:
        """Заполнить таблицу данными.

        Args:
            headers: заголовки столбцов
            rows: список строк (каждая строка — список значений)
            fmt_decimals: количество десятичных знаков для float
        """
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))

        for r, row_data in enumerate(rows):
            for c, val in enumerate(row_data):
                if isinstance(val, float):
                    text = fmt_comma(val, fmt_decimals)
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                if not self._editable:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт CSV", "", "CSV (*.csv)")
        if path:
            table_to_csv(self.table, path)

    def _copy_clipboard(self) -> None:
        text = table_to_clipboard_text(self.table)
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(text)

    def _context_menu(self, pos) -> None:
        menu = QMenu(self)
        act_csv = QAction("Экспорт CSV", self)
        act_csv.triggered.connect(self._export_csv)
        menu.addAction(act_csv)
        act_copy = QAction("Копировать в буфер", self)
        act_copy.triggered.connect(self._copy_clipboard)
        menu.addAction(act_copy)
        menu.exec(self.table.mapToGlobal(pos))
