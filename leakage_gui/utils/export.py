"""Экспорт таблиц и графиков."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from PyQt6.QtWidgets import QTableWidget


def table_to_csv(table: QTableWidget, path: str) -> None:
    """Экспорт QTableWidget в CSV (разделитель ;, запятая в числах)."""
    rows = table.rowCount()
    cols = table.columnCount()
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        headers = []
        for c in range(cols):
            item = table.horizontalHeaderItem(c)
            headers.append(item.text() if item else f"Col{c}")
        writer.writerow(headers)
        for r in range(rows):
            row_data = []
            for c in range(cols):
                item = table.item(r, c)
                text = item.text() if item else ""
                row_data.append(text)
            writer.writerow(row_data)


def table_to_clipboard_text(table: QTableWidget) -> str:
    """Формирование текста Tab-separated для вставки в Word/Excel."""
    rows = table.rowCount()
    cols = table.columnCount()
    lines = []
    headers = []
    for c in range(cols):
        item = table.horizontalHeaderItem(c)
        headers.append(item.text() if item else "")
    lines.append("\t".join(headers))
    for r in range(rows):
        row_data = []
        for c in range(cols):
            item = table.item(r, c)
            row_data.append(item.text() if item else "")
        lines.append("\t".join(row_data))
    return "\n".join(lines)


def save_figure_png(fig: Figure, path: str, dpi: int = 300) -> None:
    """Сохранить matplotlib Figure в PNG."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def save_figure_svg(fig: Figure, path: str) -> None:
    """Сохранить matplotlib Figure в SVG."""
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")


def save_figure_pdf(fig: Figure, path: str) -> None:
    """Сохранить matplotlib Figure в PDF."""
    fig.savefig(path, format="pdf", bbox_inches="tight", facecolor="white")


def figure_to_clipboard_bytes(fig: Figure, dpi: int = 150) -> bytes:
    """Рендерить Figure в PNG-байты для буфера обмена."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()
