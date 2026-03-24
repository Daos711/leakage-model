"""Точка входа GUI-приложения LeakageModel."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging() -> None:
    """Настройка логирования с ротацией."""
    log_dir = Path.home() / ".leakage_model"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "leakage_gui.log"

    handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=2,
        encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("leakage_gui")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)

    # Консольный вывод
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


def main() -> None:
    """Запуск приложения."""
    setup_logging()
    logger = logging.getLogger("leakage_gui")
    logger.info("Запуск LeakageModel GUI")

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("LeakageModel")
    app.setStyle("Fusion")

    from .main_window import MainWindow

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
