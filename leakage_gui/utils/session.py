"""Сохранение и загрузка JSON-сессии проекта."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SESSION_VERSION = "1.0"


def save_session(path: str | Path, state: dict[str, Any]) -> None:
    """Сохранить состояние приложения в JSON."""
    data = {"version": SESSION_VERSION}
    data.update(state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_session(path: str | Path) -> dict[str, Any]:
    """Загрузить состояние из JSON-файла."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def default_state() -> dict[str, Any]:
    """Состояние по умолчанию."""
    return {
        "geometry": {
            "D": 8.0,
            "b_ok": 3.0,
            "h_ok": 4.0,
            "beta_deg": 45.0,
            "L_up": 111.5,
            "eps": 0.002,
            "nu": 1.5e-5,
            "rho": 1.2,
        },
        "calibration": {
            "source": "water",
            "a_xi": 38.51,
            "b_xi": -2.664,
            "c0": 1.983,
        },
        "data": {
            "calibration": [],
            "validation": [],
        },
        "optimization_results": {},
        "settings": {
            "decimal_separator": ",",
            "export_dpi": 300,
            "theme": "light",
        },
    }
