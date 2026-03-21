"""Сводная таблица всех моделей и выбор лучшей."""

import logging
import os

import pandas as pd

from .alternatives import DirectFitResult
from ..core.validation import Metrics

logger = logging.getLogger(__name__)

from ..core.config import OUTPUT_STAGE1_1

OUTPUT_DIR = OUTPUT_STAGE1_1


def build_comparison_table(
    metrics_base_A: Metrics,
    R2_cal_base_A: float,
    metrics_base_B: Metrics,
    R2_cal_base_B: float,
    alt_fits: list[DirectFitResult],
) -> pd.DataFrame:
    """Сводная таблица всех моделей."""
    rows = [
        {
            "Модель": "Δζ(Re), вар. A",
            "Калибровка R²": R2_cal_base_A,
            "Валидация RMSE": metrics_base_A.RMSE,
            "Валидация R²": metrics_base_A.R2,
            "Физ. адекватность": "Да",
            "Комментарий": "Базовая, провалена",
        },
        {
            "Модель": "Δζ(Re), вар. B",
            "Калибровка R²": R2_cal_base_B,
            "Валидация RMSE": metrics_base_B.RMSE,
            "Валидация R²": metrics_base_B.R2,
            "Физ. адекватность": "Да",
            "Комментарий": "Переобучена",
        },
    ]

    for fit in alt_fits:
        phys = "Да" if fit.physical_ok else "Нет"
        comment = ""
        if not fit.converged:
            comment = "Не сошёлся"
        elif not fit.physical_ok:
            comment = "; ".join(fit.physical_notes)

        rows.append({
            "Модель": fit.name,
            "Калибровка R²": fit.R2_cal,
            "Валидация RMSE": fit.metrics_val.RMSE if fit.metrics_val else None,
            "Валидация R²": fit.metrics_val.R2 if fit.metrics_val else None,
            "Физ. адекватность": phys,
            "Комментарий": comment,
        })

    df = pd.DataFrame(rows)
    return df


def select_best_model(alt_fits: list[DirectFitResult]) -> DirectFitResult | None:
    """Выбор лучшей модели: RMSE валидации → физ. адекватность → R² калибровки."""
    candidates = [
        f for f in alt_fits
        if f.converged and f.metrics_val is not None
    ]

    if not candidates:
        logger.warning("Нет пригодных альтернативных моделей.")
        return None

    # Сначала физически адекватные
    physical = [f for f in candidates if f.physical_ok]
    pool = physical if physical else candidates

    best = min(pool, key=lambda f: f.metrics_val.RMSE)
    logger.info(
        "Лучшая альтернативная модель: %s (RMSE_val=%.6f, физ.=%s)",
        best.name, best.metrics_val.RMSE, best.physical_ok,
    )
    return best


def save_comparison_csv(df: pd.DataFrame) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df.to_csv(path, index=False)
    return path
