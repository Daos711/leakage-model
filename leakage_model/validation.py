"""Валидация модели и расчёт метрик."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import calc_r_explicit

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    RMSE: float
    MAE: float
    R2: float
    max_abs_error: float


def compute_metrics(r_exp: np.ndarray, r_pred: np.ndarray) -> Metrics:
    err = r_exp - r_pred
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((r_exp - np.mean(r_exp)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    max_err = np.max(np.abs(err))
    return Metrics(rmse, mae, R2, max_err)


def validate(
    df_val: pd.DataFrame,
    geom: dict,
    dz_func_A,
    dz_func_B,
) -> pd.DataFrame:
    """Валидация на воздушной модели.

    Возвращает DataFrame с предсказаниями и ошибками.
    """
    u1 = df_val["u1"].values

    r_pred_A = calc_r_explicit(u1, geom, dz_func_A)
    r_pred_B = calc_r_explicit(u1, geom, dz_func_B)

    result = df_val.copy()
    result["r_pred_A"] = r_pred_A
    result["r_pred_B"] = r_pred_B
    result["error_A"] = df_val["r"].values - r_pred_A
    result["error_B"] = df_val["r"].values - r_pred_B

    r_exp = df_val["r"].values

    metrics_A = compute_metrics(r_exp, r_pred_A)
    metrics_B = compute_metrics(r_exp, r_pred_B)

    logger.info("Метрики валидации (вариант A):")
    logger.info("  RMSE=%.6f, MAE=%.6f, R²=%.6f, max|err|=%.6f",
                metrics_A.RMSE, metrics_A.MAE, metrics_A.R2, metrics_A.max_abs_error)
    logger.info("Метрики валидации (вариант B):")
    logger.info("  RMSE=%.6f, MAE=%.6f, R²=%.6f, max|err|=%.6f",
                metrics_B.RMSE, metrics_B.MAE, metrics_B.R2, metrics_B.max_abs_error)

    return result, metrics_A, metrics_B
