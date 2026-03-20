"""Валидация и экспорт результатов полуэмпирической модели (этап 3)."""

import json
import logging
import os

import numpy as np
import pandas as pd

from .idelchik import EPS_DEFAULT, L_UPPER_DEFAULT
from .model import calc_Re
from .physics_model import solve_all, PhysicsResult
from .validation import compute_metrics, Metrics

logger = logging.getLogger(__name__)


def validate(u1_val, r_val, geom_val, a_xi, b_xi, beta,
             L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
             R_down=0.0, criterion="Re"):
    """Валидация на воздушных данных.

    Возвращает (metrics, result_val).
    """
    u1_val = np.asarray(u1_val, dtype=float)
    r_val = np.asarray(r_val, dtype=float)

    result_val = solve_all(u1_val, geom_val, a_xi, b_xi, beta,
                           L_upper, eps, R_down, criterion)
    metrics_val = compute_metrics(r_val, result_val.r_pred)

    logger.info(f"Метрики валидации: RMSE={metrics_val.RMSE:.4f}, "
                f"MAE={metrics_val.MAE:.4f}, R²={metrics_val.R2:.4f}, "
                f"max|err|={metrics_val.max_abs_error:.4f}")

    return metrics_val, result_val


def _build_prediction_df(u1, r_exp, result, geom, label):
    """Собрать DataFrame с предсказаниями для одного набора."""
    nu = geom["nu"]
    D_h = geom["D_h"]
    Re = u1 * D_h / nu
    sigma = geom["A_ok"] / geom["A_s"]

    df = pd.DataFrame({
        "dataset": label,
        "u1_m_s": u1,
        "Re": Re,
        "sigma": sigma,
        "xi": result.xi,
        "phi_up": result.phi_up,
        "phi_down": result.phi_down,
        "r_exp": r_exp,
        "r_pred": result.r_pred,
        "error": result.r_pred - r_exp,
        "Q1_m3s": u1 * geom["A_ok"],
        "Q2_pred_m3s": result.r_pred * u1 * geom["A_ok"],
    })
    return df


def export_results(u1_cal, r_cal, result_cal, geom_cal, metrics_cal,
                   u1_val, r_val, result_val, geom_val, metrics_val,
                   a_xi, b_xi, criterion, output_dir):
    """Экспорт CSV + JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # prediction CSV
    df_cal = _build_prediction_df(u1_cal, r_cal, result_cal, geom_cal, "water")
    df_val = _build_prediction_df(u1_val, r_val, result_val, geom_val, "air")
    df_all = pd.concat([df_cal, df_val], ignore_index=True)

    csv_path = os.path.join(output_dir, "physics_prediction.csv")
    df_all.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"Сохранено: {csv_path}")

    # parameters JSON
    params = {
        "a_xi": float(a_xi),
        "b_xi": float(b_xi),
        "criterion": criterion,
        "calibration": {
            "RMSE": float(metrics_cal.RMSE),
            "MAE": float(metrics_cal.MAE),
            "R2": float(metrics_cal.R2),
            "max_abs_error": float(metrics_cal.max_abs_error),
        },
        "validation": {
            "RMSE": float(metrics_val.RMSE),
            "MAE": float(metrics_val.MAE),
            "R2": float(metrics_val.R2),
            "max_abs_error": float(metrics_val.max_abs_error),
        },
    }
    json_path = os.path.join(output_dir, "physics_parameters.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    logger.info(f"Сохранено: {json_path}")

    return df_all
