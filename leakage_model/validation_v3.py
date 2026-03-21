"""Валидация импульсной модели (этап 3)."""

import logging
from typing import Callable, Union

import numpy as np
import pandas as pd

from .config import RHO
from .model import calc_Re, calc_k_ut
from .solver import solve_r_brent, solve_r_newton
from .validation import Metrics, compute_metrics

logger = logging.getLogger(__name__)


def forward_solve(df: pd.DataFrame, geom: dict,
                  C_M: Union[float, Callable],
                  method: str = "brent") -> pd.DataFrame:
    """Прямая задача: решить F(r)=0 для каждого режима.

    Parameters
    ----------
    df : pd.DataFrame
        Данные с колонками Q, u1, r.
    geom : dict
        Геометрические параметры.
    C_M : float or callable
        Калибруемый коэффициент. Если callable — функция Re -> C_M.
    method : str
        "brent" или "newton".
    """
    solver = solve_r_brent if method == "brent" else solve_r_newton
    results = []

    for _, row in df.iterrows():
        Q1 = row["Q"]
        u1 = row["u1"]
        r_exp = row["r"]
        Re = calc_Re(np.float64(u1), geom["D_h"], geom["nu"])

        if callable(C_M):
            cm_val = float(C_M(float(Re)))
        else:
            cm_val = C_M

        r_calc = solver(Q1, geom, RHO, cm_val)
        error = r_calc - r_exp
        k_ut_calc = calc_k_ut(np.float64(r_calc))

        results.append({
            "Q": Q1,
            "u1": u1,
            "Re": float(Re),
            "r_exp": r_exp,
            "r_calc": r_calc,
            "error": error,
            "k_ut_calc": float(k_ut_calc),
        })

    return pd.DataFrame(results)


def validate_v3(df_cal: pd.DataFrame, df_val: pd.DataFrame,
                geom_cal: dict, geom_val: dict,
                C_M: Union[float, Callable]) -> tuple:
    """Полная валидация: калибровка + валидация.

    Returns
    -------
    tuple
        (df_cal_result, df_val_result, metrics_cal, metrics_val)
    """
    label = f"C_M={'func' if callable(C_M) else f'{C_M:.6f}'}"
    logger.info("Прямая задача — калибровочный набор (%s)...", label)
    df_cal_res = forward_solve(df_cal, geom_cal, C_M, method="brent")
    metrics_cal = compute_metrics(
        df_cal_res["r_exp"].values, df_cal_res["r_calc"].values
    )
    logger.info("Калибровка: RMSE=%.6f, MAE=%.6f, R²=%.6f, max|err|=%.6f",
                metrics_cal.RMSE, metrics_cal.MAE,
                metrics_cal.R2, metrics_cal.max_abs_error)

    logger.info("Прямая задача — валидационный набор (%s)...", label)
    df_val_res = forward_solve(df_val, geom_val, C_M, method="brent")
    metrics_val = compute_metrics(
        df_val_res["r_exp"].values, df_val_res["r_calc"].values
    )
    logger.info("Валидация: RMSE=%.6f, MAE=%.6f, R²=%.6f, max|err|=%.6f",
                metrics_val.RMSE, metrics_val.MAE,
                metrics_val.R2, metrics_val.max_abs_error)

    return df_cal_res, df_val_res, metrics_cal, metrics_val


def verify_newton_vs_brent(df: pd.DataFrame, geom: dict,
                           C_M: Union[float, Callable]) -> float:
    """Проверка совпадения Brent и Newton-Raphson.

    Returns
    -------
    float
        Максимальное расхождение.
    """
    max_diff = 0.0
    for _, row in df.iterrows():
        Q1 = row["Q"]
        u1 = row["u1"]
        Re = calc_Re(np.float64(u1), geom["D_h"], geom["nu"])
        cm_val = float(C_M(float(Re))) if callable(C_M) else C_M
        r_brent = solve_r_brent(Q1, geom, RHO, cm_val)
        r_newton = solve_r_newton(Q1, geom, RHO, cm_val)
        diff = abs(r_brent - r_newton)
        max_diff = max(max_diff, diff)

    logger.info("Макс. расхождение Brent vs Newton: %.2e (порог: 1e-10)",
                max_diff)
    if max_diff < 1e-10:
        logger.info("Newton-Raphson верифицирован")
    else:
        logger.warning("Расхождение превышает порог!")

    return max_diff
