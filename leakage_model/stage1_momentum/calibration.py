"""Калибровка C_M: обратный расчёт и анализ (этап 3)."""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..core.config import RHO
from .model import calc_C_M
from ..stage1_energy.model import calc_Re

logger = logging.getLogger(__name__)


def compute_C_M_table(df_cal: pd.DataFrame, geom: dict) -> pd.DataFrame:
    """Рассчитать C_M для каждой калибровочной точки.

    Parameters
    ----------
    df_cal : pd.DataFrame
        Калибровочные данные с колонками Q, u1, r.
    geom : dict
        Геометрические параметры.

    Returns
    -------
    pd.DataFrame
        Таблица с Re, u1, r_exp, C_M.
    """
    results = []
    for _, row in df_cal.iterrows():
        Q1 = row["Q"]
        u1 = row["u1"]
        r_exp = row["r"]
        Re = calc_Re(np.float64(u1), geom["D_h"], geom["nu"])
        C_M_val = calc_C_M(r_exp, Q1, geom, RHO)
        results.append({
            "Q": Q1,
            "u1": u1,
            "Re": float(Re),
            "r_exp": r_exp,
            "C_M": C_M_val,
        })

    return pd.DataFrame(results)


def analyze_C_M(df_cm: pd.DataFrame) -> dict:
    """Анализ стабильности C_M.

    Parameters
    ----------
    df_cm : pd.DataFrame
        Таблица с колонкой C_M.

    Returns
    -------
    dict
        Статистика: mean, std, CV, min, max, all_positive.
    """
    cm_vals = df_cm["C_M"].values
    mean = float(np.mean(cm_vals))
    std = float(np.std(cm_vals, ddof=1))
    cv = std / abs(mean) if abs(mean) > 1e-15 else float("inf")

    stats = {
        "mean": mean,
        "std": std,
        "CV": cv,
        "min": float(np.min(cm_vals)),
        "max": float(np.max(cm_vals)),
        "all_positive": bool(np.all(cm_vals > 0)),
    }

    logger.info("C_M среднее: %.6f", stats["mean"])
    logger.info("C_M стд. откл.: %.6f", stats["std"])
    logger.info("C_M CV: %.2f%%", stats["CV"] * 100)
    logger.info("C_M диапазон: [%.6f, %.6f]", stats["min"], stats["max"])
    logger.info("C_M все положительны: %s", stats["all_positive"])

    if cv < 0.20:
        logger.info("CV < 20%% → C_M = const допустимо")
    else:
        logger.warning("CV >= 20%% → рассмотреть C_M = f(Re)")

    return stats


@dataclass
class CM_FitResult:
    name: str
    params: dict
    R2: float
    cm_func: Callable  # Re -> C_M
    converged: bool = True


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_C_M_power(Re: np.ndarray, C_M: np.ndarray) -> CM_FitResult:
    """C_M = a · Re^b (степенной закон)."""
    ln_Re = np.log(Re)
    ln_CM = np.log(C_M)
    coeffs = np.polyfit(ln_Re, ln_CM, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])

    cm_func = lambda Re, _a=a, _b=b: _a * np.asarray(Re, dtype=float) ** _b
    R2 = _r_squared(C_M, cm_func(Re))

    logger.info("C_M = a·Re^b: a=%.6g, b=%.6g, R²=%.6f", a, b, R2)
    return CM_FitResult("power", {"a": a, "b": b}, R2, cm_func)


def fit_C_M_asymptotic(Re: np.ndarray, C_M: np.ndarray) -> CM_FitResult:
    """C_M = C_M_inf + c / Re^n (асимптотический)."""
    def model(Re, cm_inf, c, n):
        return cm_inf + c / Re ** n

    p0 = [C_M.min(), C_M.max() * Re.min(), 1.0]
    bounds = ([0, 0, 0.01], [np.inf, np.inf, 10.0])

    try:
        popt, _ = curve_fit(model, Re, C_M, p0=p0, bounds=bounds, maxfev=10000)
        cm_inf, c, n = popt
        cm_func = lambda Re, _ci=cm_inf, _c=c, _n=n: _ci + _c / np.asarray(Re, dtype=float) ** _n
        R2 = _r_squared(C_M, cm_func(Re))
        logger.info("C_M = C_M_inf + c/Re^n: C_M_inf=%.6g, c=%.6g, n=%.6g, R²=%.6f",
                     cm_inf, c, n, R2)
        return CM_FitResult("asymptotic", {"cm_inf": cm_inf, "c": c, "n": n},
                            R2, cm_func)
    except (RuntimeError, ValueError) as e:
        logger.error("C_M asymptotic fit не сошёлся: %s", e)
        return CM_FitResult("asymptotic", {}, 0.0,
                            lambda Re: np.full_like(Re, np.nan, dtype=float),
                            converged=False)
