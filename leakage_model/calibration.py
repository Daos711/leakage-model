"""Калибровка Δζ(Re): варианты A (степенной) и B (асимптотический)."""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    name: str
    params: dict
    R2: float
    dz_func: Callable
    converged: bool = True


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_power_law(Re: np.ndarray, dz: np.ndarray) -> FitResult:
    """Вариант A: Δζ = a · Re^b.

    Калибровка через линейную регрессию ln(Δζ) vs ln(Re).
    """
    mask = dz > 0
    if mask.sum() < 2:
        logger.warning("Недостаточно точек с Δζ > 0 для степенного фита")
        return FitResult("A", {"a": np.nan, "b": np.nan}, 0.0,
                         lambda Re: np.full_like(Re, np.nan, dtype=float),
                         converged=False)

    if (~mask).any():
        logger.warning(
            "Точки с Δζ ≤ 0 исключены из log-fit: %d шт.",
            (~mask).sum(),
        )

    ln_Re = np.log(Re[mask])
    ln_dz = np.log(dz[mask])

    # линейная регрессия: ln(Δζ) = ln(a) + b·ln(Re)
    coeffs = np.polyfit(ln_Re, ln_dz, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])

    dz_func = lambda Re, _a=a, _b=b: _a * np.asarray(Re, dtype=float) ** _b

    dz_pred = dz_func(Re)
    R2 = _r_squared(dz, dz_pred)

    logger.info("Вариант A: a=%.6g, b=%.6g, R²=%.6f", a, b, R2)
    return FitResult("A", {"a": a, "b": b}, R2, dz_func)


def fit_asymptotic(Re: np.ndarray, dz: np.ndarray) -> FitResult:
    """Вариант B: Δζ = Δζ_∞ + c / Re^n."""

    def model(Re, dz_inf, c, n):
        return dz_inf + c / Re**n

    # Δζ растёт с Re и асимптотически стремится к Δζ_∞ → c < 0
    p0 = [dz.max() * 1.2, -dz.max() * Re.min(), 1.0]
    bounds = ([0, -np.inf, 0.01], [np.inf, 0, 10.0])

    try:
        popt, _ = curve_fit(model, Re, dz, p0=p0, bounds=bounds, maxfev=10000)
        dz_inf, c, n = popt

        dz_func = lambda Re, _di=dz_inf, _c=c, _n=n: _di + _c / np.asarray(Re, dtype=float) ** _n

        dz_pred = dz_func(Re)
        R2 = _r_squared(dz, dz_pred)

        # Проверка физичности
        if dz_inf < -1.0 or n > 5.0:
            logger.warning(
                "Вариант B: нефизичные параметры (Δζ_∞=%.4g, n=%.4g)", dz_inf, n
            )

        logger.info(
            "Вариант B: Δζ_∞=%.6g, c=%.6g, n=%.6g, R²=%.6f",
            dz_inf, c, n, R2,
        )
        return FitResult("B", {"dz_inf": dz_inf, "c": c, "n": n}, R2, dz_func)

    except (RuntimeError, ValueError) as e:
        logger.error("Вариант B: curve_fit не сошёлся: %s", e)
        return FitResult(
            "B", {"dz_inf": np.nan, "c": np.nan, "n": np.nan}, 0.0,
            lambda Re: np.full_like(Re, np.nan, dtype=float),
            converged=False,
        )
