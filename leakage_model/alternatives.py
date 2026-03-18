"""Альтернативные замыкания: прямые модели r(Re) и r(u₁)."""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .model import calc_Re
from .validation import Metrics, compute_metrics

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "plots")


@dataclass
class DirectFitResult:
    name: str
    params: dict
    R2_cal: float
    r_func: Callable  # r(u1, geom) -> np.ndarray
    converged: bool = True
    metrics_cal: Metrics | None = None
    metrics_val: Metrics | None = None
    physical_ok: bool = True
    physical_notes: list[str] = field(default_factory=list)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _check_physical(
    r_pred: np.ndarray,
    u1: np.ndarray,
    label: str,
) -> tuple[bool, list[str]]:
    """Проверка физической адекватности: 0 < r < 1, монотонное убывание."""
    notes = []
    ok = True

    if np.any(r_pred <= 0) or np.any(r_pred >= 1):
        notes.append(f"{label}: r выходит за (0, 1)")
        ok = False

    order = np.argsort(u1)
    r_sorted = r_pred[order]
    diffs = np.diff(r_sorted)
    if np.any(diffs > 1e-10):
        notes.append(f"{label}: r не монотонно убывает")
        ok = False

    return ok, notes


# ---------------------------------------------------------------------------
# Вариант 2A: r = a · Re^b (степенной)
# ---------------------------------------------------------------------------
def fit_r_power_Re(
    u1_cal: np.ndarray,
    r_cal: np.ndarray,
    geom_cal: dict,
) -> DirectFitResult:
    """Вариант 2A: r = a · Re^b — степенное замыкание по Re."""
    Re_cal = calc_Re(u1_cal, geom_cal["D_h"], geom_cal["nu"])

    ln_Re = np.log(Re_cal)
    ln_r = np.log(r_cal)

    coeffs = np.polyfit(ln_Re, ln_r, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])

    def r_func(u1, geom, _a=a, _b=b):
        Re = calc_Re(np.asarray(u1, dtype=float), geom["D_h"], geom["nu"])
        return _a * Re ** _b

    r_pred = r_func(u1_cal, geom_cal)
    R2 = _r_squared(r_cal, r_pred)
    metrics = compute_metrics(r_cal, r_pred)

    logger.info("Вариант 2A: a=%.6g, b=%.6g, R²_cal=%.6f", a, b, R2)
    return DirectFitResult(
        name="r(Re) степенной",
        params={"a": a, "b": b},
        R2_cal=R2,
        r_func=r_func,
        metrics_cal=metrics,
    )


# ---------------------------------------------------------------------------
# Вариант 2B: r = r_∞ + c / Re^n (асимптотический)
# ---------------------------------------------------------------------------
def fit_r_asymptotic_Re(
    u1_cal: np.ndarray,
    r_cal: np.ndarray,
    geom_cal: dict,
) -> DirectFitResult:
    """Вариант 2B: r = r_∞ + c / Re^n — асимптотическое замыкание по Re."""
    Re_cal = calc_Re(u1_cal, geom_cal["D_h"], geom_cal["nu"])

    def model(Re, r_inf, c, n):
        return r_inf + c / Re ** n

    # r убывает с ростом Re → c > 0, r_inf > 0
    p0 = [r_cal.min() * 0.5, r_cal.max() * Re_cal.min() ** 0.5, 0.5]
    bounds = ([0, 0, 0.01], [1.0, np.inf, 10.0])

    try:
        popt, _ = curve_fit(model, Re_cal, r_cal, p0=p0, bounds=bounds, maxfev=10000)
        r_inf, c, n = popt

        def r_func(u1, geom, _ri=r_inf, _c=c, _n=n):
            Re = calc_Re(np.asarray(u1, dtype=float), geom["D_h"], geom["nu"])
            return _ri + _c / Re ** _n

        r_pred = r_func(u1_cal, geom_cal)
        R2 = _r_squared(r_cal, r_pred)
        metrics = compute_metrics(r_cal, r_pred)

        logger.info(
            "Вариант 2B: r_∞=%.6g, c=%.6g, n=%.6g, R²_cal=%.6f",
            r_inf, c, n, R2,
        )
        return DirectFitResult(
            name="r(Re) асимптотический",
            params={"r_inf": r_inf, "c": c, "n": n},
            R2_cal=R2,
            r_func=r_func,
            metrics_cal=metrics,
        )

    except (RuntimeError, ValueError) as e:
        logger.error("Вариант 2B: curve_fit не сошёлся: %s", e)
        return DirectFitResult(
            name="r(Re) асимптотический",
            params={"r_inf": np.nan, "c": np.nan, "n": np.nan},
            R2_cal=0.0,
            r_func=lambda u1, geom: np.full(np.asarray(u1).shape, np.nan),
            converged=False,
        )


# ---------------------------------------------------------------------------
# Вариант 2C: r = a · u₁^b (степенной по скорости)
# ---------------------------------------------------------------------------
def fit_r_power_u1(
    u1_cal: np.ndarray,
    r_cal: np.ndarray,
) -> DirectFitResult:
    """Вариант 2C: r = a · u₁^b — степенное замыкание по скорости."""
    ln_u1 = np.log(u1_cal)
    ln_r = np.log(r_cal)

    coeffs = np.polyfit(ln_u1, ln_r, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])

    def r_func(u1, geom, _a=a, _b=b):
        return _a * np.asarray(u1, dtype=float) ** _b

    r_pred = r_func(u1_cal, None)
    R2 = _r_squared(r_cal, r_pred)
    metrics = compute_metrics(r_cal, r_pred)

    logger.info("Вариант 2C: a=%.6g, b=%.6g, R²_cal=%.6f", a, b, R2)
    return DirectFitResult(
        name="r(u₁) степенной",
        params={"a": a, "b": b},
        R2_cal=R2,
        r_func=r_func,
        metrics_cal=metrics,
    )


# ---------------------------------------------------------------------------
# Валидация и проверки
# ---------------------------------------------------------------------------
def validate_alternative(
    fit: DirectFitResult,
    u1_cal: np.ndarray,
    r_cal: np.ndarray,
    geom_cal: dict,
    u1_val: np.ndarray,
    r_val: np.ndarray,
    geom_val: dict,
) -> DirectFitResult:
    """Валидация альтернативной модели + проверки физической адекватности."""
    if not fit.converged:
        return fit

    # Предсказания
    r_pred_cal = fit.r_func(u1_cal, geom_cal)
    r_pred_val = fit.r_func(u1_val, geom_val)

    # Метрики валидации
    fit.metrics_val = compute_metrics(r_val, r_pred_val)

    logger.info(
        "Валидация %s: RMSE=%.6f, MAE=%.6f, R²=%.6f, max|err|=%.6f",
        fit.name,
        fit.metrics_val.RMSE,
        fit.metrics_val.MAE,
        fit.metrics_val.R2,
        fit.metrics_val.max_abs_error,
    )

    # Проверки физической адекватности
    all_notes = []

    ok_cal, notes_cal = _check_physical(r_pred_cal, u1_cal, "Калибровка")
    all_notes.extend(notes_cal)

    ok_val, notes_val = _check_physical(r_pred_val, u1_val, "Валидация")
    all_notes.extend(notes_val)

    fit.physical_ok = ok_cal and ok_val
    fit.physical_notes = all_notes

    if not fit.physical_ok:
        for note in all_notes:
            logger.warning("[%s] %s", fit.name, note)
    else:
        logger.info("[%s] Физические проверки пройдены.", fit.name)

    return fit


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------
def plot_all_models_calibration(
    u1_cal: np.ndarray,
    r_cal: np.ndarray,
    geom_cal: dict,
    fits: list[DirectFitResult],
) -> str:
    """График 7: r(u₁) — все модели, калибровка."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(u1_cal, r_cal, "ko", ms=8, zorder=5, label="Эксперимент (вода)")

    u_fine = np.linspace(u1_cal.min() * 0.9, u1_cal.max() * 1.1, 200)
    colors = ["b-", "r--", "g-."]
    for fit, style in zip(fits, colors):
        if fit.converged:
            r_pred = fit.r_func(u_fine, geom_cal)
            ax.plot(u_fine, r_pred, style, lw=2, label=fit.name)

    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_ylabel("Доля утечек r")
    ax.set_title("Калибровка: все альтернативные модели r(u₁)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "07_all_models_calibration.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all_models_validation(
    u1_cal: np.ndarray,
    r_cal: np.ndarray,
    geom_cal: dict,
    u1_val: np.ndarray,
    r_val: np.ndarray,
    geom_val: dict,
    fits: list[DirectFitResult],
) -> str:
    """График 8: r(u₁) — все модели, валидация."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(u1_cal, r_cal, "bs", ms=7, alpha=0.4, label="Эксперимент (вода)")
    ax.plot(u1_val, r_val, "ro", ms=8, zorder=5, label="Эксперимент (воздух)")

    u_fine = np.linspace(
        min(u1_cal.min(), u1_val.min()) * 0.9,
        max(u1_cal.max(), u1_val.max()) * 1.1,
        200,
    )
    colors = ["b-", "r--", "g-."]
    for fit, style in zip(fits, colors):
        if fit.converged:
            r_pred = fit.r_func(u_fine, geom_val)
            ax.plot(u_fine, r_pred, style, lw=2, label=f"{fit.name} (воздух)")

    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_ylabel("Доля утечек r")
    ax.set_title("Валидация: все альтернативные модели r(u₁)\n(воздушная модель, A_ок=20 м²)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "08_all_models_validation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parity_best(
    u1_cal: np.ndarray,
    r_cal: np.ndarray,
    geom_cal: dict,
    u1_val: np.ndarray,
    r_val: np.ndarray,
    geom_val: dict,
    best_fit: DirectFitResult,
) -> str:
    """График 9: Parity plot лучшей альтернативной модели."""
    fig, ax = plt.subplots(figsize=(6, 6))

    r_pred_cal = best_fit.r_func(u1_cal, geom_cal)
    r_pred_val = best_fit.r_func(u1_val, geom_val)

    ax.plot(r_cal, r_pred_cal, "bs", ms=7, label="Калибровка (вода)")
    ax.plot(r_val, r_pred_val, "ro", ms=7, label="Валидация (воздух)")

    all_r = np.concatenate([r_cal, r_val, r_pred_cal, r_pred_val])
    lims = [0, max(all_r.max(), 0.5) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Идеальное совпадение")

    ax.set_xlabel("r эксперимент")
    ax.set_ylabel("r расчёт")
    ax.set_title(f"Parity plot: {best_fit.name}")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "09_parity_best.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
