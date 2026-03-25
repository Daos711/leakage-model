"""Суррогатные модели для параметров пластин (этап 4.1).

Связывают ζ_пл и Δc₀ с геометрией пластин по сериям 1, 3, 4.
"""

import logging

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Серия 3: угол наклона (вставки 23–29)
# ---------------------------------------------------------------------------

def fit_series3_angle(angles, zeta_pl, delta_c0):
    """Квадратичная регрессия по углу для серии 3.

    Δc₀(α) = p₀ + p₁·(α − 40) + p₂·(α − 40)²
    ζ_пл(α) = q₀ + q₁·(α − 40) + q₂·(α − 40)²

    Возвращает dict с коэффициентами, R², оптимальный угол.
    """
    angles = np.asarray(angles, dtype=float)
    zeta_pl = np.asarray(zeta_pl, dtype=float)
    delta_c0 = np.asarray(delta_c0, dtype=float)

    x = angles - 40.0  # центрированный угол

    # Δc₀(α): квадратичная регрессия
    p = np.polyfit(x, delta_c0, 2)
    p2, p1, p0 = p
    dc0_pred = np.polyval(p, x)
    ss_res = np.sum((delta_c0 - dc0_pred) ** 2)
    ss_tot = np.sum((delta_c0 - np.mean(delta_c0)) ** 2)
    R2_dc0 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    rmse_dc0 = np.sqrt(np.mean((delta_c0 - dc0_pred) ** 2))

    # Оптимальный угол: максимум Δc₀ → dΔc₀/dα = 0 → α* = −p₁/(2p₂) + 40
    alpha_opt_dc0 = -p1 / (2 * p2) + 40.0 if abs(p2) > 1e-15 else np.nan

    # ζ_пл(α): квадратичная регрессия
    q = np.polyfit(x, zeta_pl, 2)
    q2, q1, q0 = q
    zeta_pred = np.polyval(q, x)
    ss_res_z = np.sum((zeta_pl - zeta_pred) ** 2)
    ss_tot_z = np.sum((zeta_pl - np.mean(zeta_pl)) ** 2)
    R2_zeta = 1.0 - ss_res_z / ss_tot_z if ss_tot_z > 1e-30 else 0.0
    rmse_zeta = np.sqrt(np.mean((zeta_pl - zeta_pred) ** 2))

    result = {
        "series": 3,
        "dc0_coeffs": {"p0": p0, "p1": p1, "p2": p2},
        "dc0_R2": R2_dc0,
        "dc0_RMSE": rmse_dc0,
        "alpha_opt_dc0": alpha_opt_dc0,
        "zeta_coeffs": {"q0": q0, "q1": q1, "q2": q2},
        "zeta_R2": R2_zeta,
        "zeta_RMSE": rmse_zeta,
    }

    logger.info("Серия 3 (угол):")
    logger.info("  Δc₀(α) = %.4f + %.4f·(α−40) + %.6f·(α−40)²", p0, p1, p2)
    logger.info("  R²=%.4f, RMSE=%.4f, α*(Δc₀ макс)=%.1f°", R2_dc0, rmse_dc0, alpha_opt_dc0)
    logger.info("  ζ_пл(α) = %.4f + %.4f·(α−40) + %.6f·(α−40)²", q0, q1, q2)
    logger.info("  R²=%.4f, RMSE=%.4f", R2_zeta, rmse_zeta)

    return result


def predict_series3(alpha, surr):
    """Предсказать ζ_пл и Δc₀ по углу α из суррогата серии 3."""
    x = alpha - 40.0
    dc = surr["dc0_coeffs"]
    zc = surr["zeta_coeffs"]
    dc0 = dc["p0"] + dc["p1"] * x + dc["p2"] * x ** 2
    zeta = zc["q0"] + zc["q1"] * x + zc["q2"] * x ** 2
    zeta = max(zeta, 0.0)  # ζ_пл не может быть отрицательным
    return zeta, dc0


# ---------------------------------------------------------------------------
# Серия 4: ширина пластины (вставки 30–32)
# ---------------------------------------------------------------------------

def fit_series4_width(widths, zeta_pl, delta_c0):
    """Степенной закон по ширине для серии 4.

    ζ_пл(x) = A · x^m,  x = b / 1000
    Δc₀(x) = d₀ + d₁·ln(x)

    Возвращает dict с коэффициентами, R².
    """
    widths = np.asarray(widths, dtype=float)
    zeta_pl = np.asarray(zeta_pl, dtype=float)
    delta_c0 = np.asarray(delta_c0, dtype=float)

    x = widths / 1000.0  # нормированная ширина

    # ζ_пл: степенной закон ln(ζ) = ln(A) + m·ln(x)
    mask_pos = zeta_pl > 1e-10
    if mask_pos.sum() >= 2:
        slope = linregress(np.log(x[mask_pos]), np.log(zeta_pl[mask_pos]))
        m = slope.slope
        A = np.exp(slope.intercept)
        R2_zeta = slope.rvalue ** 2
        zeta_pred = A * x ** m
        rmse_zeta = np.sqrt(np.mean((zeta_pl - zeta_pred) ** 2))
    else:
        A, m = np.nan, np.nan
        R2_zeta = 0.0
        rmse_zeta = np.nan

    # Δc₀: лог-линейная модель Δc₀(x) = d₀ + d₁·ln(x)
    ln_x = np.log(x)
    slope_dc = linregress(ln_x, delta_c0)
    d0 = slope_dc.intercept
    d1 = slope_dc.slope
    R2_dc0 = slope_dc.rvalue ** 2
    dc0_pred = d0 + d1 * ln_x
    rmse_dc0 = np.sqrt(np.mean((delta_c0 - dc0_pred) ** 2))

    result = {
        "series": 4,
        "zeta_coeffs": {"A": A, "m": m},
        "zeta_R2": R2_zeta,
        "zeta_RMSE": rmse_zeta,
        "dc0_coeffs": {"d0": d0, "d1": d1},
        "dc0_R2": R2_dc0,
        "dc0_RMSE": rmse_dc0,
    }

    logger.info("Серия 4 (ширина):")
    logger.info("  ζ_пл(x) = %.4f · x^%.4f  (x = b/1000)", A, m)
    logger.info("  R²=%.4f, RMSE=%.4f", R2_zeta, rmse_zeta)
    logger.info("  Δc₀(x) = %.4f + %.4f·ln(x)", d0, d1)
    logger.info("  R²=%.4f, RMSE=%.4f", R2_dc0, rmse_dc0)

    return result


def predict_series4(width_mm, surr):
    """Предсказать ζ_пл и Δc₀ по ширине b (мм) из суррогата серии 4."""
    x = width_mm / 1000.0
    zc = surr["zeta_coeffs"]
    dc = surr["dc0_coeffs"]

    # ζ_пл: если степенной закон не подогнался (NaN коэффициенты) → ζ_пл = 0
    if np.isnan(zc["A"]) or np.isnan(zc["m"]):
        zeta = 0.0
    else:
        zeta = max(zc["A"] * x ** zc["m"], 0.0)

    dc0 = dc["d0"] + dc["d1"] * np.log(x)
    return zeta, dc0


# ---------------------------------------------------------------------------
# Серия 1: количество пластин (вставки 3, 5, 4 — 45°)
# ---------------------------------------------------------------------------

def _saturation_model(n, y_inf, n0):
    """y(n) = y∞ · n / (n + n₀)."""
    return y_inf * n / (n + n0)


def fit_series1_nplates(n_plates, zeta_pl, delta_c0):
    """Закон насыщения по числу пластин для серии 1 (только 45°).

    y(n) = y∞ · n / (n + n₀)

    Возвращает dict с коэффициентами.
    """
    n_plates = np.asarray(n_plates, dtype=float)
    zeta_pl = np.asarray(zeta_pl, dtype=float)
    delta_c0 = np.asarray(delta_c0, dtype=float)

    result = {"series": 1}

    # ζ_пл(n): насыщение
    try:
        popt_z, _ = curve_fit(_saturation_model, n_plates, zeta_pl,
                              p0=[10.0, 2.0], bounds=([0, 0.01], [100, 50]),
                              maxfev=5000)
        y_inf_z, n0_z = popt_z
        zeta_pred = _saturation_model(n_plates, y_inf_z, n0_z)
        ss_res = np.sum((zeta_pl - zeta_pred) ** 2)
        ss_tot = np.sum((zeta_pl - np.mean(zeta_pl)) ** 2)
        R2_z = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
        rmse_z = np.sqrt(np.mean((zeta_pl - zeta_pred) ** 2))
        result["zeta_coeffs"] = {"y_inf": y_inf_z, "n0": n0_z}
        result["zeta_R2"] = R2_z
        result["zeta_RMSE"] = rmse_z
        logger.info("Серия 1 (n пластин):")
        logger.info("  ζ_пл(n) = %.4f · n / (n + %.4f)", y_inf_z, n0_z)
        logger.info("  R²=%.4f, RMSE=%.4f", R2_z, rmse_z)
    except (RuntimeError, ValueError) as e:
        logger.warning("Серия 1: curve_fit для ζ_пл не сошёлся: %s", e)
        result["zeta_coeffs"] = {"y_inf": np.nan, "n0": np.nan}
        result["zeta_R2"] = 0.0
        result["zeta_RMSE"] = np.nan

    # Δc₀(n): насыщение
    try:
        popt_d, _ = curve_fit(_saturation_model, n_plates, delta_c0,
                              p0=[0.2, 1.0], bounds=([0, 0.01], [10, 50]),
                              maxfev=5000)
        y_inf_d, n0_d = popt_d
        dc0_pred = _saturation_model(n_plates, y_inf_d, n0_d)
        ss_res = np.sum((delta_c0 - dc0_pred) ** 2)
        ss_tot = np.sum((delta_c0 - np.mean(delta_c0)) ** 2)
        R2_d = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
        rmse_d = np.sqrt(np.mean((delta_c0 - dc0_pred) ** 2))
        result["dc0_coeffs"] = {"y_inf": y_inf_d, "n0": n0_d}
        result["dc0_R2"] = R2_d
        result["dc0_RMSE"] = rmse_d
        logger.info("  Δc₀(n) = %.4f · n / (n + %.4f)", y_inf_d, n0_d)
        logger.info("  R²=%.4f, RMSE=%.4f", R2_d, rmse_d)
    except (RuntimeError, ValueError) as e:
        logger.warning("Серия 1: curve_fit для Δc₀ не сошёлся: %s", e)
        result["dc0_coeffs"] = {"y_inf": np.nan, "n0": np.nan}
        result["dc0_R2"] = 0.0
        result["dc0_RMSE"] = np.nan

    return result


def predict_series1(n, surr):
    """Предсказать ζ_пл и Δc₀ по числу пластин n из суррогата серии 1."""
    zc = surr["zeta_coeffs"]
    dc = surr["dc0_coeffs"]
    zeta = _saturation_model(n, zc["y_inf"], zc["n0"])
    dc0 = _saturation_model(n, dc["y_inf"], dc["n0"])
    return max(zeta, 0.0), dc0


# ---------------------------------------------------------------------------
# Универсальный предиктор
# ---------------------------------------------------------------------------

def predict_zeta_dc0(alpha=None, width=None, n_plates=None, surrogates=None):
    """По суррогатам предсказать ζ_пл и Δc₀ для заданной геометрии."""
    if surrogates is None:
        raise ValueError("surrogates dict required")

    if alpha is not None and 3 in surrogates:
        return predict_series3(alpha, surrogates[3])
    if width is not None and 4 in surrogates:
        return predict_series4(width, surrogates[4])
    if n_plates is not None and 1 in surrogates:
        return predict_series1(n_plates, surrogates[1])

    raise ValueError("No matching surrogate for given parameters")
