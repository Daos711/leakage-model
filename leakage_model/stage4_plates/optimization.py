"""Оптимизация конфигурации пластин (этап 4.2).

Подставляет суррогатные модели в уравнение F_пл(r) = 0
и находит конфигурацию, минимизирующую r.
"""

import logging

import numpy as np
from scipy.optimize import minimize_scalar, minimize

from .model import solve_r_plates
from .surrogates import predict_series3, predict_series4

logger = logging.getLogger(__name__)


def _solve_r_for_params(zeta_pl, delta_c0, u1, geom, base_params, beta, L, eps,
                        criterion="Re"):
    """Решить F_пл(r)=0 для заданных ζ_пл и Δc₀."""
    a_xi, b_xi, c0 = base_params
    r, conv, note = solve_r_plates(
        u1, geom, a_xi, b_xi, c0, beta, L, eps,
        zeta_pl=zeta_pl, delta_c0=delta_c0, criterion=criterion,
    )
    if not conv:
        return 1.0  # штраф: максимальная утечка
    return r


def optimize_angle(surrogates, u1, geom, base_params, beta, L, eps,
                   criterion="Re"):
    """Найти оптимальный угол α*, минимизирующий r.

    Возвращает (alpha_opt, r_opt, details).
    """
    surr3 = surrogates[3]

    def objective(alpha):
        zeta, dc0 = predict_series3(alpha, surr3)
        return _solve_r_for_params(zeta, dc0, u1, geom, base_params, beta, L, eps,
                                   criterion)

    result = minimize_scalar(objective, bounds=(20.0, 65.0), method="bounded",
                             options={"xatol": 0.01})
    alpha_opt = result.x
    r_opt = result.fun

    # Построить профиль r(α)
    alphas = np.linspace(20, 65, 91)
    r_profile = np.array([objective(a) for a in alphas])

    details = {
        "alpha_range": alphas.tolist(),
        "r_profile": r_profile.tolist(),
        "u1": u1,
    }

    logger.info("Оптимизация по углу: α*=%.1f°, r*=%.4f (u₁=%.2f м/с)",
                alpha_opt, r_opt, u1)

    return alpha_opt, r_opt, details


def optimize_width(surrogates, u1, geom, base_params, beta, L, eps,
                   criterion="Re"):
    """Найти оптимальную ширину b*.

    Возвращает (width_opt, r_opt, details).
    """
    surr4 = surrogates[4]

    def objective(width_mm):
        zeta, dc0 = predict_series4(width_mm, surr4)
        return _solve_r_for_params(zeta, dc0, u1, geom, base_params, beta, L, eps,
                                   criterion)

    result = minimize_scalar(objective, bounds=(100.0, 1500.0), method="bounded",
                             options={"xatol": 1.0})
    width_opt = result.x
    r_opt = result.fun

    # Профиль r(b)
    widths = np.linspace(100, 1500, 141)
    r_profile = np.array([objective(b) for b in widths])

    details = {
        "width_range": widths.tolist(),
        "r_profile": r_profile.tolist(),
        "u1": u1,
    }

    logger.info("Оптимизация по ширине: b*=%.0f мм, r*=%.4f (u₁=%.2f м/с)",
                width_opt, r_opt, u1)

    return width_opt, r_opt, details


def optimize_joint(surrogates, u1, geom, base_params, beta, L, eps,
                   criterion="Re"):
    """Совместная оптимизация (α, b).

    ИССЛЕДОВАТЕЛЬСКИЙ СЦЕНАРИЙ: допущение — ζ_пл и Δc₀ мультипликативно
    разделимы по α и b. Не подтверждено данными.

    Возвращает (alpha_opt, width_opt, r_opt, details).
    """
    surr3 = surrogates[3]
    surr4 = surrogates[4]

    # Референсные значения при α=45°, b=1000 мм
    zeta_ref_a, dc0_ref_a = predict_series3(45.0, surr3)
    zeta_ref_b, dc0_ref_b = predict_series4(1000.0, surr4)

    # Избегаем деления на 0
    zeta_ref_a = max(zeta_ref_a, 1e-10)
    dc0_ref_a = dc0_ref_a if abs(dc0_ref_a) > 1e-10 else 1e-10
    zeta_ref_b = max(zeta_ref_b, 1e-10)
    dc0_ref_b = dc0_ref_b if abs(dc0_ref_b) > 1e-10 else 1e-10

    def objective(params):
        alpha, width_mm = params
        zeta_a, dc0_a = predict_series3(alpha, surr3)
        zeta_b, dc0_b = predict_series4(width_mm, surr4)

        # Мультипликативная модель: масштабирование от референса
        zeta = zeta_a * (zeta_b / zeta_ref_b)
        dc0 = dc0_a * (dc0_b / dc0_ref_b)

        return _solve_r_for_params(max(zeta, 0.0), dc0, u1, geom, base_params,
                                   beta, L, eps, criterion)

    result = minimize(objective, x0=[40.0, 750.0],
                      bounds=[(20.0, 65.0), (100.0, 1500.0)],
                      method="L-BFGS-B")
    alpha_opt, width_opt = result.x
    r_opt = result.fun

    details = {
        "u1": u1,
        "note": "Мультипликативное допущение — не подтверждено данными",
    }

    logger.info("Совместная оптимизация: α*=%.1f°, b*=%.0f мм, r*=%.4f (u₁=%.2f м/с)",
                alpha_opt, width_opt, r_opt, u1)

    return alpha_opt, width_opt, r_opt, details
