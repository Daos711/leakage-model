"""Модель с направляющими пластинами (этап 4).

Расширение базовой модели этапа 3: добавление ζ_пл (сопротивление пластин)
и Δc₀ (направляющий эффект) к уравнению баланса энергии.
"""

import logging

import numpy as np
from scipy.optimize import brentq

from .idelchik import churchill_friction, EPS_DEFAULT, L_UPPER_DEFAULT
from .model import calc_Re
from .physics_closures import calc_xi, calc_phi, calc_C_beta
from .physics_model import borda_carnot_loss_coeff

logger = logging.getLogger(__name__)


def residual_F_plates(r, u1, geom, a_xi, b_xi, c0, beta,
                      L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
                      R_down=0.0, criterion="Re",
                      zeta_pl=0.0, delta_c0=0.0):
    """Невязка F̃_pl(r) = 0 с учётом пластин.

    F̃_pl(r) = r²·[1 + (1/φ₂−1)² + λL/D + ζ_пл]
             − (1−r)²·[1 + (1/φ₃−1)²]
             + [c₀ + Δc₀]·cos²β·(1 − ξ)
    """
    sigma = geom["A_ok"] / geom["A_s"]
    D = geom["D"]
    nu = geom["nu"]

    # Критерий для ξ
    if criterion == "Re":
        crit_val = u1 * geom["D_h"] / nu
    elif criterion == "Fr":
        g = 9.81
        crit_val = u1 / np.sqrt(g * geom["D_h"])
    elif criterion == "u1":
        crit_val = u1
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    xi = calc_xi(crit_val, a_xi, b_xi)
    phi_up = calc_phi(xi, sigma, beta, "up")
    phi_down = calc_phi(xi, sigma, beta, "down")

    # Скорость в верхней ветви
    u2 = r * sigma * u1
    Re2 = abs(u2) * D / nu
    Re2 = max(Re2, 1.0)
    lam = churchill_friction(Re2, D, eps)

    # Коэффициенты потерь Борда-Карно
    bc_up = borda_carnot_loss_coeff(phi_up)
    bc_down = borda_carnot_loss_coeff(phi_down)

    # Асимметричный член с добавкой Δc₀
    C_b = (c0 + delta_c0) * np.cos(beta) ** 2 * (1.0 - xi)

    # F̃(r) = 0
    term_up = r ** 2 * (1.0 + bc_up + lam * L_upper / D + zeta_pl)
    term_down = (1.0 - r) ** 2 * (1.0 + bc_down)

    return term_up - term_down + C_b


def solve_r_plates(u1, geom, a_xi, b_xi, c0, beta,
                   L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
                   zeta_pl=0.0, delta_c0=0.0,
                   criterion="Re"):
    """Brent solver с пластинами.

    Возвращает (r, converged, note).
    """
    r_lo, r_hi = 1e-8, 1.0 - 1e-8

    args = (u1, geom, a_xi, b_xi, c0, beta, L_upper, eps, 0.0,
            criterion, zeta_pl, delta_c0)

    f_lo = residual_F_plates(r_lo, *args)
    f_hi = residual_F_plates(r_hi, *args)

    if np.isnan(f_lo) or np.isnan(f_hi):
        return np.nan, False, f"nan_in_residual: u₁={u1:.2f}"

    if f_lo * f_hi > 0:
        return np.nan, False, (
            f"no_bracket: F({r_lo:.1e})={f_lo:.4f}, "
            f"F({r_hi:.1e})={f_hi:.4f}. u₁={u1:.2f}"
        )

    try:
        r_sol = brentq(residual_F_plates, r_lo, r_hi, args=args, xtol=1e-12)
        return r_sol, True, ""
    except ValueError as e:
        return np.nan, False, f"solver_error: u₁={u1:.2f}: {e}"


def predict_plates(u1_arr, geom, a_xi, b_xi, c0, beta,
                   L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
                   zeta_pl=0.0, delta_c0=0.0, criterion="Re"):
    """Предсказание r для массива u₁ с пластинами.

    Возвращает (r_pred, converged, notes).
    """
    u1_arr = np.asarray(u1_arr, dtype=float)
    n = len(u1_arr)
    r_pred = np.full(n, np.nan)
    converged = np.zeros(n, dtype=bool)
    notes = []

    for i, u1 in enumerate(u1_arr):
        r, conv, note = solve_r_plates(
            u1, geom, a_xi, b_xi, c0, beta, L_upper, eps,
            zeta_pl, delta_c0, criterion,
        )
        r_pred[i] = r
        converged[i] = conv
        if note:
            notes.append(note)

    return r_pred, converged, notes
