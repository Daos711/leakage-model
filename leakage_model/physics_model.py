"""Полуэмпирическая модель разделения потока (этап 3).

Решатель F(r) = 0 на основе баланса энергии с потерями Борда-Карно,
трением Чёрчилля, скрытым параметром блокировки ξ и асимметричным членом C_β.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .idelchik import churchill_friction, EPS_DEFAULT, L_UPPER_DEFAULT
from .model import calc_Re
from .physics_closures import calc_xi, calc_phi, calc_C_beta

logger = logging.getLogger(__name__)


@dataclass
class PhysicsResult:
    """Результат расчёта для набора точек."""

    r_pred: np.ndarray
    xi: np.ndarray
    phi_up: np.ndarray
    phi_down: np.ndarray
    C_beta: np.ndarray
    u1: np.ndarray
    converged: np.ndarray
    notes: list = field(default_factory=list)


def borda_carnot_loss_coeff(phi):
    """Безразмерный коэффициент потерь Борда-Карно: (1/φ − 1)²."""
    return (1.0 / phi - 1.0) ** 2


def residual_F(r, u1, geom, a_xi, b_xi, c0, beta,
               L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
               R_down=0.0, criterion="Re"):
    """Невязка F̃(r) = 0 — безразмерное замыкающее уравнение.

    F̃(r) = r²·[1 + (1/φ₂−1)² + λL/D]
          − (1−r)²·[1 + (1/φ₃−1)²]
          + c₀·cos²β·(1−ξ)
          + R̃_down·(1−r)²

    Параметры
    ---------
    r : float
        Доля утечек (0, 1).
    u1 : float
        Скорость в окне, м/с.
    geom : dict
        Геометрия (A_ok, A_s, D, D_h, nu).
    a_xi, b_xi : float
        Параметры сигмоиды ξ.
    c0 : float
        Коэффициент асимметричного члена C_β.
    beta : float
        Угол подвода, рад.
    L_upper : float
        Длина верхней ветви, м.
    eps : float
        Шероховатость, м.
    R_down : float
        Сопротивление нижней сети, Н·с²/м⁵.
    criterion : str
        Аргумент ξ: 'Re', 'Fr' или 'u1'.
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

    # Скорость в верхней ветви после расширения
    u2 = r * sigma * u1
    # Re₂ для трения
    Re2 = abs(u2) * D / nu
    Re2 = max(Re2, 1.0)
    lam = churchill_friction(Re2, D, eps)

    # Коэффициенты потерь Борда-Карно
    bc_up = borda_carnot_loss_coeff(phi_up)
    bc_down = borda_carnot_loss_coeff(phi_down)

    # Асимметричный член ядра
    C_b = calc_C_beta(xi, beta, c0)

    # Безразмерное сопротивление нижней сети
    if R_down > 0.0:
        A_s = geom["A_s"]
        R_down_dimless = 2.0 * R_down * (sigma * u1 * A_s) ** 2 / (sigma * u1) ** 2
    else:
        R_down_dimless = 0.0

    # F̃(r) = 0
    term_up = r ** 2 * (1.0 + bc_up + lam * L_upper / D)
    term_down = (1.0 - r) ** 2 * (1.0 + bc_down)
    term_Rdown = R_down_dimless * (1.0 - r) ** 2

    return term_up - term_down + C_b + term_Rdown


def solve_r(u1, geom, a_xi, b_xi, c0, beta,
            L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
            R_down=0.0, criterion="Re"):
    """Решить F(r)=0 методом Brent.

    Возвращает (r, phi_up, phi_down, xi, C_beta, converged, note).
    """
    r_lo, r_hi = 1e-8, 1.0 - 1e-8
    sigma = geom["A_ok"] / geom["A_s"]
    nu = geom["nu"]

    # Вычислить ξ, φ, C_β для возврата
    if criterion == "Re":
        crit_val = u1 * geom["D_h"] / nu
    elif criterion == "Fr":
        crit_val = u1 / np.sqrt(9.81 * geom["D_h"])
    elif criterion == "u1":
        crit_val = u1
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    xi = float(calc_xi(crit_val, a_xi, b_xi))
    phi_up = float(calc_phi(xi, sigma, beta, "up"))
    phi_down = float(calc_phi(xi, sigma, beta, "down"))
    C_b = float(calc_C_beta(xi, beta, c0))

    args = (u1, geom, a_xi, b_xi, c0, beta, L_upper, eps, R_down, criterion)

    f_lo = residual_F(r_lo, *args)
    f_hi = residual_F(r_hi, *args)

    if np.isnan(f_lo) or np.isnan(f_hi):
        note = f"nan_in_residual: u₁={u1:.2f} м/с"
        logger.warning(note)
        return np.nan, phi_up, phi_down, xi, C_b, False, note

    if f_lo * f_hi > 0:
        note = (f"no_bracket: F({r_lo:.1e})={f_lo:.4f}, "
                f"F({r_hi:.1e})={f_hi:.4f}. u₁={u1:.2f} м/с.")
        logger.warning(note)
        return np.nan, phi_up, phi_down, xi, C_b, False, note

    try:
        r_sol = brentq(residual_F, r_lo, r_hi, args=args, xtol=1e-12)
        return r_sol, phi_up, phi_down, xi, C_b, True, ""
    except ValueError as e:
        note = f"solver_error: u₁={u1:.2f}: {e}"
        logger.warning(note)
        return np.nan, phi_up, phi_down, xi, C_b, False, note


def solve_all(u1_array, geom, a_xi, b_xi, c0, beta,
              L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
              R_down=0.0, criterion="Re"):
    """Решить для массива u₁. Возвращает PhysicsResult."""
    u1_arr = np.asarray(u1_array, dtype=float)
    n = len(u1_arr)
    r_pred = np.full(n, np.nan)
    xi_arr = np.full(n, np.nan)
    phi_up_arr = np.full(n, np.nan)
    phi_down_arr = np.full(n, np.nan)
    C_beta_arr = np.full(n, np.nan)
    converged = np.zeros(n, dtype=bool)
    notes = []

    for i, u1 in enumerate(u1_arr):
        r, pu, pd, xi, cb, conv, note = solve_r(
            u1, geom, a_xi, b_xi, c0, beta, L_upper, eps, R_down, criterion
        )
        r_pred[i] = r
        xi_arr[i] = xi
        phi_up_arr[i] = pu
        phi_down_arr[i] = pd
        C_beta_arr[i] = cb
        converged[i] = conv
        if note:
            notes.append(note)

    n_conv = int(converged.sum())
    logger.debug(f"Сошлось {n_conv}/{n} точек (criterion={criterion})")

    return PhysicsResult(
        r_pred=r_pred,
        xi=xi_arr,
        phi_up=phi_up_arr,
        phi_down=phi_down_arr,
        C_beta=C_beta_arr,
        u1=u1_arr,
        converged=converged,
        notes=notes,
    )
