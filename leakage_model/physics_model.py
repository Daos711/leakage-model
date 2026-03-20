"""Полуэмпирическая модель разделения потока (этап 3).

Решатель F(r) = 0 на основе баланса энергии с потерями Борда-Карно,
трением Чёрчилля и скрытым параметром блокировки ξ.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .idelchik import churchill_friction, EPS_DEFAULT, L_UPPER_DEFAULT
from .model import calc_Re
from .physics_closures import calc_xi, calc_phi

logger = logging.getLogger(__name__)


@dataclass
class PhysicsResult:
    """Результат расчёта для набора точек."""

    r_pred: np.ndarray
    xi: np.ndarray
    phi_up: np.ndarray
    phi_down: np.ndarray
    u1: np.ndarray
    converged: np.ndarray
    notes: list = field(default_factory=list)


def borda_carnot_loss_coeff(phi):
    """Безразмерный коэффициент потерь Борда-Карно: (1/φ − 1)²."""
    return (1.0 / phi - 1.0) ** 2


def residual_F(r, u1, geom, a_xi, b_xi, beta,
               L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
               R_down=0.0, criterion="Re"):
    """Невязка F̃(r) = 0 — безразмерное замыкающее уравнение (§3.8).

    F̃(r) = r²·[1 + (1/φ₂−1)² + λL/D]
          − (1−r)²·[1 + (1/φ₃−1)²]
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

    # Безразмерное сопротивление нижней сети
    A_s = geom["A_s"]
    if R_down > 0.0:
        # R̃_down = R_down · 2·(σ·u₁·A_с)² / (ρ·(σ·u₁)²)
        # = R_down · 2·A_с²  (ρ сокращается в безразмерной форме)
        # Но здесь мы уже поделили на ½ρ(σu₁)², поэтому:
        # R_down·Q₃² / (½ρ(σu₁)²) = R_down·((1-r)·σ·u1·A_s)² / (½ρ(σu₁)²)
        # Для безразмерности нужна плотность, которой нет в geom.
        # В лабораторном варианте R_down = 0, поэтому пока заглушка.
        R_down_dimless = 0.0
    else:
        R_down_dimless = 0.0

    # F̃(r) = 0
    term_up = r ** 2 * (1.0 + bc_up + lam * L_upper / D)
    term_down = (1.0 - r) ** 2 * (1.0 + bc_down)
    term_Rdown = R_down_dimless * (1.0 - r) ** 2

    return term_up - term_down + term_Rdown


def solve_r(u1, geom, a_xi, b_xi, beta,
            L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
            R_down=0.0, criterion="Re"):
    """Решить F(r)=0 методом Brent.

    Возвращает (r, phi_up, phi_down, xi, converged, note).
    """
    r_lo, r_hi = 1e-8, 1.0 - 1e-8
    sigma = geom["A_ok"] / geom["A_s"]
    nu = geom["nu"]

    # Вычислить ξ, φ для возврата
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

    args = (u1, geom, a_xi, b_xi, beta, L_upper, eps, R_down, criterion)

    f_lo = residual_F(r_lo, *args)
    f_hi = residual_F(r_hi, *args)

    if f_lo * f_hi > 0:
        note = (f"Нет смены знака F(r): F({r_lo:.1e})={f_lo:.4f}, "
                f"F({r_hi:.1e})={f_hi:.4f}. u₁={u1:.2f} м/с.")
        logger.warning(note)
        return np.nan, phi_up, phi_down, xi, False, note

    try:
        r_sol = brentq(residual_F, r_lo, r_hi, args=args, xtol=1e-12)
        return r_sol, phi_up, phi_down, xi, True, ""
    except ValueError as e:
        note = f"Brent не сошёлся для u₁={u1:.2f}: {e}"
        logger.warning(note)
        return np.nan, phi_up, phi_down, xi, False, note


def solve_all(u1_array, geom, a_xi, b_xi, beta,
              L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
              R_down=0.0, criterion="Re"):
    """Решить для массива u₁. Возвращает PhysicsResult."""
    u1_arr = np.asarray(u1_array, dtype=float)
    n = len(u1_arr)
    r_pred = np.full(n, np.nan)
    xi_arr = np.full(n, np.nan)
    phi_up_arr = np.full(n, np.nan)
    phi_down_arr = np.full(n, np.nan)
    converged = np.zeros(n, dtype=bool)
    notes = []

    for i, u1 in enumerate(u1_arr):
        r, pu, pd, xi, conv, note = solve_r(
            u1, geom, a_xi, b_xi, beta, L_upper, eps, R_down, criterion
        )
        r_pred[i] = r
        xi_arr[i] = xi
        phi_up_arr[i] = pu
        phi_down_arr[i] = pd
        converged[i] = conv
        if note:
            notes.append(note)

    n_conv = int(converged.sum())
    logger.info(f"Сошлось {n_conv}/{n} точек (criterion={criterion})")

    return PhysicsResult(
        r_pred=r_pred,
        xi=xi_arr,
        phi_up=phi_up_arr,
        phi_down=phi_down_arr,
        u1=u1_arr,
        converged=converged,
        notes=notes,
    )
