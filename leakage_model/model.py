"""Математическая модель: явная формула, обратная задача, Newton-Raphson."""

import numpy as np

from .config import NR_MAX_ITER, NR_MIN_DERIV, NR_TOL


def calc_delta_zeta(r: np.ndarray, A_ok: float, A_s: float) -> np.ndarray:
    """Обратная задача: Δζ из экспериментального r.

    Δζ = (1 - 2r) · (A_ок / A_с)²
    """
    ratio_sq = (A_ok / A_s) ** 2
    return (1.0 - 2.0 * r) * ratio_sq


def calc_Re(u1: np.ndarray, D_h: float, nu: float) -> np.ndarray:
    """Число Рейнольдса: Re = u₁ · D_h / ν."""
    return u1 * D_h / nu


def calc_r_explicit(u1: np.ndarray, geom: dict, dz_func) -> np.ndarray:
    """Явное решение для r.

    r = 0.5 - Δζ(Re) / (2 · (A_ок/A_с)²)
    """
    Re = calc_Re(u1, geom["D_h"], geom["nu"])
    dz = dz_func(Re)
    ratio_sq = (geom["A_ok"] / geom["A_s"]) ** 2
    return 0.5 - dz / (2.0 * ratio_sq)


def calc_r_newton(u1: float, geom: dict, dz_func) -> float:
    """Newton-Raphson для F(r) = 0.

    F(r) = r²/A_с² + ζ₁₂/A_ок² - (1-r)²/A_с² - ζ₁₃/A_ок²

    На этапе 1 Δζ не зависит от r, поэтому:
    F(r) = (r² - (1-r)²)/A_с² + Δζ/A_ок²
         = (2r - 1)/A_с² + Δζ/A_ок²

    dF/dr = 2/A_с²
    """
    Re = calc_Re(np.float64(u1), geom["D_h"], geom["nu"])
    dz = dz_func(Re)
    A_s = geom["A_s"]
    A_ok = geom["A_ok"]

    r = 0.5  # начальное приближение

    for _ in range(NR_MAX_ITER):
        F = (2.0 * r - 1.0) / A_s**2 + dz / A_ok**2
        dF_dr = 2.0 / A_s**2

        if abs(dF_dr) < NR_MIN_DERIV:
            raise RuntimeError("Слишком малая производная в Newton-Raphson")

        r_new = r - F / dF_dr
        r_new = min(max(r_new, 1e-10), 1.0 - 1e-10)

        if abs(r_new - r) < NR_TOL:
            return r_new
        r = r_new

    return r


def calc_k_ut(r: np.ndarray) -> np.ndarray:
    """Коэффициент утечек: k_ут = r / (1 - r)."""
    return r / (1.0 - r)
