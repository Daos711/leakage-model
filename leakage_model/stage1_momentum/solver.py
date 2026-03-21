"""Решатели: Brent (основной) и Newton-Raphson (дополнительный)."""

from scipy.optimize import brentq

from ..core.config import NR_MAX_ITER, NR_TOL
from .model import momentum_residual


def solve_r_brent(Q1: float, geom: dict, rho: float, C_M: float) -> float:
    """Решение F(r) = 0 методом Brent (bracketing) на интервале (0, 1).

    Parameters
    ----------
    Q1 : float
        Расход на входе, м³/с.
    geom : dict
        Геометрические параметры.
    rho : float
        Плотность воздуха, кг/м³.
    C_M : float
        Коэффициент силовой реакции узла.

    Returns
    -------
    float
        Доля утечек r.
    """
    a, b = 1e-8, 1 - 1e-8
    fa = momentum_residual(a, Q1, geom, rho, C_M)
    fb = momentum_residual(b, Q1, geom, rho, C_M)

    if fa * fb > 0:
        raise RuntimeError(
            f"Нет смены знака F(r) на (0,1): F({a})={fa:.4e}, F({b})={fb:.4e}"
        )

    r = brentq(momentum_residual, a, b, args=(Q1, geom, rho, C_M),
               xtol=1e-14, rtol=1e-14)
    return r


def solve_r_newton(Q1: float, geom: dict, rho: float, C_M: float,
                   r0: float = 0.3) -> float:
    """Решение F(r) = 0 методом Newton-Raphson с численной производной.

    Parameters
    ----------
    Q1 : float
        Расход на входе, м³/с.
    geom : dict
        Геометрические параметры.
    rho : float
        Плотность воздуха, кг/м³.
    C_M : float
        Коэффициент силовой реакции узла.
    r0 : float
        Начальное приближение.

    Returns
    -------
    float
        Доля утечек r.
    """
    r = r0
    dr = 1e-8

    for _ in range(NR_MAX_ITER):
        F_r = momentum_residual(r, Q1, geom, rho, C_M)

        # Численная производная (центральные разности)
        F_plus = momentum_residual(r + dr, Q1, geom, rho, C_M)
        F_minus = momentum_residual(r - dr, Q1, geom, rho, C_M)
        dF_dr = (F_plus - F_minus) / (2 * dr)

        if abs(dF_dr) < 1e-30:
            raise RuntimeError("Слишком малая производная в Newton-Raphson")

        r_new = r - F_r / dF_dr
        r_new = min(max(r_new, 1e-10), 1.0 - 1e-10)

        if abs(r_new - r) < NR_TOL:
            return r_new
        r = r_new

    return r
