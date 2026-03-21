"""Импульсная 0D-модель: баланс импульса, обратная задача для C_M."""

import math

from .friction import churchill_friction


def momentum_residual(r: float, Q1: float, geom: dict,
                      rho: float, C_M: float) -> float:
    """Невязка баланса импульса F(r) = LHS - RHS.

    Уравнение: (p̃₃ − p̃₂)·A_с + C_M·(ρu₁²/2)·A_ок = ρQ₂u₂ − ρQ₃u₃ + ρQ₁u₁γ

    Parameters
    ----------
    r : float
        Доля утечек Q₂/Q₁.
    Q1 : float
        Расход на входе, м³/с.
    geom : dict
        Геометрические параметры.
    rho : float
        Плотность воздуха, кг/м³.
    C_M : float
        Калибруемый коэффициент локальной силовой реакции узла.

    Returns
    -------
    float
        Невязка F(r) = LHS - RHS.
    """
    A_ok = geom["A_ok"]
    A_s = geom["A_s"]
    D = geom["D"]
    L_up = geom["L_up"]
    beta = geom["beta"]
    nu = geom["nu"]

    gamma = math.cos(beta)

    u1 = Q1 / A_ok
    Q2 = r * Q1
    Q3 = (1 - r) * Q1
    u2 = Q2 / A_s
    u3 = Q3 / A_s

    # Верхняя ветвь: трение
    Re2 = abs(u2) * D / nu
    lam2 = churchill_friction(Re2)
    dp_upper = lam2 * (L_up / D) * rho * u2 ** 2 / 2  # p̃₂

    # Нижняя ветвь: базовая постановка
    dp_lower = 0.0  # p̃₃

    # Баланс импульса
    LHS = (dp_lower - dp_upper) * A_s + C_M * (rho * u1 ** 2 / 2) * A_ok
    RHS = rho * Q2 * u2 - rho * Q3 * u3 + rho * Q1 * u1 * gamma

    return LHS - RHS


def calc_C_M(r_exp: float, Q1: float, geom: dict, rho: float) -> float:
    """Обратная задача: вычисление C_M из баланса импульса при известном r_exp.

    Parameters
    ----------
    r_exp : float
        Экспериментальная доля утечек.
    Q1 : float
        Расход на входе, м³/с.
    geom : dict
        Геометрические параметры.
    rho : float
        Плотность воздуха, кг/м³.

    Returns
    -------
    float
        Значение C_M.
    """
    A_ok = geom["A_ok"]
    A_s = geom["A_s"]
    D = geom["D"]
    L_up = geom["L_up"]
    beta = geom["beta"]
    nu = geom["nu"]

    gamma = math.cos(beta)

    u1 = Q1 / A_ok
    Q2 = r_exp * Q1
    Q3 = (1 - r_exp) * Q1
    u2 = Q2 / A_s
    u3 = Q3 / A_s

    Re2 = abs(u2) * D / nu
    lam2 = churchill_friction(Re2)
    dp_upper = lam2 * (L_up / D) * rho * u2 ** 2 / 2
    dp_lower = 0.0

    # RHS - (p̃₃ − p̃₂)·A_с = C_M · (ρu₁²/2) · A_ок
    RHS = rho * Q2 * u2 - rho * Q3 * u3 + rho * Q1 * u1 * gamma
    pressure_term = (dp_lower - dp_upper) * A_s
    denom = (rho * u1 ** 2 / 2) * A_ok

    C_M = (RHS - pressure_term) / denom
    return C_M
