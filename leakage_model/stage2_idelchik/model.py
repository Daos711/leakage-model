"""Система уравнений и решатель Brent для модели Идельчика.

Физическая схема: поток Q₁ входит через окно и делится на два:
  Q₂ = r·Q₁   — утечки вверх (боковое ответвление)
  Q₃ = (1−r)·Q₁ — шахта вниз (прямой проход)

Уравнения энергии (после исключения p̃₁):

  F(r) = [p̃₂ + ρu₂²/2 + ζ_утеч(r)·ρw_c²/2]
       − [p̃₃ + ρu₃²/2 + ζ_шахта(r)·ρw_c²/2] = 0

где p̃₂ = λ·(L/D)·ρu₂²/2 (трение в верхнем участке ствола), p̃₃ = 0.

После деления на ρw_c²/2 (w_c = u₁ = Q₁/A_ок):

  F(r) = σ²·r²·(1 + λ·L/D) − σ²·(1−r)² + ζ_б(r) − ζ_п(r) = 0

где σ = A_ок/A_с, λ = λ(Re₂) по Черчиллю, Re₂ = r·σ·u₁·D/ν.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from ..core.friction import churchill_friction
from .coefficients import (
    VARIANTS,
    L_UPPER_DEFAULT,
    EPS_DEFAULT,
)

logger = logging.getLogger(__name__)


@dataclass
class IdelchikResult:
    """Результат расчёта по модели Идельчика для одного варианта."""

    variant: str
    variant_name: str
    r_pred: np.ndarray
    u1: np.ndarray
    zeta_branch: np.ndarray
    zeta_straight: np.ndarray
    converged: np.ndarray  # bool array: сошёлся ли решатель для каждой точки
    notes: list[str] = field(default_factory=list)


def residual_F(r, u1, geom, coeff_func, L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT):
    """Невязка F(r) = 0 — нелинейное уравнение для доли утечек.

    Параметры
    ---------
    r : float
        Доля утечек (0 < r < 1)
    u1 : float
        Скорость в окне, м/с
    geom : dict
        Геометрия (A_ok, A_s, D, nu, ...)
    coeff_func : callable
        Стратегия выбора коэффициентов (variant A/B/C)
    L_upper : float
        Длина верхнего участка ствола, м
    eps : float
        Шероховатость стенок, м
    """
    A_ok = geom["A_ok"]
    A_s = geom["A_s"]
    D = geom["D"]
    nu = geom["nu"]

    sigma = A_ok / A_s

    # Скорости в стволе
    u2 = r * sigma * u1        # верхняя часть (утечки)
    u3 = (1.0 - r) * sigma * u1  # нижняя часть (шахта)

    # Коэффициент трения в верхнем стволе
    Re2 = abs(u2) * D / nu if abs(u2) > 1e-12 else 1.0
    lam = churchill_friction(Re2, D, eps)

    # Коэффициенты Идельчика (зависят от r)
    z_b, z_s = coeff_func(r, A_ok, A_s)

    # F(r) / w_c² = σ²·r²·(1 + λ·L/D) − σ²·(1−r)² + ζ_б − ζ_п
    F = (
        sigma ** 2 * r ** 2 * (1.0 + lam * L_upper / D)
        - sigma ** 2 * (1.0 - r) ** 2
        + z_b - z_s
    )
    return F


def solve_r(u1, geom, coeff_func, L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT):
    """Найти r из F(r) = 0 методом Brent.

    Возвращает
    ----------
    r : float или nan
        Доля утечек
    z_b : float
        ζ бокового ответвления в найденной точке
    z_s : float
        ζ прямого прохода в найденной точке
    converged : bool
    note : str
    """
    r_lo, r_hi = 1e-8, 1.0 - 1e-8

    # Проверка смены знака
    F_lo = residual_F(r_lo, u1, geom, coeff_func, L_upper, eps)
    F_hi = residual_F(r_hi, u1, geom, coeff_func, L_upper, eps)

    if F_lo * F_hi > 0:
        # Нет смены знака — модель неприменима
        note = (
            f"Нет смены знака F(r): F({r_lo:.1e})={F_lo:.4f}, "
            f"F({r_hi:.1e})={F_hi:.4f}. "
            f"Эквивалентная схема неприменима для u₁={u1:.2f} м/с."
        )
        logger.warning(note)
        # Грубая оценка: найти минимум |F| для информации
        r_scan = np.linspace(r_lo, r_hi, 1000)
        F_scan = [residual_F(ri, u1, geom, coeff_func, L_upper, eps) for ri in r_scan]
        r_best = r_scan[np.argmin(np.abs(F_scan))]
        z_b, z_s = coeff_func(r_best, geom["A_ok"], geom["A_s"])
        return r_best, z_b, z_s, False, note

    try:
        r_root = brentq(
            residual_F, r_lo, r_hi,
            args=(u1, geom, coeff_func, L_upper, eps),
            xtol=1e-12, rtol=1e-12, maxiter=200,
        )
        z_b, z_s = coeff_func(r_root, geom["A_ok"], geom["A_s"])
        return r_root, z_b, z_s, True, ""

    except ValueError as e:
        note = f"Brent не сошёлся для u₁={u1:.2f}: {e}"
        logger.warning(note)
        return np.nan, np.nan, np.nan, False, note


def solve_all(
    u1_array,
    geom,
    coeff_func,
    variant_key,
    variant_name,
    L_upper=L_UPPER_DEFAULT,
    eps=EPS_DEFAULT,
):
    """Решить F(r)=0 для массива скоростей u₁.

    Возвращает IdelchikResult.
    """
    u1_arr = np.asarray(u1_array, dtype=float)
    n = len(u1_arr)
    r_pred = np.full(n, np.nan)
    z_b_arr = np.full(n, np.nan)
    z_s_arr = np.full(n, np.nan)
    conv = np.zeros(n, dtype=bool)
    notes = []

    for i, u1_i in enumerate(u1_arr):
        r_i, zb_i, zs_i, ok_i, note_i = solve_r(
            u1_i, geom, coeff_func, L_upper, eps,
        )
        r_pred[i] = r_i
        z_b_arr[i] = zb_i
        z_s_arr[i] = zs_i
        conv[i] = ok_i
        if note_i:
            notes.append(note_i)

    n_conv = conv.sum()
    logger.info(
        "Вариант %s (%s): сошлось %d/%d точек",
        variant_key, variant_name, n_conv, n,
    )

    return IdelchikResult(
        variant=variant_key,
        variant_name=variant_name,
        r_pred=r_pred,
        u1=u1_arr,
        zeta_branch=z_b_arr,
        zeta_straight=z_s_arr,
        converged=conv,
        notes=notes,
    )


def run_variant(variant_key, u1_array, geom, L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT):
    """Запустить расчёт для указанного варианта ('A', 'B' или 'C')."""
    name, func = VARIANTS[variant_key]
    return solve_all(u1_array, geom, func, variant_key, name, L_upper, eps)


def sensitivity_L_upper(
    u1_ref, geom, coeff_func, L_values, eps=EPS_DEFAULT,
):
    """Чувствительность r к длине верхнего участка L_верх.

    Возвращает массив r для каждого значения L_верх при заданной u₁.
    """
    r_values = np.full(len(L_values), np.nan)
    for i, L in enumerate(L_values):
        r_i, _, _, ok, _ = solve_r(u1_ref, geom, coeff_func, L, eps)
        if ok:
            r_values[i] = r_i
    return r_values


def sensitivity_coefficients(
    u1_ref, geom, K_b_values, K_pp_values,
    L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
):
    """Чувствительность r к поправочным коэффициентам K_б и K''_п.

    Варьируется K_б при K''_п=0 и наоборот.
    Возвращает два массива r: r(K_б) и r(K''_п).
    """
    from .coefficients import zeta_branch, zeta_straight

    A_ok = geom["A_ok"]
    A_s = geom["A_s"]

    def make_func_Kb(K_b_val):
        def f(r, A_ok_, A_s_):
            z_b = zeta_branch(r, A_ok_, A_s_, A_coeff=1.0, K_b=K_b_val)
            z_s = zeta_straight(r, A_ok_, A_s_, K_pp=0.0)
            return z_b, z_s
        return f

    def make_func_Kpp(K_pp_val):
        def f(r, A_ok_, A_s_):
            z_b = zeta_branch(r, A_ok_, A_s_, A_coeff=1.0, K_b=0.0)
            z_s = zeta_straight(r, A_ok_, A_s_, K_pp=K_pp_val)
            return z_b, z_s
        return f

    r_Kb = np.full(len(K_b_values), np.nan)
    for i, Kb in enumerate(K_b_values):
        r_i, _, _, ok, _ = solve_r(u1_ref, geom, make_func_Kb(Kb), L_upper, eps)
        if ok:
            r_Kb[i] = r_i

    r_Kpp = np.full(len(K_pp_values), np.nan)
    for i, Kpp in enumerate(K_pp_values):
        r_i, _, _, ok, _ = solve_r(u1_ref, geom, make_func_Kpp(Kpp), L_upper, eps)
        if ok:
            r_Kpp[i] = r_i

    return r_Kb, r_Kpp
