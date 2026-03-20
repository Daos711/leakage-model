"""Замыкающие соотношения полуэмпирической модели (этап 3).

Скрытый параметр ξ (блокировка сечения вихрями) и коэффициенты сжатия φ.
"""

import numpy as np


def calc_xi(criterion_value, a_xi, b_xi):
    """Скрытый параметр состояния ξ (сигмоида).

    ξ = 1 / (1 + exp(-(a_ξ + b_ξ·ln(criterion_value))))

    ξ → 1: сильная блокировка (мощные вихри).
    ξ → 0: слабая блокировка (вихри подавлены).

    Параметры
    ---------
    criterion_value : float или ndarray
        Критерий подобия: Re (по умолчанию), Fr или u₁.
    a_xi : float
        Сдвиг сигмоиды.
    b_xi : float
        Наклон по ln(criterion_value). Ожидается b_ξ < 0.

    Возвращает
    ----------
    float или ndarray
        Значение ξ ∈ (0, 1).
    """
    criterion_value = np.asarray(criterion_value, dtype=float)
    z = a_xi + b_xi * np.log(criterion_value)
    return 1.0 / (1.0 + np.exp(-z))


def _k_direction(beta, direction):
    """Коэффициент асимметрии для направления 'up' или 'down'.

    k_down = (1 + cos β) / 2
    k_up   = (1 − cos β) / 2
    """
    cos_b = np.cos(beta)
    if direction == "down":
        return (1.0 + cos_b) / 2.0
    elif direction == "up":
        return (1.0 - cos_b) / 2.0
    else:
        raise ValueError(f"direction must be 'up' or 'down', got '{direction}'")


def calc_phi(xi, sigma, beta, direction):
    """Коэффициент сжатия φ.

    φ = σ + (1 − σ)·(1 − ξ)·k(β, direction)

    Параметры
    ---------
    xi : float или ndarray
        Параметр блокировки ξ ∈ (0, 1).
    sigma : float
        Отношение площадей σ = A_ок / A_с.
    beta : float
        Угол подвода потока, рад.
    direction : str
        'up' или 'down'.

    Возвращает
    ----------
    float или ndarray
        Коэффициент сжатия φ ∈ (σ, 1).
    """
    k = _k_direction(beta, direction)
    return sigma + (1.0 - sigma) * (1.0 - xi) * k


def calc_C_beta(xi, beta, c0):
    """Асимметричный член ядра: C_β = c₀·cos²β·(1−ξ).

    Параметры
    ---------
    xi : float или ndarray
        Параметр блокировки ξ ∈ (0, 1).
    beta : float
        Угол подвода потока, рад.
    c0 : float
        Калибруемый коэффициент асимметрии.

    Возвращает
    ----------
    float или ndarray
        Значение C_β.
    """
    return c0 * np.cos(beta) ** 2 * (1.0 - xi)


def calc_u_contracted(u_branch, phi):
    """Скорость в сжатом сечении: u_c = u_branch / φ."""
    return u_branch / phi
