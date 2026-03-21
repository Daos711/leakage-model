"""Калибровка полуэмпирической модели (этап 3).

Подбор a_ξ, b_ξ и c₀ по водяным данным (минимизация суммы квадратов по r).
"""

import logging

import numpy as np
from scipy.optimize import minimize, differential_evolution

from ..stage2_idelchik.coefficients import EPS_DEFAULT, L_UPPER_DEFAULT
from .model import solve_all
from ..core.validation import compute_metrics

logger = logging.getLogger(__name__)

# Штраф за несходимость
_PENALTY = 10.0

# Сетка начальных приближений для multi-start (a_ξ, b_ξ, c₀)
_INITIAL_GRID = [
    (10.0, -0.7, 2.0),
    (20.0, -1.5, 1.5),
    (40.0, -2.5, 2.0),
    (60.0, -4.0, 1.8),
    (80.0, -5.0, 2.5),
    (5.0, -0.3, 1.0),
    (50.0, -3.5, 2.2),
    (100.0, -7.0, 1.5),
    (30.0, -2.0, 1.0),
    (35.0, -2.5, 2.5),
]


def calibrate(u1_cal, r_cal, geom_cal, beta,
              L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT,
              R_down=0.0, criterion="Re"):
    """Калибровка a_ξ, b_ξ и c₀ по водяным данным.

    Параметры
    ---------
    u1_cal : array-like
        Скорости в окне (калибровочный набор).
    r_cal : array-like
        Экспериментальные r (калибровочный набор).
    geom_cal : dict
        Геометрия калибровочной модели.
    beta : float
        Угол подвода, рад.
    L_upper, eps, R_down : float
        Параметры верхней ветви.
    criterion : str
        Аргумент ξ: 'Re', 'Fr' или 'u1'.

    Возвращает
    ----------
    tuple (a_xi, b_xi, c0, metrics_cal, result_cal)
    """
    u1_cal = np.asarray(u1_cal, dtype=float)
    r_cal = np.asarray(r_cal, dtype=float)

    def loss(params):
        a_xi, b_xi, c0 = params
        res = solve_all(u1_cal, geom_cal, a_xi, b_xi, c0, beta,
                        L_upper, eps, R_down, criterion)
        err = 0.0
        for i in range(len(u1_cal)):
            if res.converged[i]:
                err += (res.r_pred[i] - r_cal[i]) ** 2
            else:
                err += _PENALTY
        return err

    # Multi-start Nelder-Mead
    best_loss = np.inf
    best_params = None

    for a0, b0, c0_0 in _INITIAL_GRID:
        try:
            opt = minimize(loss, [a0, b0, c0_0], method="Nelder-Mead",
                           options={"maxiter": 20000, "xatol": 1e-12,
                                    "fatol": 1e-14, "adaptive": True})
            if opt.fun < best_loss:
                best_loss = opt.fun
                best_params = opt.x
                logger.info(f"  Старт ({a0}, {b0}, {c0_0}): loss={opt.fun:.6e}, "
                            f"a_ξ={opt.x[0]:.4f}, b_ξ={opt.x[1]:.4f}, "
                            f"c₀={opt.x[2]:.4f}")
        except Exception:
            pass

    # Также попробовать глобальную оптимизацию
    try:
        bounds = [(-50, 200), (-10, 10), (-5, 10)]
        de_result = differential_evolution(loss, bounds, seed=42,
                                           maxiter=500, tol=1e-12,
                                           polish=True)
        if de_result.fun < best_loss:
            best_loss = de_result.fun
            best_params = de_result.x
            logger.info(f"  DE: loss={de_result.fun:.6e}, "
                        f"a_ξ={de_result.x[0]:.4f}, b_ξ={de_result.x[1]:.4f}, "
                        f"c₀={de_result.x[2]:.4f}")
    except Exception:
        pass

    a_xi, b_xi, c0 = best_params
    logger.info(f"Калибровка: a_ξ={a_xi:.6f}, b_ξ={b_xi:.6f}, "
                f"c₀={c0:.6f}, loss={best_loss:.6e}")

    # Финальный расчёт с оптимальными параметрами
    result_cal = solve_all(u1_cal, geom_cal, a_xi, b_xi, c0, beta,
                           L_upper, eps, R_down, criterion)
    metrics_cal = compute_metrics(r_cal, result_cal.r_pred)

    logger.info(f"Метрики калибровки: RMSE={metrics_cal.RMSE:.4f}, "
                f"MAE={metrics_cal.MAE:.4f}, R²={metrics_cal.R2:.4f}, "
                f"max|err|={metrics_cal.max_abs_error:.4f}")

    return a_xi, b_xi, c0, metrics_cal, result_cal
