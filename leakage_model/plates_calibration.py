"""Калибровка параметров пластин для моделей M1, M2, M3 (этап 4).

M1: только ζ_пл (≥ 0, softplus-параметризация)
M2: только Δc₀ (свободный знак)
M3: ζ_пл + Δc₀ (два параметра)
"""

import logging

import numpy as np
from scipy.optimize import minimize

from .plates_model import predict_plates

logger = logging.getLogger(__name__)

# Штраф за несходимость одной точки
_PENALTY = 10.0


def _softplus(theta):
    """ζ_пл = ln(1 + exp(θ)) — гарантирует ζ_пл ≥ 0."""
    return np.log1p(np.exp(theta))


def _softplus_inv(zeta):
    """Обратная: θ = ln(exp(ζ) − 1)."""
    if zeta > 20.0:
        return zeta
    return np.log(np.expm1(max(zeta, 1e-12)))


def _loss(r_pred, r_exp, converged):
    """Сумма квадратов + штраф за несходимости."""
    total = 0.0
    for i in range(len(r_exp)):
        if converged[i] and np.isfinite(r_pred[i]):
            total += (r_pred[i] - r_exp[i]) ** 2
        else:
            total += _PENALTY
    return total


def calibrate_insert_M1(u1, r_exp, geom, base_params, beta, L, eps,
                        criterion="Re"):
    """Калибровка ζ_пл для одной вставки (модель M1).

    Возвращает (zeta_pl, rmse).
    """
    a_xi, b_xi, c0 = base_params

    def objective(theta_arr):
        zeta_pl = _softplus(theta_arr[0])
        r_pred, conv, _ = predict_plates(
            u1, geom, a_xi, b_xi, c0, beta, L, eps,
            zeta_pl=zeta_pl, delta_c0=0.0, criterion=criterion,
        )
        return _loss(r_pred, r_exp, conv)

    theta0 = [0.0]
    result = minimize(objective, theta0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10})

    zeta_pl = _softplus(result.x[0])
    r_pred, conv, _ = predict_plates(
        u1, geom, a_xi, b_xi, c0, beta, L, eps,
        zeta_pl=zeta_pl, delta_c0=0.0, criterion=criterion,
    )
    n_valid = conv.sum()
    if n_valid > 0:
        rmse = np.sqrt(np.mean((r_pred[conv] - r_exp[conv]) ** 2))
    else:
        rmse = np.nan

    return zeta_pl, rmse


def calibrate_insert_M2(u1, r_exp, geom, base_params, beta, L, eps,
                        criterion="Re"):
    """Калибровка Δc₀ для одной вставки (модель M2).

    Возвращает (delta_c0, rmse).
    """
    a_xi, b_xi, c0 = base_params

    def objective(params):
        delta_c0 = params[0]
        r_pred, conv, _ = predict_plates(
            u1, geom, a_xi, b_xi, c0, beta, L, eps,
            zeta_pl=0.0, delta_c0=delta_c0, criterion=criterion,
        )
        return _loss(r_pred, r_exp, conv)

    x0 = [0.0]
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10})

    delta_c0 = result.x[0]
    if abs(delta_c0) > 5.0:
        logger.warning("M2: |Δc₀| = %.2f > 5.0 — необычно большое значение", abs(delta_c0))

    r_pred, conv, _ = predict_plates(
        u1, geom, a_xi, b_xi, c0, beta, L, eps,
        zeta_pl=0.0, delta_c0=delta_c0, criterion=criterion,
    )
    n_valid = conv.sum()
    if n_valid > 0:
        rmse = np.sqrt(np.mean((r_pred[conv] - r_exp[conv]) ** 2))
    else:
        rmse = np.nan

    return delta_c0, rmse


def calibrate_insert_M3(u1, r_exp, geom, base_params, beta, L, eps,
                        criterion="Re"):
    """Калибровка (ζ_пл, Δc₀) для одной вставки (модель M3).

    Возвращает (zeta_pl, delta_c0, rmse).
    """
    a_xi, b_xi, c0 = base_params

    def objective(params):
        zeta_pl = _softplus(params[0])
        delta_c0 = params[1]
        r_pred, conv, _ = predict_plates(
            u1, geom, a_xi, b_xi, c0, beta, L, eps,
            zeta_pl=zeta_pl, delta_c0=delta_c0, criterion=criterion,
        )
        return _loss(r_pred, r_exp, conv)

    x0 = [0.0, 0.0]
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})

    zeta_pl = _softplus(result.x[0])
    delta_c0 = result.x[1]

    if abs(delta_c0) > 5.0:
        logger.warning("M3: |Δc₀| = %.2f > 5.0 — необычно большое значение", abs(delta_c0))

    r_pred, conv, _ = predict_plates(
        u1, geom, a_xi, b_xi, c0, beta, L, eps,
        zeta_pl=zeta_pl, delta_c0=delta_c0, criterion=criterion,
    )
    n_valid = conv.sum()
    if n_valid > 0:
        rmse = np.sqrt(np.mean((r_pred[conv] - r_exp[conv]) ** 2))
    else:
        rmse = np.nan

    return zeta_pl, delta_c0, rmse


def compute_aicc(mse, n, k):
    """AICc = n·ln(MSE_eff) + 2k + 2k(k+1)/(n−k−1).

    MSE_eff = max(MSE, 1e-12) для устойчивости ln.
    """
    mse_eff = max(mse, 1e-12)
    aicc = n * np.log(mse_eff) + 2 * k
    denom = n - k - 1
    if denom > 0:
        aicc += 2 * k * (k + 1) / denom
    return aicc


def calibrate_all(plates_df, geom, base_params, beta, L, eps,
                  criterion="Re", exclude_insert_1=True):
    """Калибровка всех вставок, все три модели.

    Параметры
    ---------
    plates_df : pd.DataFrame
        Данные с колонками insert_id, insert_name, u1, r_exp.
    geom : dict
        Геометрия (GEOM_WATER).
    base_params : tuple
        (a_xi, b_xi, c0) из этапа 3.
    exclude_insert_1 : bool
        Исключить вставку №1 (базовая, без пластин).

    Возвращает
    ----------
    list[dict]
        Результаты по каждой вставке.
    """
    results = []
    insert_ids = sorted(plates_df["insert_id"].unique())

    for iid in insert_ids:
        if exclude_insert_1 and iid == 1:
            continue

        sub = plates_df[plates_df["insert_id"] == iid].sort_values("u1")
        u1 = sub["u1"].values
        r_exp = sub["r_exp"].values
        name = sub["insert_name"].iloc[0]
        n = len(u1)

        logger.debug("Калибровка вставки %d (%s), %d точек", iid, name, n)

        # M1
        zeta_m1, rmse_m1 = calibrate_insert_M1(
            u1, r_exp, geom, base_params, beta, L, eps, criterion)
        mse_m1 = rmse_m1 ** 2 if np.isfinite(rmse_m1) else np.nan
        aicc_m1 = compute_aicc(mse_m1, n, k=1) if np.isfinite(mse_m1) else np.nan

        # M2
        dc0_m2, rmse_m2 = calibrate_insert_M2(
            u1, r_exp, geom, base_params, beta, L, eps, criterion)
        mse_m2 = rmse_m2 ** 2 if np.isfinite(rmse_m2) else np.nan
        aicc_m2 = compute_aicc(mse_m2, n, k=1) if np.isfinite(mse_m2) else np.nan

        # M3
        zeta_m3, dc0_m3, rmse_m3 = calibrate_insert_M3(
            u1, r_exp, geom, base_params, beta, L, eps, criterion)
        mse_m3 = rmse_m3 ** 2 if np.isfinite(rmse_m3) else np.nan
        aicc_m3 = compute_aicc(mse_m3, n, k=2) if np.isfinite(mse_m3) else np.nan

        results.append({
            "insert_id": iid,
            "insert_name": name,
            "n_points": n,
            # M1
            "zeta_pl_M1": zeta_m1,
            "RMSE_M1": rmse_m1,
            "AICc_M1": aicc_m1,
            # M2
            "delta_c0_M2": dc0_m2,
            "RMSE_M2": rmse_m2,
            "AICc_M2": aicc_m2,
            # M3
            "zeta_pl_M3": zeta_m3,
            "delta_c0_M3": dc0_m3,
            "RMSE_M3": rmse_m3,
            "AICc_M3": aicc_m3,
        })

    return results
