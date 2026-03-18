"""Главный модуль: запуск полного расчёта (шаги 1–7)."""

import json
import logging
import os

import numpy as np
import pandas as pd

from .calibration import fit_asymptotic, fit_power_law
from .checks import run_all_checks
from .config import GEOM_AIR, GEOM_WATER
from .data import load_calibration_data, load_validation_data
from .model import calc_Re, calc_delta_zeta, calc_k_ut, calc_r_explicit, calc_r_newton
from .plots import (
    plot_dz_Re,
    plot_k_ut,
    plot_parity,
    plot_r_calibration,
    plot_r_validation,
)
from .validation import compute_metrics, validate

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Шаг 1. Загрузка данных ===
    logger.info("=== Шаг 1: Загрузка данных ===")
    df_cal = load_calibration_data()
    df_val = load_validation_data()
    logger.info("Калибровка: %d точек, валидация: %d точек", len(df_cal), len(df_val))

    # === Шаг 2. Обратный расчёт Δζ ===
    logger.info("=== Шаг 2: Обратный расчёт Δζ ===")
    r_cal = df_cal["r"].values
    u1_cal = df_cal["u1"].values

    dz_exp = calc_delta_zeta(r_cal, GEOM_WATER["A_ok"], GEOM_WATER["A_s"])
    Re_cal = calc_Re(u1_cal, GEOM_WATER["D_h"], GEOM_WATER["nu"])

    df_cal["dz_exp"] = dz_exp
    df_cal["Re"] = Re_cal

    logger.info("Re: %s", np.round(Re_cal, 0))
    logger.info("Δζ: %s", np.round(dz_exp, 6))

    # === Шаг 3. Калибровка Δζ(Re) ===
    logger.info("=== Шаг 3: Калибровка Δζ(Re) ===")
    fit_A = fit_power_law(Re_cal, dz_exp)
    fit_B = fit_asymptotic(Re_cal, dz_exp)

    logger.info("R² (A): %.6f, R² (B): %.6f", fit_A.R2, fit_B.R2)

    # Проверки на калибровочных данных
    logger.info("=== Проверки физической корректности (калибровка) ===")
    run_all_checks(r_cal, u1_cal, dz_exp, label="Калибровка")

    # === Шаг 4–5. Валидация ===
    logger.info("=== Шаг 5: Валидация ===")
    df_val_result, metrics_A, metrics_B = validate(
        df_val, GEOM_AIR, fit_A.dz_func, fit_B.dz_func
    )

    # Проверки на валидационных данных (вариант A)
    r_val_pred_A = df_val_result["r_pred_A"].values
    u1_val = df_val["u1"].values
    Re_val = calc_Re(u1_val, GEOM_AIR["D_h"], GEOM_AIR["nu"])
    dz_val = calc_delta_zeta(r_val_pred_A, GEOM_AIR["A_ok"], GEOM_AIR["A_s"])
    run_all_checks(r_val_pred_A, u1_val, dz_val, label="Валидация A")

    # Выбор лучшего варианта по метрикам валидации
    best = "A" if metrics_A.RMSE <= metrics_B.RMSE else "B"
    logger.info("Лучший вариант по RMSE валидации: %s", best)

    # === Шаг 6. Графики ===
    logger.info("=== Шаг 6: Графики ===")
    p1 = plot_r_calibration(df_cal, GEOM_WATER, fit_A.dz_func, fit_B.dz_func)
    p2 = plot_r_validation(df_cal, df_val, GEOM_WATER, GEOM_AIR,
                           fit_A.dz_func, fit_B.dz_func)
    p3 = plot_dz_Re(Re_cal, dz_exp, fit_A.dz_func, fit_B.dz_func)
    p4 = plot_k_ut(df_cal, df_val, GEOM_WATER, GEOM_AIR, fit_A.dz_func)
    p5 = plot_parity(df_cal, df_val, GEOM_WATER, GEOM_AIR, fit_A.dz_func)
    logger.info("Графики сохранены: %s", [p1, p2, p3, p4, p5])

    # === Экспорт CSV ===
    logger.info("=== Экспорт результатов ===")

    # calibration_results.csv
    cal_out = pd.DataFrame({
        "Re": Re_cal,
        "dz_exp": dz_exp,
        "dz_fit_A": fit_A.dz_func(Re_cal),
        "dz_fit_B": fit_B.dz_func(Re_cal),
        "r_exp": r_cal,
        "r_fit_A": calc_r_explicit(u1_cal, GEOM_WATER, fit_A.dz_func),
        "r_fit_B": calc_r_explicit(u1_cal, GEOM_WATER, fit_B.dz_func),
    })
    cal_out.to_csv(os.path.join(OUTPUT_DIR, "calibration_results.csv"), index=False)

    # validation_results.csv
    val_out = pd.DataFrame({
        "u1": u1_val,
        "Re": Re_val,
        "r_exp": df_val["r"].values,
        "r_pred_A": df_val_result["r_pred_A"].values,
        "r_pred_B": df_val_result["r_pred_B"].values,
        "error_A": df_val_result["error_A"].values,
        "error_B": df_val_result["error_B"].values,
    })
    val_out.to_csv(os.path.join(OUTPUT_DIR, "validation_results.csv"), index=False)

    # fitted_parameters.json
    params = {
        "A": {"a": fit_A.params["a"], "b": fit_A.params["b"], "R2": fit_A.R2},
        "B": {
            "dz_inf": fit_B.params["dz_inf"],
            "c": fit_B.params["c"],
            "n": fit_B.params["n"],
            "R2": fit_B.R2,
        },
        "best_variant": best,
        "validation_metrics": {
            "A": {"RMSE": metrics_A.RMSE, "MAE": metrics_A.MAE,
                   "R2": metrics_A.R2, "max_abs_error": metrics_A.max_abs_error},
            "B": {"RMSE": metrics_B.RMSE, "MAE": metrics_B.MAE,
                   "R2": metrics_B.R2, "max_abs_error": metrics_B.max_abs_error},
        },
    }
    with open(os.path.join(OUTPUT_DIR, "fitted_parameters.json"), "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # === Шаг 7. Newton-Raphson — верификация ===
    logger.info("=== Шаг 7: Верификация Newton-Raphson ===")
    best_func = fit_A.dz_func if best == "A" else fit_B.dz_func
    max_diff = 0.0
    for u1_i in u1_cal:
        r_explicit = calc_r_explicit(np.float64(u1_i), GEOM_WATER, best_func)
        r_nr = calc_r_newton(u1_i, GEOM_WATER, best_func)
        diff = abs(float(r_explicit) - r_nr)
        max_diff = max(max_diff, diff)

    logger.info(
        "Макс. расхождение явная формула vs Newton-Raphson: %.2e (порог: 1e-10)",
        max_diff,
    )
    if max_diff < 1e-10:
        logger.info("✓ Newton-Raphson верифицирован")
    else:
        logger.warning("✗ Расхождение превышает порог!")

    # === Итоговый отчёт ===
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ")
    logger.info("=" * 60)
    logger.info("Калибровка R² (A): %.6f, R² (B): %.6f", fit_A.R2, fit_B.R2)
    logger.info("Валидация RMSE (A): %.6f, RMSE (B): %.6f",
                metrics_A.RMSE, metrics_B.RMSE)
    logger.info("Лучший вариант: %s", best)

    cal_ok = max(fit_A.R2, fit_B.R2) > 0.95
    val_ok = min(metrics_A.RMSE, metrics_B.RMSE) < 0.05
    logger.info("Критерий калибровки (R² > 0.95): %s", "✓" if cal_ok else "✗")
    logger.info("Критерий валидации (RMSE < 0.05): %s", "✓" if val_ok else "✗")


if __name__ == "__main__":
    main()
