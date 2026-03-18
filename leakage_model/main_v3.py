"""Главный модуль импульсной модели (этап 3): шаги 1–8."""

import json
import logging
import os

import numpy as np
import pandas as pd

from .calibration import fit_power_law
from .calibration_v3 import (
    CM_FitResult,
    analyze_C_M,
    compute_C_M_table,
    fit_C_M_asymptotic,
    fit_C_M_power,
)
from .checks import check_monotonic_decrease, check_r_range
from .config import GEOM_AIR, GEOM_WATER, RHO
from .data import load_calibration_data, load_validation_data
from .friction import churchill_friction
from .model import calc_Re, calc_delta_zeta, calc_k_ut
from .plots_v3 import (
    plot_C_M_diagnostics,
    plot_C_M_Re,
    plot_comparison_v3,
    plot_k_ut_v3,
    plot_parity_v3,
    plot_r_calibration_v3,
    plot_r_validation_v3,
)
from .validation_v3 import forward_solve, validate_v3, verify_newton_vs_brent

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Шаг 1. Загрузка данных и геометрии ===
    logger.info("=== Шаг 1: Загрузка данных ===")
    df_cal = load_calibration_data()
    df_val = load_validation_data()
    logger.info("Калибровка: %d точек, валидация: %d точек",
                len(df_cal), len(df_val))

    # === Шаг 2. Обратный расчёт C_M для каждой калибровочной точки ===
    logger.info("=== Шаг 2: Обратный расчёт C_M ===")
    df_cm = compute_C_M_table(df_cal, GEOM_WATER)
    logger.info("Таблица C_M:\n%s", df_cm.to_string(index=False))

    # === Шаг 3. Анализ C_M ===
    logger.info("=== Шаг 3: Анализ C_M ===")
    cm_stats = analyze_C_M(df_cm)

    # Scatter plot: C_M(Re), C_M(r), C_M(u₁)
    p1 = plot_C_M_Re(df_cm, cm_stats["mean"])
    p1a = plot_C_M_diagnostics(df_cm, cm_stats["mean"])
    logger.info("Графики C_M: %s, %s", p1, p1a)

    # === Шаг 4. Калибровка ===
    logger.info("=== Шаг 4: Калибровка ===")

    # --- Вариант 1: C_M = const ---
    C_M_const = cm_stats["mean"]
    logger.info("Вариант 1: C_M = %.6f (среднее)", C_M_const)

    logger.info("--- Прямая задача с C_M = const ---")
    df_cal_res_c, df_val_res_c, met_cal_c, met_val_c = validate_v3(
        df_cal, df_val, GEOM_WATER, GEOM_AIR, C_M_const
    )

    # --- Вариант 2: C_M = f(Re) ---
    logger.info("--- Калибровка C_M = f(Re) ---")
    Re_cm = df_cm["Re"].values
    CM_cm = df_cm["C_M"].values

    fit_pow = fit_C_M_power(Re_cm, CM_cm)
    fit_asym = fit_C_M_asymptotic(Re_cm, CM_cm)

    # Выбираем лучший фит по R²
    best_cm_fit = fit_pow if fit_pow.R2 >= fit_asym.R2 else fit_asym
    if not best_cm_fit.converged:
        best_cm_fit = fit_pow if fit_pow.converged else fit_asym
    logger.info("Лучший фит C_M(Re): %s (R²=%.6f)",
                best_cm_fit.name, best_cm_fit.R2)

    logger.info("--- Прямая задача с C_M = f(Re) ---")
    df_cal_res_f, df_val_res_f, met_cal_f, met_val_f = validate_v3(
        df_cal, df_val, GEOM_WATER, GEOM_AIR, best_cm_fit.cm_func
    )

    # === Выбор варианта ===
    logger.info("=== Выбор варианта ===")
    logger.info("C_M=const: RMSE_val=%.6f, R²_val=%.6f",
                met_val_c.RMSE, met_val_c.R2)
    logger.info("C_M=f(Re): RMSE_val=%.6f, R²_val=%.6f",
                met_val_f.RMSE, met_val_f.R2)

    if met_val_f.RMSE < met_val_c.RMSE:
        logger.info("C_M=f(Re) лучше → используется вариант 2")
        C_M_final = best_cm_fit.cm_func
        df_cal_res = df_cal_res_f
        df_val_res = df_val_res_f
        metrics_cal = met_cal_f
        metrics_val = met_val_f
        variant_used = f"C_M = f(Re) [{best_cm_fit.name}]"
    else:
        logger.info("C_M=const лучше → используется вариант 1")
        C_M_final = C_M_const
        df_cal_res = df_cal_res_c
        df_val_res = df_val_res_c
        metrics_cal = met_cal_c
        metrics_val = met_val_c
        variant_used = "C_M = const"

    # === Шаг 5–6. Метрики ===
    logger.info("=== Метрики выбранного варианта (%s) ===", variant_used)
    logger.info("Калибровка: RMSE=%.6f, MAE=%.6f, R²=%.6f, max|err|=%.6f",
                metrics_cal.RMSE, metrics_cal.MAE,
                metrics_cal.R2, metrics_cal.max_abs_error)
    logger.info("Валидация:  RMSE=%.6f, MAE=%.6f, R²=%.6f, max|err|=%.6f",
                metrics_val.RMSE, metrics_val.MAE,
                metrics_val.R2, metrics_val.max_abs_error)

    # === Проверки физической корректности ===
    logger.info("=== Проверки физической корректности ===")
    r_cal_calc = df_cal_res["r_calc"].values
    r_val_calc = df_val_res["r_calc"].values
    u1_cal = df_cal_res["u1"].values
    u1_val = df_val_res["u1"].values

    check_r_range(r_cal_calc, "v3 калибровка")
    check_r_range(r_val_calc, "v3 валидация")
    check_monotonic_decrease(r_cal_calc, u1_cal, "v3 калибровка")
    check_monotonic_decrease(r_val_calc, u1_val, "v3 валидация")

    if cm_stats["all_positive"]:
        logger.info("C_M все положительны — ОК")
    else:
        logger.warning("C_M имеет разные знаки!")

    # Проверка давлений
    logger.info("=== Проверка давлений ===")
    for _, row in df_cal_res.iterrows():
        Q1 = row["Q"]
        u1 = row["u1"]
        r = row["r_calc"]
        Q2 = r * Q1
        u2 = Q2 / GEOM_WATER["A_s"]
        Re2 = abs(u2) * GEOM_WATER["D"] / GEOM_WATER["nu"]
        lam2 = churchill_friction(Re2)
        dp_upper = lam2 * (GEOM_WATER["L_up"] / GEOM_WATER["D"]) * RHO * u2**2 / 2
        logger.info("  u1=%.2f: p̃₂=%.2f Па, p̃₃=0.00 Па", u1, dp_upper)

    # === Верификация Newton-Raphson vs Brent ===
    logger.info("=== Верификация Newton-Raphson vs Brent ===")
    max_diff = verify_newton_vs_brent(df_cal, GEOM_WATER, C_M_final)

    # === Шаг 7. Сравнение с прежней моделью ===
    logger.info("=== Шаг 7: Сравнение с Δζ-моделью (этап 1) ===")
    dz_exp = calc_delta_zeta(
        df_cal["r"].values, GEOM_WATER["A_ok"], GEOM_WATER["A_s"]
    )
    Re_cal = calc_Re(u1_cal, GEOM_WATER["D_h"], GEOM_WATER["nu"])
    fit_A = fit_power_law(Re_cal, dz_exp)
    dz_func_best = fit_A.dz_func if fit_A.converged else None

    # === Шаг 8. Графики и экспорт ===
    logger.info("=== Шаг 8: Графики и экспорт ===")
    p2 = plot_r_calibration_v3(df_cal_res, GEOM_WATER, C_M_final)
    p3 = plot_r_validation_v3(df_cal_res, df_val_res,
                               GEOM_WATER, GEOM_AIR, C_M_final)
    p4 = plot_k_ut_v3(df_cal_res, df_val_res,
                       GEOM_WATER, GEOM_AIR, C_M_final)
    p5 = plot_parity_v3(df_cal_res, df_val_res)
    p6 = plot_comparison_v3(df_cal_res, df_val_res,
                             GEOM_WATER, GEOM_AIR, C_M_final, dz_func_best)
    logger.info("Графики сохранены: %s",
                [p1, p1a, p2, p3, p4, p5, p6])

    # Экспорт CSV
    df_cm.to_csv(os.path.join(OUTPUT_DIR, "C_M_analysis.csv"), index=False)
    df_cal_res.to_csv(
        os.path.join(OUTPUT_DIR, "calibration_results_v3.csv"), index=False
    )
    df_val_res.to_csv(
        os.path.join(OUTPUT_DIR, "validation_results_v3.csv"), index=False
    )

    # model_parameters_v3.json
    params_v3 = {
        "variant_used": variant_used,
        "C_M_mean": cm_stats["mean"],
        "C_M_std": cm_stats["std"],
        "C_M_cv": cm_stats["CV"],
        "C_M_min": cm_stats["min"],
        "C_M_max": cm_stats["max"],
        "C_M_all_positive": cm_stats["all_positive"],
        "C_M_fit_power": {
            "a": fit_pow.params.get("a"),
            "b": fit_pow.params.get("b"),
            "R2": fit_pow.R2,
        },
        "C_M_fit_asymptotic": {
            "cm_inf": fit_asym.params.get("cm_inf"),
            "c": fit_asym.params.get("c"),
            "n": fit_asym.params.get("n"),
            "R2": fit_asym.R2,
        } if fit_asym.converged else {"converged": False},
        "metrics_const": {
            "calibration": {
                "RMSE": met_cal_c.RMSE, "MAE": met_cal_c.MAE,
                "R2": met_cal_c.R2, "max_abs_error": met_cal_c.max_abs_error,
            },
            "validation": {
                "RMSE": met_val_c.RMSE, "MAE": met_val_c.MAE,
                "R2": met_val_c.R2, "max_abs_error": met_val_c.max_abs_error,
            },
        },
        "metrics_func": {
            "calibration": {
                "RMSE": met_cal_f.RMSE, "MAE": met_cal_f.MAE,
                "R2": met_cal_f.R2, "max_abs_error": met_cal_f.max_abs_error,
            },
            "validation": {
                "RMSE": met_val_f.RMSE, "MAE": met_val_f.MAE,
                "R2": met_val_f.R2, "max_abs_error": met_val_f.max_abs_error,
            },
        },
        "metrics_selected": {
            "calibration": {
                "RMSE": metrics_cal.RMSE, "MAE": metrics_cal.MAE,
                "R2": metrics_cal.R2, "max_abs_error": metrics_cal.max_abs_error,
            },
            "validation": {
                "RMSE": metrics_val.RMSE, "MAE": metrics_val.MAE,
                "R2": metrics_val.R2, "max_abs_error": metrics_val.max_abs_error,
            },
        },
        "newton_vs_brent_max_diff": max_diff,
    }
    with open(os.path.join(OUTPUT_DIR, "model_parameters_v3.json"), "w",
              encoding="utf-8") as f:
        json.dump(params_v3, f, indent=2, ensure_ascii=False)

    # === Итоговый отчёт ===
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ — ИМПУЛЬСНАЯ МОДЕЛЬ (v3)")
    logger.info("=" * 60)
    logger.info("Выбранный вариант: %s", variant_used)
    logger.info("C_M среднее = %.6f (CV = %.1f%%)",
                cm_stats["mean"], cm_stats["CV"] * 100)
    if best_cm_fit.converged:
        logger.info("C_M fit (%s): R²=%.6f, params=%s",
                     best_cm_fit.name, best_cm_fit.R2, best_cm_fit.params)
    logger.info("--- C_M=const ---")
    logger.info("  Калибровка: RMSE=%.6f, R²=%.6f",
                met_cal_c.RMSE, met_cal_c.R2)
    logger.info("  Валидация:  RMSE=%.6f, R²=%.6f",
                met_val_c.RMSE, met_val_c.R2)
    logger.info("--- C_M=f(Re) ---")
    logger.info("  Калибровка: RMSE=%.6f, R²=%.6f",
                met_cal_f.RMSE, met_cal_f.R2)
    logger.info("  Валидация:  RMSE=%.6f, R²=%.6f",
                met_val_f.RMSE, met_val_f.R2)

    val_ok = metrics_val.RMSE < 0.05
    logger.info("Критерий валидации (RMSE < 0.05): %s (RMSE=%.6f)",
                "PASS" if val_ok else "FAIL", metrics_val.RMSE)

    logger.info("Экспорт завершён. Файлы в: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
