"""Главный модуль: запуск полного расчёта (шаги 1–12)."""

import argparse
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
from .diagnostics import compute_dz_exp, plot_dz_diagnostic
from .alternatives import (
    fit_r_power_Re,
    fit_r_asymptotic_Re,
    fit_r_power_u1,
    validate_alternative,
    plot_all_models_calibration,
    plot_all_models_validation,
    plot_parity_best,
)
from .comparison import build_comparison_table, select_best_model, save_comparison_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from .config import OUTPUT_STAGE1, OUTPUT_STAGE1_1

OUTPUT_DIR = OUTPUT_STAGE1
OUTPUT_DIR_V2 = OUTPUT_STAGE1_1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_V2, exist_ok=True)

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
    with open(os.path.join(OUTPUT_DIR, "fitted_parameters.json"), "w",
              encoding="utf-8") as f:
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

    # === Шаг 8. Диагностический график Δζ_exp ===
    logger.info("=== Шаг 8: Диагностика Δζ_exp(Re) — вода vs воздух ===")
    Re_water, dz_water = compute_dz_exp(df_cal, GEOM_WATER)
    Re_air, dz_air = compute_dz_exp(df_val, GEOM_AIR)

    logger.info("Δζ_exp вода: %s", np.round(dz_water, 6))
    logger.info("Δζ_exp воздух: %s", np.round(dz_air, 6))

    p6 = plot_dz_diagnostic(Re_water, dz_water, Re_air, dz_air)
    logger.info("Диагностический график: %s", p6)

    # Интерпретация: сравнение диапазонов
    dz_w_range = (dz_water.min(), dz_water.max())
    dz_a_range = (dz_air.min(), dz_air.max())
    logger.info(
        "Диапазон Δζ_exp: вода [%.4f, %.4f], воздух [%.4f, %.4f]",
        dz_w_range[0], dz_w_range[1], dz_a_range[0], dz_a_range[1],
    )
    if dz_a_range[0] > dz_w_range[1] or dz_w_range[0] > dz_a_range[1]:
        logger.info("Ветви воды и воздуха НЕ ПЕРЕСЕКАЮТСЯ — подтверждена непереносимость Δζ(Re).")
    else:
        logger.info("Ветви воды и воздуха частично пересекаются.")

    # === Шаг 9. Альтернативные замыкания ===
    logger.info("=== Шаг 9: Альтернативные замыкания ===")
    u1_cal = df_cal["u1"].values
    r_cal = df_cal["r"].values
    u1_val_arr = df_val["u1"].values
    r_val_arr = df_val["r"].values

    fit_2A = fit_r_power_Re(u1_cal, r_cal, GEOM_WATER)
    fit_2B = fit_r_asymptotic_Re(u1_cal, r_cal, GEOM_WATER)
    fit_2C = fit_r_power_u1(u1_cal, r_cal)

    alt_fits = [fit_2A, fit_2B, fit_2C]

    # Валидация + физические проверки
    logger.info("=== Шаг 10: Валидация альтернативных моделей ===")
    for fit in alt_fits:
        validate_alternative(
            fit, u1_cal, r_cal, GEOM_WATER,
            u1_val_arr, r_val_arr, GEOM_AIR,
        )

    # === Шаг 11. Сводная таблица и выбор лучшей модели ===
    logger.info("=== Шаг 11: Сводная таблица ===")
    df_comparison = build_comparison_table(
        metrics_A, fit_A.R2,
        metrics_B, fit_B.R2,
        alt_fits,
    )
    comp_path = save_comparison_csv(df_comparison)
    logger.info("Сводная таблица:\n%s", df_comparison.to_string(index=False))
    logger.info("Сохранена: %s", comp_path)

    best_alt = select_best_model(alt_fits)

    # === Шаг 12. Графики и экспорт (этап 1.1) ===
    logger.info("=== Шаг 12: Графики и экспорт (этап 1.1) ===")
    p7 = plot_all_models_calibration(u1_cal, r_cal, GEOM_WATER, alt_fits)
    p8 = plot_all_models_validation(
        u1_cal, r_cal, GEOM_WATER,
        u1_val_arr, r_val_arr, GEOM_AIR,
        alt_fits,
    )
    logger.info("Графики моделей: %s, %s", p7, p8)

    if best_alt:
        p9 = plot_parity_best(
            u1_cal, r_cal, GEOM_WATER,
            u1_val_arr, r_val_arr, GEOM_AIR,
            best_alt,
        )
        logger.info("Parity plot лучшей модели: %s", p9)

    # Экспорт calibration_results_v2.csv
    cal_v2 = {"u1": u1_cal, "Re": Re_cal, "r_exp": r_cal}
    for fit in alt_fits:
        if fit.converged:
            cal_v2[f"r_pred_{fit.name}"] = fit.r_func(u1_cal, GEOM_WATER)
    pd.DataFrame(cal_v2).to_csv(
        os.path.join(OUTPUT_DIR_V2, "calibration_results_v2.csv"), index=False,
    )

    # Экспорт validation_results_v2.csv
    val_v2 = {"u1": u1_val_arr, "Re": Re_val, "r_exp": r_val_arr}
    for fit in alt_fits:
        if fit.converged:
            val_v2[f"r_pred_{fit.name}"] = fit.r_func(u1_val_arr, GEOM_AIR)
    pd.DataFrame(val_v2).to_csv(
        os.path.join(OUTPUT_DIR_V2, "validation_results_v2.csv"), index=False,
    )

    # Экспорт fitted_parameters_v2.json
    params_v2 = {}
    for fit in alt_fits:
        params_v2[fit.name] = {
            "params": {k: float(v) if not isinstance(v, str) else v
                       for k, v in fit.params.items()},
            "R2_cal": fit.R2_cal,
            "converged": fit.converged,
            "physical_ok": fit.physical_ok,
        }
        if fit.metrics_val:
            params_v2[fit.name]["validation"] = {
                "RMSE": fit.metrics_val.RMSE,
                "MAE": fit.metrics_val.MAE,
                "R2": fit.metrics_val.R2,
                "max_abs_error": fit.metrics_val.max_abs_error,
            }
    if best_alt:
        params_v2["best_model"] = best_alt.name

    with open(os.path.join(OUTPUT_DIR_V2, "fitted_parameters_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(params_v2, f, indent=2, ensure_ascii=False)

    logger.info("Экспорт v2 завершён.")

    # === Итоговый отчёт ===
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ")
    logger.info("=" * 60)
    logger.info("--- Базовые модели (Δζ-замыкание) ---")
    logger.info("Калибровка R² (A): %.6f, R² (B): %.6f", fit_A.R2, fit_B.R2)
    logger.info("Валидация RMSE (A): %.6f, RMSE (B): %.6f",
                metrics_A.RMSE, metrics_B.RMSE)
    logger.info("Лучший базовый вариант: %s", best)

    logger.info("--- Альтернативные модели (прямое замыкание) ---")
    for fit in alt_fits:
        if fit.metrics_val:
            logger.info(
                "  %s: R²_cal=%.4f, RMSE_val=%.6f, R²_val=%.4f, физ.=%s",
                fit.name, fit.R2_cal,
                fit.metrics_val.RMSE, fit.metrics_val.R2,
                "Да" if fit.physical_ok else "Нет",
            )

    if best_alt:
        logger.info("Лучшая альтернативная модель: %s", best_alt.name)

    cal_ok = max(fit_A.R2, fit_B.R2) > 0.95
    best_val_rmse = min(
        metrics_A.RMSE, metrics_B.RMSE,
        *(f.metrics_val.RMSE for f in alt_fits if f.metrics_val),
    )
    val_ok = best_val_rmse < 0.05
    logger.info("Критерий калибровки (R² > 0.95): %s", "✓" if cal_ok else "✗")
    logger.info(
        "Критерий валидации (RMSE < 0.05): %s (лучший RMSE: %.6f)",
        "✓" if val_ok else "✗", best_val_rmse,
    )
    if not val_ok:
        logger.info(
            "Прямые однопараметрические замыкания также недостаточны "
            "для переноса между объектами с разной геометрией окна."
        )


def _parse_args():
    parser = argparse.ArgumentParser(description="Leakage model (v1)")
    parser.add_argument("--no-titles", action="store_true",
                        help="Генерировать графики без заголовков")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.no_titles:
        from leakage_model import config as _cfg
        _cfg.NO_TITLES = True
    main()
