"""Главный модуль: расчёт по модели Идельчика (этап 2).

Предсказательная модель на основе справочных коэффициентов Идельчика
для стандартных тройников. Коэффициенты из независимого справочника,
наши экспериментальные данные — только для валидации.
"""

import logging
import os

import numpy as np

from ..core.config import GEOM_AIR, GEOM_WATER
from ..core.data import load_calibration_data, load_validation_data
from .coefficients import VARIANTS, L_UPPER_DEFAULT, EPS_DEFAULT
from .model import (
    run_variant,
    sensitivity_L_upper,
    sensitivity_coefficients,
)
from .validation import (
    compute_variant_metrics,
    plot_r_prediction,
    plot_zeta_curves,
    plot_parity_idelchik,
    plot_all_models_comparison,
    plot_sensitivity,
    export_prediction_csv,
    export_coefficients_csv,
    export_parameters_json,
    export_all_models_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # === Шаг 1. Загрузка данных ===
    logger.info("=" * 60)
    logger.info("МОДЕЛЬ ИДЕЛЬЧИКА — ПРЕДСКАЗАТЕЛЬНЫЙ РАСЧЁТ")
    logger.info("=" * 60)

    df_cal = load_calibration_data()
    df_val = load_validation_data()

    u1_water = df_cal["u1"].values
    u1_air = df_val["u1"].values
    r_exp_water = df_cal["r"].values
    r_exp_air = df_val["r"].values

    logger.info(
        "Данные: вода %d точек (A_ок=%.0f м²), воздух %d точек (A_ок=%.0f м²)",
        len(df_cal), GEOM_WATER["A_ok"], len(df_val), GEOM_AIR["A_ok"],
    )

    # Информация о геометрии
    for label, geom in [("Вода", GEOM_WATER), ("Воздух", GEOM_AIR)]:
        fb_fc = geom["A_s"] / geom["A_ok"]
        sigma = geom["A_ok"] / geom["A_s"]
        logger.info(
            "  %s: F_б/F_c = %.2f (таблицы до 1.0!), σ = F_c/F_б = %.4f",
            label, fb_fc, sigma,
        )

    # === Шаг 2. Расчёт по трём вариантам ===
    logger.info("=== Шаг 2: Расчёт по формулам Идельчика ===")
    logger.info("L_верх = %.0f м, ε = %.4f м", L_UPPER_DEFAULT, EPS_DEFAULT)

    results_water = {}
    results_air = {}

    for key in ["A", "B", "C"]:
        name, _ = VARIANTS[key]
        logger.info("--- Вариант %s: %s ---", key, name)

        results_water[key] = run_variant(key, u1_water, GEOM_WATER)
        results_air[key] = run_variant(key, u1_air, GEOM_AIR)

    # === Шаг 3. Валидация — метрики ===
    logger.info("=== Шаг 3: Метрики валидации ===")

    metrics_water = {}
    metrics_air = {}

    for key in results_water:
        res_w = results_water[key]
        res_a = results_air[key]

        m_w = compute_variant_metrics(res_w, r_exp_water)
        m_a = compute_variant_metrics(res_a, r_exp_air)

        metrics_water[key] = m_w
        metrics_air[key] = m_a

        if m_w:
            logger.info(
                "  Вариант %s, вода:   RMSE=%.4f, MAE=%.4f, R²=%.4f, max|err|=%.4f",
                key, m_w.RMSE, m_w.MAE, m_w.R2, m_w.max_abs_error,
            )
        else:
            logger.warning("  Вариант %s, вода: метрики не вычислены (нет сходимости)", key)

        if m_a:
            logger.info(
                "  Вариант %s, воздух: RMSE=%.4f, MAE=%.4f, R²=%.4f, max|err|=%.4f",
                key, m_a.RMSE, m_a.MAE, m_a.R2, m_a.max_abs_error,
            )
        else:
            logger.warning("  Вариант %s, воздух: метрики не вычислены", key)

    # === Шаг 4. Анализ коэффициентов ===
    logger.info("=== Шаг 4: Анализ коэффициентов Идельчика ===")
    for key in results_water:
        res = results_water[key]
        mask = res.converged
        if mask.any():
            logger.info(
                "  Вариант %s, вода: r_pred = %s",
                key, np.round(res.r_pred[mask], 4),
            )
            logger.info(
                "    ζ_утеч = %s", np.round(res.zeta_branch[mask], 4),
            )
            logger.info(
                "    ζ_шахта = %s", np.round(res.zeta_straight[mask], 4),
            )

    # === Шаг 5. Графики ===
    logger.info("=== Шаг 5: Графики ===")

    p1 = plot_r_prediction(results_water, results_air, df_cal, df_val)
    logger.info("График 1 (r vs u₁): %s", p1)

    p2 = plot_zeta_curves(GEOM_WATER, GEOM_AIR)
    logger.info("График 2 (ζ curves): %s", p2)

    p3 = plot_parity_idelchik(results_water, results_air, df_cal, df_val)
    logger.info("График 3 (parity): %s", p3)

    # График 4: сравнение с предыдущими моделями
    # Попробуем загрузить предыдущие модели
    prev_funcs = _load_previous_models()
    p4 = plot_all_models_comparison(
        results_water, results_air, df_cal, df_val,
        GEOM_WATER, GEOM_AIR,
        prev_model_funcs=prev_funcs,
    )
    logger.info("График 4 (сравнение моделей): %s", p4)

    # === Шаг 6. Чувствительность ===
    logger.info("=== Шаг 6: Анализ чувствительности ===")

    for geom, label in [(GEOM_WATER, "water"), (GEOM_AIR, "air")]:
        u1_ref = 10.0  # референсная скорость

        # Чувствительность к L_верх
        L_values = [0, 10, 25, 50, 100, 200, 500, 1000]
        _, coeff_func_a = VARIANTS["A"]
        r_vs_L = sensitivity_L_upper(u1_ref, geom, coeff_func_a, L_values)

        # Чувствительность к K_б и K''_п
        Kb_values = np.linspace(-1.0, 1.0, 21).tolist()
        Kpp_values = np.linspace(-1.0, 1.0, 21).tolist()
        r_vs_Kb, r_vs_Kpp = sensitivity_coefficients(
            u1_ref, geom, Kb_values, Kpp_values,
        )

        p5 = plot_sensitivity(r_vs_L, L_values, r_vs_Kb, Kb_values,
                              r_vs_Kpp, Kpp_values, label)
        logger.info("График 5 (чувствительность %s): %s", label, p5)

        # Логирование ключевых значений
        mask_L = ~np.isnan(r_vs_L)
        if mask_L.any():
            logger.info(
                "  %s: r(L=0)=%.4f, r(L=100)=%.4f, r(L=1000)=%.4f",
                label,
                r_vs_L[0] if not np.isnan(r_vs_L[0]) else -1,
                r_vs_L[4] if not np.isnan(r_vs_L[4]) else -1,
                r_vs_L[7] if not np.isnan(r_vs_L[7]) else -1,
            )

    # === Шаг 7. Экспорт ===
    logger.info("=== Шаг 7: Экспорт результатов ===")

    export_prediction_csv(results_water, results_air, df_cal, df_val)
    export_coefficients_csv(GEOM_WATER, GEOM_AIR)
    export_parameters_json(
        results_water, results_air,
        metrics_water, metrics_air,
        GEOM_WATER, GEOM_AIR,
    )
    export_all_models_csv(
        results_water, results_air,
        metrics_water, metrics_air,
        df_cal, df_val,
    )

    # === Итоговый отчёт ===
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ — МОДЕЛЬ ИДЕЛЬЧИКА")
    logger.info("=" * 60)

    # Найти лучший вариант по RMSE
    best_key = None
    best_rmse = np.inf

    for key in metrics_water:
        for m_dict, label in [(metrics_water, "вода"), (metrics_air, "воздух")]:
            m = m_dict.get(key)
            if m and m.RMSE < best_rmse:
                best_rmse = m.RMSE
                best_key = key

    logger.info("Лучший вариант по RMSE: %s (RMSE=%.4f)", best_key, best_rmse)

    # Оценка по критериям из ТЗ
    if best_rmse < 0.05:
        logger.info("Результат: ОТЛИЧНЫЙ (RMSE < 0.05)")
    elif best_rmse < 0.10:
        logger.info("Результат: ХОРОШИЙ (RMSE < 0.10)")
    else:
        logger.info("Результат: ПРИЕМЛЕМЫЙ (RMSE ≥ 0.10)")

    # Проверка тренда r(u₁) — должен убывать
    for key in results_water:
        res = results_water[key]
        mask = res.converged
        if mask.sum() >= 2:
            r_conv = res.r_pred[mask]
            dr = np.diff(r_conv)
            monotone = np.all(dr <= 0.001)  # допуск на числ. шум
            logger.info(
                "  Вариант %s: тренд r(u₁) %s",
                key,
                "убывает ✓" if monotone else "НЕ убывает ✗",
            )

    logger.info("-" * 60)
    logger.info("ВЫВОД:")
    logger.info(
        "Модель Идельчика для стандартных тройников применена к нестандартной"
    )
    logger.info(
        "геометрии шахтного узла (F_б/F_c >> 1). Результат показывает"
    )
    logger.info(
        "насколько справочные данные применимы к данной задаче."
    )
    logger.info(
        "Основное ограничение: слабая зависимость r от скорости u₁"
    )
    logger.info(
        "(коэффициенты Идельчика не зависят от Re в турбулентном режиме)."
    )
    logger.info("=" * 60)


def _load_previous_models():
    """Попытка загрузить предыдущие модели для сравнения."""
    funcs = []
    try:
        from ..stage1_energy.calibration import fit_power_law
        from ..stage1_energy.model import calc_Re, calc_delta_zeta, calc_r_explicit
        from ..core.data import load_calibration_data

        df_cal = load_calibration_data()
        from ..core.config import GEOM_WATER
        r_cal = df_cal["r"].values
        u1_cal = df_cal["u1"].values
        dz_exp = calc_delta_zeta(r_cal, GEOM_WATER["A_ok"], GEOM_WATER["A_s"])
        Re_cal = calc_Re(u1_cal, GEOM_WATER["D_h"], GEOM_WATER["nu"])

        fit_A = fit_power_law(Re_cal, dz_exp)

        def model_dz_A(u1, geom, _f=fit_A):
            return calc_r_explicit(u1, geom, _f.dz_func)

        funcs.append(("Δζ(Re) степ.", model_dz_A, "b--"))

    except Exception as e:
        logger.debug("Не удалось загрузить Δζ-модель: %s", e)

    try:
        from ..stage1_energy.alternatives import fit_r_power_u1
        from ..core.data import load_calibration_data

        df_cal = load_calibration_data()
        fit_2C = fit_r_power_u1(df_cal["u1"].values, df_cal["r"].values)
        funcs.append(("r(u₁) степ.", fit_2C.r_func, "g-."))

    except Exception as e:
        logger.debug("Не удалось загрузить r(u₁)-модель: %s", e)

    return funcs if funcs else None


if __name__ == "__main__":
    main()
