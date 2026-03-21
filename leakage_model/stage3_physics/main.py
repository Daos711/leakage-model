"""Главный модуль этапа 3 — полуэмпирическая физическая модель.

Запуск: python -m leakage_model.stage3_physics.main
"""

import logging
import os

import numpy as np

from ..core.config import (
    GEOM_WATER, GEOM_AIR, BETA_RAD, BETA_DEG,
    OUTPUT_STAGE3, OUTPUT_STAGE3_PLOTS,
)
from ..core.data import load_calibration_data, load_validation_data
from ..stage2_idelchik.coefficients import L_UPPER_DEFAULT, EPS_DEFAULT
from ..stage1_energy.model import calc_Re
from .closures import calc_xi, calc_phi
from .model import solve_all, borda_carnot_loss_coeff
from .calibration import calibrate
from .validation import (
    validate, export_results, check_dimensionless_invariance,
)
from .plots import (
    plot_r_vs_u1,
    plot_parity,
    plot_xi_and_phi,
    plot_sensitivity,
    plot_F_residual,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Параметры расчёта
L_UPPER = L_UPPER_DEFAULT   # 100 м
EPS = EPS_DEFAULT            # 0.002 м
R_DOWN = 0.0                 # открытый выход
CRITERION = "Re"             # аргумент ξ


def main():
    logger.info("=" * 60)
    logger.info("ЭТАП 3 — ПОЛУЭМПИРИЧЕСКАЯ ФИЗИЧЕСКАЯ МОДЕЛЬ")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Шаг 1. Загрузка данных
    # ------------------------------------------------------------------
    logger.info("=== Шаг 1: Загрузка данных ===")
    df_cal = load_calibration_data()
    df_val = load_validation_data()

    u1_water = df_cal["u1"].values
    r_water = df_cal["r"].values
    u1_air = df_val["u1"].values
    r_air = df_val["r"].values

    sigma_w = GEOM_WATER["A_ok"] / GEOM_WATER["A_s"]
    sigma_a = GEOM_AIR["A_ok"] / GEOM_AIR["A_s"]

    logger.info(f"Данные: вода {len(u1_water)} точек (A_ок={GEOM_WATER['A_ok']:.0f} м²), "
                f"воздух {len(u1_air)} точек (A_ок={GEOM_AIR['A_ok']:.0f} м²)")
    logger.info(f"  Вода:   σ = {sigma_w:.4f}, Re = {calc_Re(u1_water, GEOM_WATER['D_h'], GEOM_WATER['nu']).min():.0f}–"
                f"{calc_Re(u1_water, GEOM_WATER['D_h'], GEOM_WATER['nu']).max():.0f}")
    logger.info(f"  Воздух: σ = {sigma_a:.4f}, Re = {calc_Re(u1_air, GEOM_AIR['D_h'], GEOM_AIR['nu']).min():.0f}–"
                f"{calc_Re(u1_air, GEOM_AIR['D_h'], GEOM_AIR['nu']).max():.0f}")
    logger.info(f"  β = {BETA_DEG}°, L_верх = {L_UPPER} м, ε = {EPS} м")
    logger.info(f"  Критерий ξ: {CRITERION}")

    # ------------------------------------------------------------------
    # Шаг 2. Калибровка a_ξ, b_ξ, c₀ по водяным данным
    # ------------------------------------------------------------------
    logger.info("=== Шаг 2: Калибровка a_ξ, b_ξ, c₀ ===")
    a_xi, b_xi, c0, metrics_cal, result_cal = calibrate(
        u1_water, r_water, GEOM_WATER, BETA_RAD,
        L_UPPER, EPS, R_DOWN, CRITERION,
    )
    logger.info(f"  Результат: a_ξ = {a_xi:.6f}, b_ξ = {b_xi:.6f}, c₀ = {c0:.6f}")

    # ------------------------------------------------------------------
    # Шаг 3. Предсказание r для обоих наборов
    # ------------------------------------------------------------------
    logger.info("=== Шаг 3: Предсказание ===")

    # Калибровочный набор уже решён
    logger.info(f"  Вода: r_pred = {np.array2string(result_cal.r_pred, precision=4)}")
    logger.info(f"    ξ     = {np.array2string(result_cal.xi, precision=4)}")
    logger.info(f"    φ_up  = {np.array2string(result_cal.phi_up, precision=4)}")
    logger.info(f"    φ_down= {np.array2string(result_cal.phi_down, precision=4)}")
    logger.info(f"    C_β   = {np.array2string(result_cal.C_beta, precision=4)}")

    # Валидационный набор
    result_val = solve_all(
        u1_air, GEOM_AIR, a_xi, b_xi, c0, BETA_RAD,
        L_UPPER, EPS, R_DOWN, CRITERION,
    )
    logger.info(f"  Воздух: r_pred = {np.array2string(result_val.r_pred, precision=4)}")
    logger.info(f"    ξ     = {np.array2string(result_val.xi, precision=4)}")
    logger.info(f"    φ_up  = {np.array2string(result_val.phi_up, precision=4)}")
    logger.info(f"    φ_down= {np.array2string(result_val.phi_down, precision=4)}")
    logger.info(f"    C_β   = {np.array2string(result_val.C_beta, precision=4)}")

    # ------------------------------------------------------------------
    # Шаг 4. Валидация на воздушных данных
    # ------------------------------------------------------------------
    logger.info("=== Шаг 4: Валидация ===")
    metrics_val, result_val = validate(
        u1_air, r_air, GEOM_AIR, a_xi, b_xi, c0, BETA_RAD,
        L_UPPER, EPS, R_DOWN, CRITERION,
    )

    logger.info(f"  Калибровка (вода):   RMSE={metrics_cal.RMSE:.4f}, "
                f"MAE={metrics_cal.MAE:.4f}, R²={metrics_cal.R2:.4f}, "
                f"max|err|={metrics_cal.max_abs_error:.4f}")
    logger.info(f"  Валидация (воздух):  RMSE={metrics_val.RMSE:.4f}, "
                f"MAE={metrics_val.MAE:.4f}, R²={metrics_val.R2:.4f}, "
                f"max|err|={metrics_val.max_abs_error:.4f}")

    # ------------------------------------------------------------------
    # Шаг 5. Проверки
    # ------------------------------------------------------------------
    logger.info("=== Шаг 5: Проверки ===")

    # dr/du₁ < 0
    for name, u1, r_pred in [("вода", u1_water, result_cal.r_pred),
                              ("воздух", u1_air, result_val.r_pred)]:
        dr = np.diff(r_pred)
        du = np.diff(u1)
        slopes = dr / du
        monotone = np.all(slopes <= 0)
        logger.info(f"  {name}: dr/du₁ < 0 → {'✓' if monotone else '✗'} "
                    f"(slopes: {np.array2string(slopes, precision=4)})")

    # 0 < r < 1
    for name, r_pred in [("вода", result_cal.r_pred), ("воздух", result_val.r_pred)]:
        valid_range = np.all((r_pred > 0) & (r_pred < 1))
        logger.info(f"  {name}: 0 < r < 1 → {'✓' if valid_range else '✗'} "
                    f"(min={np.nanmin(r_pred):.4f}, max={np.nanmax(r_pred):.4f})")

    # Сходимость
    for name, res in [("вода", result_cal), ("воздух", result_val)]:
        n_conv = int(res.converged.sum())
        n_tot = len(res.converged)
        logger.info(f"  {name}: сходимость {n_conv}/{n_tot} "
                    f"→ {'✓' if n_conv == n_tot else '✗'}")

    # ------------------------------------------------------------------
    # Шаг 6. Диагностика: безразмерная инвариантность
    # ------------------------------------------------------------------
    logger.info("=== Шаг 6: Диагностика (ξ=const → r=const) ===")
    check_dimensionless_invariance(GEOM_WATER, a_xi, b_xi, c0, BETA_RAD,
                                   L_UPPER, EPS)

    # ------------------------------------------------------------------
    # Шаг 7. Графики
    # ------------------------------------------------------------------
    logger.info("=== Шаг 7: Графики ===")
    os.makedirs(OUTPUT_STAGE3_PLOTS, exist_ok=True)

    path1 = plot_r_vs_u1(
        u1_water, r_water, result_cal,
        u1_air, r_air, result_val,
        GEOM_WATER, GEOM_AIR, a_xi, b_xi, c0, BETA_RAD,
        L_UPPER, EPS, CRITERION,
    )
    logger.info(f"  График 1 (r vs u₁): {path1}")

    path2 = plot_parity(r_water, result_cal, r_air, result_val)
    logger.info(f"  График 2 (parity): {path2}")

    path3 = plot_xi_and_phi(
        u1_water, result_cal, u1_air, result_val,
        GEOM_WATER, GEOM_AIR, a_xi, b_xi, c0, BETA_RAD, CRITERION,
    )
    logger.info(f"  График 3 (ξ, φ, C_β): {path3}")

    path4 = plot_sensitivity(
        u1_water, r_water, GEOM_WATER, a_xi, b_xi, c0, BETA_RAD,
        L_UPPER, EPS, CRITERION,
    )
    logger.info(f"  График 4 (чувствительность): {path4}")

    u1_for_F = [u1_water[0], u1_water[len(u1_water)//2], u1_water[-1]]
    path5 = plot_F_residual(
        GEOM_WATER, a_xi, b_xi, c0, BETA_RAD, u1_for_F,
        L_UPPER, EPS, CRITERION,
    )
    logger.info(f"  График 5 (F̃(r)): {path5}")

    # ------------------------------------------------------------------
    # Шаг 8. Экспорт
    # ------------------------------------------------------------------
    logger.info("=== Шаг 8: Экспорт результатов ===")
    os.makedirs(OUTPUT_STAGE3, exist_ok=True)

    export_results(
        u1_water, r_water, result_cal, GEOM_WATER, metrics_cal,
        u1_air, r_air, result_val, GEOM_AIR, metrics_val,
        a_xi, b_xi, c0, CRITERION, OUTPUT_STAGE3,
    )

    # ------------------------------------------------------------------
    # Шаг 9. Итоговый отчёт
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ — ЭТАП 3")
    logger.info("=" * 60)
    logger.info(f"Параметры модели: a_ξ = {a_xi:.6f}, b_ξ = {b_xi:.6f}, c₀ = {c0:.6f}")
    logger.info(f"Критерий: {CRITERION}")
    logger.info(f"  Калибровка (вода):   RMSE = {metrics_cal.RMSE:.4f}")
    logger.info(f"  Валидация (воздух):  RMSE = {metrics_val.RMSE:.4f}")

    # Сравнение с этапами 1–2
    logger.info("-" * 60)
    logger.info("Сравнение с предыдущими этапами:")
    logger.info("  Этап 1:  степенной закон   — R²_cal ≈ 0.86, валидация не работает")
    logger.info("  Этап 1.1: асимптотический   — R²_cal ≈ 0.93, валидация не работает")
    logger.info("  Этап 2:  Идельчик (вар. C) — RMSE ≈ 0.19 (r ≈ const)")
    logger.info(f"  Этап 3:  физическая модель — RMSE_cal = {metrics_cal.RMSE:.4f}, "
                f"RMSE_val = {metrics_val.RMSE:.4f}")

    # Оценка результата
    cal_ok = metrics_cal.RMSE < 0.05
    val_ok = metrics_val.RMSE < 0.10
    if cal_ok and val_ok:
        verdict = "ОТЛИЧНЫЙ"
    elif cal_ok or val_ok:
        verdict = "ПРИЕМЛЕМЫЙ"
    else:
        verdict = "ТРЕБУЕТ ДОРАБОТКИ"
    logger.info(f"Результат: {verdict}")
    logger.info(f"  RMSE калибровки < 0.05: {'✓' if cal_ok else '✗'} ({metrics_cal.RMSE:.4f})")
    logger.info(f"  RMSE валидации < 0.10:  {'✓' if val_ok else '✗'} ({metrics_val.RMSE:.4f})")

    logger.info("-" * 60)
    logger.info("ВЫВОД:")
    logger.info("Полуэмпирическая модель с параметром блокировки ξ(Re)")
    logger.info("и асимметричным членом C_β = c₀·cos²β·(1−ξ)")
    logger.info("воспроизводит убывание r(u₁) через физический механизм:")
    logger.info("рост инерции струи → подавление вихрей → уменьшение блокировки")
    logger.info("→ снижение потерь Борда-Карно → меньше утечек вверх.")
    logger.info("Три калибруемых параметра (a_ξ, b_ξ, c₀) обеспечивают перенос")
    logger.info("на другую геометрию (σ) через явную зависимость φ(σ, ξ).")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
