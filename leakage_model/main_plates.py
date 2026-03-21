"""Этап 4 — учёт направляющих пластин.

Запуск: python -m leakage_model.main_plates [путь_к_xlsx]
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

from .config import GEOM_WATER, BETA_RAD, OUTPUT_STAGE4, OUTPUT_STAGE4_PLOTS
from .idelchik import L_UPPER_DEFAULT, EPS_DEFAULT
from .physics_model import solve_all
from .plates_model import predict_plates
from .plates_calibration import calibrate_all, compute_aicc
from .plates_plots import (
    plot_rmse_comparison,
    plot_zeta_by_insert,
    plot_dc0_by_insert,
    plot_angle_effect,
    plot_width_effect,
    plot_r_prediction_best,
)
from .validation import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_XLSX = os.path.join(
    os.path.dirname(__file__), "..",
    "Расчет_параметров_модели_и_результаты_моделирования___ВОДА.xlsx",
)

CRITERION = "Re"
L_UPPER = L_UPPER_DEFAULT
EPS = EPS_DEFAULT


def _load_base_params():
    """Загрузить a_ξ, b_ξ, c₀ из physics_parameters.json."""
    params_path = os.path.join(
        os.path.dirname(__file__), "output", "stage3_physics", "physics_parameters.json"
    )
    with open(params_path) as f:
        params = json.load(f)
    a_xi = params["a_xi"]
    b_xi = params["b_xi"]
    c0 = params["c0"]
    logger.info("Базовые параметры: a_ξ=%.4f, b_ξ=%.4f, c₀=%.4f", a_xi, b_xi, c0)
    return a_xi, b_xi, c0


def _diagnose_q25(plates_df, geom, base_params, beta):
    """Диагностика точки Q=25 на вставке №1.

    Сравнивает базовую модель с 8 и 7 точками.
    Возвращает True, если Q=25 следует исключить.
    """
    a_xi, b_xi, c0 = base_params
    ins1 = plates_df[plates_df["insert_id"] == 1].sort_values("u1")

    if len(ins1) == 0:
        logger.warning("Вставка №1 не найдена в данных")
        return False

    u1_all = ins1["u1"].values
    r_exp_all = ins1["r_exp"].values

    # Базовая модель на всех 8 точках
    result_8 = solve_all(u1_all, geom, a_xi, b_xi, c0, beta,
                         L_UPPER, EPS, criterion=CRITERION)
    err_8 = np.abs(result_8.r_pred - r_exp_all)
    rmse_8 = np.sqrt(np.mean((result_8.r_pred - r_exp_all) ** 2))

    logger.info("Вставка №1 (8 точек): RMSE=%.4f", rmse_8)
    for i in range(len(u1_all)):
        logger.info("  u₁=%.2f: r_exp=%.4f, r_pred=%.4f, |err|=%.4f",
                     u1_all[i], r_exp_all[i], result_8.r_pred[i], err_8[i])

    # Найти Q=25 (минимальная скорость)
    idx_q25 = np.argmin(u1_all)
    err_q25 = err_8[idx_q25]
    r_pred_q25 = result_8.r_pred[idx_q25]

    # Без Q=25
    mask_no_q25 = np.ones(len(u1_all), dtype=bool)
    mask_no_q25[idx_q25] = False
    u1_7 = u1_all[mask_no_q25]
    r_exp_7 = r_exp_all[mask_no_q25]

    result_7 = solve_all(u1_7, geom, a_xi, b_xi, c0, beta,
                         L_UPPER, EPS, criterion=CRITERION)
    rmse_7 = np.sqrt(np.mean((result_7.r_pred - r_exp_7) ** 2))

    logger.info("Вставка №1 (7 точек, без Q=25): RMSE=%.4f", rmse_7)

    # Критерий: Q=25 — выброс, если ошибка > 0.1 или r_pred ≈ 0.5
    is_outlier = err_q25 > 0.1 or r_pred_q25 > 0.45
    if is_outlier:
        logger.info(
            "РЕШЕНИЕ: Q=25 исключена глобально "
            "(r_pred=%.4f, |err|=%.4f — систематический выброс)",
            r_pred_q25, err_q25,
        )
    else:
        logger.info(
            "РЕШЕНИЕ: Q=25 оставлена (r_pred=%.4f, |err|=%.4f — в пределах нормы)",
            r_pred_q25, err_q25,
        )

    return is_outlier


def main(xlsx_path=None):
    if xlsx_path is None:
        xlsx_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_XLSX

    os.makedirs(OUTPUT_STAGE4, exist_ok=True)
    os.makedirs(OUTPUT_STAGE4_PLOTS, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ЭТАП 4 — УЧЁТ НАПРАВЛЯЮЩИХ ПЛАСТИН")
    logger.info("=" * 60)

    # --- Шаг 1: Загрузка данных ---
    logger.info("=== Шаг 1: Загрузка данных из Excel ===")
    from .plates_data import load_plates_with_geometry
    plates_df = load_plates_with_geometry(xlsx_path)
    n_inserts = plates_df["insert_id"].nunique()
    n_points = len(plates_df)
    logger.info("Загружено: %d вставок, %d точек", n_inserts, n_points)

    # Сохранить сырые данные
    plates_df.to_csv(os.path.join(OUTPUT_STAGE4, "plates_data.csv"), index=False)

    # --- Загрузка базовых параметров ---
    a_xi, b_xi, c0 = _load_base_params()
    base_params = (a_xi, b_xi, c0)
    geom = GEOM_WATER

    # --- Шаг 2: Диагностика Q=25 на вставке №1 ---
    logger.info("=== Шаг 2: Проверка базовой модели на вставке №1 ===")
    exclude_q25 = _diagnose_q25(plates_df, geom, base_params, BETA_RAD)

    if exclude_q25:
        # Найти минимальный Q для каждой вставки и исключить
        min_u1_per_insert = plates_df.groupby("insert_id")["u1"].transform("min")
        n_before = len(plates_df)
        plates_df = plates_df[plates_df["u1"] != min_u1_per_insert].copy()
        n_after = len(plates_df)
        logger.info("Исключено %d точек (Q=25) из всех вставок", n_before - n_after)

    # --- Шаг 3: Калибровка M1, M2, M3 ---
    logger.info("=== Шаг 3: Калибровка M1, M2, M3 для каждой вставки ===")
    cal_results = calibrate_all(
        plates_df, geom, base_params, BETA_RAD, L_UPPER, EPS,
        criterion=CRITERION, exclude_insert_1=True,
    )
    results_df = pd.DataFrame(cal_results)
    logger.info("Откалибровано %d вставок", len(results_df))

    # --- Шаг 4: Сравнение моделей ---
    logger.info("=== Шаг 4: Сравнение моделей ===")
    for model in ["M1", "M2", "M3"]:
        rmse_col = f"RMSE_{model}"
        aicc_col = f"AICc_{model}"
        mean_rmse = results_df[rmse_col].mean()
        median_rmse = results_df[rmse_col].median()
        mean_aicc = results_df[aicc_col].mean()
        logger.info(
            "  %s: средний RMSE=%.4f, медианный RMSE=%.4f, средний AICc=%.2f",
            model, mean_rmse, median_rmse, mean_aicc,
        )

    # Лучшая модель по среднему RMSE
    mean_rmse = {m: results_df[f"RMSE_{m}"].mean() for m in ["M1", "M2", "M3"]}
    best_model = min(mean_rmse, key=mean_rmse.get)
    logger.info("Лучшая модель по среднему RMSE: %s (%.4f)", best_model, mean_rmse[best_model])

    # Сводная таблица
    comparison_cols = ["insert_id", "insert_name",
                       "RMSE_M1", "RMSE_M2", "RMSE_M3",
                       "AICc_M1", "AICc_M2", "AICc_M3"]
    comparison_df = results_df[comparison_cols].copy()
    comparison_df.to_csv(os.path.join(OUTPUT_STAGE4, "plates_comparison.csv"), index=False)

    # --- Шаг 5: Анализ параметров ---
    logger.info("=== Шаг 5: Анализ параметров ===")
    params_cols = ["insert_id", "insert_name", "zeta_pl_M3", "delta_c0_M3", "RMSE_M3"]
    params_df = results_df[params_cols].copy()
    logger.info("Параметры M3:")
    for _, row in params_df.iterrows():
        logger.info(
            "  Вставка %2d: ζ_пл=%.4f, Δc₀=%+.4f, RMSE=%.4f  %s",
            row["insert_id"], row["zeta_pl_M3"], row["delta_c0_M3"],
            row["RMSE_M3"], row["insert_name"][:50],
        )

    # Проверка «вредных» вставок (28, 29)
    for iid in [28, 29]:
        sub = results_df[results_df["insert_id"] == iid]
        if len(sub) > 0:
            dc0 = sub["delta_c0_M3"].iloc[0]
            sign = "< 0 (вредная)" if dc0 < 0 else ">= 0"
            logger.info("  Вставка %d: Δc₀ = %+.4f %s", iid, dc0, sign)

    # Серия 3 (23-29): зависимость Δc₀(α)
    series3 = results_df[results_df["insert_id"].between(23, 29)]
    if len(series3) > 0:
        logger.info("Серия 3 (вставки 23-29, разные углы):")
        for _, row in series3.iterrows():
            logger.info("  Вставка %d: Δc₀=%+.4f, ζ_пл=%.4f",
                         row["insert_id"], row["delta_c0_M3"], row["zeta_pl_M3"])

    # Серия 4 (30-32): зависимость ζ_пл(b)
    series4 = results_df[results_df["insert_id"].between(30, 32)]
    if len(series4) > 0:
        logger.info("Серия 4 (вставки 30-32, разная ширина):")
        for _, row in series4.iterrows():
            logger.info("  Вставка %d: ζ_пл=%.4f, Δc₀=%+.4f",
                         row["insert_id"], row["zeta_pl_M3"], row["delta_c0_M3"])

    # --- Шаг 6: Графики ---
    logger.info("=== Шаг 6: Графики ===")
    p1 = plot_rmse_comparison(results_df, OUTPUT_STAGE4_PLOTS)
    logger.info("  График 1: %s", p1)

    p2 = plot_zeta_by_insert(results_df, OUTPUT_STAGE4_PLOTS)
    logger.info("  График 2: %s", p2)

    p3 = plot_dc0_by_insert(results_df, OUTPUT_STAGE4_PLOTS)
    logger.info("  График 3: %s", p3)

    p4 = plot_angle_effect(results_df, plates_df, OUTPUT_STAGE4_PLOTS)
    logger.info("  График 4: %s", p4)

    p5 = plot_width_effect(results_df, plates_df, OUTPUT_STAGE4_PLOTS)
    logger.info("  График 5: %s", p5)

    p6 = plot_r_prediction_best(
        results_df, plates_df, geom, base_params, BETA_RAD,
        L_UPPER, EPS, CRITERION, OUTPUT_STAGE4_PLOTS,
    )
    logger.info("  График 6: %s", p6)

    # --- Шаг 7: Экспорт ---
    logger.info("=== Шаг 7: Экспорт ===")

    # plates_parameters_M1.csv
    m1_df = results_df[["insert_id", "insert_name", "zeta_pl_M1", "RMSE_M1"]].copy()
    m1_df.columns = ["insert_id", "insert_name", "zeta_pl", "RMSE"]
    m1_df.to_csv(os.path.join(OUTPUT_STAGE4, "plates_parameters_M1.csv"), index=False)

    # plates_parameters_M2.csv
    m2_df = results_df[["insert_id", "insert_name", "delta_c0_M2", "RMSE_M2"]].copy()
    m2_df.columns = ["insert_id", "insert_name", "delta_c0", "RMSE"]
    m2_df.to_csv(os.path.join(OUTPUT_STAGE4, "plates_parameters_M2.csv"), index=False)

    # plates_parameters_M3.csv
    m3_df = results_df[["insert_id", "insert_name", "zeta_pl_M3", "delta_c0_M3",
                         "RMSE_M3", "AICc_M3"]].copy()
    m3_df.columns = ["insert_id", "insert_name", "zeta_pl", "delta_c0", "RMSE", "AICc"]
    m3_df.to_csv(os.path.join(OUTPUT_STAGE4, "plates_parameters_M3.csv"), index=False)

    # plates_prediction_M3.csv — по точкам
    pred_records = []
    for _, row in results_df.iterrows():
        iid = row["insert_id"]
        sub = plates_df[plates_df["insert_id"] == iid].sort_values("u1")
        u1 = sub["u1"].values
        r_exp = sub["r_exp"].values

        r_pred, conv, _ = predict_plates(
            u1, geom, a_xi, b_xi, c0, BETA_RAD, L_UPPER, EPS,
            zeta_pl=row["zeta_pl_M3"], delta_c0=row["delta_c0_M3"],
            criterion=CRITERION,
        )
        for j in range(len(u1)):
            pred_records.append({
                "insert_id": iid,
                "insert_name": row["insert_name"],
                "u1": u1[j],
                "r_exp": r_exp[j],
                "r_pred_M3": r_pred[j],
                "error_M3": r_pred[j] - r_exp[j],
            })

    pred_df = pd.DataFrame(pred_records)
    pred_df.to_csv(os.path.join(OUTPUT_STAGE4, "plates_prediction_M3.csv"), index=False)

    logger.info("Экспортировано в %s", OUTPUT_STAGE4)

    # --- Шаг 8: Итоговый отчёт ---
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ — ЭТАП 4 (НАПРАВЛЯЮЩИЕ ПЛАСТИНЫ)")
    logger.info("=" * 60)

    for model in ["M1", "M2", "M3"]:
        col = f"RMSE_{model}"
        logger.info(
            "%s: средний RMSE=%.4f, медианный=%.4f, мин=%.4f, макс=%.4f",
            model,
            results_df[col].mean(), results_df[col].median(),
            results_df[col].min(), results_df[col].max(),
        )

    # Лучшие и худшие вставки (M3)
    best3 = results_df.nsmallest(3, "RMSE_M3")
    worst3 = results_df.nlargest(3, "RMSE_M3")

    logger.info("Лучшие вставки (M3):")
    for _, row in best3.iterrows():
        logger.info("  Вставка %d: RMSE=%.4f, ζ_пл=%.4f, Δc₀=%+.4f",
                     row["insert_id"], row["RMSE_M3"],
                     row["zeta_pl_M3"], row["delta_c0_M3"])

    logger.info("Худшие вставки (M3):")
    for _, row in worst3.iterrows():
        logger.info("  Вставка %d: RMSE=%.4f, ζ_пл=%.4f, Δc₀=%+.4f",
                     row["insert_id"], row["RMSE_M3"],
                     row["zeta_pl_M3"], row["delta_c0_M3"])

    logger.info("Лучшая модель по среднему RMSE: %s", best_model)
    logger.info("Q=25 исключена: %s", "Да" if exclude_q25 else "Нет")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
