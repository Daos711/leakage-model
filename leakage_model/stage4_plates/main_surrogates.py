"""Этап 4.1/4.2 — суррогатные модели и оптимизация пластин.

Запуск: python -m leakage_model.stage4_plates.main_surrogates
"""

import json
import logging
import os

import numpy as np
import pandas as pd

from ..core.config import (
    GEOM_WATER, BETA_RAD,
    OUTPUT_STAGE4, OUTPUT_STAGE4_SURROGATES, OUTPUT_STAGE4_SURR_PLOTS,
)
from ..stage2_idelchik.coefficients import L_UPPER_DEFAULT, EPS_DEFAULT
from .data import load_plates_with_geometry
from .surrogates import (
    fit_series3_angle,
    fit_series4_width,
    fit_series1_nplates,
)
from .optimization import optimize_angle, optimize_width, optimize_joint
from .surrogate_plots import (
    plot_dc0_vs_angle,
    plot_zeta_vs_angle,
    plot_zeta_vs_width,
    plot_dc0_vs_width,
    plot_r_vs_angle,
    plot_r_vs_width,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

CRITERION = "Re"
L_UPPER = L_UPPER_DEFAULT
EPS = EPS_DEFAULT
U1_WORK = 12.5  # рабочая скорость, м/с


def _load_base_params():
    """Загрузить a_ξ, b_ξ, c₀ из physics_parameters.json."""
    params_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "output",
        "stage3_physics", "physics_parameters.json"
    )
    with open(params_path) as f:
        params = json.load(f)
    return params["a_xi"], params["b_xi"], params["c0"]


def _load_m3_params():
    """Загрузить параметры M3 из plates_parameters_M3.csv."""
    path = os.path.join(OUTPUT_STAGE4, "plates_parameters_M3.csv")
    return pd.read_csv(path)


def _json_safe(obj):
    """Рекурсивно привести значения к JSON-совместимым типам."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj


def main():
    os.makedirs(OUTPUT_STAGE4_SURROGATES, exist_ok=True)
    os.makedirs(OUTPUT_STAGE4_SURR_PLOTS, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ЭТАП 4.1 — СУРРОГАТНЫЕ МОДЕЛИ ДЛЯ ПАРАМЕТРОВ ПЛАСТИН")
    logger.info("=" * 60)

    # === Шаг 1: Загрузка данных ===
    logger.info("=== Шаг 1: Загрузка данных ===")
    m3_df = _load_m3_params()
    plates_df = load_plates_with_geometry()
    a_xi, b_xi, c0 = _load_base_params()
    base_params = (a_xi, b_xi, c0)
    geom = GEOM_WATER

    logger.info("Загружено %d вставок из M3", len(m3_df))
    logger.info("Базовые параметры: a_ξ=%.4f, b_ξ=%.4f, c₀=%.4f", a_xi, b_xi, c0)

    # Объединить M3 параметры с геометрией
    geom_info = plates_df.drop_duplicates("insert_id")[
        ["insert_id", "series_id", "n_plates", "angle_deg", "width_mm", "topology"]
    ]
    m3_df = m3_df.merge(geom_info, on="insert_id", how="left")

    # === Шаг 2: Суррогаты по сериям ===
    logger.info("=== Шаг 2: Построение суррогатов ===")
    surrogates = {}

    # --- Серия 3: угол (вставки 23–29) ---
    s3 = m3_df[m3_df["insert_id"].between(23, 29)].sort_values("insert_id")
    angles_s3 = np.array([25, 30, 35, 40, 50, 55, 60], dtype=float)
    zeta_s3 = s3["zeta_pl"].values
    dc0_s3 = s3["delta_c0"].values
    surr3 = fit_series3_angle(angles_s3, zeta_s3, dc0_s3)
    surrogates[3] = surr3

    # --- Серия 4: ширина (вставки 30–32) ---
    s4 = m3_df[m3_df["insert_id"].between(30, 32)].sort_values("insert_id")
    widths_s4 = np.array([250, 500, 750], dtype=float)
    zeta_s4 = s4["zeta_pl"].values
    dc0_s4 = s4["delta_c0"].values
    surr4 = fit_series4_width(widths_s4, zeta_s4, dc0_s4)
    surrogates[4] = surr4

    # --- Серия 1: количество пластин (вставки 3, 5, 4 — 45°) ---
    s1_ids = [3, 5, 4]  # n=3, 5, 10
    s1 = m3_df[m3_df["insert_id"].isin(s1_ids)].set_index("insert_id").loc[s1_ids]
    n_plates_s1 = np.array([3, 5, 10], dtype=float)
    zeta_s1 = s1["zeta_pl"].values
    dc0_s1 = s1["delta_c0"].values
    surr1 = fit_series1_nplates(n_plates_s1, zeta_s1, dc0_s1)
    surrogates[1] = surr1

    # --- Серия 2: дискретная таблица ---
    logger.info("Серия 2 (дискретная таблица): суррогат не строится")
    s2_short = m3_df[m3_df["insert_id"].between(9, 16)].sort_values("insert_id")
    s2_long = m3_df[m3_df["insert_id"].between(17, 22)].sort_values("insert_id")

    logger.info("  Подсерия «короткий ток» (вставки 9–16):")
    logger.info("    Средние: ζ_пл=%.4f, Δc₀=%.4f, RMSE=%.4f",
                s2_short["zeta_pl"].mean(), s2_short["delta_c0"].mean(),
                s2_short["RMSE"].mean())
    logger.info("  Подсерия «длинный ток» (вставки 17–22):")
    logger.info("    Средние: ζ_пл=%.4f, Δc₀=%.4f, RMSE=%.4f",
                s2_long["zeta_pl"].mean(), s2_long["delta_c0"].mean(),
                s2_long["RMSE"].mean())
    if s2_short["delta_c0"].mean() > s2_long["delta_c0"].mean():
        logger.info("    Короткий ток эффективнее по Δc₀")
    else:
        logger.info("    Длинный ток эффективнее по Δc₀")

    # === Шаг 3: Оптимизация ===
    logger.info("=" * 60)
    logger.info("ЭТАП 4.2 — ОПТИМИЗАЦИЯ")
    logger.info("=" * 60)

    # Оптимизация по углу
    logger.info("=== Оптимизация по углу (серия 3) ===")
    alpha_opt, r_opt_a, det_a = optimize_angle(
        surrogates, U1_WORK, geom, base_params, BETA_RAD, L_UPPER, EPS, CRITERION,
    )

    # Оптимизация по ширине
    logger.info("=== Оптимизация по ширине (серия 4) ===")
    width_opt, r_opt_w, det_w = optimize_width(
        surrogates, U1_WORK, geom, base_params, BETA_RAD, L_UPPER, EPS, CRITERION,
    )

    # Совместная оптимизация
    logger.info("=== Совместная оптимизация (исследовательский сценарий) ===")
    alpha_j, width_j, r_j, det_j = optimize_joint(
        surrogates, U1_WORK, geom, base_params, BETA_RAD, L_UPPER, EPS, CRITERION,
    )

    # === Шаг 4: Графики ===
    logger.info("=== Шаг 4: Графики ===")
    p1 = plot_dc0_vs_angle(angles_s3, dc0_s3, surr3, OUTPUT_STAGE4_SURR_PLOTS)
    logger.info("  %s", p1)
    p2 = plot_zeta_vs_angle(angles_s3, zeta_s3, surr3, OUTPUT_STAGE4_SURR_PLOTS)
    logger.info("  %s", p2)
    p3 = plot_zeta_vs_width(widths_s4, zeta_s4, surr4, OUTPUT_STAGE4_SURR_PLOTS)
    logger.info("  %s", p3)
    p4 = plot_dc0_vs_width(widths_s4, dc0_s4, surr4, OUTPUT_STAGE4_SURR_PLOTS)
    logger.info("  %s", p4)
    p5 = plot_r_vs_angle(det_a, alpha_opt, r_opt_a, OUTPUT_STAGE4_SURR_PLOTS)
    logger.info("  %s", p5)
    p6 = plot_r_vs_width(det_w, width_opt, r_opt_w, OUTPUT_STAGE4_SURR_PLOTS)
    logger.info("  %s", p6)

    # === Шаг 5: Экспорт ===
    logger.info("=== Шаг 5: Экспорт ===")

    # surrogate_params.json
    surr_export = {
        "series_3_angle": {
            "dc0_coeffs": surr3["dc0_coeffs"],
            "dc0_R2": surr3["dc0_R2"],
            "dc0_RMSE": surr3["dc0_RMSE"],
            "alpha_opt_dc0": surr3["alpha_opt_dc0"],
            "zeta_coeffs": surr3["zeta_coeffs"],
            "zeta_R2": surr3["zeta_R2"],
            "zeta_RMSE": surr3["zeta_RMSE"],
        },
        "series_4_width": {
            "zeta_coeffs": surr4["zeta_coeffs"],
            "zeta_R2": surr4["zeta_R2"],
            "zeta_RMSE": surr4["zeta_RMSE"],
            "dc0_coeffs": surr4["dc0_coeffs"],
            "dc0_R2": surr4["dc0_R2"],
            "dc0_RMSE": surr4["dc0_RMSE"],
        },
        "series_1_nplates": {
            "zeta_coeffs": surr1.get("zeta_coeffs", {}),
            "zeta_R2": surr1.get("zeta_R2", 0.0),
            "dc0_coeffs": surr1.get("dc0_coeffs", {}),
            "dc0_R2": surr1.get("dc0_R2", 0.0),
        },
    }

    surr_path = os.path.join(OUTPUT_STAGE4_SURROGATES, "surrogate_params.json")
    with open(surr_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(surr_export), f, indent=2, ensure_ascii=False)
    logger.info("  %s", surr_path)

    # optimization_results.json
    opt_export = {
        "u1_work": U1_WORK,
        "angle_optimization": {
            "alpha_opt": alpha_opt,
            "r_opt": r_opt_a,
        },
        "width_optimization": {
            "width_opt_mm": width_opt,
            "r_opt": r_opt_w,
        },
        "joint_optimization": {
            "alpha_opt": alpha_j,
            "width_opt_mm": width_j,
            "r_opt": r_j,
            "note": det_j.get("note", ""),
        },
    }

    opt_path = os.path.join(OUTPUT_STAGE4_SURROGATES, "optimization_results.json")
    with open(opt_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(opt_export), f, indent=2, ensure_ascii=False)
    logger.info("  %s", opt_path)

    # === Шаг 6: Итоговый отчёт ===
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ — ЭТАП 4.1/4.2")
    logger.info("=" * 60)

    logger.info("Суррогаты:")
    logger.info("  Серия 3 (угол): R²(Δc₀)=%.4f, R²(ζ)=%.4f, α*(Δc₀)=%.1f°",
                surr3["dc0_R2"], surr3["zeta_R2"], surr3["alpha_opt_dc0"])
    logger.info("  Серия 4 (ширина): R²(ζ)=%.4f, m=%.2f, R²(Δc₀)=%.4f",
                surr4["zeta_R2"], surr4["zeta_coeffs"]["m"], surr4["dc0_R2"])
    logger.info("  Серия 1 (n пластин): R²(ζ)=%.4f, R²(Δc₀)=%.4f",
                surr1.get("zeta_R2", 0), surr1.get("dc0_R2", 0))

    logger.info("Оптимизация (u₁=%.1f м/с):", U1_WORK)
    logger.info("  По углу: α*=%.1f°, r*=%.4f", alpha_opt, r_opt_a)
    logger.info("  По ширине: b*=%.0f мм, r*=%.4f", width_opt, r_opt_w)
    logger.info("  Совместная: α*=%.1f°, b*=%.0f мм, r*=%.4f", alpha_j, width_j, r_j)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
