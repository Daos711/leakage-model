"""Графики для этапа 4 — направляющие пластины."""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_rmse_comparison(results_df, output_dir):
    """30_plates_rmse_comparison.png — RMSE по трём моделям (bar chart)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results_df))
    w = 0.25

    ax.bar(x - w, results_df["RMSE_M1"], w, label="M1 (ζ_пл)", color="#4C72B0")
    ax.bar(x, results_df["RMSE_M2"], w, label="M2 (Δc₀)", color="#DD8452")
    ax.bar(x + w, results_df["RMSE_M3"], w, label="M3 (ζ_пл + Δc₀)", color="#55A868")

    ax.set_xlabel("Вставка")
    ax.set_ylabel("RMSE")
    ax.set_title("Сравнение моделей M1, M2, M3 по RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["insert_id"].astype(str), rotation=90, fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "30_plates_rmse_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_zeta_by_insert(results_df, output_dir):
    """31_plates_zeta_by_insert.png — ζ_пл для каждой вставки (M3)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(results_df["insert_id"].astype(str), results_df["zeta_pl_M3"],
           color="#4C72B0")
    ax.set_xlabel("Вставка")
    ax.set_ylabel("ζ_пл")
    ax.set_title("Коэффициент сопротивления пластин ζ_пл (модель M3)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "31_plates_zeta_by_insert.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dc0_by_insert(results_df, output_dir):
    """32_plates_dc0_by_insert.png — Δc₀ для каждой вставки (M3)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#DD8452" if v >= 0 else "#C44E52" for v in results_df["delta_c0_M3"]]
    ax.bar(results_df["insert_id"].astype(str), results_df["delta_c0_M3"],
           color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Вставка")
    ax.set_ylabel("Δc₀")
    ax.set_title("Направляющий эффект Δc₀ (модель M3)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "32_plates_dc0_by_insert.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_angle_effect(results_df, plates_df, output_dir):
    """33_plates_angle_effect.png — Δc₀(α) для серии 3 (вставки 23–29)."""
    # Серия 3: вставки 23-29, разные углы при фиксированном количестве и ширине
    series3 = results_df[results_df["insert_id"].between(23, 29)].copy()

    if len(series3) == 0:
        logger.warning("Серия 3 (вставки 23-29) не найдена, график не построен")
        return None

    # Извлечь углы из plates_df
    angles = []
    for iid in series3["insert_id"]:
        sub = plates_df[plates_df["insert_id"] == iid]
        if "angle_deg" in sub.columns and len(sub) > 0:
            angles.append(sub["angle_deg"].iloc[0])
        else:
            angles.append(np.nan)
    series3 = series3.copy()
    series3["angle_deg"] = angles

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(series3["angle_deg"], series3["delta_c0_M3"], "o-", color="#4C72B0",
            markersize=8, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Угол наклона α, °")
    ax.set_ylabel("Δc₀")
    ax.set_title("Серия 3: влияние угла пластин на направляющий эффект")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "33_plates_angle_effect.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_width_effect(results_df, plates_df, output_dir):
    """34_plates_width_effect.png — ζ_пл(b) для серии 4 (вставки 30–32)."""
    series4 = results_df[results_df["insert_id"].between(30, 32)].copy()

    if len(series4) == 0:
        logger.warning("Серия 4 (вставки 30-32) не найдена, график не построен")
        return None

    widths = []
    for iid in series4["insert_id"]:
        sub = plates_df[plates_df["insert_id"] == iid]
        if "width_mm" in sub.columns and len(sub) > 0:
            widths.append(sub["width_mm"].iloc[0])
        else:
            widths.append(np.nan)
    series4 = series4.copy()
    series4["width_mm"] = widths

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(series4["width_mm"], series4["zeta_pl_M3"], "s-", color="#55A868",
            markersize=8, linewidth=2)
    ax.set_xlabel("Ширина пластины b, мм")
    ax.set_ylabel("ζ_пл")
    ax.set_title("Серия 4: влияние ширины пластин на сопротивление")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "34_plates_width_effect.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_r_prediction_best(results_df, plates_df, geom, base_params, beta,
                           L, eps, criterion, output_dir):
    """35_plates_r_prediction_best.png — r(u₁) для лучших 3-4 вставок (M3)."""
    from .model import predict_plates

    # Выбрать 4 лучшие вставки по RMSE_M3
    best = results_df.nsmallest(4, "RMSE_M3")
    a_xi, b_xi, c0 = base_params

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (_, row) in enumerate(best.iterrows()):
        iid = row["insert_id"]
        ax = axes[idx]

        sub = plates_df[plates_df["insert_id"] == iid].sort_values("u1")
        u1 = sub["u1"].values
        r_exp = sub["r_exp"].values

        # Предсказание M3
        u1_fine = np.linspace(u1.min() * 0.9, u1.max() * 1.1, 50)
        r_pred_fine, _, _ = predict_plates(
            u1_fine, geom, a_xi, b_xi, c0, beta, L, eps,
            zeta_pl=row["zeta_pl_M3"], delta_c0=row["delta_c0_M3"],
            criterion=criterion,
        )

        ax.plot(u1, r_exp, "ko", markersize=6, label="Эксперимент")
        ax.plot(u1_fine, r_pred_fine, "-", color="#4C72B0", linewidth=2,
                label=f"M3 (RMSE={row['RMSE_M3']:.4f})")
        ax.set_xlabel("u₁, м/с")
        ax.set_ylabel("r")
        ax.set_title(f"Вставка {iid}: {row['insert_name'][:40]}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "35_plates_r_prediction_best.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
