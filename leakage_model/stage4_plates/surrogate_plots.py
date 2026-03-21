"""Графики суррогатных моделей (этап 4.1)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .surrogates import predict_series3, predict_series4

plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dc0_vs_angle(angles, delta_c0, surr3, output_dir):
    """40_surr_dc0_vs_angle.png — Δc₀(α): данные + квадратичная регрессия + оптимум."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(angles, delta_c0, "ko", ms=8, zorder=5, label="Данные (серия 3)")

    a_fine = np.linspace(20, 65, 200)
    dc0_fine = []
    for a in a_fine:
        _, dc = predict_series3(a, surr3)
        dc0_fine.append(dc)
    dc0_fine = np.array(dc0_fine)
    ax.plot(a_fine, dc0_fine, "r-", lw=2, label="Квадратичная регрессия")

    # Оптимум
    alpha_opt = surr3["alpha_opt_dc0"]
    if np.isfinite(alpha_opt) and 20 <= alpha_opt <= 65:
        _, dc0_opt = predict_series3(alpha_opt, surr3)
        ax.axvline(alpha_opt, ls="--", color="blue", alpha=0.7)
        ax.plot(alpha_opt, dc0_opt, "b*", ms=15, zorder=6,
                label=f"α* = {alpha_opt:.1f}°")

    ax.set_xlabel("Угол наклона α, °")
    ax.set_ylabel("Δc₀")
    ax.set_title("Серия 3: Δc₀(α) — направляющий эффект")
    ax.legend()
    return _save(fig, output_dir, "40_surr_dc0_vs_angle.png")


def plot_zeta_vs_angle(angles, zeta_pl, surr3, output_dir):
    """41_surr_zeta_vs_angle.png — ζ_пл(α): данные + регрессия."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(angles, zeta_pl, "ko", ms=8, zorder=5, label="Данные (серия 3)")

    a_fine = np.linspace(20, 65, 200)
    zeta_fine = []
    for a in a_fine:
        z, _ = predict_series3(a, surr3)
        zeta_fine.append(z)
    zeta_fine = np.array(zeta_fine)
    ax.plot(a_fine, zeta_fine, "b-", lw=2, label="Квадратичная регрессия")

    ax.set_xlabel("Угол наклона α, °")
    ax.set_ylabel("ζ_пл")
    ax.set_title("Серия 3: ζ_пл(α) — сопротивление пластины")
    ax.legend()
    return _save(fig, output_dir, "41_surr_zeta_vs_angle.png")


def plot_zeta_vs_width(widths, zeta_pl, surr4, output_dir):
    """42_surr_zeta_vs_width.png — ζ_пл(b): данные + степенной закон."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(widths, zeta_pl, "ko", ms=8, zorder=5, label="Данные (серия 4)")

    b_fine = np.linspace(100, 1500, 200)
    zeta_fine = []
    for b in b_fine:
        z, _ = predict_series4(b, surr4)
        zeta_fine.append(z)
    zeta_fine = np.array(zeta_fine)
    ax.plot(b_fine, zeta_fine, "b-", lw=2, label="Степенной закон")

    ax.set_xlabel("Ширина пластины b, мм")
    ax.set_ylabel("ζ_пл")
    ax.set_title("Серия 4: ζ_пл(b) — сопротивление пластины")
    ax.legend()
    return _save(fig, output_dir, "42_surr_zeta_vs_width.png")


def plot_dc0_vs_width(widths, delta_c0, surr4, output_dir):
    """43_surr_dc0_vs_width.png — Δc₀(b): данные + лог-линейный закон."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(widths, delta_c0, "ko", ms=8, zorder=5, label="Данные (серия 4)")

    b_fine = np.linspace(100, 1500, 200)
    dc0_fine = []
    for b in b_fine:
        _, dc = predict_series4(b, surr4)
        dc0_fine.append(dc)
    dc0_fine = np.array(dc0_fine)
    ax.plot(b_fine, dc0_fine, "r-", lw=2, label="Лог-линейная модель")

    ax.set_xlabel("Ширина пластины b, мм")
    ax.set_ylabel("Δc₀")
    ax.set_title("Серия 4: Δc₀(b) — направляющий эффект")
    ax.legend()
    return _save(fig, output_dir, "43_surr_dc0_vs_width.png")


def plot_r_vs_angle(opt_details, alpha_opt, r_opt, output_dir):
    """44_surr_r_vs_angle.png — r(α) при u₁=12.5 м/с."""
    fig, ax = plt.subplots(figsize=(8, 5))

    alphas = np.array(opt_details["alpha_range"])
    r_profile = np.array(opt_details["r_profile"])
    u1 = opt_details["u1"]

    ax.plot(alphas, r_profile, "b-", lw=2, label=f"r(α), u₁={u1} м/с")
    ax.axvline(alpha_opt, ls="--", color="red", alpha=0.7)
    ax.plot(alpha_opt, r_opt, "r*", ms=15, zorder=5,
            label=f"α* = {alpha_opt:.1f}°, r* = {r_opt:.4f}")

    ax.axhline(0.138, ls=":", color="gray", lw=1.5, label="без пластин (r = 0.138)")

    ax.set_xlabel("Угол наклона α, °")
    ax.set_ylabel("Доля утечек r")
    ax.set_title(f"Оптимизация: r(α) при u₁ = {u1} м/с")
    ax.legend()
    return _save(fig, output_dir, "44_surr_r_vs_angle.png")


def plot_r_vs_width(opt_details, width_opt, r_opt, output_dir):
    """45_surr_r_vs_width.png — r(b) при α=45°, u₁=12.5 м/с."""
    fig, ax = plt.subplots(figsize=(8, 5))

    widths = np.array(opt_details["width_range"])
    r_profile = np.array(opt_details["r_profile"])
    u1 = opt_details["u1"]

    ax.plot(widths, r_profile, "b-", lw=2, label=f"r(b), α=45°, u₁={u1} м/с")
    ax.axvline(width_opt, ls="--", color="red", alpha=0.7)
    ax.plot(width_opt, r_opt, "r*", ms=15, zorder=5,
            label=f"b* = {width_opt:.0f} мм, r* = {r_opt:.4f}")

    ax.axhline(0.138, ls=":", color="gray", lw=1.5, label="без пластин (r = 0.138)")

    ax.set_xlabel("Ширина пластины b, мм")
    ax.set_ylabel("Доля утечек r")
    ax.set_title(f"Оптимизация: r(b) при α=45°, u₁ = {u1} м/с")
    ax.legend()
    return _save(fig, output_dir, "45_surr_r_vs_width.png")
