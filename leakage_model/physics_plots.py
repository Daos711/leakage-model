"""Графики полуэмпирической модели (этап 3)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_STAGE3_PLOTS
from .physics_closures import calc_xi, calc_phi, calc_C_beta
from .physics_model import solve_all, residual_F
from .idelchik import L_UPPER_DEFAULT, EPS_DEFAULT

plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUTPUT_DIR = OUTPUT_STAGE3_PLOTS


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_r_vs_u1(u1_cal, r_cal, result_cal, u1_val, r_val, result_val,
                 geom_cal, geom_val, a_xi, b_xi, c0, beta,
                 L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT, criterion="Re"):
    """График 1: r(u₁) — эксперимент и модель для обоих наборов."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Эксперимент
    ax.plot(u1_cal, r_cal, "ko", ms=8, label="Эксперимент (вода)", zorder=5)
    ax.plot(u1_val, r_val, "rs", ms=8, label="Эксперимент (воздух)", zorder=5)

    # Модельные кривые (гладкие)
    u_fine_w = np.linspace(min(u1_cal) * 0.8, max(u1_cal) * 1.1, 200)
    res_w = solve_all(u_fine_w, geom_cal, a_xi, b_xi, c0, beta,
                      L_upper, eps, criterion=criterion)
    ax.plot(u_fine_w, res_w.r_pred, "b-", lw=2, label="Модель (вода)")

    u_fine_a = np.linspace(min(u1_val) * 0.8, max(u1_val) * 1.1, 200)
    res_a = solve_all(u_fine_a, geom_val, a_xi, b_xi, c0, beta,
                      L_upper, eps, criterion=criterion)
    ax.plot(u_fine_a, res_a.r_pred, "r--", lw=2, label="Модель (воздух)")

    # Предсказания в точках
    ax.plot(u1_cal, result_cal.r_pred, "b^", ms=7, mfc="none", lw=1.5)
    ax.plot(u1_val, result_val.r_pred, "rv", ms=7, mfc="none", lw=1.5)

    ax.set_xlabel("u₁, м/с")
    ax.set_ylabel("r = Q₂/Q₁")
    ax.set_title("Полуэмпирическая модель: r(u₁)")
    ax.legend()
    ax.set_ylim(bottom=0)
    return _save(fig, "20_physics_r_prediction.png")


def plot_parity(r_cal, result_cal, r_val, result_val):
    """График 2: parity plot (r_pred vs r_exp)."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(r_cal, result_cal.r_pred, "bo", ms=8, label="Вода (калибровка)")
    ax.plot(r_val, result_val.r_pred, "rs", ms=8, label="Воздух (валидация)")

    lims = [0, max(max(r_cal), max(r_val)) * 1.2]
    ax.plot(lims, lims, "k--", lw=1, label="Идеал")
    # ±10% полоса
    x_band = np.linspace(lims[0], lims[1], 100)
    ax.fill_between(x_band, x_band * 0.9, x_band * 1.1,
                    alpha=0.1, color="gray", label="±10%")

    ax.set_xlabel("r_exp")
    ax.set_ylabel("r_pred")
    ax.set_title("Parity plot")
    ax.legend(loc="upper left")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    return _save(fig, "21_physics_parity.png")


def plot_xi_and_phi(u1_cal, result_cal, u1_val, result_val,
                    geom_cal, geom_val, a_xi, b_xi, c0, beta, criterion="Re"):
    """График 3: ξ(u₁), φ₂(u₁), φ₃(u₁), C_β(u₁)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    u_range = np.linspace(1.0, 20.0, 300)

    # Левый: ξ(u₁) для обоих
    for geom, label, ls in [(geom_cal, "вода", "-"), (geom_val, "воздух", "--")]:
        nu = geom["nu"]
        D_h = geom["D_h"]
        if criterion == "Re":
            crit = u_range * D_h / nu
        elif criterion == "Fr":
            crit = u_range / np.sqrt(9.81 * D_h)
        else:
            crit = u_range
        xi = calc_xi(crit, a_xi, b_xi)
        axes[0].plot(u_range, xi, ls, lw=2, label=f"ξ ({label})")

    axes[0].plot(u1_cal, result_cal.xi, "bo", ms=6)
    axes[0].plot(u1_val, result_val.xi, "rs", ms=6)
    axes[0].set_xlabel("u₁, м/с")
    axes[0].set_ylabel("ξ")
    axes[0].set_title("Параметр блокировки ξ(u₁)")
    axes[0].legend()
    axes[0].set_ylim(-0.05, 1.05)

    # Средний: φ_up и φ_down
    for geom, label, ls in [(geom_cal, "вода", "-"), (geom_val, "воздух", "--")]:
        sigma = geom["A_ok"] / geom["A_s"]
        nu = geom["nu"]
        D_h = geom["D_h"]
        if criterion == "Re":
            crit = u_range * D_h / nu
        elif criterion == "Fr":
            crit = u_range / np.sqrt(9.81 * D_h)
        else:
            crit = u_range
        xi = calc_xi(crit, a_xi, b_xi)
        phi_up = calc_phi(xi, sigma, beta, "up")
        phi_down = calc_phi(xi, sigma, beta, "down")
        axes[1].plot(u_range, phi_up, f"b{ls}", lw=2, label=f"φ_up ({label})")
        axes[1].plot(u_range, phi_down, f"r{ls}", lw=2, label=f"φ_down ({label})")

    axes[1].set_xlabel("u₁, м/с")
    axes[1].set_ylabel("φ")
    axes[1].set_title("Коэффициенты сжатия φ(u₁)")
    axes[1].legend(fontsize=9)

    # Правый: C_β(u₁)
    for geom, label, ls in [(geom_cal, "вода", "-"), (geom_val, "воздух", "--")]:
        nu = geom["nu"]
        D_h = geom["D_h"]
        if criterion == "Re":
            crit = u_range * D_h / nu
        elif criterion == "Fr":
            crit = u_range / np.sqrt(9.81 * D_h)
        else:
            crit = u_range
        xi = calc_xi(crit, a_xi, b_xi)
        cb = calc_C_beta(xi, beta, c0)
        axes[2].plot(u_range, cb, ls, lw=2, label=f"C_β ({label})")

    axes[2].plot(u1_cal, result_cal.C_beta, "bo", ms=6)
    axes[2].plot(u1_val, result_val.C_beta, "rs", ms=6)
    axes[2].set_xlabel("u₁, м/с")
    axes[2].set_ylabel("C_β")
    axes[2].set_title("Асимметричный член C_β(u₁)")
    axes[2].legend()

    fig.tight_layout()
    return _save(fig, "22_physics_xi_phi.png")


def plot_sensitivity(u1_cal, r_cal, geom_cal, a_xi, b_xi, c0, beta,
                     L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT, criterion="Re"):
    """График 4: чувствительность r к a_ξ, b_ξ и c₀."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    u_fine = np.linspace(min(u1_cal) * 0.8, max(u1_cal) * 1.1, 150)

    # Левый: варьируем a_ξ
    for da in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        a_var = a_xi + da
        res = solve_all(u_fine, geom_cal, a_var, b_xi, c0, beta,
                        L_upper, eps, criterion=criterion)
        lbl = f"a_ξ = {a_var:.1f}"
        lw = 2.5 if da == 0 else 1.0
        axes[0].plot(u_fine, res.r_pred, lw=lw, label=lbl)
    axes[0].plot(u1_cal, r_cal, "ko", ms=7, label="Эксп.", zorder=10)
    axes[0].set_xlabel("u₁, м/с")
    axes[0].set_ylabel("r")
    axes[0].set_title(f"Чувствительность к a_ξ (b_ξ={b_xi:.3f}, c₀={c0:.3f})")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(bottom=0)

    # Средний: варьируем b_ξ
    for db in [-0.3, -0.15, 0.0, 0.15, 0.3]:
        b_var = b_xi + db
        res = solve_all(u_fine, geom_cal, a_xi, b_var, c0, beta,
                        L_upper, eps, criterion=criterion)
        lbl = f"b_ξ = {b_var:.3f}"
        lw = 2.5 if db == 0 else 1.0
        axes[1].plot(u_fine, res.r_pred, lw=lw, label=lbl)
    axes[1].plot(u1_cal, r_cal, "ko", ms=7, label="Эксп.", zorder=10)
    axes[1].set_xlabel("u₁, м/с")
    axes[1].set_ylabel("r")
    axes[1].set_title(f"Чувствительность к b_ξ (a_ξ={a_xi:.3f}, c₀={c0:.3f})")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(bottom=0)

    # Правый: варьируем c₀
    for dc in [-0.5, -0.25, 0.0, 0.25, 0.5]:
        c_var = c0 + dc
        res = solve_all(u_fine, geom_cal, a_xi, b_xi, c_var, beta,
                        L_upper, eps, criterion=criterion)
        lbl = f"c₀ = {c_var:.3f}"
        lw = 2.5 if dc == 0 else 1.0
        axes[2].plot(u_fine, res.r_pred, lw=lw, label=lbl)
    axes[2].plot(u1_cal, r_cal, "ko", ms=7, label="Эксп.", zorder=10)
    axes[2].set_xlabel("u₁, м/с")
    axes[2].set_ylabel("r")
    axes[2].set_title(f"Чувствительность к c₀ (a_ξ={a_xi:.3f}, b_ξ={b_xi:.3f})")
    axes[2].legend(fontsize=9)
    axes[2].set_ylim(bottom=0)

    fig.tight_layout()
    return _save(fig, "23_physics_sensitivity.png")


def plot_F_residual(geom, a_xi, b_xi, c0, beta, u1_values,
                    L_upper=L_UPPER_DEFAULT, eps=EPS_DEFAULT, criterion="Re"):
    """График 5: F(r) для нескольких u₁ — проверка единственности корня."""
    fig, ax = plt.subplots(figsize=(9, 6))
    r_range = np.linspace(0.001, 0.999, 500)

    for u1 in u1_values:
        F_vals = np.array([
            residual_F(r, u1, geom, a_xi, b_xi, c0, beta,
                       L_upper, eps, criterion=criterion)
            for r in r_range
        ])
        ax.plot(r_range, F_vals, lw=1.5, label=f"u₁={u1:.1f} м/с")

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("r")
    ax.set_ylabel("F̃(r)")
    ax.set_title("Невязка F̃(r) — проверка единственности корня")
    ax.legend()
    return _save(fig, "24_physics_residual_F.png")
