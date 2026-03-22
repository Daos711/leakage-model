"""Графики импульсной модели (этап 3)."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.config import RHO
from ..core.plot_style import setup_matplotlib, apply_comma_ticks
from ..stage1_energy.model import calc_k_ut, calc_Re
from .solver import solve_r_brent

setup_matplotlib()

from ..core.config import OUTPUT_STAGE1_1_PLOTS
OUTPUT_DIR = OUTPUT_STAGE1_1_PLOTS


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    apply_comma_ticks(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _solve_curve(u1_arr, geom, C_M):
    """Решить для массива u1 (через Q = u1 * A_ok).

    C_M может быть float или callable (Re -> C_M).
    """
    r_arr = []
    for u1 in u1_arr:
        Q1 = u1 * geom["A_ok"]
        if callable(C_M):
            Re = float(calc_Re(np.float64(u1), geom["D_h"], geom["nu"]))
            cm_val = float(C_M(Re))
        else:
            cm_val = C_M
        r_arr.append(solve_r_brent(Q1, geom, RHO, cm_val))
    return np.array(r_arr)


def plot_C_M_Re(df_cm: pd.DataFrame, cm_mean: float):
    """График 1: C_M(Re) — scatter + среднее."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_cm["Re"], df_cm["C_M"], "ko", ms=8, label="C_M по точкам")
    ax.axhline(cm_mean, color="r", ls="--", lw=2,
               label=f"C_M среднее = {cm_mean:.4f}")
    ax.set_xlabel("Число Рейнольдса Re")
    ax.set_ylabel("C_M")
    ax.legend()
    return _save(fig, "v3_01_C_M_Re.png")


def plot_C_M_diagnostics(df_cm: pd.DataFrame, cm_mean: float):
    """Графики 1a: C_M(r) и C_M(u₁) — диагностика."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df_cm["r_exp"], df_cm["C_M"], "ko", ms=8)
    ax1.axhline(cm_mean, color="r", ls="--", lw=2,
                label=f"Среднее = {cm_mean:.4f}")
    ax1.set_xlabel("r (эксперимент)")
    ax1.set_ylabel("C_M")
    ax1.legend()

    ax2.plot(df_cm["u1"], df_cm["C_M"], "ko", ms=8)
    ax2.axhline(cm_mean, color="r", ls="--", lw=2,
                label=f"Среднее = {cm_mean:.4f}")
    ax2.set_xlabel("Скорость u₁, м/с")
    ax2.set_ylabel("C_M")
    ax2.legend()

    fig.tight_layout()
    return _save(fig, "v3_01a_C_M_diagnostics.png")


def plot_r_calibration_v3(df_cal_res: pd.DataFrame, geom: dict, C_M: float):
    """График 2: r(u₁) — калибровка — точки + расчётная кривая."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df_cal_res["u1"], df_cal_res["r_exp"], "ko", ms=8,
            label="Эксперимент (вода)")
    ax.plot(df_cal_res["u1"], df_cal_res["r_calc"], "b^", ms=8,
            label="Модель (вода)")

    u_fine = np.linspace(
        df_cal_res["u1"].min() * 0.9,
        df_cal_res["u1"].max() * 1.1, 200)
    r_fine = _solve_curve(u_fine, geom, C_M)
    ax.plot(u_fine, r_fine, "b-", lw=2, alpha=0.7, label="Импульсная модель")

    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_ylabel("Доля утечек r")
    ax.legend()
    return _save(fig, "v3_02_r_calibration.png")


def plot_r_validation_v3(df_cal_res: pd.DataFrame, df_val_res: pd.DataFrame,
                         geom_cal: dict, geom_val: dict, C_M: float):
    """График 3: r(u₁) — валидация — оба набора + расчётные кривые."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df_cal_res["u1"], df_cal_res["r_exp"], "bs", ms=7,
            label="Эксперимент — вода (A_ок=12 м²)")
    ax.plot(df_val_res["u1"], df_val_res["r_exp"], "ro", ms=7,
            label="Эксперимент — воздух (A_ок=20 м²)")

    u_min = min(df_cal_res["u1"].min(), df_val_res["u1"].min()) * 0.9
    u_max = max(df_cal_res["u1"].max(), df_val_res["u1"].max()) * 1.1
    u_fine = np.linspace(u_min, u_max, 200)

    r_fine_w = _solve_curve(u_fine, geom_cal, C_M)
    r_fine_a = _solve_curve(u_fine, geom_val, C_M)

    ax.plot(u_fine, r_fine_w, "b-", lw=2, alpha=0.7, label="Модель — вода")
    ax.plot(u_fine, r_fine_a, "r-", lw=2, alpha=0.7, label="Модель — воздух")

    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_ylabel("Доля утечек r")
    ax.legend(fontsize=10)
    return _save(fig, "v3_03_r_validation.png")


def plot_k_ut_v3(df_cal_res: pd.DataFrame, df_val_res: pd.DataFrame,
                 geom_cal: dict, geom_val: dict, C_M: float):
    """График 4: k_ут(u₁) — коэффициент утечек."""
    fig, ax = plt.subplots(figsize=(8, 5))

    k_ut_cal_exp = df_cal_res["r_exp"] / (1 - df_cal_res["r_exp"])
    k_ut_val_exp = df_val_res["r_exp"] / (1 - df_val_res["r_exp"])

    ax.plot(df_cal_res["u1"], k_ut_cal_exp, "bs", ms=7,
            label="Эксперимент (вода)")
    ax.plot(df_val_res["u1"], k_ut_val_exp, "ro", ms=7,
            label="Эксперимент (воздух)")

    u_fine = np.linspace(3, 18, 200)
    r_w = _solve_curve(u_fine, geom_cal, C_M)
    r_a = _solve_curve(u_fine, geom_val, C_M)

    ax.plot(u_fine, calc_k_ut(r_w), "b-", lw=2, alpha=0.7,
            label="Модель (вода)")
    ax.plot(u_fine, calc_k_ut(r_a), "r-", lw=2, alpha=0.7,
            label="Модель (воздух)")

    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_ylabel("Коэффициент утечек k_ут")
    ax.legend(fontsize=10)
    return _save(fig, "v3_04_k_ut.png")


def plot_parity_v3(df_cal_res: pd.DataFrame, df_val_res: pd.DataFrame):
    """График 5: Parity plot r_расч vs r_эксп."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(df_cal_res["r_exp"], df_cal_res["r_calc"], "bs", ms=8,
            label="Калибровка (вода)")
    ax.plot(df_val_res["r_exp"], df_val_res["r_calc"], "ro", ms=8,
            label="Валидация (воздух)")

    all_r = np.concatenate([
        df_cal_res["r_exp"].values, df_cal_res["r_calc"].values,
        df_val_res["r_exp"].values, df_val_res["r_calc"].values,
    ])
    lims = [0, max(all_r) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Идеальное совпадение")

    ax.set_xlabel("r эксперимент")
    ax.set_ylabel("r расчёт")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    return _save(fig, "v3_05_parity.png")


def plot_comparison_v3(df_cal_res: pd.DataFrame, df_val_res: pd.DataFrame,
                       geom_cal: dict, geom_val: dict, C_M: float,
                       dz_func_best=None):
    """График 6: Сравнение импульсной модели vs прежняя r(Re)."""
    from ..stage1_energy.model import calc_r_explicit

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Левый: калибровка (вода)
    ax1.plot(df_cal_res["u1"], df_cal_res["r_exp"], "ko", ms=8,
             label="Эксперимент")
    u_fine = np.linspace(
        df_cal_res["u1"].min() * 0.9,
        df_cal_res["u1"].max() * 1.1, 200)
    r_imp = _solve_curve(u_fine, geom_cal, C_M)
    ax1.plot(u_fine, r_imp, "b-", lw=2, label="Импульсная (v3)")
    if dz_func_best is not None:
        r_old = calc_r_explicit(u_fine, geom_cal, dz_func_best)
        ax1.plot(u_fine, r_old, "g--", lw=2, label="Δζ-модель (v1)")
    ax1.set_xlabel("u₁, м/с")
    ax1.set_ylabel("r")
    ax1.legend()

    # Правый: валидация (воздух)
    ax2.plot(df_val_res["u1"], df_val_res["r_exp"], "ko", ms=8,
             label="Эксперимент")
    u_fine_v = np.linspace(
        df_val_res["u1"].min() * 0.9,
        df_val_res["u1"].max() * 1.1, 200)
    r_imp_v = _solve_curve(u_fine_v, geom_val, C_M)
    ax2.plot(u_fine_v, r_imp_v, "b-", lw=2, label="Импульсная (v3)")
    if dz_func_best is not None:
        r_old_v = calc_r_explicit(u_fine_v, geom_val, dz_func_best)
        ax2.plot(u_fine_v, r_old_v, "g--", lw=2, label="Δζ-модель (v1)")
    ax2.set_xlabel("u₁, м/с")
    ax2.set_ylabel("r")
    ax2.legend()

    fig.tight_layout()
    return _save(fig, "v3_06_comparison.png")
