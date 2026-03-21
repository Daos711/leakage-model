"""Графики модели."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

from ..core.config import OUTPUT_STAGE1_PLOTS

OUTPUT_DIR = OUTPUT_STAGE1_PLOTS


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_r_calibration(df_cal, geom_w, dz_func_A, dz_func_B):
    """График 1: r(u₁) — калибровка."""
    from .model import calc_r_explicit

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df_cal["u1"], df_cal["r"], "ko", ms=7, label="Эксперимент (вода)")

    u_fine = np.linspace(df_cal["u1"].min() * 0.9, df_cal["u1"].max() * 1.1, 200)
    r_A = calc_r_explicit(u_fine, geom_w, dz_func_A)
    r_B = calc_r_explicit(u_fine, geom_w, dz_func_B)

    ax.plot(u_fine, r_A, "b-", lw=2, label="Вариант A (степенной)")
    ax.plot(u_fine, r_B, "r--", lw=2, label="Вариант B (асимптотический)")

    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_ylabel("Доля утечек r")
    ax.legend()
    return _save(fig, "01_r_calibration.png")


def plot_r_validation(df_cal, df_val, geom_w, geom_a, dz_func_A, dz_func_B):
    """График 2: r(u₁) — валидация (оба набора на одном графике)."""
    from .model import calc_r_explicit

    fig, ax = plt.subplots(figsize=(8, 5))

    # Экспериментальные точки
    ax.plot(df_cal["u1"], df_cal["r"], "bs", ms=7,
            label="Эксперимент — вода (A_ок=12 м²)")
    ax.plot(df_val["u1"], df_val["r"], "ro", ms=7,
            label="Эксперимент — воздух (A_ок=20 м²)")

    # Кривая модели для воздушной геометрии
    u_fine = np.linspace(
        min(df_cal["u1"].min(), df_val["u1"].min()) * 0.9,
        max(df_cal["u1"].max(), df_val["u1"].max()) * 1.1,
        200,
    )
    r_A_air = calc_r_explicit(u_fine, geom_a, dz_func_A)
    r_A_wat = calc_r_explicit(u_fine, geom_w, dz_func_A)

    ax.plot(u_fine, r_A_wat, "b-", lw=1.5, alpha=0.5, label="Модель A — вода")
    ax.plot(u_fine, r_A_air, "r-", lw=2, label="Модель A — воздух")

    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_ylabel("Доля утечек r")
    ax.legend(fontsize=10)
    return _save(fig, "02_r_validation.png")


def plot_dz_Re(Re, dz_exp, dz_func_A, dz_func_B):
    """График 3: Δζ(Re) — калиброванные зависимости."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(Re, dz_exp, "ko", ms=7, label="Δζ из эксперимента")

    Re_fine = np.linspace(Re.min() * 0.8, Re.max() * 1.2, 200)
    ax.plot(Re_fine, dz_func_A(Re_fine), "b-", lw=2, label="Вариант A (степенной)")
    ax.plot(Re_fine, dz_func_B(Re_fine), "r--", lw=2, label="Вариант B (асимптотический)")

    ax.set_xlabel("Число Рейнольдса Re")
    ax.set_ylabel("Δζ")
    ax.legend()
    return _save(fig, "03_dz_Re.png")


def plot_k_ut(df_cal, df_val, geom_w, geom_a, dz_func_A):
    """График 4: k_ут(u₁)."""
    from .model import calc_k_ut, calc_r_explicit

    fig, ax = plt.subplots(figsize=(8, 5))

    # Эксперимент
    ax.plot(df_cal["u1"], df_cal["k_ut"], "bs", ms=7,
            label="Эксперимент (вода)")
    k_ut_val = df_val["r"] / (1 - df_val["r"])
    ax.plot(df_val["u1"], k_ut_val, "ro", ms=7,
            label="Эксперимент (воздух)")

    # Модель
    u_fine = np.linspace(3, 18, 200)
    r_w = calc_r_explicit(u_fine, geom_w, dz_func_A)
    r_a = calc_r_explicit(u_fine, geom_a, dz_func_A)

    ax.plot(u_fine, calc_k_ut(r_w), "b-", lw=1.5, alpha=0.5, label="Модель A (вода)")
    ax.plot(u_fine, calc_k_ut(r_a), "r-", lw=2, label="Модель A (воздух)")

    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_ylabel("Коэффициент утечек k_ут")
    ax.legend(fontsize=10)
    return _save(fig, "04_k_ut.png")


def plot_parity(df_cal, df_val, geom_w, geom_a, dz_func_A):
    """График 5: Parity plot r_расч vs r_эксп."""
    from .model import calc_r_explicit

    fig, ax = plt.subplots(figsize=(6, 6))

    r_cal_pred = calc_r_explicit(df_cal["u1"].values, geom_w, dz_func_A)
    r_val_pred = calc_r_explicit(df_val["u1"].values, geom_a, dz_func_A)

    ax.plot(df_cal["r"], r_cal_pred, "bs", ms=7, label="Калибровка (вода)")
    ax.plot(df_val["r"], r_val_pred, "ro", ms=7, label="Валидация (воздух)")

    lims = [0, max(df_cal["r"].max(), df_val["r"].max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Идеальное совпадение")

    ax.set_xlabel("r эксперимент")
    ax.set_ylabel("r расчёт")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    return _save(fig, "05_parity.png")
