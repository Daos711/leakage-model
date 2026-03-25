"""Диагностика: расчёт Δζ_exp для обоих наборов и диагностический график."""

import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import calc_delta_zeta, calc_Re
from ..core.plot_style import setup_matplotlib, apply_comma_ticks

setup_matplotlib()

logger = logging.getLogger(__name__)

from ..core.config import OUTPUT_STAGE1_PLOTS

OUTPUT_DIR = OUTPUT_STAGE1_PLOTS


def compute_dz_exp(df: pd.DataFrame, geom: dict) -> tuple[np.ndarray, np.ndarray]:
    """Вычислить экспериментальное Δζ и Re для набора данных.

    Δζ_exp = (1 - 2r) · (A_ок / A_с)²
    Re = u₁ · D_h / ν
    """
    r = df["r"].values
    u1 = df["u1"].values
    dz_exp = calc_delta_zeta(r, geom["A_ok"], geom["A_s"])
    Re = calc_Re(u1, geom["D_h"], geom["nu"])
    return Re, dz_exp


def plot_dz_diagnostic(
    Re_water: np.ndarray,
    dz_water: np.ndarray,
    Re_air: np.ndarray,
    dz_air: np.ndarray,
) -> str:
    """График 6: Δζ_exp(Re) — вода и воздух на одном поле."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(Re_water, dz_water, "bs", ms=8, label="Вода (A_ок=12 м²)")
    ax.plot(Re_air, dz_air, "ro", ms=8, label="Воздух (A_ок=20 м²)")

    ax.set_xlabel("Число Рейнольдса Re")
    ax.set_ylabel("Δζ_exp")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "06_dz_diagnostic.png")
    apply_comma_ticks(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
