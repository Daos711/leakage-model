"""Геометрические параметры и константы модели."""

import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# Пути вывода — разделение по этапам
# ---------------------------------------------------------------------------
_OUTPUT_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

OUTPUT_STAGE1 = os.path.join(_OUTPUT_BASE, "stage1_delta_zeta")
OUTPUT_STAGE1_PLOTS = os.path.join(OUTPUT_STAGE1, "plots")

OUTPUT_STAGE1_1 = os.path.join(_OUTPUT_BASE, "stage1_1_impulse")
OUTPUT_STAGE1_1_PLOTS = os.path.join(OUTPUT_STAGE1_1, "plots")

OUTPUT_STAGE2 = os.path.join(_OUTPUT_BASE, "stage2_idelchik")
OUTPUT_STAGE2_PLOTS = os.path.join(OUTPUT_STAGE2, "plots")

OUTPUT_STAGE3 = os.path.join(_OUTPUT_BASE, "stage3_physics")
OUTPUT_STAGE3_PLOTS = os.path.join(OUTPUT_STAGE3, "plots")

OUTPUT_STAGE4 = os.path.join(_OUTPUT_BASE, "stage4_plates")
OUTPUT_STAGE4_PLOTS = os.path.join(OUTPUT_STAGE4, "plots")

OUTPUT_STAGE4_SURROGATES = os.path.join(OUTPUT_STAGE4, "surrogates")
OUTPUT_STAGE4_SURR_PLOTS = os.path.join(OUTPUT_STAGE4_SURROGATES, "plots")

OUTPUT_STAGE4_JOINT = os.path.join(_OUTPUT_BASE, "stage4_plates_joint")
OUTPUT_STAGE4_JOINT_PLOTS = os.path.join(OUTPUT_STAGE4_JOINT, "plots")
OUTPUT_STAGE4_JOINT_SURROGATES = os.path.join(OUTPUT_STAGE4_JOINT, "surrogates")
OUTPUT_STAGE4_JOINT_SURR_PLOTS = os.path.join(OUTPUT_STAGE4_JOINT_SURROGATES, "plots")

# Водяная модель (калибровка) — натурный эквивалент
GEOM_WATER = {
    "D": 8.0,        # м, диаметр ствола
    "A_s": 50.0,     # м², площадь ствола
    "b_ok": 3.0,     # м, ширина окна
    "h_ok": 4.0,     # м, высота окна
    "A_ok": 12.0,    # м², площадь окна
    "D_h": 3.43,     # м, гидравл. диаметр окна
    "nu": 1.5e-5,    # м²/с, кинематическая вязкость воздуха (Re натуры)
    "L_up": 111.5,   # м, длина верхнего участка (от КО до устья)
    "beta": math.radians(45),  # рад, угол канала к вертикали
}

# Воздушная модель (валидация) — натурный эквивалент
GEOM_AIR = {
    "D": 8.0,        # м, диаметр ствола
    "A_s": 50.3,     # м², площадь ствола
    "b_ok": 4.0,     # м, ширина окна
    "h_ok": 5.0,     # м, высота окна
    "A_ok": 20.0,    # м², площадь окна
    "D_h": 4.44,     # м, гидравл. диаметр окна
    "nu": 1.5e-5,    # м²/с, кинематическая вязкость воздуха
    "L_up": 111.5,   # м, длина верхнего участка (от КО до устья)
    "beta": math.radians(45),  # рад, угол канала к вертикали
}

# Свойства среды (воздух натуры)
RHO = 1.2          # кг/м³, плотность воздуха
P_ATM = 101325     # Па, атмосферное давление

# Параметры Newton-Raphson
NR_TOL = 1e-12
NR_MAX_ITER = 100
NR_MIN_DERIV = 1e-12

# ---------------------------------------------------------------------------
# Этап 3 — полуэмпирическая физическая модель
# ---------------------------------------------------------------------------

BETA_DEG = 45.0
BETA_RAD = np.radians(BETA_DEG)
