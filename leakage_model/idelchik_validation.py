"""Валидация модели Идельчика: сравнение с экспериментом, графики, экспорт."""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .idelchik import VARIANTS, zeta_branch, zeta_straight, COS_ALPHA
from .idelchik_model import IdelchikResult
from .validation import Metrics, compute_metrics

logger = logging.getLogger(__name__)

from .config import OUTPUT_STAGE2, OUTPUT_STAGE2_PLOTS

OUTPUT_DIR = OUTPUT_STAGE2
PLOTS_DIR = OUTPUT_STAGE2_PLOTS

plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def compute_variant_metrics(result, r_exp):
    """Вычислить метрики для результата варианта."""
    mask = result.converged
    if mask.sum() == 0:
        return None
    return compute_metrics(r_exp[mask], result.r_pred[mask])


# ---------------------------------------------------------------------------
# График 1: r(u₁) — предсказание vs эксперимент
# ---------------------------------------------------------------------------

def plot_r_prediction(
    results_water,
    results_air,
    df_cal,
    df_val,
):
    """r(u₁) — все варианты Идельчика vs эксперимент, оба набора."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    colors = {"A": "b", "B": "g", "C": "r"}
    styles = {"A": "-", "B": "--", "C": "-."}

    # Левый: водяная модель
    ax = axes[0]
    ax.plot(df_cal["u1"], df_cal["r"], "ko", ms=8, zorder=5,
            label="Эксперимент (вода)")
    for key, res in results_water.items():
        mask = res.converged
        if mask.any():
            ax.plot(res.u1[mask], res.r_pred[mask],
                    color=colors[key], linestyle=styles[key],
                    marker="s", ms=5, lw=2,
                    label=f"Идельчик {key}: {res.variant_name}")
    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_ylabel("Доля утечек r")
    ax.set_title("Водяная модель (A_ок=12 м², F_б/F_c≈4,17)")
    ax.legend(fontsize=9)

    # Правый: воздушная модель
    ax = axes[1]
    ax.plot(df_val["u1"], df_val["r"], "ko", ms=8, zorder=5,
            label="Эксперимент (воздух)")
    for key, res in results_air.items():
        mask = res.converged
        if mask.any():
            ax.plot(res.u1[mask], res.r_pred[mask],
                    color=colors[key], linestyle=styles[key],
                    marker="s", ms=5, lw=2,
                    label=f"Идельчик {key}: {res.variant_name}")
    ax.set_xlabel("Скорость u₁, м/с")
    ax.set_title("Воздушная модель (A_ок=20 м², F_б/F_c≈2,51)")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Предсказание модели Идельчика vs эксперимент",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "10_idelchik_r_prediction.png")


# ---------------------------------------------------------------------------
# График 2: ζ(r) — коэффициенты Идельчика
# ---------------------------------------------------------------------------

def plot_zeta_curves(geom_water, geom_air):
    """ζ_утеч(r) и ζ_шахта(r) — кривые коэффициентов для обоих наборов."""
    r_range = np.linspace(0.01, 0.99, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (geom, title_suffix) in enumerate([
        (geom_water, "Вода (A_ок=12)"),
        (geom_air, "Воздух (A_ок=20)"),
    ]):
        ax = axes[idx]
        A_ok, A_s = geom["A_ok"], geom["A_s"]

        # Вариант A (чистые формулы)
        zb_A = [zeta_branch(r, A_ok, A_s) for r in r_range]
        zs_A = [zeta_straight(r, A_ok, A_s) for r in r_range]

        ax.plot(r_range, zb_A, "b-", lw=2, label="ζ_утеч (форм. 7-1)")
        ax.plot(r_range, zs_A, "r-", lw=2, label="ζ_шахта (форм. 7-2)")
        ax.axhline(0, color="k", lw=0.5, ls=":")

        sigma = A_ok / A_s
        fb_fc = A_s / A_ok
        ax.set_xlabel("Доля утечек r")
        ax.set_ylabel("ζ")
        ax.set_title(f"{title_suffix}\nσ={sigma:.3f}, F_б/F_c={fb_fc:.2f}")
        ax.legend(fontsize=10)

    fig.suptitle(
        "Коэффициенты сопротивления Идельчика (вариант А)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "11_idelchik_zeta_curves.png")


# ---------------------------------------------------------------------------
# График 3: Parity plot
# ---------------------------------------------------------------------------

def plot_parity_idelchik(results_water, results_air, df_cal, df_val):
    """Parity plot: r_расч vs r_эксп (оба набора, лучший вариант)."""
    fig, ax = plt.subplots(figsize=(6, 6))

    r_cal_exp = df_cal["r"].values
    r_val_exp = df_val["r"].values

    markers_w = {"A": "bs", "B": "g^", "C": "rD"}
    markers_a = {"A": "bo", "B": "gv", "C": "r*"}

    for key in results_water:
        res_w = results_water[key]
        res_a = results_air[key]
        m_w = res_w.converged
        m_a = res_a.converged

        if m_w.any():
            ax.plot(r_cal_exp[m_w], res_w.r_pred[m_w], markers_w[key],
                    ms=7, label=f"Вода, вар. {key}")
        if m_a.any():
            ax.plot(r_val_exp[m_a], res_a.r_pred[m_a], markers_a[key],
                    ms=7, label=f"Воздух, вар. {key}")

    all_r = np.concatenate([r_cal_exp, r_val_exp])
    lims = [0, max(all_r.max(), 0.5) * 1.2]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Идеальное совпадение")

    ax.set_xlabel("r эксперимент")
    ax.set_ylabel("r Идельчик")
    ax.set_title("Parity plot: модель Идельчика")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    return _save(fig, "12_idelchik_parity.png")


# ---------------------------------------------------------------------------
# График 4: Сравнение всех моделей
# ---------------------------------------------------------------------------

def plot_all_models_comparison(
    results_water,
    results_air,
    df_cal,
    df_val,
    geom_water,
    geom_air,
    prev_model_funcs=None,
):
    """Сводный график: Идельчик vs предыдущие модели vs эксперимент."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Подготовка тонкой сетки
    u_fine_w = np.linspace(
        df_cal["u1"].min() * 0.9, df_cal["u1"].max() * 1.1, 200,
    )
    u_fine_a = np.linspace(
        df_val["u1"].min() * 0.9, df_val["u1"].max() * 1.1, 200,
    )

    for ax_idx, (ax, df, geom, results, u_fine, label_data) in enumerate([
        (axes[0], df_cal, geom_water, results_water, u_fine_w, "Вода"),
        (axes[1], df_val, geom_air, results_air, u_fine_a, "Воздух"),
    ]):
        ax.plot(df["u1"], df["r"], "ko", ms=8, zorder=5,
                label=f"Эксперимент ({label_data})")

        # Идельчик — лучший вариант (C, если сошёлся)
        for key in ["C", "A"]:
            res = results[key]
            mask = res.converged
            if mask.any():
                ax.plot(res.u1[mask], res.r_pred[mask],
                        "r-s", ms=5, lw=2,
                        label=f"Идельчик {key}")
                break

        # Предыдущие модели (если переданы)
        if prev_model_funcs:
            for pname, pfunc, pstyle in prev_model_funcs:
                try:
                    r_prev = pfunc(u_fine, geom)
                    ax.plot(u_fine, r_prev, pstyle, lw=1.5, label=pname)
                except Exception:
                    pass

        ax.set_xlabel("Скорость u₁, м/с")
        if ax_idx == 0:
            ax.set_ylabel("Доля утечек r")
        ax.set_title(f"{label_data}")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Сравнение моделей: Идельчик vs эмпирические",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "13_idelchik_all_models.png")


# ---------------------------------------------------------------------------
# График 5: Чувствительность
# ---------------------------------------------------------------------------

def plot_sensitivity(
    r_vs_L, L_values,
    r_vs_Kb, Kb_values,
    r_vs_Kpp, Kpp_values,
    geom_label,
):
    """Графики чувствительности r к L_верх, K_б, K''_п."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # r(L_верх)
    ax = axes[0]
    mask = ~np.isnan(r_vs_L)
    if mask.any():
        ax.plot(np.array(L_values)[mask], r_vs_L[mask], "b-o", lw=2)
    ax.set_xlabel("L_верх, м")
    ax.set_ylabel("r")
    ax.set_title("Чувствительность к L_верх")

    # r(K_б)
    ax = axes[1]
    mask = ~np.isnan(r_vs_Kb)
    if mask.any():
        ax.plot(np.array(Kb_values)[mask], r_vs_Kb[mask], "g-o", lw=2)
    ax.set_xlabel("K_б")
    ax.set_ylabel("r")
    ax.set_title("Чувствительность к K_б")

    # r(K''_п)
    ax = axes[2]
    mask = ~np.isnan(r_vs_Kpp)
    if mask.any():
        ax.plot(np.array(Kpp_values)[mask], r_vs_Kpp[mask], "r-o", lw=2)
    ax.set_xlabel("K''_п")
    ax.set_ylabel("r")
    ax.set_title("Чувствительность к K''_п")

    fig.suptitle(
        f"Чувствительность модели Идельчика ({geom_label})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, f"14_idelchik_sensitivity_{geom_label}.png")


# ---------------------------------------------------------------------------
# Экспорт
# ---------------------------------------------------------------------------

def export_prediction_csv(
    results_water, results_air, df_cal, df_val,
):
    """Экспорт idelchik_prediction.csv — r_exp vs r_idelchik."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Водяная модель
    rows_w = {
        "dataset": "water",
        "u1": df_cal["u1"].values,
        "r_exp": df_cal["r"].values,
    }
    for key, res in results_water.items():
        rows_w[f"r_idelchik_{key}"] = res.r_pred
        rows_w[f"converged_{key}"] = res.converged

    # Воздушная модель
    rows_a = {
        "dataset": "air",
        "u1": df_val["u1"].values,
        "r_exp": df_val["r"].values,
    }
    for key, res in results_air.items():
        rows_a[f"r_idelchik_{key}"] = res.r_pred
        rows_a[f"converged_{key}"] = res.converged

    df_w = pd.DataFrame(rows_w)
    df_a = pd.DataFrame(rows_a)
    df_out = pd.concat([df_w, df_a], ignore_index=True)

    path = os.path.join(OUTPUT_DIR, "idelchik_prediction.csv")
    df_out.to_csv(path, index=False)
    logger.info("Сохранено: %s", path)
    return path


def export_coefficients_csv(geom_water, geom_air):
    """Экспорт idelchik_coefficients.csv — ζ(r) для диапазона r."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    r_range = np.linspace(0.01, 0.99, 100)
    rows = []
    for geom, label in [(geom_water, "water"), (geom_air, "air")]:
        A_ok, A_s = geom["A_ok"], geom["A_s"]
        for r in r_range:
            zb = zeta_branch(r, A_ok, A_s)
            zs = zeta_straight(r, A_ok, A_s)
            rows.append({
                "dataset": label,
                "r": r,
                "zeta_branch": zb,
                "zeta_straight": zs,
                "zeta_diff": zb - zs,
            })

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "idelchik_coefficients.csv")
    df.to_csv(path, index=False)
    logger.info("Сохранено: %s", path)
    return path


def export_parameters_json(
    results_water, results_air,
    metrics_water, metrics_air,
    geom_water, geom_air,
):
    """Экспорт idelchik_parameters.json."""
    import json

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    params = {
        "geometry": {
            "water": {
                "A_ok": geom_water["A_ok"],
                "A_s": geom_water["A_s"],
                "Fb_Fc": geom_water["A_s"] / geom_water["A_ok"],
                "sigma": geom_water["A_ok"] / geom_water["A_s"],
            },
            "air": {
                "A_ok": geom_air["A_ok"],
                "A_s": geom_air["A_s"],
                "Fb_Fc": geom_air["A_s"] / geom_air["A_ok"],
                "sigma": geom_air["A_ok"] / geom_air["A_s"],
            },
        },
        "alpha_deg": 45.0,
        "cos_alpha": float(COS_ALPHA),
        "variants": {},
    }

    for key in results_water:
        res_w = results_water[key]
        res_a = results_air[key]
        m_w = metrics_water.get(key)
        m_a = metrics_air.get(key)

        entry = {
            "name": res_w.variant_name,
            "water": {
                "r_pred": res_w.r_pred.tolist(),
                "converged": res_w.converged.tolist(),
                "n_converged": int(res_w.converged.sum()),
            },
            "air": {
                "r_pred": res_a.r_pred.tolist(),
                "converged": res_a.converged.tolist(),
                "n_converged": int(res_a.converged.sum()),
            },
        }
        if m_w:
            entry["water"]["metrics"] = {
                "RMSE": m_w.RMSE, "MAE": m_w.MAE,
                "R2": m_w.R2, "max_abs_error": m_w.max_abs_error,
            }
        if m_a:
            entry["air"]["metrics"] = {
                "RMSE": m_a.RMSE, "MAE": m_a.MAE,
                "R2": m_a.R2, "max_abs_error": m_a.max_abs_error,
            }
        if res_w.notes:
            entry["water"]["notes"] = res_w.notes
        if res_a.notes:
            entry["air"]["notes"] = res_a.notes

        params["variants"][key] = entry

    path = os.path.join(OUTPUT_DIR, "idelchik_parameters.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    logger.info("Сохранено: %s", path)
    return path


def export_all_models_csv(
    results_water, results_air,
    metrics_water, metrics_air,
    df_cal, df_val,
):
    """Экспорт all_models_comparison.csv — сводная таблица по всем моделям."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    for key in results_water:
        m_w = metrics_water.get(key)
        m_a = metrics_air.get(key)
        res_w = results_water[key]

        rows.append({
            "Модель": f"Идельчик {key}: {res_w.variant_name}",
            "Тип": "Справочная (предсказательная)",
            "Калибровка": "Нет",
            "Вода RMSE": m_w.RMSE if m_w else None,
            "Вода R²": m_w.R2 if m_w else None,
            "Воздух RMSE": m_a.RMSE if m_a else None,
            "Воздух R²": m_a.R2 if m_a else None,
            "Сходимость": f"{res_w.converged.sum()}/{len(res_w.converged)}",
        })

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "all_models_comparison.csv")
    df.to_csv(path, index=False)
    logger.info("Сохранено: %s", path)
    return path
