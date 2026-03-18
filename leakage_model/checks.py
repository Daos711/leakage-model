"""Проверки физической корректности."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_r_range(r: np.ndarray, label: str = "") -> bool:
    """Проверка 0 < r < 1."""
    ok = True
    if np.any(r <= 0):
        logger.warning("[%s] Обнаружены r ≤ 0: %s", label, r[r <= 0])
        ok = False
    if np.any(r >= 1):
        logger.warning("[%s] Обнаружены r ≥ 1: %s", label, r[r >= 1])
        ok = False
    return ok


def check_k_ut_nonneg(k_ut: np.ndarray, label: str = "") -> bool:
    """Проверка k_ут ≥ 0."""
    if np.any(k_ut < 0):
        logger.warning("[%s] Обнаружены k_ут < 0: %s", label, k_ut[k_ut < 0])
        return False
    return True


def check_dz_positive(dz: np.ndarray, label: str = "") -> bool:
    """Проверка Δζ > 0."""
    if np.any(dz <= 0):
        logger.warning("[%s] Обнаружены Δζ ≤ 0: %s", label, dz[dz <= 0])
        return False
    return True


def check_monotonic_decrease(r: np.ndarray, u1: np.ndarray, label: str = "") -> bool:
    """Проверка монотонного убывания r(u₁)."""
    order = np.argsort(u1)
    r_sorted = r[order]
    diffs = np.diff(r_sorted)
    if np.any(diffs > 0):
        logger.warning(
            "[%s] r(u₁) не монотонно убывает: приращения %s", label, diffs[diffs > 0]
        )
        return False
    return True


def run_all_checks(r: np.ndarray, u1: np.ndarray, dz: np.ndarray, label: str = ""):
    """Запуск всех проверок."""
    from .model import calc_k_ut

    k_ut = calc_k_ut(r)

    results = {
        "r_range": check_r_range(r, label),
        "k_ut_nonneg": check_k_ut_nonneg(k_ut, label),
        "dz_positive": check_dz_positive(dz, label),
        "monotonic": check_monotonic_decrease(r, u1, label),
    }

    all_ok = all(results.values())
    if all_ok:
        logger.info("[%s] Все проверки пройдены.", label)
    else:
        failed = [k for k, v in results.items() if not v]
        logger.warning("[%s] Не пройдены проверки: %s", label, failed)

    return results
