"""Загрузка экспериментальных данных по вставкам с направляющими пластинами (этап 4).

Источник: Excel-файл «Расчет_параметров_модели_и_результаты_моделирования___ВОДА.xlsx»,
лист «Расчеты и журнал моделирования».
"""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Количество вставок и точек
N_INSERTS = 32
N_SPEEDS = 8
BLOCK_ROWS = 14  # строк на блок одной вставки


def load_plates_data(xlsx_path: str) -> pd.DataFrame:
    """Прочитать Excel, вернуть DataFrame со всеми 256 точками.

    Колонки: insert_id, insert_name, u1, r_exp, Q_nat, k_p.

    Параметры
    ---------
    xlsx_path : str
        Путь к xlsx-файлу.

    Возвращает
    ----------
    pd.DataFrame
        256 строк (32 вставки × 8 скоростей).
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError("openpyxl необходим: pip install openpyxl")

    df_raw = pd.read_excel(
        xlsx_path,
        sheet_name="Расчеты и журнал моделирования",
        header=None,
        engine="openpyxl",
    )

    records = []
    insert_id = 0

    for idx in range(len(df_raw)):
        cell_c = df_raw.iloc[idx, 2]  # столбец C (0-indexed = 2)
        if cell_c is None:
            continue
        cell_str = str(cell_c).strip().upper()
        if "ВСТАВКА" not in cell_str:
            continue

        insert_id += 1
        insert_name = str(df_raw.iloc[idx, 2]).strip()

        # Данные начинаются через 3 строки после заголовка вставки
        data_start = idx + 3
        for j in range(N_SPEEDS):
            row = data_start + j
            if row >= len(df_raw):
                logger.warning(
                    "Вставка %d (%s): недостаточно строк данных", insert_id, insert_name
                )
                break

            Q_nat = df_raw.iloc[row, 3]   # столбец D
            u1 = df_raw.iloc[row, 4]      # столбец E
            k_p = df_raw.iloc[row, 23]    # столбец X (0-indexed = 23)

            # Преобразование к числам
            try:
                Q_nat = float(Q_nat)
                u1 = float(u1)
                k_p = float(k_p)
            except (TypeError, ValueError):
                logger.warning(
                    "Вставка %d, строка %d: не удалось преобразовать данные "
                    "(Q=%s, u1=%s, k_p=%s)",
                    insert_id, row, Q_nat, u1, k_p,
                )
                continue

            r_exp = k_p / (1.0 + k_p)

            records.append({
                "insert_id": insert_id,
                "insert_name": insert_name,
                "Q_nat": Q_nat,
                "u1": u1,
                "k_p": k_p,
                "r_exp": r_exp,
            })

        if insert_id >= N_INSERTS:
            break

    df = pd.DataFrame(records)
    logger.info("Загружено %d точек из %d вставок", len(df), df["insert_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# Парсинг геометрии вставки из названия
# ---------------------------------------------------------------------------

# Шаблоны названий вставок (примеры):
# ВСТАВКА №1 - 1.0=без пластин
# ВСТАВКА №2 - 1.1=1 кольцо,1000 мм
# ВСТАВКА №3 - 1.3=3х45°,1000 мм
# ВСТАВКА №5 - 1.5=5х45°,1000 мм
# ВСТАВКА №12 - 2.3=3х45°,500 мм
# ВСТАВКА №23 - 3.1=3х25°,1000 мм
# ВСТАВКА №29 - 3.7=3х55°,1000 мм
# ВСТАВКА №30 - 4.1=3х45°,250 мм
# ВСТАВКА №32 - 4.3=3х45°,750 мм

_RE_INSERT_NUM = re.compile(r"ВСТАВКА\s*[№#]?\s*(\d+)", re.IGNORECASE)
_RE_SERIES = re.compile(r"(\d+)\.(\d+)\s*=")
_RE_PLATES = re.compile(r"(\d+)\s*[хxXХ×]\s*(\d+)\s*°")
_RE_WIDTH = re.compile(r"(\d+)\s*мм")
_RE_RING = re.compile(r"кольц", re.IGNORECASE)
_RE_NONE = re.compile(r"без\s*пластин", re.IGNORECASE)


def parse_insert_geometry(insert_name: str) -> dict:
    """Разобрать название вставки в структурированные признаки.

    Возвращает dict с полями:
      insert_num   — номер вставки
      series_id    — номер серии (1, 2, 3, 4)
      n_plates     — количество пластин (0, 1, 3, 5, 10)
      angle_deg    — угол наклона пластины (25, 30, 35, 40, 45, 50, 55, 60, 90)
      width_mm     — ширина пластины (0, 250, 500, 750, 1000)
      topology     — тип: 'none' / 'ring' / 'inclined' / 'unknown'
    """
    result = {
        "insert_num": 0,
        "series_id": 0,
        "n_plates": 0,
        "angle_deg": 0,
        "width_mm": 0,
        "topology": "unknown",
    }

    m = _RE_INSERT_NUM.search(insert_name)
    if m:
        result["insert_num"] = int(m.group(1))

    m = _RE_SERIES.search(insert_name)
    if m:
        result["series_id"] = int(m.group(1))

    if _RE_NONE.search(insert_name):
        result["topology"] = "none"
        result["n_plates"] = 0
        result["angle_deg"] = 0
        result["width_mm"] = 0
        return result

    if _RE_RING.search(insert_name):
        result["topology"] = "ring"
        result["n_plates"] = 1
        result["angle_deg"] = 90
        m = _RE_WIDTH.search(insert_name)
        if m:
            result["width_mm"] = int(m.group(1))
        return result

    m = _RE_PLATES.search(insert_name)
    if m:
        result["topology"] = "inclined"
        result["n_plates"] = int(m.group(1))
        result["angle_deg"] = int(m.group(2))
        mw = _RE_WIDTH.search(insert_name)
        if mw:
            result["width_mm"] = int(mw.group(1))
        return result

    logger.warning("Не удалось разобрать геометрию вставки: %s", insert_name)
    return result


def load_plates_with_geometry(xlsx_path: str) -> pd.DataFrame:
    """load_plates_data + parse_insert_geometry для каждой вставки.

    Возвращает DataFrame с дополнительными колонками геометрии.
    """
    df = load_plates_data(xlsx_path)

    # Добавить геометрические признаки
    geom_cols = ["insert_num", "series_id", "n_plates", "angle_deg", "width_mm", "topology"]
    for col in geom_cols:
        df[col] = np.nan if col != "topology" else ""

    insert_names = df.drop_duplicates("insert_id")[["insert_id", "insert_name"]]
    for _, row in insert_names.iterrows():
        geom = parse_insert_geometry(row["insert_name"])
        mask = df["insert_id"] == row["insert_id"]
        for col in geom_cols:
            df.loc[mask, col] = geom[col]

    for col in ["insert_num", "series_id", "n_plates", "angle_deg", "width_mm"]:
        df[col] = df[col].astype(int)

    return df
