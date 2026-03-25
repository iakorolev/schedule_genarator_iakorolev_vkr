"""Разбор РУН/УН и построение атомарной модели учебной нагрузки преподавателей."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .normalize import _txt, extract_group_parts, normalize_disc, normalize_group, normalize_kind_un, teacher_lastname


def _norm_sheet_name(s: str) -> str:
    """Нормализует имя листа Excel-книги для поиска нужного листа."""
    return _txt(s).lower().replace("ё", "е").strip()



def pick_un_sheet(excel_path: Path) -> str:
    """Выбирает подходящий лист РУН/УН для чтения."""
    xls = pd.ExcelFile(excel_path)
    sheets = list(xls.sheet_names)
    norm_map = {_norm_sheet_name(x): x for x in sheets}

    target = _norm_sheet_name("ун сводная")
    if target in norm_map:
        return norm_map[target]

    for sh in sheets:
        n = _norm_sheet_name(sh)
        if "ун" in n and "свод" in n:
            return sh

    return sheets[0]



def expand_group_numbers(num_str: str) -> list[str]:
    """Разворачивает строку с несколькими группами в список отдельных кодов."""
    s = _txt(num_str).replace(" ", "")
    if not s:
        return []

    tokens = [t for t in s.split(",") if t]
    if not tokens:
        return []

    suffix = None
    for t in tokens:
        if "-" in t:
            suffix = t.split("-", 1)[1]
            break

    out = []
    for t in tokens:
        if "-" in t:
            out.append(t)
        else:
            out.append(f"{t}-{suffix}" if suffix else t)

    return list(dict.fromkeys(out))



def hours_to_units(hours: float) -> float:
    """Преобразует часы нагрузки в условные единицы распределения."""
    try:
        h = float(hours or 0)
    except Exception:
        h = 0.0
    return h / 36.0



def _read_un_base(run_xlsx: Path) -> pd.DataFrame:
    """Читает базовую таблицу РУН/УН без разворачивания по группам."""
    sheet = pick_un_sheet(run_xlsx)
    excel_data = pd.read_excel(run_xlsx, sheet_name=sheet, header=None)

    header_rows = excel_data.iloc[:5]
    combined_headers = header_rows.fillna("").astype(str).agg(" ".join)
    combined_headers = combined_headers.str.replace(r"\s+", " ", regex=True).str.strip()

    df = excel_data.iloc[5:].copy()
    df.columns = combined_headers

    def find_col(pattern: str) -> str:
        """Выполняет операцию `find col`."""
        for col in df.columns:
            if pattern in col:
                return col
        return None

    spec = {
        "Код_ООП": "Код Направление /специальность /образовательная программа",
        "ООП": "Образовательная программа",
        "Дисциплина": "Наименование дисциплины или вида учебной работы",
        "Семестр": "Семестр",
        "Вид_работы": "Вид учебной работы",
        "Код_группы": "Учебная группа",
        "Номер_группы": "Номер группы",
        "Количество_студентов": "Кол-во чел. в группе (потоке) Всего",
        "Кафедра": "Сведения о ППС Кафедра",
        "Должность": "должность",
        "Тип_занятости": "штатн.",
        "Преподаватель": "Фамилия И.О. преподавателя",
        "Лекции_часы": "Объём учебной работы ППС Лекции",
        "Практика_часы": "Практика / Семинары",
        "Лабораторные_часы": "Лаб. работы / Клинические занятия",
        "Всего_часов": "Всего часов",
    }

    rename_map = {}
    for new_name, pattern in spec.items():
        col = find_col(pattern)
        if col is not None:
            rename_map[col] = new_name

    df = df[list(rename_map.keys())].rename(columns=rename_map)

    numeric_cols = [c for c in ["Лекции_часы", "Практика_часы", "Лабораторные_часы", "Всего_часов", "Количество_студентов"] if c in df.columns]
    text_cols = [c for c in df.columns if c not in numeric_cols]

    for col in text_cols:
        df[col] = df[col].apply(_txt).replace({"nan": ""})

    for col in numeric_cols:
        df[col] = (
            df[col].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "Преподаватель" in df.columns:
        df = df[(df["Преподаватель"] != "") & (~df["Преподаватель"].str.match(r"^\d+(\.\d+)?$", na=False))]
    return df.reset_index(drop=True)



def read_un(run_xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает базовую и развёрнутую таблицы РУН/УН."""
    un_raw = _read_un_base(run_xlsx)
    run_atoms = read_un_atoms(run_xlsx)
    un_expanded = run_atoms.rename(columns={"kind_norm": "Вид_работы_норм"}).copy()
    return un_raw, un_expanded



def read_un_atoms(run_xlsx: Path) -> pd.DataFrame:
    """Преобразует РУН/УН в атомы нагрузки преподавателей."""
    un_raw = _read_un_base(run_xlsx)
    rows = []

    for _, row in un_raw.iterrows():
        disc = _txt(row.get("Дисциплина"))
        kind_norm = normalize_kind_un(row.get("Вид_работы"))
        if not disc or not kind_norm:
            continue

        kod = _txt(row.get("Код_группы"))
        nums = expand_group_numbers(row.get("Номер_группы"))
        if not kod or not nums:
            continue

        teacher = _txt(row.get("Преподаватель"))
        if not teacher:
            continue

        if kind_norm == "лек":
            kind_hours = float(row.get("Лекции_часы", 0) or 0)
        elif kind_norm == "сем":
            kind_hours = float(row.get("Практика_часы", 0) or 0)
        else:
            kind_hours = float(row.get("Лабораторные_часы", 0) or 0)

        disc_key = normalize_disc(disc)
        for gn in nums:
            group = normalize_group(f"{kod}-{gn}")
            gp = extract_group_parts(group)
            rows.append(
                {
                    "Учебная группа": group,
                    "group_norm": group,
                    "group_prefix": gp.get("group_prefix", ""),
                    "group_num": gp.get("group_num", ""),
                    "course_year": gp.get("course_year", ""),
                    "Дисциплина": disc,
                    "discipline_norm": disc_key,
                    "disc_key": disc_key,
                    "Вид_работы": _txt(row.get("Вид_работы")),
                    "kind_norm": kind_norm,
                    "Преподаватель": teacher,
                    "teacher_norm": teacher,
                    "teacher_lastname": teacher_lastname(teacher),
                    "Кафедра": _txt(row.get("Кафедра")),
                    "Должность": _txt(row.get("Должность")),
                    "Тип_занятости": _txt(row.get("Тип_занятости")),
                    "ООП": _txt(row.get("ООП")),
                    "Код_ООП": _txt(row.get("Код_ООП")),
                    "Семестр": _txt(row.get("Семестр")),
                    "hours_kind": kind_hours,
                    "hours_total": float(row.get("Всего_часов", 0) or 0),
                    "capacity_units": hours_to_units(kind_hours),
                    "capacity_units_total": hours_to_units(float(row.get("Всего_часов", 0) or 0)),
                    "source_row_kind": _txt(row.get("Вид_работы")),
                }
            )

    run_atoms = pd.DataFrame(rows)
    if len(run_atoms) == 0:
        return pd.DataFrame(
            columns=[
                "Учебная группа",
                "disc_key",
                "kind_norm",
                "Преподаватель",
                "hours_kind",
                "capacity_units",
            ]
        )
    return run_atoms



def build_teacher_capacity(run_atoms: pd.DataFrame) -> pd.DataFrame:
    """Строит сводную таблицу доступной нагрузки преподавателей."""
    if run_atoms is None or len(run_atoms) == 0:
        return pd.DataFrame(columns=["Преподаватель", "capacity_total_units", "remaining_total_units"])

    base = run_atoms.groupby("Преподаватель", as_index=False).agg(
        capacity_total_units=("capacity_units", "sum"),
        capacity_total_hours=("hours_kind", "sum"),
        disc_count=("disc_key", lambda s: len(set([x for x in s if _txt(x)]))),
        group_count=("Учебная группа", lambda s: len(set([x for x in s if _txt(x)]))),
    )

    for kind in ["лек", "сем", "лаб"]:
        sub = run_atoms[run_atoms["kind_norm"] == kind].groupby("Преподаватель", as_index=False).agg(
            units=("capacity_units", "sum"),
            hours=("hours_kind", "sum"),
        )
        base = base.merge(sub.rename(columns={"units": f"capacity_{kind}_units", "hours": f"capacity_{kind}_hours"}), on="Преподаватель", how="left")

    for c in base.columns:
        if c.startswith("capacity_"):
            base[c] = base[c].fillna(0.0)

    base["remaining_total_units"] = base["capacity_total_units"]
    return base



def build_teacher_skills(run_atoms: pd.DataFrame) -> pd.DataFrame:
    """Формирует сводку навыков преподавателя по дисциплинам и видам занятий."""
    if run_atoms is None or len(run_atoms) == 0:
        return pd.DataFrame(columns=["Преподаватель", "disc_key", "kind_norm", "skill_weight"])

    skills = run_atoms.groupby(["Преподаватель", "disc_key", "kind_norm"], as_index=False).agg(
        skill_hours=("hours_kind", "sum"),
        skill_groups=("Учебная группа", lambda s: len(set(s))),
    )
    skills["skill_weight"] = skills["skill_hours"].fillna(0.0) + skills["skill_groups"].fillna(0.0) * 4.0
    return skills



def build_teacher_group_links(run_atoms: pd.DataFrame) -> pd.DataFrame:
    """Строит связи преподавателя с группами, дисциплинами и видами занятий."""
    if run_atoms is None or len(run_atoms) == 0:
        return pd.DataFrame(columns=["Преподаватель", "Учебная группа", "disc_key", "kind_norm", "hours_kind"])
    return run_atoms.groupby(["Преподаватель", "Учебная группа", "disc_key", "kind_norm"], as_index=False).agg(
        hours_kind=("hours_kind", "sum"),
        capacity_units=("capacity_units", "sum"),
    )
