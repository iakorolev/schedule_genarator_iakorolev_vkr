"""Экспорт преподавательского расписания и диагностических таблиц в Excel-формат."""

import pandas as pd

from .normalize import _txt, normalize_day, normalize_time



def make_cell_text(group: pd.DataFrame) -> str:
    """Собирает текст одной ячейки преподавательского расписания из нескольких записей."""
    rows = []
    has_week_types = group.get("week_type") is not None and group["week_type"].astype(str).str.strip().ne("").any()
    for _, r in group.iterrows():
        disc = _txt(r.get("Дисциплина_out"))
        kind = _txt(r.get("Вид_занятия_out"))
        room = _txt(r.get("Аудитория_out"))
        g = _txt(r.get("Учебная группа"))
        week_type = _txt(r.get("week_type"))

        if not disc and not g:
            continue

        s = f"{disc}"
        if kind:
            s += f"; {kind}"
        if room:
            s += f"; {room}"
        if g:
            s += f"; {g}"
        if has_week_types and week_type:
            s = f"[{week_type}] {s}"

        rows.append(s)

    rows = [x.strip() for x in rows if x.strip() and not x.strip().startswith(";")]
    rows = sorted(dict.fromkeys(rows))
    return "\n".join(rows)



def build_teacher_timetable(schedule_with_teachers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит pivot-таблицу преподавательского расписания и список конфликтов."""
    df = schedule_with_teachers.copy()
    df["День недели"] = df["День недели"].apply(normalize_day)
    df["Время"] = df["Время"].apply(normalize_time)
    if "week_type" not in df.columns:
        df["week_type"] = ""

    if "Дисциплина_sched" in df.columns:
        df["Дисциплина_out"] = df["Дисциплина_sched"].apply(_txt)
    else:
        df["Дисциплина_out"] = df.get("Дисциплина", "").apply(_txt) if "Дисциплина" in df.columns else ""

    df["Вид_занятия_out"] = df.get("Вид_занятия", "").apply(_txt) if "Вид_занятия" in df.columns else ""
    df["Аудитория_out"] = df.get("Аудитория", "").apply(_txt) if "Аудитория" in df.columns else ""

    ok = df[df["Преподаватель"].notna()].copy()

    conflicts = (
        ok.groupby(["День недели", "Пара", "Время", "week_type", "Преподаватель"])["Аудитория_out"]
        .nunique()
        .reset_index(name="rooms")
    )
    conflicts = conflicts[conflicts["rooms"] > 1].copy()

    grouped = ok.groupby(["День недели", "Пара", "Время", "week_type", "Преподаватель"], dropna=False).apply(
        lambda g: make_cell_text(g), include_groups=False
    )
    grouped = grouped.reset_index()
    value_col = grouped.columns[-1]
    if value_col != "cell":
        grouped = grouped.rename(columns={value_col: "cell"})

    day_order = {"ПН": 1, "ВТ": 2, "СР": 3, "ЧТ": 4, "ПТ": 5, "СБ": 6, "ВС": 7}
    grouped["day_order"] = grouped["День недели"].map(day_order).fillna(99)
    grouped = grouped.sort_values(["day_order", "Пара", "Время", "week_type"])

    pivot = grouped.pivot(
        index=["День недели", "Пара", "Время", "week_type"],
        columns="Преподаватель",
        values="cell",
    ).reset_index()

    return pivot, conflicts
