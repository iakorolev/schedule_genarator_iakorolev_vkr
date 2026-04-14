"""Поиск точных совпадений и вспомогательные операции с результатами сопоставления."""

from __future__ import annotations

import pandas as pd

from .normalize import _txt, best_disc_match, teacher_lastname, extract_group_parts



def dedup_un_for_merge(un_expanded: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дубли из таблицы нагрузки перед объединением с расписанием."""
    u = un_expanded.copy()
    if len(u) == 0:
        return u
    if "hours_kind" in u.columns:
        score_col = "hours_kind"
    elif "capacity_units" in u.columns:
        score_col = "capacity_units"
    else:
        score_col = None
    kind_col = "kind_norm" if "kind_norm" in u.columns else "Вид_работы_норм"
    if score_col:
        u = u.sort_values(["Учебная группа", "disc_key", kind_col, score_col], ascending=[True, True, True, False])
    cols = [c for c in ["Учебная группа", "disc_key", kind_col, "Преподаватель"] if c in u.columns]
    if not cols:
        return u
    return u.drop_duplicates(subset=["Учебная группа", "disc_key", kind_col, "Преподаватель"], keep="first")



def _is_stream_like_text(group: str, source_text: str = "", source_block: str = "") -> bool:
    """Грубая эвристика для потоковых/объединённых слотов."""
    raw = " ".join([_txt(group), _txt(source_text), _txt(source_block)]).lower()
    if not raw:
        return False
    return any(sep in raw for sep in [",", "/", " пот", "поток"])



def build_exact_candidates(schedule_atoms: pd.DataFrame, run_atoms: pd.DataFrame) -> pd.DataFrame:
    """Строит кандидатов по точным совпадениям группы, дисциплины и вида занятия.

    Дополнительно считает признаки для более строгого exact-lock: если по дисциплине и виду
    в РУН есть несколько преподавателей, такие слоты не должны автоматически фиксироваться
    слишком рано и должны идти в обычный кандидатный отбор.
    """
    if schedule_atoms is None or len(schedule_atoms) == 0 or run_atoms is None or len(run_atoms) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "score", "reason"])

    s = schedule_atoms.copy()
    u = run_atoms.copy()

    # признаки группы для расписания
    s_parts = s["Учебная группа"].apply(extract_group_parts).apply(pd.Series)
    for col in ["group_prefix", "course_year"]:
        if col in s_parts.columns:
            s[col] = s_parts[col]

    disc_kind_teachers = (
        u.groupby(["disc_key", "kind_norm"], as_index=False)
        .agg(disc_kind_teacher_count=("Преподаватель", "nunique"))
    )
    prefix_teachers = (
        u.groupby(["disc_key", "kind_norm", "group_prefix", "course_year"], as_index=False)
        .agg(prefix_teacher_count=("Преподаватель", "nunique"))
    )

    merged = s.merge(
        u,
        how="left",
        left_on=["Учебная группа", "disc_key", "Вид_занятия_норм"],
        right_on=["Учебная группа", "disc_key", "kind_norm"],
        suffixes=("_sched", "_un"),
    )
    merged = merged[merged["Преподаватель"].notna()].copy()
    if len(merged) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "score", "reason"])

    merged = merged.merge(disc_kind_teachers, on=["disc_key", "kind_norm"], how="left")
    left_prefix_col = "group_prefix_sched" if "group_prefix_sched" in merged.columns else "group_prefix"
    left_year_col = "course_year_sched" if "course_year_sched" in merged.columns else "course_year"
    merged = merged.merge(
        prefix_teachers,
        left_on=["disc_key", "kind_norm", left_prefix_col, left_year_col],
        right_on=["disc_key", "kind_norm", "group_prefix", "course_year"],
        how="left",
    )

    merged["disc_kind_teacher_count"] = merged["disc_kind_teacher_count"].fillna(0).astype(int)
    merged["prefix_teacher_count"] = merged["prefix_teacher_count"].fillna(0).astype(int)
    merged["hint_match"] = merged.apply(
        lambda r: teacher_lastname(r.get("teacher_hint")) == teacher_lastname(r.get("Преподаватель")), axis=1
    )
    merged["stream_like"] = merged.apply(
        lambda r: _is_stream_like_text(r.get("Учебная группа"), r.get("source_text"), r.get("source_block")), axis=1
    )
    merged["score"] = 100.0 + merged.get("capacity_units", 0).fillna(0)
    merged["reason"] = "exact group+disc+kind"
    cols = [
        "slot_id", "Преподаватель", "score", "reason", "disc_key", "kind_norm",
        "disc_kind_teacher_count", "prefix_teacher_count", "hint_match", "stream_like"
    ]
    if "Вид_занятия_норм" in merged.columns:
        cols.append("Вид_занятия_норм")
    return merged[cols].drop_duplicates()



def lock_teacher_hints(schedule_atoms: pd.DataFrame, run_atoms: pd.DataFrame) -> pd.DataFrame:
    """Фиксирует назначения по однозначным teacher hints из расписания."""
    if schedule_atoms is None or len(schedule_atoms) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "assign_type", "confidence", "reason"])
    s = schedule_atoms.copy()
    s = s[s["teacher_hint"].fillna("").astype(str).str.strip() != ""].copy()
    if len(s) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "assign_type", "confidence", "reason"])

    teacher_pool = []
    if run_atoms is not None and len(run_atoms) > 0:
        teacher_pool = sorted(run_atoms["Преподаватель"].dropna().astype(str).unique().tolist())

    rows = []
    for _, r in s.iterrows():
        hint = _txt(r.get("teacher_hint"))
        if not hint:
            continue

        raw_parts = [h.strip() for h in str(hint).split("/") if h.strip()]
        matched = []
        for part in raw_parts:
            hint_last = teacher_lastname(part)
            if not hint_last:
                continue
            hit = None
            for t in teacher_pool:
                if teacher_lastname(t) == hint_last:
                    hit = t
                    break
            if hit:
                matched.append(hit)

        matched = list(dict.fromkeys(matched))

        # Надёжно фиксируем только если подсказка однозначно ведёт к одному преподавателю.
        if len(matched) == 1:
            teacher = matched[0]
        else:
            continue

        rows.append(
            {
                "slot_id": r["slot_id"],
                "Преподаватель": teacher,
                "assign_type": "locked_teacher_hint",
                "confidence": 0.98,
                "reason": "teacher hint from schedule",
                "score": 999.0,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.drop_duplicates(subset=["slot_id"], keep="first")



def select_locked_exact_matches(schedule_atoms: pd.DataFrame, run_atoms: pd.DataFrame) -> pd.DataFrame:
    """Выбирает слоты, для которых найден ровно один действительно сильный точный кандидат.

    Если по дисциплине и виду занятия в РУН есть несколько преподавателей, такой слот не должен
    автоматически фиксироваться слишком рано: его нужно отдать в общий отбор, чтобы правило про
    предметных соседей смогло сработать.
    """
    cand = build_exact_candidates(schedule_atoms, run_atoms)
    if len(cand) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "assign_type", "confidence", "reason", "score"])

    grp = cand.groupby("slot_id").agg(n_teachers=("Преподаватель", "nunique"), best_score=("score", "max")).reset_index()
    unique_slots = grp[grp["n_teachers"] == 1]["slot_id"].tolist()
    locked = cand[cand["slot_id"].isin(unique_slots)].copy()
    if len(locked) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "assign_type", "confidence", "reason", "score"])

    kind_col = "Вид_занятия_норм" if "Вид_занятия_норм" in locked.columns else "kind_norm"
    strong_exact = (
        (locked["disc_kind_teacher_count"] <= 1)
        | locked["hint_match"].fillna(False).astype(bool)
        | (
            locked[kind_col].eq("лек")
            & locked["stream_like"].fillna(False).astype(bool)
            & (locked["prefix_teacher_count"] <= 1)
        )
    )
    locked = locked[strong_exact].copy()
    if len(locked) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "assign_type", "confidence", "reason", "score"])

    locked = locked.sort_values(["slot_id", "score"], ascending=[True, False]).drop_duplicates("slot_id", keep="first")
    locked["assign_type"] = "locked_exact"
    locked["confidence"] = 0.95
    locked.loc[locked["hint_match"].fillna(False).astype(bool), "reason"] = locked["reason"] + "; strong by teacher hint"
    locked.loc[(locked["disc_kind_teacher_count"] <= 1), "reason"] = locked["reason"] + "; unique teacher for disc+kind"
    stream_mask = locked[kind_col].eq("лек") & locked["stream_like"].fillna(False).astype(bool) & (locked["prefix_teacher_count"] <= 1)
    locked.loc[stream_mask, "reason"] = locked["reason"] + "; strong stream lecture"
    return locked[["slot_id", "Преподаватель", "assign_type", "confidence", "reason", "score"]]



def fuzzy_fill_unmatched(merged: pd.DataFrame, un_dedup: pd.DataFrame, threshold_ok: float = 0.55) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Пытается заполнить оставшиеся слоты по нестрогому совпадению дисциплины."""
    lowconf_rows = []

    bucket = {}
    kind_col = "kind_norm" if "kind_norm" in un_dedup.columns else "Вид_работы_норм"
    for _, r in un_dedup.iterrows():
        g = _txt(r.get("Учебная группа"))
        k = _txt(r.get(kind_col))
        d = _txt(r.get("disc_key"))
        if not g or not k or not d:
            continue
        bucket.setdefault((g, k), []).append(d)

    merged = merged.copy()
    if "Преподаватель" not in merged.columns:
        merged["Преподаватель"] = None
    m_unmatched = merged["Преподаватель"].isna()

    for idx in merged[m_unmatched].index:
        g = _txt(merged.at[idx, "Учебная группа"])
        k = _txt(merged.at[idx, "Вид_занятия_норм"])
        disc = _txt(merged.at[idx, "disc_key"])

        cands = bucket.get((g, k), [])
        if not cands:
            continue

        best, score = best_disc_match(disc, cands)
        if not best or score <= 0:
            continue

        if score >= threshold_ok:
            hit = un_dedup[
                (un_dedup["Учебная группа"] == g)
                & (un_dedup[kind_col] == k)
                & (un_dedup["disc_key"] == best)
            ]
            if len(hit) > 0:
                merged.at[idx, "Преподаватель"] = hit.iloc[0].get("Преподаватель")
        else:
            lowconf_rows.append(
                {
                    "День недели": merged.at[idx, "День недели"],
                    "Пара": merged.at[idx, "Пара"],
                    "Время": merged.at[idx, "Время"],
                    "Учебная группа": g,
                    "disc_key_sched": disc,
                    "disc_key_best": best,
                    "score": score,
                }
            )

    return merged, pd.DataFrame(lowconf_rows)



def merge_schedule_with_teachers(sched_norm: pd.DataFrame, un_expanded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Объединяет расписание и нагрузку в legacy-формате для сравнения."""
    un_dedup = dedup_un_for_merge(un_expanded)
    kind_col = "kind_norm" if "kind_norm" in un_dedup.columns else "Вид_работы_норм"

    merged = sched_norm.merge(
        un_dedup,
        how="left",
        left_on=["Учебная группа", "disc_key", "Вид_занятия_норм"],
        right_on=["Учебная группа", "disc_key", kind_col],
        suffixes=("_sched", "_un"),
    )

    if "Преподаватель" not in merged.columns:
        merged["Преподаватель"] = None

    merged, lowconf = fuzzy_fill_unmatched(merged, un_dedup)
    merged["Преподаватель"] = merged["Преподаватель"].apply(lambda x: x if _txt(x) else None)

    return merged, lowconf
