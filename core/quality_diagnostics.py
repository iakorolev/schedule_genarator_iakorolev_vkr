"""Встроенная автономная диагностика качества распределения преподавателей."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .load_model import build_teacher_state
from .normalize import _txt, normalize_week_type

LEVEL_LABELS = {3: "строгий", 2: "ограниченный", 1: "резервный", 0: "неопределён"}


def _safe_to_excel(df: pd.DataFrame, path: Path) -> None:
    if df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


def _top2_by_slot(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates is None or len(candidates) == 0:
        return pd.DataFrame(columns=["slot_id", "top_score", "second_score", "candidate_count"])
    rows = []
    for slot_id, grp in candidates.groupby("slot_id"):
        scores = grp["score"].astype(float).sort_values(ascending=False).tolist()
        rows.append({
            "slot_id": slot_id,
            "top_score": float(scores[0]) if scores else 0.0,
            "second_score": float(scores[1]) if len(scores) > 1 else 0.0,
            "candidate_count": int(len(scores)),
        })
    return pd.DataFrame(rows)


def _merge_assignment_candidate_info(final_assignments: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    assigned = final_assignments[final_assignments["Преподаватель"].notna()].copy()
    if len(assigned) == 0:
        return assigned

    if candidates is not None and len(candidates) > 0:
        chosen = candidates.rename(columns={"teacher": "Преподаватель"}).copy()
        chosen = chosen.sort_values(["slot_id", "Преподаватель", "score"], ascending=[True, True, False])
        chosen = chosen.drop_duplicates(["slot_id", "Преподаватель"], keep="first")
        cols = [
            "slot_id", "Преподаватель", "heuristic_score", "admissibility_level", "subject_fit",
            "day_fit", "load_fit", "continuity_fit", "hint_fit", "new_day_penalty", "weighted_score",
            "topsis_score", "math_score",
        ]
        cols = [c for c in cols if c in chosen.columns]
        assigned = assigned.merge(chosen[cols], on=["slot_id", "Преподаватель"], how="left")

    top2 = _top2_by_slot(candidates)
    assigned = assigned.merge(top2, on="slot_id", how="left")
    assigned["candidate_count"] = assigned["candidate_count"].fillna(0).astype(int)
    assigned["top_score"] = pd.to_numeric(assigned["top_score"], errors="coerce").fillna(pd.to_numeric(assigned.get("score", 0), errors="coerce") if hasattr(pd, "to_numeric") else 0).astype(float)
    assigned["second_score"] = assigned["second_score"].fillna(0.0).astype(float)
    assigned["score_gap"] = (assigned["top_score"] - assigned["second_score"]).round(3)
    assigned["admissibility_level"] = assigned["admissibility_level"].fillna(
        assigned["assign_type"].fillna("").map({
            "locked_exact": 3,
            "locked_teacher_hint": 3,
            "manual_force": 3,
            "auto_allocated": 2,
        }).fillna(0)
    ).astype(int)
    assigned["admissibility_label"] = assigned["admissibility_level"].map(LEVEL_LABELS).fillna("неопределён")
    for col in ["subject_fit", "day_fit", "load_fit", "continuity_fit", "hint_fit", "new_day_penalty", "weighted_score", "topsis_score", "math_score"]:
        if col not in assigned.columns:
            assigned[col] = 0.0
        assigned[col] = assigned[col].fillna(0.0).astype(float)
    assigned["week_type"] = assigned.get("week_type", "").apply(normalize_week_type)
    return assigned


def _risk_for_row(row: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    assign_type = _txt(row.get("assign_type"))
    level = int(row.get("admissibility_level", 0) or 0)
    math_score = float(row.get("math_score", 0.0) or 0.0)
    score_gap = float(row.get("score_gap", 0.0) or 0.0)
    new_day_penalty = float(row.get("new_day_penalty", 0.0) or 0.0)
    subject_fit = float(row.get("subject_fit", 0.0) or 0.0)
    candidate_count = int(row.get("candidate_count", 0) or 0)
    confidence = float(row.get("confidence", 0.0) or 0.0)

    if assign_type in {"locked_exact", "locked_teacher_hint", "manual_force"}:
        if assign_type == "locked_teacher_hint":
            return "низкий", "жёсткое назначение по teacher hint"
        if assign_type == "manual_force":
            return "низкий", "ручное жёсткое правило"
        return "низкий", "жёсткое точное назначение"

    if level <= 1:
        reasons.append("резервный уровень допустимости")
    elif level == 2:
        reasons.append("ограниченный уровень допустимости")
    if new_day_penalty >= 0.99:
        reasons.append("открывается новый день")
    if math_score < 0.45:
        reasons.append("низкая математическая оценка")
    elif math_score < 0.60:
        reasons.append("умеренная математическая оценка")
    if score_gap < 4.0 and candidate_count > 1:
        reasons.append("слабый отрыв от следующего кандидата")
    elif score_gap < 10.0 and candidate_count > 1:
        reasons.append("небольшой отрыв от следующего кандидата")
    if subject_fit < 0.55:
        reasons.append("слабое предметное соответствие")
    elif subject_fit < 0.70:
        reasons.append("неполное предметное соответствие")
    if confidence and confidence < 0.45:
        reasons.append("низкая внутренняя уверенность")

    if level <= 1 or math_score < 0.45 or (score_gap < 4.0 and candidate_count > 1):
        return "высокий", "; ".join(reasons) if reasons else "рискованное автоматическое назначение"
    if level == 2 or new_day_penalty >= 0.99 or math_score < 0.60 or subject_fit < 0.70:
        return "средний", "; ".join(reasons) if reasons else "назначение требует дополнительной проверки"
    return "низкий", "; ".join(reasons) if reasons else "устойчивое автоматическое назначение"


def _teacher_day_load(final_assignments: pd.DataFrame) -> pd.DataFrame:
    assigned = final_assignments[final_assignments["Преподаватель"].notna()].copy()
    if len(assigned) == 0:
        return pd.DataFrame(columns=["Преподаватель", "День недели", "slot_count", "disc_count", "week_types"])
    assigned["week_type"] = assigned.get("week_type", "").apply(normalize_week_type)
    rows = []
    for (teacher, day), grp in assigned.groupby(["Преподаватель", "День недели"], dropna=False):
        weeks = sorted(set(_txt(x) for x in grp.get("week_type", []) if _txt(x)))
        rows.append({
            "Преподаватель": teacher,
            "День недели": day,
            "slot_count": int(grp["slot_id"].nunique()),
            "disc_count": int(grp.get("disc_key", pd.Series(dtype=str)).astype(str).replace("", pd.NA).dropna().nunique()),
            "week_types": ", ".join(weeks),
        })
    return pd.DataFrame(rows).sort_values(["Преподаватель", "День недели"]).reset_index(drop=True)


def _teacher_quality_summary(assigned: pd.DataFrame, teacher_capacity: pd.DataFrame) -> pd.DataFrame:
    if len(assigned) == 0:
        return pd.DataFrame(columns=["Преподаватель", "assigned_slots", "day_count", "high_risk_slots"])
    base = (
        assigned.groupby("Преподаватель", as_index=False)
        .agg(
            assigned_slots=("slot_id", "nunique"),
            day_count=("День недели", lambda s: s.astype(str).replace("", pd.NA).dropna().nunique()),
            high_risk_slots=("risk_level", lambda s: int((s == "высокий").sum())),
            medium_risk_slots=("risk_level", lambda s: int((s == "средний").sum())),
            strict_slots=("admissibility_level", lambda s: int((s == 3).sum())),
            limited_slots=("admissibility_level", lambda s: int((s == 2).sum())),
            reserve_slots=("admissibility_level", lambda s: int((s <= 1).sum())),
            avg_math_score=("math_score", "mean"),
            avg_score_gap=("score_gap", "mean"),
            opened_new_day_slots=("new_day_penalty", lambda s: int((s >= 0.99).sum())),
        )
    )
    if teacher_capacity is not None and len(teacher_capacity) > 0:
        cols = [c for c in ["Преподаватель", "capacity_total_units", "remaining_total_units"] if c in teacher_capacity.columns]
        base = base.merge(teacher_capacity[cols], on="Преподаватель", how="left")
    return base.sort_values(["high_risk_slots", "reserve_slots", "assigned_slots"], ascending=[False, False, False]).reset_index(drop=True)




def _discipline_risk_clusters(assigned: pd.DataFrame) -> pd.DataFrame:
    """Сводит риск в кластеры по преподавателю и дисциплине."""
    if assigned is None or len(assigned) == 0:
        return pd.DataFrame(columns=[
            "Преподаватель", "disc_key", "slot_count", "group_count", "day_count",
            "high_risk_slots", "medium_risk_slots", "opened_new_day_slots", "limited_slots",
            "avg_math_score", "risk_cluster_score",
        ])
    work = assigned.copy()
    work["Учебная группа"] = work.get("Учебная группа", "").astype(str)
    out = work.groupby(["Преподаватель", "disc_key"], as_index=False).agg(
        slot_count=("slot_id", "nunique"),
        group_count=("Учебная группа", lambda s: s.replace("", pd.NA).dropna().nunique()),
        day_count=("День недели", lambda s: s.astype(str).replace("", pd.NA).dropna().nunique()),
        high_risk_slots=("risk_level", lambda s: int((s == "высокий").sum())),
        medium_risk_slots=("risk_level", lambda s: int((s == "средний").sum())),
        opened_new_day_slots=("new_day_penalty", lambda s: int((s >= 0.99).sum())),
        limited_slots=("admissibility_level", lambda s: int((s == 2).sum())),
        avg_math_score=("math_score", "mean"),
    )
    out["risk_cluster_score"] = (
        4.5 * out["high_risk_slots"]
        + 1.8 * out["medium_risk_slots"]
        + 1.5 * out["opened_new_day_slots"]
        + 0.8 * out["limited_slots"]
        + 0.6 * out["group_count"].clip(lower=0)
        + 0.4 * out["day_count"].clip(lower=0)
        + (1.0 - out["avg_math_score"].fillna(0.0).clip(lower=0.0, upper=1.0)) * 12.0
    ).round(3)
    return out.sort_values(["risk_cluster_score", "high_risk_slots", "slot_count"], ascending=[False, False, False]).reset_index(drop=True)

def build_quality_diagnostics(*, final_assignments: pd.DataFrame, candidates: pd.DataFrame, teacher_capacity: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Строит автономную диагностику качества без эталонного файла."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assigned = _merge_assignment_candidate_info(final_assignments, candidates)
    for col, default in {
        "admissibility_level": 0,
        "new_day_penalty": 0.0,
        "math_score": 0.0,
        "score_gap": 0.0,
        "assign_type": "",
        "risk_level": "",
        "risk_reason": "",
    }.items():
        if col not in assigned.columns:
            assigned[col] = default

    if len(assigned) > 0:
        risks = assigned.apply(_risk_for_row, axis=1, result_type="expand")
        assigned["risk_level"] = risks[0]
        assigned["risk_reason"] = risks[1]
    else:
        assigned["risk_level"] = pd.Series(dtype=str)
        assigned["risk_reason"] = pd.Series(dtype=str)

    high_risk = assigned[assigned["risk_level"] == "высокий"].copy()
    reserve = assigned[assigned["admissibility_level"] <= 1].copy()
    limited = assigned[assigned["admissibility_level"] == 2].copy()
    teacher_day = _teacher_day_load(final_assignments)
    teacher_quality = _teacher_quality_summary(assigned, teacher_capacity)
    discipline_clusters = _discipline_risk_clusters(assigned)

    final_state = build_teacher_state(teacher_capacity, final_assignments[final_assignments["Преподаватель"].notna()].copy())
    overloaded = sum(1 for _, v in final_state.get("remaining_total", {}).items() if float(v or 0.0) < -0.25)

    summary_rows = [
        {"metric": "total_slots", "value": int(len(final_assignments))},
        {"metric": "assigned_slots", "value": int(len(assigned))},
        {"metric": "unmatched_slots", "value": int(len(final_assignments) - len(assigned))},
        {"metric": "strict_assignments", "value": int((assigned["admissibility_level"] == 3).sum())},
        {"metric": "limited_assignments", "value": int((assigned["admissibility_level"] == 2).sum())},
        {"metric": "reserve_assignments", "value": int((assigned["admissibility_level"] <= 1).sum())},
        {"metric": "low_risk_assignments", "value": int((assigned["risk_level"] == "низкий").sum())},
        {"metric": "medium_risk_assignments", "value": int((assigned["risk_level"] == "средний").sum())},
        {"metric": "high_risk_assignments", "value": int((assigned["risk_level"] == "высокий").sum())},
        {"metric": "opened_new_day_assignments", "value": int((assigned["new_day_penalty"] >= 0.99).sum())},
        {"metric": "auto_allocated_assignments", "value": int((assigned["assign_type"] == "auto_allocated").sum())},
        {"metric": "locked_assignments", "value": int((assigned["assign_type"].isin(["locked_exact", "locked_teacher_hint", "manual_force"])).sum())},
        {"metric": "overloaded_teachers", "value": int(overloaded)},
        {"metric": "avg_math_score_auto", "value": round(float(assigned.loc[assigned["assign_type"] == "auto_allocated", "math_score"].mean() or 0.0), 4)},
        {"metric": "avg_score_gap_auto", "value": round(float(assigned.loc[assigned["assign_type"] == "auto_allocated", "score_gap"].mean() or 0.0), 4)},
    ]
    summary = pd.DataFrame(summary_rows)

    quality_summary_path = out_dir / "quality_summary.xlsx"
    teacher_day_path = out_dir / "teacher_day_load.xlsx"
    teacher_quality_path = out_dir / "assignment_quality_by_teacher.xlsx"
    high_risk_path = out_dir / "high_risk_assignments.xlsx"
    reserve_path = out_dir / "reserve_assignments.xlsx"
    limited_path = out_dir / "limited_assignments.xlsx"
    discipline_clusters_path = out_dir / "discipline_risk_clusters.xlsx"

    _safe_to_excel(summary, quality_summary_path)
    _safe_to_excel(teacher_day, teacher_day_path)
    _safe_to_excel(teacher_quality, teacher_quality_path)
    _safe_to_excel(high_risk, high_risk_path)
    _safe_to_excel(reserve, reserve_path)
    _safe_to_excel(limited, limited_path)
    _safe_to_excel(discipline_clusters, discipline_clusters_path)

    return {
        "summary": {
            "strict_assignments": int((assigned["admissibility_level"] == 3).sum()),
            "limited_assignments": int((assigned["admissibility_level"] == 2).sum()),
            "reserve_assignments": int((assigned["admissibility_level"] <= 1).sum()),
            "low_risk_assignments": int((assigned["risk_level"] == "низкий").sum()),
            "medium_risk_assignments": int((assigned["risk_level"] == "средний").sum()),
            "high_risk_assignments": int((assigned["risk_level"] == "высокий").sum()),
            "opened_new_day_assignments": int((assigned["new_day_penalty"] >= 0.99).sum()),
            "overloaded_teachers": int(overloaded),
        },
        "files": {
            "quality_summary": str(quality_summary_path),
            "teacher_day_load": str(teacher_day_path),
            "assignment_quality_by_teacher": str(teacher_quality_path),
            "high_risk_assignments": str(high_risk_path),
            "reserve_assignments": str(reserve_path),
            "limited_assignments": str(limited_path),
            "discipline_risk_clusters": str(discipline_clusters_path),
        },
    }
