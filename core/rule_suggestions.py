"""Генерация рекомендаций по пользовательским правилам для спорных и неназначенных слотов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .normalize import _txt, normalize_week_type
from .quality_diagnostics import _merge_assignment_candidate_info, _risk_for_row, _discipline_risk_clusters


def _safe_to_excel(sheets: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        wrote = False
        for name, df in sheets.items():
            if df is None:
                continue
            out = df.copy()
            if len(out) == 0:
                out = pd.DataFrame({"info": ["нет данных"]})
            out.to_excel(writer, sheet_name=name[:31], index=False)
            wrote = True
        if not wrote:
            pd.DataFrame({"info": ["нет данных"]}).to_excel(writer, sheet_name="summary", index=False)


def _json_rule(mode: str, when: dict[str, Any], teacher: str | None = None) -> str:
    rule = {
        "mode": mode,
        "when": {k: v for k, v in when.items() if _txt(v)},
        "assign": {"teacher": teacher} if _txt(teacher) else {},
    }
    return json.dumps(rule, ensure_ascii=False)


def _slot_when(slot: pd.Series, *, with_time_scope: bool = False, use_prefix: bool = False) -> dict[str, str]:
    when = {
        "disc_key": _txt(slot.get("disc_key")),
        "kind": _txt(slot.get("Вид_занятия_норм")),
    }
    if use_prefix:
        when["group_prefix"] = _txt(slot.get("group_prefix"))
    else:
        when["group"] = _txt(slot.get("Учебная группа"))
    if with_time_scope:
        when["day"] = _txt(slot.get("День недели"))
        when["week_type"] = normalize_week_type(slot.get("week_type"))
    return {k: v for k, v in when.items() if _txt(v)}


def _top_by_slot(candidates: pd.DataFrame, slot_id: Any, n: int = 3) -> list[dict[str, Any]]:
    if candidates is None or len(candidates) == 0:
        return []
    grp = candidates[candidates["slot_id"] == slot_id].sort_values(["score", "math_score", "heuristic_score"], ascending=False).head(n)
    rows: list[dict[str, Any]] = []
    for rank, (_, r) in enumerate(grp.iterrows(), start=1):
        rows.append({
            "rank": rank,
            "teacher": _txt(r.get("teacher")),
            "score": float(r.get("score", 0.0) or 0.0),
            "math_score": float(r.get("math_score", 0.0) or 0.0),
            "admissibility_level": int(r.get("admissibility_level", 0) or 0),
            "reason": _txt(r.get("reason")),
        })
    return rows


def _flatten_top(top: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in top:
        prefix = f"top{item['rank']}"
        out[f"{prefix}_teacher"] = item["teacher"]
        out[f"{prefix}_score"] = item["score"]
        out[f"{prefix}_math_score"] = item["math_score"]
        out[f"{prefix}_level"] = item["admissibility_level"]
    return out


def _suggest_for_unmatched(slot: pd.Series, top: list[dict[str, Any]]) -> dict[str, Any]:
    base = {
        "source": "неназначенный слот",
        "slot_id": slot.get("slot_id"),
        "День недели": _txt(slot.get("День недели")),
        "Пара": slot.get("Пара"),
        "Учебная группа": _txt(slot.get("Учебная группа")),
        "disc_key": _txt(slot.get("disc_key")),
        "Вид_занятия_норм": _txt(slot.get("Вид_занятия_норм")),
        "teacher_hint": _txt(slot.get("teacher_hint")),
    }
    base.update(_flatten_top(top))
    if not top:
        base.update({
            "priority": "high",
            "recommendation": "manual_review",
            "mode": "",
            "teacher": "",
            "reason": "для слота не найдено ни одного кандидата; нужен ручной анализ данных или новое правило",
            "rule_json": "",
        })
        return base

    best = top[0]
    gap = best["score"] - (top[1]["score"] if len(top) > 1 else 0.0)
    hint_last = _txt(slot.get("teacher_hint"))
    mode = "prefer_teacher"
    when = _slot_when(slot)
    priority = "medium"
    reason = ""
    if hint_last and best["teacher"] and hint_last.lower() in best["teacher"].lower() and best["admissibility_level"] >= 2 and gap >= 8.0:
        mode = "force_teacher"
        when = _slot_when(slot, with_time_scope=True)
        priority = "high"
        reason = "teacher hint согласуется с лучшим кандидатом, отрыв высокий"
    elif best["admissibility_level"] >= 3 and gap >= 10.0:
        mode = "prefer_teacher"
        priority = "high"
        reason = "есть сильный строгий кандидат для пустого слота"
    elif best["admissibility_level"] >= 2:
        mode = "prefer_teacher"
        priority = "medium"
        reason = "есть ограниченно допустимый кандидат, который выглядит лучшим для слота"
    elif _txt(slot.get("group_prefix")):
        mode = "prefer_teacher_scope"
        when = _slot_when(slot, use_prefix=True)
        priority = "low"
        reason = "точного кандидата нет; возможна мягкая настройка по префиксу группы"
    else:
        priority = "low"
        reason = "лучший кандидат слабый, рекомендация требует ручной проверки"
    base.update({
        "priority": priority,
        "recommendation": "rule_candidate",
        "mode": mode,
        "teacher": best["teacher"],
        "reason": reason,
        "rule_json": _json_rule(mode, when, best["teacher"]),
    })
    return base


def _suggest_for_high_risk(row: pd.Series, top: list[dict[str, Any]]) -> dict[str, Any]:
    base = {
        "source": "высокий риск",
        "slot_id": row.get("slot_id"),
        "День недели": _txt(row.get("День недели")),
        "Пара": row.get("Пара"),
        "Учебная группа": _txt(row.get("Учебная группа")),
        "disc_key": _txt(row.get("disc_key")),
        "Вид_занятия_норм": _txt(row.get("Вид_занятия_норм")),
        "current_teacher": _txt(row.get("Преподаватель")),
        "risk_reason": _txt(row.get("risk_reason")),
    }
    base.update(_flatten_top(top))
    current = _txt(row.get("Преподаватель"))
    best = top[0] if top else None
    best_other = None
    for item in top:
        if _txt(item.get("teacher")) != current:
            best_other = item
            break
    if best_other and best_other["admissibility_level"] >= int(row.get("admissibility_level", 0) or 0) and best_other["score"] >= float(row.get("score", 0.0) or 0.0) + 4.0:
        when = _slot_when(row)
        base.update({
            "priority": "high",
            "recommendation": "switch_preference",
            "mode": "prefer_teacher",
            "teacher": best_other["teacher"],
            "reason": "для рискованного слота найден альтернативный кандидат с не худшей допустимостью и заметным преимуществом",
            "rule_json": _json_rule("prefer_teacher", when, best_other["teacher"]),
        })
        return base
    if current:
        when = _slot_when(row)
        base.update({
            "priority": "medium",
            "recommendation": "review_assignment",
            "mode": "prefer_teacher",
            "teacher": current,
            "reason": "слот остаётся рискованным; можно зафиксировать текущего преподавателя правилом только после проверки кафедрой",
            "rule_json": _json_rule("prefer_teacher", when, current),
        })
        return base
    base.update({
        "priority": "medium",
        "recommendation": "manual_review",
        "mode": "",
        "teacher": "",
        "reason": "рискованный слот требует ручного решения",
        "rule_json": "",
    })
    return base


def _suggest_for_cluster(row: pd.Series) -> dict[str, Any]:
    teacher = _txt(row.get("Преподаватель"))
    disc = _txt(row.get("disc_key"))
    high_risk_slots = int(row.get("high_risk_slots", 0) or 0)
    opened_new_day_slots = int(row.get("opened_new_day_slots", 0) or 0)
    limited_slots = int(row.get("limited_slots", 0) or 0)
    priority = "high" if high_risk_slots >= 4 else "medium"
    mode = "ban_teacher_scope" if high_risk_slots >= 4 else "prefer_teacher_scope"
    reason = (
        "по связке преподаватель–дисциплина накопился рискованный кластер; "
        "стоит сузить или явно зафиксировать область назначения через правило"
    )
    when = {"disc_key": disc}
    if mode == "ban_teacher_scope":
        rule_json = _json_rule(mode, when, teacher)
    else:
        rule_json = _json_rule(mode, when, teacher)
    return {
        "source": "кластер дисциплины",
        "slot_id": "",
        "День недели": "",
        "Пара": "",
        "Учебная группа": "",
        "disc_key": disc,
        "Вид_занятия_норм": "",
        "current_teacher": teacher,
        "priority": priority,
        "recommendation": "cluster_rule",
        "mode": mode,
        "teacher": teacher,
        "reason": f"{reason}; high_risk={high_risk_slots}, new_days={opened_new_day_slots}, limited={limited_slots}",
        "rule_json": rule_json,
        "cluster_slot_count": int(row.get("slot_count", 0) or 0),
        "cluster_group_count": int(row.get("group_count", 0) or 0),
        "cluster_day_count": int(row.get("day_count", 0) or 0),
        "cluster_risk_score": float(row.get("risk_cluster_score", 0.0) or 0.0),
    }




def _load_suggestion_sheets(path: Path) -> pd.DataFrame:
    """Читает лист all_suggestions из файла рекомендаций."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, sheet_name="all_suggestions").fillna("")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _normalize_json_rule(raw: str) -> dict[str, Any] | None:
    """Безопасно разбирает JSON правила из строки рекомендации."""
    if not _txt(raw):
        return None
    try:
        rule = json.loads(str(raw))
    except Exception:
        return None
    if not isinstance(rule, dict):
        return None
    when = rule.get("when", {}) if isinstance(rule.get("when", {}), dict) else {}
    assign = rule.get("assign", {}) if isinstance(rule.get("assign", {}), dict) else {}
    mode = _txt(rule.get("mode")) or ""
    return {"mode": mode, "when": when, "assign": assign}


def apply_rule_suggestions(
    *,
    suggestions_path: Path,
    mappings_path: Path,
    out_dir: Path,
    priority: str = "high",
    modes: list[str] | None = None,
    sources: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Применяет часть рекомендаций в mappings.json и сохраняет отчёт."""
    df = _load_suggestion_sheets(Path(suggestions_path))
    if len(df) == 0:
        report_path = Path(out_dir) / "accepted_rule_suggestions.xlsx"
        _safe_to_excel({"summary": pd.DataFrame([{"metric": "applied", "value": 0}, {"metric": "skipped", "value": 0}])}, report_path)
        return {
            "summary": {"applied_rule_suggestions": 0, "skipped_rule_suggestions": 0},
            "files": {"accepted_rule_suggestions": str(report_path)},
        }

    if _txt(priority):
        df = df[df.get("priority", pd.Series(dtype=str)).astype(str).eq(priority)]
    if modes:
        allow = {str(m).strip() for m in modes if _txt(m)}
        if allow:
            df = df[df.get("mode", pd.Series(dtype=str)).astype(str).isin(allow)]
    if sources:
        allow_src = {str(s).strip() for s in sources if _txt(s)}
        if allow_src:
            df = df[df.get("source", pd.Series(dtype=str)).astype(str).isin(allow_src)]

    df = df[df.get("rule_json", pd.Series(dtype=str)).astype(str).ne("")].copy().reset_index(drop=True)
    if limit is not None and limit > 0:
        df = df.head(int(limit)).copy()

    try:
        from .mappings import load_mappings, save_mappings, _normalize_rule, _same_rule
    except Exception:
        from core.mappings import load_mappings, save_mappings, _normalize_rule, _same_rule  # type: ignore

    mappings = load_mappings(Path(mappings_path))
    rules = [r for r in mappings.get("rules", []) if isinstance(r, dict)]

    applied_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    def conflict_reason(rule: dict[str, Any]) -> str | None:
        mode = _txt(rule.get("mode")) or ""
        when = rule.get("when", {}) if isinstance(rule.get("when", {}), dict) else {}
        teacher = _txt((rule.get("assign", {}) or {}).get("teacher"))
        for ex in rules:
            ex_when = ex.get("when", {}) if isinstance(ex.get("when", {}), dict) else {}
            ex_mode = _txt(ex.get("mode")) or ""
            ex_teacher = _txt((ex.get("assign", {}) or {}).get("teacher"))
            if ex_when != when:
                continue
            if _same_rule(ex, rule):
                return "duplicate"
            same_prefer_family = mode in {"force_teacher", "prefer_teacher", "prefer_teacher_scope"} and ex_mode in {"force_teacher", "prefer_teacher", "prefer_teacher_scope"}
            if same_prefer_family and teacher and ex_teacher and teacher != ex_teacher:
                return f"conflict with existing {ex_mode}: {ex_teacher}"
            if mode.startswith("ban_") and teacher and ex_mode in {"force_teacher", "prefer_teacher", "prefer_teacher_scope"} and ex_teacher == teacher:
                return f"conflict with existing {ex_mode}: {ex_teacher}"
            if ex_mode.startswith("ban_") and teacher and mode in {"force_teacher", "prefer_teacher", "prefer_teacher_scope"} and ex_teacher == teacher:
                return f"conflict with existing {ex_mode}: {ex_teacher}"
        return None

    for _, row in df.iterrows():
        raw_rule = _normalize_json_rule(row.get("rule_json"))
        if not raw_rule:
            skipped_rows.append({**row.to_dict(), "skip_reason": "invalid rule_json"})
            continue
        _normalize_rule(raw_rule)
        reason = conflict_reason(raw_rule)
        if reason == "duplicate":
            skipped_rows.append({**row.to_dict(), "skip_reason": "duplicate"})
            continue
        if reason:
            skipped_rows.append({**row.to_dict(), "skip_reason": reason})
            continue
        rules.append(raw_rule)
        applied_rows.append({**row.to_dict(), "applied_mode": raw_rule.get("mode", "")})

    mappings["rules"] = rules
    save_mappings(Path(mappings_path), mappings)

    report_path = Path(out_dir) / "accepted_rule_suggestions.xlsx"
    summary = pd.DataFrame([
        {"metric": "applied", "value": int(len(applied_rows))},
        {"metric": "skipped", "value": int(len(skipped_rows))},
        {"metric": "rules_total_after_apply", "value": int(len(rules))},
    ])
    _safe_to_excel({
        "summary": summary,
        "applied": pd.DataFrame(applied_rows) if applied_rows else pd.DataFrame({"info": ["нет применённых рекомендаций"]}),
        "skipped": pd.DataFrame(skipped_rows) if skipped_rows else pd.DataFrame({"info": ["нет пропущенных рекомендаций"]}),
    }, report_path)

    return {
        "summary": {
            "applied_rule_suggestions": int(len(applied_rows)),
            "skipped_rule_suggestions": int(len(skipped_rows)),
            "rules_total_after_apply": int(len(rules)),
        },
        "files": {"accepted_rule_suggestions": str(report_path)},
    }

def build_rule_suggestions(*, final_assignments: pd.DataFrame, candidates: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assigned = _merge_assignment_candidate_info(final_assignments, candidates)
    if len(assigned) > 0:
        risks = assigned.apply(_risk_for_row, axis=1, result_type="expand")
        assigned["risk_level"] = risks[0]
        assigned["risk_reason"] = risks[1]
    else:
        assigned["risk_level"] = pd.Series(dtype=str)
        assigned["risk_reason"] = pd.Series(dtype=str)

    unmatched = final_assignments[final_assignments["Преподаватель"].isna()].copy()
    high_risk = assigned[assigned["risk_level"] == "высокий"].copy()
    clusters = _discipline_risk_clusters(assigned)
    clusters = clusters[(clusters["risk_cluster_score"] >= 18.0) | (clusters["high_risk_slots"] >= 3)].copy()

    unmatched_rows = []
    for _, slot in unmatched.iterrows():
        top = _top_by_slot(candidates, slot.get("slot_id"), n=3)
        unmatched_rows.append(_suggest_for_unmatched(slot, top))
    unmatched_df = pd.DataFrame(unmatched_rows)

    high_risk_rows = []
    for _, row in high_risk.iterrows():
        top = _top_by_slot(candidates, row.get("slot_id"), n=3)
        high_risk_rows.append(_suggest_for_high_risk(row, top))
    high_risk_df = pd.DataFrame(high_risk_rows)

    cluster_rows = []
    for _, row in clusters.iterrows():
        cluster_rows.append(_suggest_for_cluster(row))
    cluster_df = pd.DataFrame(cluster_rows)

    all_suggestions = pd.concat([df for df in [unmatched_df, high_risk_df, cluster_df] if df is not None and len(df) > 0], ignore_index=True) if any(len(df) > 0 for df in [unmatched_df, high_risk_df, cluster_df]) else pd.DataFrame()
    if len(all_suggestions) > 0:
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_suggestions["_priority_order"] = all_suggestions.get("priority", "low").map(priority_order).fillna(9)
        all_suggestions = all_suggestions.sort_values(["_priority_order", "source", "disc_key", "Учебная группа"], ascending=[True, True, True, True]).drop(columns=["_priority_order"], errors="ignore").reset_index(drop=True)

    summary = pd.DataFrame([
        {"metric": "total_suggestions", "value": int(len(all_suggestions))},
        {"metric": "unmatched_slot_suggestions", "value": int(len(unmatched_df))},
        {"metric": "high_risk_slot_suggestions", "value": int(len(high_risk_df))},
        {"metric": "discipline_cluster_suggestions", "value": int(len(cluster_df))},
        {"metric": "high_priority_suggestions", "value": int((all_suggestions.get("priority", pd.Series(dtype=str)) == "high").sum()) if len(all_suggestions) else 0},
    ])

    out_path = out_dir / "rule_suggestions.xlsx"
    _safe_to_excel({
        "summary": summary,
        "all_suggestions": all_suggestions,
        "unmatched_slots": unmatched_df,
        "high_risk_slots": high_risk_df,
        "discipline_clusters": cluster_df,
    }, out_path)

    return {
        "summary": {
            "rule_suggestions_total": int(len(all_suggestions)),
            "rule_suggestions_high_priority": int((all_suggestions.get("priority", pd.Series(dtype=str)) == "high").sum()) if len(all_suggestions) else 0,
        },
        "files": {
            "rule_suggestions": str(out_path),
        },
    }
