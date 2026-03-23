from __future__ import annotations

from pathlib import Path

import pandas as pd

from .allocation import build_slot_candidates, allocate_unmatched_greedy, merge_locked_and_allocated
from .export import build_teacher_timetable
from .load_model import build_teacher_state
from .mappings import apply_mappings, load_mappings
from .matching import lock_teacher_hints, merge_schedule_with_teachers, select_locked_exact_matches
from .sched_parser import read_schedule_atoms
from .un_parser import build_teacher_capacity, read_un, read_un_atoms


from .normalize import _txt, best_disc_match, teacher_lastname, extract_group_parts


def _split_hint_names(value: str) -> list[str]:
    return [x.strip() for x in str(_txt(value)).split("/") if x and str(x).strip()]


def _infer_hint_teacher_atoms(schedule_atoms: pd.DataFrame, run_atoms: pd.DataFrame) -> pd.DataFrame:
    if schedule_atoms is None or len(schedule_atoms) == 0:
        return pd.DataFrame()
    teacher_lastnames = set(_txt(x) for x in run_atoms.get("teacher_lastname", []) if _txt(x)) if run_atoms is not None and len(run_atoms) > 0 else set()
    rows = []
    charged_streams: set[tuple] = set()
    hint_name_counts: dict[str, int] = {}
    for _, r in schedule_atoms.iterrows():
        hints = _split_hint_names(r.get("teacher_hint"))
        if len(hints) != 1:
            continue
        h = hints[0]
        hint_name_counts[h] = hint_name_counts.get(h, 0) + 1

    for _, r in schedule_atoms.iterrows():
        hints = _split_hint_names(r.get("teacher_hint"))
        if len(hints) != 1:
            continue
        teacher = hints[0]
        lname = teacher_lastname(teacher)
        if not lname or lname in teacher_lastnames:
            continue
        disc = _txt(r.get("disc_key"))
        kind = _txt(r.get("Вид_занятия_норм"))
        group = _txt(r.get("Учебная группа"))
        if not disc or not kind or not group:
            continue
        # Для отсутствующих в РУН преподавателей доверяем только повторяющимся подсказкам
        # или лекциям, которые образуют явный поток.
        if hint_name_counts.get(teacher, 0) < 2 and kind != "лек":
            continue
        gp = extract_group_parts(group)
        day = _txt(r.get("День недели"))
        pair = r.get("Пара")
        time = _txt(r.get("Время"))
        room = _txt(r.get("Аудитория"))
        stream_key = (teacher, day, pair, time, disc, kind, room)
        cap_units = 0.0 if stream_key in charged_streams else 1.0
        charged_streams.add(stream_key)
        rows.append({
            "Учебная группа": group,
            "group_norm": group,
            "group_prefix": gp.get("group_prefix", ""),
            "group_num": gp.get("group_num", ""),
            "course_year": gp.get("course_year", ""),
            "Дисциплина": _txt(r.get("Дисциплина")) or disc,
            "discipline_norm": disc,
            "disc_key": disc,
            "Вид_работы": kind,
            "kind_norm": kind,
            "Преподаватель": teacher,
            "teacher_norm": teacher,
            "teacher_lastname": lname,
            "Кафедра": "ММиИИ (inferred from schedule hint)",
            "Должность": "",
            "Тип_занятости": "",
            "ООП": "",
            "Код_ООП": "",
            "Семестр": "",
            "hours_kind": 18.0 if kind == "лек" else 36.0,
            "hours_total": 18.0 if kind == "лек" else 36.0,
            "capacity_units": cap_units,
            "capacity_units_total": cap_units,
            "source_row_kind": "schedule_hint_inferred",
        })
    return pd.DataFrame(rows)


def _is_slot_departmental(slot: pd.Series, run_atoms: pd.DataFrame, mappings: list[dict] | None = None) -> tuple[bool, str]:
    disc = _txt(slot.get("disc_key"))
    kind = _txt(slot.get("Вид_занятия_норм"))
    teacher_hint = teacher_lastname(slot.get("teacher_hint"))

    if run_atoms is None or len(run_atoms) == 0:
        return False, "no run atoms"

    run_disc = sorted(set(_txt(x) for x in run_atoms.get("disc_key", []) if _txt(x)))
    run_teacher_lastnames = set(_txt(x) for x in run_atoms.get("teacher_lastname", []) if _txt(x))

    if teacher_hint and teacher_hint in run_teacher_lastnames:
        return True, "teacher_hint in department"

    if disc and disc in run_disc:
        return True, "disc_key in department load"

    if disc:
        best, score = best_disc_match(disc, run_disc)
        if best and score >= 0.82:
            return True, f"disc fuzzy match {best}:{score:.2f}"

    if mappings:
        group = _txt(slot.get("Учебная группа"))
        for rule in mappings:
            when = rule.get("when", {}) if isinstance(rule, dict) else {}
            if not isinstance(when, dict):
                continue
            if _txt(when.get("group")) == group and _txt(when.get("disc_key")) == disc and _txt(when.get("kind")) == kind:
                return True, "manual mapping exists"

    return False, "not in department load"


def _filter_department_slots(sched_norm: pd.DataFrame, run_atoms: pd.DataFrame, mappings: list[dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sched_norm is None or len(sched_norm) == 0:
        empty = sched_norm.copy()
        return empty, empty

    rows = []
    for _, r in sched_norm.iterrows():
        is_dep, reason = _is_slot_departmental(r, run_atoms, mappings)
        rr = r.copy()
        rr["is_departmental"] = bool(is_dep)
        rr["department_reason"] = reason
        rows.append(rr)
    marked = pd.DataFrame(rows)
    departmental = marked[marked["is_departmental"]].copy().reset_index(drop=True)
    external = marked[~marked["is_departmental"]].copy().reset_index(drop=True)
    return departmental, external


def _safe_to_excel(df: pd.DataFrame, path: Path):
    if df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)



def build_timetable_bundle(run_path: Path, sched_path: Path, out_dir: Path, mappings_path: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_path = Path(run_path)
    sched_path = Path(sched_path)
    mappings_path = Path(mappings_path)

    # 1) парсинг исходников
    un_raw, un_expanded = read_un(run_path)
    run_atoms_base = read_un_atoms(run_path)
    sched_parsed, sched_norm_all, sched_atoms = read_schedule_atoms(sched_path)

    mappings = load_mappings(mappings_path)
    sched_norm, external_slots = _filter_department_slots(sched_norm_all, run_atoms_base, mappings)

    hint_atoms = _infer_hint_teacher_atoms(sched_norm, run_atoms_base)
    if len(hint_atoms) > 0:
        run_atoms = pd.concat([run_atoms_base, hint_atoms], ignore_index=True)
        hint_expanded = hint_atoms.rename(columns={"kind_norm": "Вид_работы_норм"})
        un_expanded = pd.concat([un_expanded, hint_expanded], ignore_index=True)
        # после добавления подсказанных преподавателей ещё раз проверяем кафедральность
        sched_norm, external_slots = _filter_department_slots(sched_norm_all, run_atoms, mappings)
    else:
        run_atoms = run_atoms_base

    # 2) жёсткие назначения: teacher_hint + exact
    locked_hints = lock_teacher_hints(sched_norm, run_atoms)
    exact_locked = select_locked_exact_matches(sched_norm, run_atoms)

    locked = pd.concat([locked_hints, exact_locked], ignore_index=True) if (len(locked_hints) or len(exact_locked)) else pd.DataFrame(columns=["slot_id", "Преподаватель", "assign_type", "confidence", "reason", "score"])
    if len(locked) > 0:
        locked = locked.sort_values(["slot_id", "score"], ascending=[True, False]).drop_duplicates(subset=["slot_id"], keep="first")

    base_slots = sched_norm.copy()
    base_slots["Преподаватель"] = None
    base_slots["assign_type"] = None
    base_slots["confidence"] = None
    base_slots["reason"] = None
    base_slots["score"] = None

    final_pre = merge_locked_and_allocated(base_slots, locked, pd.DataFrame())
    final_pre = apply_mappings(final_pre, mappings, override=False)

    locked_after_manual = final_pre[final_pre["Преподаватель"].notna()].copy()
    teacher_capacity = build_teacher_capacity(run_atoms)
    teacher_state = build_teacher_state(teacher_capacity, locked_after_manual)

    # 3) дораспределение unmatched
    unmatched_slots = final_pre[final_pre["Преподаватель"].isna()].copy()
    candidates = build_slot_candidates(unmatched_slots, run_atoms, teacher_state, locked_after_manual, mappings=mappings)
    allocated = allocate_unmatched_greedy(unmatched_slots, candidates, teacher_state)

    final_assignments = merge_locked_and_allocated(final_pre, locked_after_manual, allocated)

    # 4) диагностический старый merge оставим для сравнения
    legacy_merged, lowconf = merge_schedule_with_teachers(sched_norm, un_expanded)

    # 5) экспорт промежуточных
    schedule_with_teachers = out_dir / "schedule_with_teachers.xlsx"
    timetable_by_teachers = out_dir / "timetable_by_teachers.xlsx"
    unmatched_slots_path = out_dir / "unmatched_slots.xlsx"
    un_expanded_path = out_dir / "un_svodnaya_expanded.xlsx"
    lowconf_path = out_dir / "lowconf_matches.xlsx"
    conflicts_path = out_dir / "teacher_conflicts.xlsx"
    un_raw_path = out_dir / "un_svodnaya.xlsx"
    sched_parsed_path = out_dir / "расписание_по_группам_разобранное.xlsx"
    sched_norm_path = out_dir / "расписание_по_группам_norm.xlsx"
    external_slots_path = out_dir / "external_non_department_slots.xlsx"
    schedule_atoms_path = out_dir / "schedule_atoms.xlsx"
    run_atoms_path = out_dir / "run_atoms.xlsx"
    locked_path = out_dir / "locked_assignments.xlsx"
    allocated_path = out_dir / "allocated_assignments.xlsx"
    candidates_path = out_dir / "candidate_scores.xlsx"
    teacher_load_path = out_dir / "teacher_load_summary.xlsx"
    legacy_path = out_dir / "legacy_schedule_with_teachers.xlsx"

    _safe_to_excel(un_raw, un_raw_path)
    _safe_to_excel(un_expanded, un_expanded_path)
    _safe_to_excel(run_atoms, run_atoms_path)
    _safe_to_excel(sched_parsed, sched_parsed_path)
    _safe_to_excel(sched_norm, sched_norm_path)
    _safe_to_excel(external_slots, external_slots_path)
    _safe_to_excel(sched_atoms, schedule_atoms_path)
    _safe_to_excel(locked_after_manual, locked_path)
    _safe_to_excel(allocated, allocated_path)
    _safe_to_excel(candidates, candidates_path)
    _safe_to_excel(legacy_merged, legacy_path)
    _safe_to_excel(final_assignments, schedule_with_teachers)

    unmatched = final_assignments[final_assignments["Преподаватель"].isna()].copy()
    cols = [
        "slot_id",
        "День недели",
        "Пара",
        "Время",
        "Учебная группа",
        "subgroup",
        "Дисциплина",
        "Вид_занятия",
        "Аудитория",
        "disc_key",
        "Вид_занятия_норм",
        "teacher_hint",
        "source_text",
    ]
    cols = [c for c in cols if c in unmatched.columns]
    _safe_to_excel(unmatched[cols], unmatched_slots_path)

    if isinstance(lowconf, pd.DataFrame) and len(lowconf) > 0:
        _safe_to_excel(lowconf, lowconf_path)
    elif lowconf_path.exists():
        lowconf_path.unlink(missing_ok=True)

    # teacher summary
    teacher_summary = teacher_capacity.copy()
    matched_counts = final_assignments[final_assignments["Преподаватель"].notna()].groupby("Преподаватель", as_index=False).agg(
        assigned_slots=("slot_id", "nunique")
    )
    teacher_summary = teacher_summary.merge(matched_counts, on="Преподаватель", how="left")
    teacher_summary["assigned_slots"] = teacher_summary["assigned_slots"].fillna(0).astype(int)
    teacher_summary["remaining_total_units_after"] = teacher_summary["Преподаватель"].map(teacher_state["remaining_total"]).fillna(teacher_summary.get("remaining_total_units", 0))
    _safe_to_excel(teacher_summary, teacher_load_path)

    # pivot + conflicts
    pivot, conflicts = build_teacher_timetable(final_assignments)
    pivot.to_excel(timetable_by_teachers, index=False)

    if isinstance(conflicts, pd.DataFrame) and len(conflicts) > 0:
        conflicts.to_excel(conflicts_path, index=False)
    elif conflicts_path.exists():
        conflicts_path.unlink(missing_ok=True)

    total = int(len(final_assignments))
    external_n = int(len(external_slots))
    matched = int(final_assignments["Преподаватель"].notna().sum()) if "Преподаватель" in final_assignments.columns else 0
    unmatched_n = int(total - matched)
    lowconf_n = int(len(lowconf)) if isinstance(lowconf, pd.DataFrame) else 0
    conflicts_n = int(len(conflicts)) if isinstance(conflicts, pd.DataFrame) else 0
    locked_n = int((final_assignments["assign_type"].fillna("").isin(["locked_teacher_hint", "locked_exact", "manual_force"]).sum()))
    auto_n = int((final_assignments["assign_type"] == "auto_allocated").sum())

    return {
        "stats": {
            "total_slots": total,
            "matched": matched,
            "unmatched": unmatched_n,
            "lowconf": lowconf_n,
            "conflicts": conflicts_n,
            "locked": locked_n,
            "auto_allocated": auto_n,
            "external_excluded": external_n,
        },
        "files": {
            "out": str(timetable_by_teachers),
            "unmatched": str(unmatched_slots_path),
            "un_expanded": str(un_expanded_path),
            "schedule_with_teachers": str(schedule_with_teachers),
            "schedule_atoms": str(schedule_atoms_path),
            "run_atoms": str(run_atoms_path),
            "teacher_loads": str(teacher_load_path),
            "external_slots": str(external_slots_path),
        },
    }
