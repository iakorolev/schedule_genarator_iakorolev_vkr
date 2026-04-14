"""Модель состояния преподавателей: доступность, нагрузка и уже сделанные назначения."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .normalize import _txt, extract_group_parts, normalize_room, normalize_week_type


SLOT_UNIT = 1.0
KIND_UNIT = {"лек": 1.0, "сем": 1.0, "лаб": 1.0}


def build_teacher_state(teacher_capacity: pd.DataFrame, locked_assignments: pd.DataFrame | None = None) -> dict[str, Any]:
    """Создаёт начальное состояние преподавателей по нагрузке и уже известным назначениям."""
    state = {
        "capacity_total": {},
        "capacity_by_kind": {},
        "remaining_total": {},
        "remaining_by_kind": {},
        "busy_slots": defaultdict(list),  # teacher -> list[assignment_meta]
        "charged_streams": defaultdict(set),  # teacher -> set[event_key]
        "assigned_disc": defaultdict(set),
        "assigned_disc_by_day": defaultdict(lambda: defaultdict(set)),
        "assigned_family_by_day": defaultdict(lambda: defaultdict(set)),
        "assigned_prefix_year_by_day": defaultdict(lambda: defaultdict(set)),
        "assigned_disc_kind": defaultdict(set),
        "assigned_group_disc_kind": defaultdict(set),
        "assigned_family_disc_kind": defaultdict(set),
        "assigned_prefix_year_disc_kind": defaultdict(set),
        "assigned_count": defaultdict(int),
        "present_days": defaultdict(set),
        "assigned_slots_by_day": defaultdict(lambda: defaultdict(set)),
        "trusted_present_days": defaultdict(set),
        "trusted_assigned_slots_by_day": defaultdict(lambda: defaultdict(set)),
        "trusted_assigned_disc_by_day": defaultdict(lambda: defaultdict(set)),
        "trusted_assigned_family_by_day": defaultdict(lambda: defaultdict(set)),
        "trusted_assigned_prefix_year_by_day": defaultdict(lambda: defaultdict(set)),
        "trusted_assigned_disc_kind": defaultdict(set),
        "trusted_assigned_group_disc_kind": defaultdict(set),
        "trusted_assigned_family_disc_kind": defaultdict(set),
        "trusted_assigned_prefix_year_disc_kind": defaultdict(set),
    }

    if teacher_capacity is not None and len(teacher_capacity) > 0:
        for _, r in teacher_capacity.iterrows():
            t = r.get("Преподаватель")
            if not t:
                continue
            total_units = float(r.get("capacity_total_units", 0) or 0)
            by_kind = {
                "лек": float(r.get("capacity_лек_units", 0) or 0),
                "сем": float(r.get("capacity_сем_units", 0) or 0),
                "лаб": float(r.get("capacity_лаб_units", 0) or 0),
            }
            state["capacity_total"][t] = total_units
            state["capacity_by_kind"][t] = by_kind.copy()
            state["remaining_total"][t] = total_units
            state["remaining_by_kind"][t] = by_kind.copy()

    if locked_assignments is not None and len(locked_assignments) > 0:
        for _, r in locked_assignments.iterrows():
            teacher = r.get("Преподаватель")
            if teacher:
                apply_assignment_to_state(state, r)

    return state



def slot_key(day: str, pair: Any, time: str, week_type: str | None = None) -> tuple[str, int | None, str, str]:
    """Формирует ключ временного слота по дню, паре, времени и типу недели."""
    return (str(day), int(pair) if pd.notna(pair) else None, str(time), str(week_type or ""))



def _room_key(room: str) -> str:
    """Нормализует аудиторию для сравнения и проверки потоковых лекций."""
    s = normalize_room(room or "").lower()
    if not s:
        return ""
    if "дистан" in s:
        return "distance"
    if "туис" in s:
        return "tuis"
    return s



def _stream_token(assignment: Any) -> tuple[str, str, str]:
    """Строит токен потокового события для учёта совместимых лекций."""
    disc = _txt(assignment.get("disc_key"))
    kind = _txt(assignment.get("Вид_занятия_норм") or assignment.get("kind_norm"))
    room = _room_key(_txt(assignment.get("Аудитория") or assignment.get("room")))
    return (disc, kind, room)



def _assignment_meta(assignment: Any) -> dict[str, Any]:
    """Преобразует назначение в компактную структуру для хранения в состоянии."""
    group = _txt(assignment.get("Учебная группа") or assignment.get("group"))
    gparts = extract_group_parts(group)
    return {
        "day": _txt(assignment.get("День недели")),
        "pair": assignment.get("Пара"),
        "time": _txt(assignment.get("Время")),
        "week_type": normalize_week_type(assignment.get("week_type")),
        "disc": _txt(assignment.get("disc_key")),
        "kind": _txt(assignment.get("Вид_занятия_норм") or assignment.get("kind_norm")),
        "room": _room_key(_txt(assignment.get("Аудитория") or assignment.get("room"))),
        "group": group,
        "family_key": gparts.get("family_key", ""),
        "prefix_year": gparts.get("prefix_year", ""),
    }



def _stream_compatible(existing: dict, incoming: dict) -> bool:
    # Одновременное ведение допускаем только для потоковых лекций.
    """Проверяет, можно ли считать два назначения совместимыми внутри одного потока."""
    if existing.get("kind") != "лек" or incoming.get("kind") != "лек":
        return False
    if existing.get("disc") != incoming.get("disc"):
        return False
    er = existing.get("room") or ""
    ir = incoming.get("room") or ""
    if er and ir and er != ir:
        if {er, ir} <= {"distance", "tuis"}:
            return True
        return False
    if existing.get("family_key") and incoming.get("family_key") and existing.get("family_key") == incoming.get("family_key"):
        return True
    if existing.get("prefix_year") and incoming.get("prefix_year") and existing.get("prefix_year") == incoming.get("prefix_year"):
        return True
    return True



def _weeks_overlap(existing_week: str, incoming_week: str) -> bool:
    """Проверяет пересечение типов недели: пустое значение считается общим слотом."""
    existing_week = normalize_week_type(existing_week)
    incoming_week = normalize_week_type(incoming_week)
    if not existing_week or not incoming_week:
        return True
    return existing_week == incoming_week



def teacher_is_available(state: dict[str, Any], teacher: str, day: str, pair: Any, time: str, disc: str | None = None, kind: str | None = None, room: str | None = None, group: str | None = None, week_type: str | None = None) -> bool:
    """Проверяет, свободен ли преподаватель в заданном временном слоте."""
    sk = slot_key(day, pair, time, week_type)
    existing = state["busy_slots"].get(teacher, [])
    same_slot = [
        m for m in existing
        if slot_key(m.get("day"), m.get("pair"), m.get("time"), m.get("week_type"))[:3] == sk[:3]
        and _weeks_overlap(_txt(m.get("week_type")), _txt(week_type))
    ]
    if not same_slot:
        return True
    incoming = {
        "day": _txt(day),
        "pair": pair,
        "time": _txt(time),
        "disc": _txt(disc),
        "kind": _txt(kind),
        "room": _room_key(_txt(room)),
        "group": _txt(group),
        "week_type": normalize_week_type(week_type),
        **extract_group_parts(_txt(group)),
    }
    return all(_stream_compatible(m, incoming) for m in same_slot)



def teacher_remaining_capacity(state: dict[str, Any], teacher: str, kind: str | None = None) -> float:
    """Возвращает остаточную нагрузку преподавателя в целом или по виду занятий."""
    if kind:
        return float(state["remaining_by_kind"].get(teacher, {}).get(kind, 0.0) or 0.0)
    return float(state["remaining_total"].get(teacher, 0.0) or 0.0)



def apply_assignment_to_state(state: dict[str, Any], assignment: Any) -> dict[str, Any]:
    """Обновляет состояние преподавателя после нового назначения."""
    teacher = assignment.get("Преподаватель")
    if not teacher:
        return state

    meta = _assignment_meta(assignment)
    disc = meta["disc"]
    kind = meta["kind"]
    day = meta["day"]
    pair = meta["pair"]
    time = meta["time"]
    group = meta["group"]
    family_key = meta["family_key"]
    prefix_year = meta["prefix_year"]

    sk = slot_key(day, pair, time, meta.get("week_type"))
    state["busy_slots"][teacher].append(meta)
    trust_for_propagation = bool(assignment.get("_trusted_propagation", True))

    if day:
        state["present_days"][teacher].add(day)
        state["assigned_slots_by_day"][teacher][day].add(sk)
        if trust_for_propagation:
            state["trusted_present_days"][teacher].add(day)
            state["trusted_assigned_slots_by_day"][teacher][day].add(sk)
    if disc:
        state["assigned_disc"][teacher].add(disc)
        state["assigned_disc_by_day"][teacher][str(day)].add(disc)
        if family_key:
            state["assigned_family_by_day"][teacher][str(day)].add(family_key)
        if prefix_year:
            state["assigned_prefix_year_by_day"][teacher][str(day)].add(prefix_year)
        if trust_for_propagation:
            state["trusted_assigned_disc_by_day"][teacher][str(day)].add(disc)
            if family_key:
                state["trusted_assigned_family_by_day"][teacher][str(day)].add(family_key)
            if prefix_year:
                state["trusted_assigned_prefix_year_by_day"][teacher][str(day)].add(prefix_year)
        if kind:
            state["assigned_disc_kind"][teacher].add((disc, kind))
            if group:
                state["assigned_group_disc_kind"][teacher].add((group, disc, kind))
            if family_key:
                state["assigned_family_disc_kind"][teacher].add((family_key, disc, kind))
            if prefix_year:
                state["assigned_prefix_year_disc_kind"][teacher].add((prefix_year, disc, kind))
            if trust_for_propagation:
                state["trusted_assigned_disc_kind"][teacher].add((disc, kind))
                if group:
                    state["trusted_assigned_group_disc_kind"][teacher].add((group, disc, kind))
                if family_key:
                    state["trusted_assigned_family_disc_kind"][teacher].add((family_key, disc, kind))
                if prefix_year:
                    state["trusted_assigned_prefix_year_disc_kind"][teacher].add((prefix_year, disc, kind))
    state["assigned_count"][teacher] += 1

    event_key = (sk, *_stream_token(assignment))
    charge_this = not (kind == "лек" and event_key in state["charged_streams"][teacher])

    if charge_this:
        unit = KIND_UNIT.get(kind, SLOT_UNIT)
        state["remaining_total"][teacher] = float(state["remaining_total"].get(teacher, 0.0) or 0.0) - unit
        if teacher not in state["remaining_by_kind"]:
            state["remaining_by_kind"][teacher] = {"лек": 0.0, "сем": 0.0, "лаб": 0.0}
        if kind:
            state["remaining_by_kind"][teacher][kind] = float(state["remaining_by_kind"][teacher].get(kind, 0.0) or 0.0) - unit
        state["charged_streams"][teacher].add(event_key)
    return state
