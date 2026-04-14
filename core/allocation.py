"""Эвристическое распределение преподавателей по неразобранным слотам расписания."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .load_model import apply_assignment_to_state, teacher_is_available, teacher_remaining_capacity
from .math_methods import clamp01, topsis_scores, weighted_sum_score
from .normalize import _txt, best_disc_match, extract_group_parts, teacher_lastname
from .un_parser import build_teacher_discipline_capacity, build_teacher_group_links, build_teacher_skills


@dataclass
class CandidateScore:
    """Хранит итоговую оценку кандидата и текстовое объяснение назначения."""
    teacher: str
    score: float
    reason: str


DAY_PRESENCE_BONUS = 24.0
NEW_DAY_PENALTY = 30.0
FOUNDATION_DISC_MARKERS = ("математический анализ", "линейная алгебра")
TRUST_MIN_MATH_SCORE = 0.62
TRUST_MIN_SCORE_GAP = 10.0
TRUST_MIN_SUBJECT_FIT = 0.74
TEACHER_HINT_STRONG_BONUS = 126.0
PREFER_RULE_BONUS = 52.0
RECOVERY_MIN_SCORE = 22.0
RECOVERY_MIN_GROUP_FIT = 0.46
RECOVERY_MIN_MATH_SCORE = 0.42


def _is_foundation_disc(disc: str) -> bool:
    """Возвращает True для массовых базовых дисциплин, где спорное автопротягивание особенно нежелательно."""
    disc = _txt(disc).lower()
    return any(marker in disc for marker in FOUNDATION_DISC_MARKERS)

CANDIDATE_CRITERIA = [
    {"name": "subject_fit", "weight": 0.35, "mode": "benefit"},
    {"name": "group_fit", "weight": 0.24, "mode": "benefit"},
    {"name": "day_fit", "weight": 0.08, "mode": "benefit"},
    {"name": "load_fit", "weight": 0.10, "mode": "benefit"},
    {"name": "continuity_fit", "weight": 0.15, "mode": "benefit"},
    {"name": "hint_fit", "weight": 0.05, "mode": "benefit"},
    {"name": "new_day_penalty", "weight": 0.03, "mode": "cost"},
]


def _slot_group_context(slot: pd.Series) -> dict[str, str]:
    """Возвращает нормализованный контекст учебной группы для текущего слота."""
    group = _txt(slot.get("Учебная группа"))
    return extract_group_parts(group)


def _narrow_disc_kind_guard(
    *,
    disc: str,
    kind: str,
    teacher: str,
    strict_skill_teachers: set[str],
    exact_group_kind: float,
    family_kind: float,
    prefix_kind: float,
    prefix_only_kind: float,
    hint_match: bool,
) -> tuple[float, str]:
    """Возвращает штраф за слишком свободные назначения на массовые и узкие дисциплины."""
    strict_pool_exists = bool(strict_skill_teachers)
    is_narrow_scope = len(strict_skill_teachers) <= 2 if strict_pool_exists else False
    has_structural_support = any(x > 0 for x in (exact_group_kind, family_kind, prefix_kind, prefix_only_kind))
    in_strict_pool = teacher in strict_skill_teachers

    if not strict_pool_exists:
        return 0.0, ""
    if _is_foundation_disc(disc) and not (in_strict_pool or has_structural_support or hint_match):
        return 24.0, "foundation discipline outside strict pool"
    if is_narrow_scope and not (in_strict_pool or has_structural_support or hint_match):
        return 14.0, "narrow discipline scope"
    return 0.0, ""


def _state_family_bonus(state: dict[str, Any], teacher: str, slot: pd.Series, disc: str, kind: str) -> tuple[float, list[str]]:
    """Рассчитывает бонус за ранее сделанные назначения в той же группе или направлении."""
    bonus = 0.0
    reasons: list[str] = []
    ctx = _slot_group_context(slot)
    family_key = ctx.get("family_key", "")
    prefix_year = ctx.get("prefix_year", "")
    group = _txt(slot.get("Учебная группа"))

    trusted_disc_kind = state.get("trusted_assigned_disc_kind", {}).get(teacher, set())
    trusted_group_disc_kind = state.get("trusted_assigned_group_disc_kind", {}).get(teacher, set())
    trusted_family_disc_kind = state.get("trusted_assigned_family_disc_kind", {}).get(teacher, set())
    trusted_prefix_disc_kind = state.get("trusted_assigned_prefix_year_disc_kind", {}).get(teacher, set())

    if (disc, kind) in trusted_disc_kind:
        bonus += 18.0
        reasons.append("trusted same disc+kind")
    if group and (group, disc, kind) in trusted_group_disc_kind:
        bonus += 96.0
        reasons.append("trusted exact group propagation")
    elif family_key and (family_key, disc, kind) in trusted_family_disc_kind:
        bonus += 48.0
        reasons.append("trusted family propagation")
    elif prefix_year and (prefix_year, disc, kind) in trusted_prefix_disc_kind:
        bonus += 24.0
        reasons.append("trusted direction/year propagation")
    return bonus, reasons


def _prep_assigned_index(assignments: pd.DataFrame) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Строит индексы уже закреплённых назначений для повторного использования в скоринге."""
    disc_kind_teachers: dict[tuple[str, str], set[str]] = {}
    disc_teachers: dict[str, set[str]] = {}
    teacher_days: dict[tuple[str, str], set[str]] = {}
    family_disc_kind_teachers: dict[tuple[str, str, str], set[str]] = {}
    family_disc_teachers: dict[tuple[str, str], set[str]] = {}
    prefix_year_disc_kind_teachers: dict[tuple[str, str, str], set[str]] = {}
    if assignments is None or len(assignments) == 0:
        return disc_kind_teachers, disc_teachers, teacher_days, family_disc_kind_teachers, family_disc_teachers, prefix_year_disc_kind_teachers
    for _, r in assignments.iterrows():
        t = _txt(r.get("Преподаватель"))
        d = _txt(r.get("disc_key"))
        k = _txt(r.get("Вид_занятия_норм"))
        day = _txt(r.get("День недели"))
        group = _txt(r.get("Учебная группа"))
        gctx = extract_group_parts(group)
        fam = gctx.get("family_key", "")
        pyear = gctx.get("prefix_year", "")
        pprefix = gctx.get("group_prefix", "")
        if t and d:
            disc_teachers.setdefault(d, set()).add(t)
            if k:
                disc_kind_teachers.setdefault((d, k), set()).add(t)
            if fam:
                family_disc_teachers.setdefault((fam, d), set()).add(t)
                if k:
                    family_disc_kind_teachers.setdefault((fam, d, k), set()).add(t)
            if pyear and k:
                prefix_year_disc_kind_teachers.setdefault((pyear, d, k), set()).add(t)
        if t and day and d:
            teacher_days.setdefault((t, day), set()).add(d)
    return disc_kind_teachers, disc_teachers, teacher_days, family_disc_kind_teachers, family_disc_teachers, prefix_year_disc_kind_teachers


def _related_kinds(kind: str) -> list[str]:
    """Возвращает родственные виды занятий для fallback-поиска кандидатов."""
    if kind == "лаб":
        return ["сем"]
    if kind == "сем":
        return ["лаб"]
    if kind == "лек":
        return ["сем"]
    return ["лек", "сем"] if kind else []


def _admissibility_level(
    *,
    exact_group_kind: float,
    exact_group_disc: float,
    family_kind: float,
    family_disc: float,
    prefix_kind: float,
    prefix_disc: float,
    prefix_only_kind: float,
    prefix_only_disc: float,
    skill_kind: float,
    skill_disc: float,
    fuzzy_similarity: float,
    has_related_kind: bool,
    hint_match: bool,
    teaches_same_disc_kind: bool,
    teaches_same_disc: bool,
) -> int:
    """Возвращает уровень допустимости: 3=strict, 2=limited, 1=fallback, 0=reject."""
    if hint_match and (skill_kind > 0 or exact_group_disc > 0 or exact_group_kind > 0):
        return 3
    if exact_group_kind > 0 or family_kind > 0 or prefix_kind > 0 or prefix_only_kind > 0 or skill_kind > 0 or teaches_same_disc_kind:
        return 3
    if exact_group_disc > 0 or family_disc > 0 or prefix_disc > 0 or prefix_only_disc > 0 or skill_disc > 0 or teaches_same_disc:
        return 2
    if has_related_kind or fuzzy_similarity >= 0.62:
        return 1
    return 0


def _bounded_group_fit(
    *,
    exact_group_kind: float,
    exact_group_disc: float,
    family_kind: float,
    family_disc: float,
    prefix_kind: float,
    prefix_disc: float,
    prefix_only_kind: float,
    prefix_only_disc: float,
    subgroup: str,
) -> float:
    """Оценивает близость кандидата к конкретной группе слота."""
    fit = 0.0
    if exact_group_kind > 0:
        fit = 1.0
    elif exact_group_disc > 0:
        fit = 0.92
    elif family_kind > 0:
        fit = 0.72
    elif family_disc > 0:
        fit = 0.58
    elif prefix_kind > 0:
        fit = 0.45
    elif prefix_only_kind > 0:
        fit = 0.40
    elif prefix_disc > 0:
        fit = 0.35
    elif prefix_only_disc > 0:
        fit = 0.30
    if subgroup and fit >= 0.92:
        fit = min(1.0, fit + 0.04)
    return clamp01(fit)


def _group_pool_penalty(
    *,
    exact_group_kind: float,
    exact_group_disc: float,
    family_kind: float,
    family_disc: float,
    prefix_kind: float,
    prefix_disc: float,
    prefix_only_kind: float,
    prefix_only_disc: float,
    exact_kind_pool_active: bool,
    exact_disc_pool_active: bool,
    family_pool_active: bool,
) -> tuple[float, list[str]]:
    """Штрафует кандидатов, если для слота есть более точные кандидаты по группе."""
    penalty = 0.0
    reasons: list[str] = []
    if exact_kind_pool_active and exact_group_kind <= 0:
        if exact_group_disc > 0:
            penalty -= 18.0
            reasons.append("penalized: exact group+kind pool active")
        elif family_kind > 0 or family_disc > 0:
            penalty -= 42.0
            reasons.append("penalized: family candidate while exact group+kind exists")
        elif prefix_kind > 0 or prefix_disc > 0:
            penalty -= 58.0
            reasons.append("penalized: direction/year candidate while exact group+kind exists")
        elif prefix_only_kind > 0 or prefix_only_disc > 0:
            penalty -= 52.0
            reasons.append("penalized: prefix candidate while exact group+kind exists")
        else:
            penalty -= 74.0
            reasons.append("penalized: non-group candidate while exact group+kind exists")
    elif exact_disc_pool_active and exact_group_disc <= 0 and exact_group_kind <= 0:
        if family_kind > 0 or family_disc > 0:
            penalty -= 24.0
            reasons.append("penalized: family candidate while exact group+disc exists")
        elif prefix_kind > 0 or prefix_disc > 0:
            penalty -= 38.0
            reasons.append("penalized: direction/year candidate while exact group+disc exists")
        elif prefix_only_kind > 0 or prefix_only_disc > 0:
            penalty -= 30.0
            reasons.append("penalized: prefix candidate while exact group+disc exists")
        else:
            penalty -= 46.0
            reasons.append("penalized: non-group candidate while exact group+disc exists")
    elif family_pool_active and family_kind <= 0 and family_disc <= 0 and (prefix_kind > 0 or prefix_disc > 0 or prefix_only_kind > 0 or prefix_only_disc > 0):
        penalty -= 12.0
        reasons.append("penalized: family pool active")
    return penalty, reasons


def _slot_group_pool_flags(candidate_teachers: set[str], run_idx: dict[str, Any], group: str, family_key: str, disc: str, kind: str) -> tuple[bool, bool, bool]:
    """Определяет, есть ли для слота точный или семейственный пул кандидатов по группе."""
    exact_kind_pool = any(run_idx["group_kind"].get((teacher, group, disc, kind), 0.0) > 0 for teacher in candidate_teachers)
    exact_disc_pool = any(run_idx["group_disc"].get((teacher, group, disc), 0.0) > 0 for teacher in candidate_teachers)
    family_pool = False
    if family_key:
        family_pool = any(
            run_idx["family_kind"].get((teacher, family_key, disc, kind), 0.0) > 0
            or run_idx["family_disc"].get((teacher, family_key, disc), 0.0) > 0
            for teacher in candidate_teachers
        )
    return exact_kind_pool, exact_disc_pool, family_pool


def _build_run_indexes(run_atoms: pd.DataFrame) -> dict[str, Any]:
    """Готовит быстрые индексы по РУН для скоринга кандидатов."""
    group_links = build_teacher_group_links(run_atoms)
    skills = build_teacher_skills(run_atoms)
    disc_capacity = build_teacher_discipline_capacity(run_atoms)

    idx: dict[str, Any] = {
        "teacher_pool": sorted(set(run_atoms["Преподаватель"].dropna().astype(str))) if len(run_atoms) else [],
        "discs": sorted(set(_txt(x) for x in run_atoms.get("disc_key", []) if _txt(x))),
        "by_kind": {k: v.copy() for k, v in run_atoms.groupby("kind_norm")},
        "by_disc": {k: v.copy() for k, v in run_atoms.groupby("disc_key")},
        "skills_kind": {},
        "skills_disc": {},
        "group_kind": {},
        "group_disc": {},
        "family_kind": {},
        "family_disc": {},
        "prefix_kind": {},
        "prefix_disc": {},
        "prefix_only_kind": {},
        "prefix_only_disc": {},
        "disc_kind_group_count": {},
        "disc_kind_prefix_count": {},
        "disc_kind_family_count": {},
        "disc_kind_capacity_units": {},
        "disc_kind_hours": {},
    }

    for _, r in skills.iterrows():
        teacher = _txt(r.get("Преподаватель"))
        disc = _txt(r.get("disc_key"))
        kind = _txt(r.get("kind_norm"))
        weight = float(r.get("skill_weight", 0) or 0)
        idx["skills_kind"][(teacher, disc, kind)] = weight
        idx["skills_disc"][(teacher, disc)] = idx["skills_disc"].get((teacher, disc), 0.0) + weight

    for _, r in group_links.iterrows():
        teacher = _txt(r.get("Преподаватель"))
        group = _txt(r.get("Учебная группа"))
        disc = _txt(r.get("disc_key"))
        kind = _txt(r.get("kind_norm"))
        units = float(r.get("capacity_units", 0) or 0)
        gctx = extract_group_parts(group)
        fam = gctx.get("family_key", "")
        pyear = gctx.get("prefix_year", "")
        pprefix = gctx.get("group_prefix", "")

        idx["group_kind"][(teacher, group, disc, kind)] = idx["group_kind"].get((teacher, group, disc, kind), 0.0) + units
        idx["group_disc"][(teacher, group, disc)] = idx["group_disc"].get((teacher, group, disc), 0.0) + units
        if fam:
            idx["family_kind"][(teacher, fam, disc, kind)] = idx["family_kind"].get((teacher, fam, disc, kind), 0.0) + units
            idx["family_disc"][(teacher, fam, disc)] = idx["family_disc"].get((teacher, fam, disc), 0.0) + units
        if pyear:
            idx["prefix_kind"][(teacher, pyear, disc, kind)] = idx["prefix_kind"].get((teacher, pyear, disc, kind), 0.0) + units
            idx["prefix_disc"][(teacher, pyear, disc)] = idx["prefix_disc"].get((teacher, pyear, disc), 0.0) + units
        if pprefix:
            idx["prefix_only_kind"][(teacher, pprefix, disc, kind)] = idx["prefix_only_kind"].get((teacher, pprefix, disc, kind), 0.0) + units
            idx["prefix_only_disc"][(teacher, pprefix, disc)] = idx["prefix_only_disc"].get((teacher, pprefix, disc), 0.0) + units

    for _, r in disc_capacity.iterrows():
        teacher = _txt(r.get("Преподаватель"))
        disc = _txt(r.get("disc_key"))
        kind = _txt(r.get("kind_norm"))
        idx["disc_kind_group_count"][(teacher, disc, kind)] = int(r.get("confirmed_group_count", 0) or 0)
        idx["disc_kind_prefix_count"][(teacher, disc, kind)] = int(r.get("confirmed_prefix_count", 0) or 0)
        idx["disc_kind_family_count"][(teacher, disc, kind)] = int(r.get("confirmed_family_count", 0) or 0)
        idx["disc_kind_capacity_units"][(teacher, disc, kind)] = float(r.get("confirmed_capacity_units", 0.0) or 0.0)
        idx["disc_kind_hours"][(teacher, disc, kind)] = float(r.get("confirmed_hours_kind", 0.0) or 0.0)
    return idx


def _teacher_day_bonus(state: dict[str, Any], teacher: str, day: str, disc: str, family_key: str, prefix_year: str) -> tuple[float, list[str]]:
    """Даёт приоритет преподавателям, которые уже работают в этот день."""
    bonus = 0.0
    reasons: list[str] = []
    present_days = state.get("trusted_present_days", {}).get(teacher, set())
    day_slots = state.get("trusted_assigned_slots_by_day", {}).get(teacher, {}).get(day, set())
    same_day_discs = state.get("trusted_assigned_disc_by_day", {}).get(teacher, {}).get(day, set())
    same_day_families = state.get("trusted_assigned_family_by_day", {}).get(teacher, {}).get(day, set())
    same_day_prefix = state.get("trusted_assigned_prefix_year_by_day", {}).get(teacher, {}).get(day, set())

    if day and day in present_days:
        bonus += DAY_PRESENCE_BONUS + min(18.0, len(day_slots) * 6.0)
        reasons.append("trusted day presence")
        if disc and disc in same_day_discs:
            bonus += 18.0
            reasons.append("trusted same discipline on this day")
        if family_key and family_key in same_day_families:
            bonus += 10.0
            reasons.append("trusted same group family on this day")
        elif prefix_year and prefix_year in same_day_prefix:
            bonus += 6.0
            reasons.append("trusted same direction/year on this day")
    elif present_days:
        penalty = NEW_DAY_PENALTY + (10.0 if _is_foundation_disc(disc) else 0.0)
        bonus -= penalty
        reasons.append("opens new trusted day")
    else:
        bonus += 4.0
        reasons.append("first trusted teaching day")
    return bonus, reasons


def _capacity_bonus(teacher_state: dict[str, Any], teacher: str, kind: str) -> tuple[float, list[str]]:
    """Предпочитает кандидатов с остающейся нагрузкой и штрафует за сильную перегрузку."""
    bonus = 0.0
    reasons: list[str] = []
    cap_total = teacher_remaining_capacity(teacher_state, teacher)
    cap_kind = teacher_remaining_capacity(teacher_state, teacher, kind)

    if cap_total >= 2.0:
        bonus += 18.0
        reasons.append("healthy remaining load")
    elif cap_total > 0:
        bonus += 8.0 + cap_total * 4.0
        reasons.append("remaining load")
    else:
        bonus -= min(90.0, abs(cap_total) * 18.0)
        reasons.append("over capacity")

    if cap_kind >= 1.0:
        bonus += 12.0
        reasons.append("kind capacity available")
    elif cap_kind > 0:
        bonus += 5.0 + cap_kind * 4.0
    elif cap_kind < -0.25:
        bonus -= min(28.0, abs(cap_kind) * 10.0)
        reasons.append("kind over capacity")
    return bonus, reasons


def _current_disc_scope(teacher_state: dict[str, Any], teacher: str, disc: str, kind: str) -> dict[str, Any]:
    """Возвращает текущий объём уже назначенной предметной области преподавателя."""
    groups = {
        g for (g, d, k) in teacher_state.get("assigned_group_disc_kind", {}).get(teacher, set())
        if d == disc and k == kind and _txt(g)
    }
    prefixes = set()
    families = set()
    days = set()
    slot_count = 0
    for meta in teacher_state.get("busy_slots", {}).get(teacher, []):
        if _txt(meta.get("disc")) != disc or _txt(meta.get("kind")) != kind:
            continue
        slot_count += 1
        if _txt(meta.get("day")):
            days.add(_txt(meta.get("day")))
    for group in groups:
        gctx = extract_group_parts(group)
        if gctx.get("group_prefix", ""):
            prefixes.add(gctx.get("group_prefix", ""))
        if gctx.get("family_key", ""):
            families.add(gctx.get("family_key", ""))
    return {
        "groups": groups,
        "group_count": len(groups),
        "prefix_count": len(prefixes),
        "family_count": len(families),
        "day_count": len(days),
        "slot_count": slot_count,
    }


def _discipline_capacity_penalty(
    teacher_state: dict[str, Any],
    run_idx: dict[str, Any],
    *,
    teacher: str,
    disc: str,
    kind: str,
    group: str,
    group_prefix: str,
    family_key: str,
    admissibility_level: int,
    hint_match: bool,
) -> tuple[float, list[str], dict[str, float]]:
    """Штрафует чрезмерное расширение преподавателя по одной дисциплине относительно РУН."""
    reasons: list[str] = []
    penalty = 0.0
    scope = _current_disc_scope(teacher_state, teacher, disc, kind)

    confirmed_groups = int(run_idx["disc_kind_group_count"].get((teacher, disc, kind), 0) or 0)
    confirmed_prefixes = int(run_idx["disc_kind_prefix_count"].get((teacher, disc, kind), 0) or 0)
    confirmed_families = int(run_idx["disc_kind_family_count"].get((teacher, disc, kind), 0) or 0)
    confirmed_units = float(run_idx["disc_kind_capacity_units"].get((teacher, disc, kind), 0.0) or 0.0)

    new_group = bool(group and group not in scope["groups"])
    new_prefix = bool(group_prefix) and group_prefix not in {extract_group_parts(g).get("group_prefix", "") for g in scope["groups"]}
    new_family = bool(family_key) and family_key not in {extract_group_parts(g).get("family_key", "") for g in scope["groups"]}

    # мягкий базовый лимит: разрешаем чуть шире, чем прямо видно в РУН, но не бесконечно
    allowed_groups = confirmed_groups if confirmed_groups > 0 else (1 if _is_foundation_disc(disc) else 2)
    allowed_prefixes = confirmed_prefixes if confirmed_prefixes > 0 else (1 if _is_foundation_disc(disc) else 2)
    allowed_families = confirmed_families if confirmed_families > 0 else (1 if _is_foundation_disc(disc) else 2)
    allowed_slots = max(1, int(round(confirmed_units + 0.5))) if confirmed_units > 0 else (3 if _is_foundation_disc(disc) else 5)

    over_groups = max(0, scope["group_count"] - allowed_groups)
    over_prefixes = max(0, scope["prefix_count"] - allowed_prefixes)
    over_families = max(0, scope["family_count"] - allowed_families)
    over_slots = max(0, scope["slot_count"] - allowed_slots)

    if new_group and scope["group_count"] >= allowed_groups:
        step = scope["group_count"] - allowed_groups + 1
        penalty += (14.0 if _is_foundation_disc(disc) else 9.0) + 6.0 * step
        reasons.append("exceeds confirmed group span")
    elif new_group and scope["group_count"] + 1 > allowed_groups and not hint_match and admissibility_level < 3:
        penalty += 5.0
        reasons.append("pushes group span boundary")

    if new_prefix and scope["prefix_count"] >= allowed_prefixes:
        penalty += (10.0 if _is_foundation_disc(disc) else 6.0) + 4.0 * (scope["prefix_count"] - allowed_prefixes + 1)
        reasons.append("exceeds confirmed prefix span")

    if new_family and scope["family_count"] >= allowed_families and admissibility_level < 3:
        penalty += 6.0
        reasons.append("exceeds confirmed family span")

    if scope["slot_count"] >= allowed_slots and admissibility_level < 3:
        penalty += 4.0 + min(10.0, 2.0 * (scope["slot_count"] - allowed_slots + 1))
        reasons.append("exceeds confirmed discipline volume")

    if _is_foundation_disc(disc) and new_group and admissibility_level < 3 and not hint_match:
        penalty += 5.0
        reasons.append("foundation discipline new group")

    meta = {
        "disc_confirmed_group_count": float(confirmed_groups),
        "disc_current_group_count": float(scope["group_count"]),
        "disc_confirmed_prefix_count": float(confirmed_prefixes),
        "disc_current_prefix_count": float(scope["prefix_count"]),
        "disc_confirmed_slot_capacity": float(allowed_slots),
        "disc_current_slot_count": float(scope["slot_count"]),
        "discipline_capacity_penalty": float(penalty),
    }
    return penalty, reasons, meta


def _teacher_hint_candidates(run_atoms: pd.DataFrame, teacher_hint: str) -> set[str]:
    """Находит преподавателей по teacher hint из расписания."""
    hint_lastname = teacher_lastname(teacher_hint)
    if not teacher_hint or not hint_lastname or run_atoms is None or len(run_atoms) == 0:
        return set()
    return set(run_atoms.loc[run_atoms["teacher_lastname"] == hint_lastname, "Преподаватель"].astype(str).tolist())




def _is_stream_like_slot(group: str, slot: pd.Series | None = None) -> bool:
    """Определяет, что слот похож на потоковую/объединённую лекцию по нескольким группам."""
    g = _txt(group)
    if not g and slot is not None:
        g = _txt(slot.get("Учебная группа"))
    raw = " ".join([g, _txt(slot.get("source_text")) if slot is not None else "", _txt(slot.get("source_block")) if slot is not None else ""]).lower()
    if not raw:
        return False
    return any(sep in raw for sep in [",", "/", " пот", "поток"])


def _subject_peer_day_bias(
    teacher_state: dict[str, Any],
    *,
    teacher: str,
    disc: str,
    kind: str,
    day: str,
    current_new_day_penalty: float,
    strict_skill_teachers: set[str],
    hint_match: bool,
) -> tuple[float, list[str]]:
    """Сравнивает кандидата с другими преподавателями по той же дисциплине и дню.

    Если по этой дисциплине есть другие допустимые преподаватели, которые уже работают в нужный день
    или уже ведут эту дисциплину в этот день, кандидат, открывающий новый день, должен проигрывать сильнее.
    """
    reasons: list[str] = []
    bonus = 0.0
    if not day or not strict_skill_teachers:
        return bonus, reasons

    trusted_present = teacher_state.get("trusted_present_days", {})
    trusted_disc_by_day = teacher_state.get("trusted_assigned_disc_by_day", {})

    peers = [t for t in strict_skill_teachers if t and t != teacher]
    peers_present_same_day = [t for t in peers if day in trusted_present.get(t, set())]
    peers_same_disc_same_day = [
        t for t in peers if disc and disc in trusted_disc_by_day.get(t, {}).get(day, set())
    ]
    teacher_present_same_day = day in trusted_present.get(teacher, set())
    teacher_same_disc_same_day = disc and disc in trusted_disc_by_day.get(teacher, {}).get(day, set())

    if teacher_same_disc_same_day:
        bonus += 12.0
        reasons.append("subject peer: already teaches this disc in day")
    elif teacher_present_same_day and peers_present_same_day:
        bonus += 6.0
        reasons.append("subject peer: already in university this day")

    if current_new_day_penalty >= 0.99 and not hint_match:
        if peers_same_disc_same_day:
            bonus -= 22.0
            reasons.append("subject peer available in same day for same disc")
        elif peers_present_same_day:
            bonus -= 14.0
            reasons.append("subject peer already present in same day")
    return bonus, reasons


def _stream_lecture_bias(
    *,
    slot: pd.Series,
    kind: str,
    teacher: str,
    disc: str,
    exact_group_kind: float,
    exact_group_disc: float,
    skill_kind: float,
    strict_skill_teachers: set[str],
    teacher_state: dict[str, Any],
) -> tuple[float, list[str]]:
    """Усиливает точные/сильные кандидаты на потоковых лекциях и ослабляет слишком случайных."""
    reasons: list[str] = []
    bonus = 0.0
    if kind != "лек" or not _is_stream_like_slot(_txt(slot.get("Учебная группа")), slot):
        return bonus, reasons

    trusted_disc = teacher_state.get("trusted_assigned_disc_kind", {}).get(teacher, set())
    if exact_group_kind > 0 or exact_group_disc > 0:
        bonus += 12.0
        reasons.append("stream lecture exact match")
    elif skill_kind > 0:
        bonus += 8.0
        reasons.append("stream lecture strong skill")

    if (disc, kind) in trusted_disc:
        bonus += 8.0
        reasons.append("stream lecture trusted disc+kind")

    if strict_skill_teachers and teacher not in strict_skill_teachers:
        bonus -= 16.0
        reasons.append("stream lecture outside strict pool")
    return bonus, reasons


def _discipline_series_growth_penalty(
    teacher_state: dict[str, Any],
    *,
    teacher: str,
    disc: str,
    kind: str,
    group: str,
    group_prefix: str,
    family_key: str,
    exact_group_kind: float,
    exact_group_disc: float,
    hint_match: bool,
) -> tuple[float, list[str], dict[str, float]]:
    """Штрафует чрезмерное разрастание преподавателя по одной дисциплине на новые группы.

    Особенно важно для лабораторных и практик, когда первые удачные назначения не должны автоматически
    тянуть за собой почти всю серию соседних групп.
    """
    reasons: list[str] = []
    penalty = 0.0
    scope = _current_disc_scope(teacher_state, teacher, disc, kind)
    groups = scope.get("groups", set())
    seen_prefixes = {extract_group_parts(g).get("group_prefix", "") for g in groups if _txt(g)}
    seen_families = {extract_group_parts(g).get("family_key", "") for g in groups if _txt(g)}
    new_group = bool(group and group not in groups)
    new_prefix = bool(group_prefix and group_prefix not in seen_prefixes)
    new_family = bool(family_key and family_key not in seen_families)

    if not new_group:
        return penalty, reasons, {"disc_series_penalty": 0.0, "disc_known_group_count": float(scope.get("group_count", 0)), "disc_known_day_count": float(scope.get("day_count", 0))}

    group_count = int(scope.get("group_count", 0) or 0)
    day_count = int(scope.get("day_count", 0) or 0)
    if kind in {"лаб", "сем"}:
        if group_count >= 2 and exact_group_kind <= 0 and exact_group_disc <= 0 and not hint_match:
            penalty += 10.0 + 4.0 * max(0, group_count - 2)
            reasons.append("discipline series expansion on new group")
        if new_prefix and group_count >= 1 and not hint_match:
            penalty += 6.0
            reasons.append("new prefix in discipline series")
        if new_family and group_count >= 2 and not hint_match:
            penalty += 6.0
            reasons.append("new family in discipline series")
        if _is_foundation_disc(disc) and group_count >= 2 and not hint_match:
            penalty += 8.0
            reasons.append("foundation discipline series guard")
    elif kind == "лек":
        if _is_foundation_disc(disc) and new_prefix and day_count >= 2 and not hint_match:
            penalty += 4.0
            reasons.append("foundation lecture new prefix")

    meta = {
        "disc_series_penalty": float(penalty),
        "disc_known_group_count": float(group_count),
        "disc_known_day_count": float(day_count),
    }
    return penalty, reasons, meta

def _bounded_subject_fit(
    exact_group_kind: float,
    exact_group_disc: float,
    family_kind: float,
    family_disc: float,
    prefix_kind: float,
    prefix_disc: float,
    prefix_only_kind: float,
    prefix_only_disc: float,
    skill_kind: float,
    skill_disc: float,
    fuzzy_similarity: float,
    has_related_kind: bool,
    teaches_same_disc_kind: bool,
    teaches_same_disc: bool,
) -> float:
    """Формирует нормированный критерий предметной пригодности [0, 1]."""
    fit = 0.0
    if exact_group_kind > 0:
        fit = max(fit, 1.00)
    elif exact_group_disc > 0:
        fit = max(fit, 0.93)
    elif family_kind > 0:
        fit = max(fit, 0.86)
    elif family_disc > 0:
        fit = max(fit, 0.76)
    elif prefix_kind > 0:
        fit = max(fit, 0.66)
    elif prefix_only_kind > 0:
        fit = max(fit, 0.62)
    elif prefix_disc > 0:
        fit = max(fit, 0.58)
    elif prefix_only_disc > 0:
        fit = max(fit, 0.52)

    if skill_kind > 0:
        fit = max(fit, clamp01(0.58 + skill_kind / 120.0))
    elif skill_disc > 0:
        fit = max(fit, clamp01(0.42 + skill_disc / 180.0))

    if fuzzy_similarity >= 0.45:
        fit = max(fit, clamp01(0.34 + 0.36 * fuzzy_similarity))
    if has_related_kind:
        fit = max(fit, 0.40)
    if teaches_same_disc_kind:
        fit = clamp01(fit + 0.08)
    elif teaches_same_disc:
        fit = clamp01(fit + 0.05)
    return clamp01(fit)


def _bounded_day_fit(state: dict[str, Any], teacher: str, day: str, disc: str, family_key: str, prefix_year: str) -> tuple[float, float]:
    """Возвращает нормированный критерий по присутствию в дне и penalty за новый день."""
    present_days = state.get("trusted_present_days", {}).get(teacher, set())
    day_slots = state.get("trusted_assigned_slots_by_day", {}).get(teacher, {}).get(day, set())
    same_day_discs = state.get("trusted_assigned_disc_by_day", {}).get(teacher, {}).get(day, set())
    same_day_families = state.get("trusted_assigned_family_by_day", {}).get(teacher, {}).get(day, set())
    same_day_prefix = state.get("trusted_assigned_prefix_year_by_day", {}).get(teacher, {}).get(day, set())

    if not present_days:
        return 0.55, 0.0
    if day and day in present_days:
        fit = 0.78 + min(0.12, len(day_slots) * 0.04)
        if disc and disc in same_day_discs:
            fit += 0.08
        if family_key and family_key in same_day_families:
            fit += 0.06
        elif prefix_year and prefix_year in same_day_prefix:
            fit += 0.03
        return clamp01(fit), 0.0
    return (0.12 if _is_foundation_disc(disc) else 0.20), 1.0


def _bounded_load_fit(state: dict[str, Any], teacher: str, kind: str) -> float:
    """Возвращает нормированный критерий по остаточной нагрузке [0, 1]."""
    remaining_total = float(state.get("remaining_total", {}).get(teacher, 0.0) or 0.0)
    total_capacity = float(state.get("capacity_total", {}).get(teacher, 0.0) or 0.0)
    remaining_kind = float(state.get("remaining_by_kind", {}).get(teacher, {}).get(kind, 0.0) or 0.0)
    kind_capacity = float(state.get("capacity_by_kind", {}).get(teacher, {}).get(kind, 0.0) or 0.0)

    total_ratio = clamp01(remaining_total / total_capacity) if total_capacity > 0 else clamp01(1.0 if remaining_total > 0 else 0.0)
    kind_ratio = clamp01(remaining_kind / kind_capacity) if kind_capacity > 0 else clamp01(total_ratio)

    fit = 0.65 * total_ratio + 0.35 * kind_ratio
    if remaining_total <= 0:
        fit *= 0.25
    elif remaining_total < 1.0:
        fit *= 0.80
    if remaining_kind < -0.25:
        fit *= 0.70
    return clamp01(fit)


def _bounded_continuity_fit(
    state: dict[str, Any],
    teacher: str,
    day: str,
    disc: str,
    kind: str,
    group: str,
    family_key: str,
    prefix_year: str,
    disc_kind_teachers: dict,
    disc_teachers: dict,
    teacher_days: dict,
) -> float:
    """Оценивает непрерывность назначений и наследование ранее найденных связей."""
    fit = 0.0
    if teacher in disc_kind_teachers.get((disc, kind), set()):
        fit = max(fit, 0.74)
    elif teacher in disc_teachers.get(disc, set()):
        fit = max(fit, 0.58)
    if disc in teacher_days.get((teacher, day), set()):
        fit = max(fit, 0.80)

    assigned_disc_kind = state.get("trusted_assigned_disc_kind", {}).get(teacher, set())
    assigned_group_disc_kind = state.get("trusted_assigned_group_disc_kind", {}).get(teacher, set())
    assigned_family_disc_kind = state.get("trusted_assigned_family_disc_kind", {}).get(teacher, set())
    assigned_prefix_year_disc_kind = state.get("trusted_assigned_prefix_year_disc_kind", {}).get(teacher, set())
    if (disc, kind) in assigned_disc_kind:
        fit = max(fit, 0.64)
    if group and (group, disc, kind) in assigned_group_disc_kind:
        fit = max(fit, 0.94)
    elif family_key and (family_key, disc, kind) in assigned_family_disc_kind:
        fit = max(fit, 0.62)
    elif prefix_year and (prefix_year, disc, kind) in assigned_prefix_year_disc_kind:
        fit = max(fit, 0.48)
    return clamp01(fit)


def _rule_matches_slot(when: dict, slot: pd.Series) -> bool:
    """Проверяет, подходит ли пользовательское правило к конкретному слоту."""
    group = _txt(slot.get("Учебная группа"))
    disc = _txt(slot.get("disc_key"))
    kind = _txt(slot.get("Вид_занятия_норм"))
    day = _txt(slot.get("День недели"))
    week_type = _txt(slot.get("week_type"))
    group_prefix = extract_group_parts(group).get("group_prefix", "")
    if when.get("group") and when.get("group") != group:
        return False
    if when.get("group_prefix") and when.get("group_prefix") != group_prefix:
        return False
    if when.get("disc_key") and when.get("disc_key") != disc:
        return False
    if when.get("kind") and when.get("kind") != kind:
        return False
    if when.get("day") and when.get("day") != day:
        return False
    if when.get("week_type") and when.get("week_type") != week_type:
        return False
    return True


def build_slot_candidates(
    unmatched_slots: pd.DataFrame,
    run_atoms: pd.DataFrame,
    teacher_state: dict,
    locked_assignments: pd.DataFrame | None = None,
    mappings: dict | None = None,
) -> pd.DataFrame:
    """Формирует и оценивает список кандидатов для каждого нераспределённого слота."""
    if unmatched_slots is None or len(unmatched_slots) == 0:
        return pd.DataFrame(columns=["slot_id", "teacher", "score", "reason"])
    if run_atoms is None or len(run_atoms) == 0:
        return pd.DataFrame(columns=["slot_id", "teacher", "score", "reason"])

    (
        disc_kind_teachers,
        disc_teachers,
        teacher_days,
        family_disc_kind_teachers,
        family_disc_teachers,
        prefix_year_disc_kind_teachers,
    ) = _prep_assigned_index(locked_assignments)
    run_idx = _build_run_indexes(run_atoms)

    prefer_rules = []
    ban_rules = []
    if isinstance(mappings, dict):
        for rule in mappings.get("rules", []):
            mode = str(rule.get("mode") or "force_teacher")
            if mode in {"prefer_teacher", "prefer_teacher_scope"}:
                prefer_rules.append(rule)
            elif mode in {"ban_teacher", "ban_teacher_scope"}:
                ban_rules.append(rule)

    all_rows: list[dict[str, Any]] = []
    for _, slot in unmatched_slots.iterrows():
        kind = _txt(slot.get("Вид_занятия_норм"))
        group = _txt(slot.get("Учебная группа"))
        disc = _txt(slot.get("disc_key"))
        day = _txt(slot.get("День недели"))
        pair = slot.get("Пара")
        time = _txt(slot.get("Время"))
        room = _txt(slot.get("Аудитория"))
        teacher_hint = _txt(slot.get("teacher_hint"))
        hint_lastname = teacher_lastname(teacher_hint)
        gctx = _slot_group_context(slot)
        family_key = gctx.get("family_key", "")
        prefix_year = gctx.get("prefix_year", "")
        group_prefix = gctx.get("group_prefix", "")
        subgroup = _txt(slot.get("subgroup"))
        strict_skill_teachers = {t for (t, d, k), _w in run_idx["skills_kind"].items() if d == disc and k == kind}

        candidate_teachers: set[str] = set()
        candidate_teachers.update(family_disc_kind_teachers.get((family_key, disc, kind), set()))
        candidate_teachers.update(family_disc_teachers.get((family_key, disc), set()))
        candidate_teachers.update(prefix_year_disc_kind_teachers.get((prefix_year, disc, kind), set()))
        if group_prefix:
            candidate_teachers.update({t for (t, p, d, k), _w in run_idx["prefix_only_kind"].items() if p == group_prefix and d == disc and k == kind})
            candidate_teachers.update({t for (t, p, d), _w in run_idx["prefix_only_disc"].items() if p == group_prefix and d == disc})
        candidate_teachers.update(disc_kind_teachers.get((disc, kind), set()))
        candidate_teachers.update(disc_teachers.get(disc, set()))
        candidate_teachers.update(_teacher_hint_candidates(run_atoms, teacher_hint))

        if disc in run_idx["by_disc"]:
            candidate_teachers.update(set(run_idx["by_disc"][disc]["Преподаватель"].astype(str).tolist()))
        if kind in run_idx["by_kind"]:
            pool_same_kind = run_idx["by_kind"][kind]
            candidate_teachers.update(set(pool_same_kind.loc[pool_same_kind["Учебная группа"] == group, "Преподаватель"].astype(str).tolist()))
            candidate_teachers.update(set(pool_same_kind.loc[pool_same_kind["disc_key"] == disc, "Преподаватель"].astype(str).tolist()))
        for rk in _related_kinds(kind):
            rpool = run_idx["by_kind"].get(rk)
            if rpool is None or len(rpool) == 0:
                continue
            candidate_teachers.update(set(rpool.loc[rpool["disc_key"] == disc, "Преподаватель"].astype(str).tolist()))
            candidate_teachers.update(set(rpool.loc[rpool["Учебная группа"] == group, "Преподаватель"].astype(str).tolist()))

        best_disc, sim = best_disc_match(disc, run_idx["discs"])
        if best_disc and sim >= 0.45 and best_disc in run_idx["by_disc"]:
            candidate_teachers.update(set(run_idx["by_disc"][best_disc]["Преподаватель"].astype(str).tolist()))

        exact_kind_pool_active, exact_disc_pool_active, family_pool_active = _slot_group_pool_flags(
            candidate_teachers, run_idx, group, family_key, disc, kind
        )

        slot_rows: list[dict[str, Any]] = []
        slot_max_level = 0
        for teacher in sorted(candidate_teachers):
            reasons: list[str] = []
            score = 0.0

            exact_group_kind = run_idx["group_kind"].get((teacher, group, disc, kind), 0.0)
            exact_group_disc = run_idx["group_disc"].get((teacher, group, disc), 0.0)
            family_kind = run_idx["family_kind"].get((teacher, family_key, disc, kind), 0.0) if family_key else 0.0
            family_disc = run_idx["family_disc"].get((teacher, family_key, disc), 0.0) if family_key else 0.0
            prefix_kind = run_idx["prefix_kind"].get((teacher, prefix_year, disc, kind), 0.0) if prefix_year else 0.0
            prefix_disc = run_idx["prefix_disc"].get((teacher, prefix_year, disc), 0.0) if prefix_year else 0.0
            prefix_only_kind = run_idx["prefix_only_kind"].get((teacher, group_prefix, disc, kind), 0.0) if group_prefix else 0.0
            prefix_only_disc = run_idx["prefix_only_disc"].get((teacher, group_prefix, disc), 0.0) if group_prefix else 0.0
            skill_kind = run_idx["skills_kind"].get((teacher, disc, kind), 0.0)
            skill_disc = run_idx["skills_disc"].get((teacher, disc), 0.0)

            if teacher in family_disc_kind_teachers.get((family_key, disc, kind), set()):
                score = max(score, 132.0)
                reasons.append("seeded from family assignment")
            if teacher in family_disc_teachers.get((family_key, disc), set()):
                score = max(score, 118.0)
                reasons.append("seeded from family discipline assignment")
            if teacher in prefix_year_disc_kind_teachers.get((prefix_year, disc, kind), set()):
                score = max(score, 104.0)
                reasons.append("seeded from direction/year assignment")
            if teacher in disc_kind_teachers.get((disc, kind), set()):
                score = max(score, 96.0)
                reasons.append("seeded from assigned same disc+kind")
            if teacher in disc_teachers.get(disc, set()):
                score = max(score, 84.0)
                reasons.append("seeded from assigned same discipline")
            if teacher_hint and hint_lastname and hint_lastname == teacher_lastname(teacher):
                score = max(score, 138.0)
                reasons.append("seeded from teacher hint")

            if exact_group_kind > 0:
                score += 150.0 + min(12.0, exact_group_kind * 4.0)
                reasons.append("exact group+disc+kind")
            elif exact_group_disc > 0:
                score += 128.0 + min(10.0, exact_group_disc * 3.0)
                reasons.append("exact group+discipline")
            elif family_kind > 0:
                score += 116.0 + min(14.0, family_kind * 4.0)
                reasons.append("same family+disc+kind in RUN")
            elif family_disc > 0:
                score += 98.0 + min(12.0, family_disc * 3.0)
                reasons.append("same family+discipline in RUN")
            elif prefix_kind > 0:
                score += 84.0 + min(10.0, prefix_kind * 3.0)
                reasons.append("same direction/year+disc+kind in RUN")
            elif prefix_disc > 0:
                score += 72.0 + min(8.0, prefix_disc * 2.5)
                reasons.append("same direction/year+discipline in RUN")
            elif prefix_only_kind > 0:
                score += 88.0 + min(10.0, prefix_only_kind * 3.0)
                reasons.append("same group prefix+disc+kind in RUN")
            elif prefix_only_disc > 0:
                score += 66.0 + min(8.0, prefix_only_disc * 2.5)
                reasons.append("same group prefix+discipline in RUN")

            if skill_kind > 0:
                score += min(22.0, 8.0 + skill_kind / 5.0)
                reasons.append("teacher can teach this disc+kind")
            elif skill_disc > 0:
                score += min(18.0, 6.0 + skill_disc / 7.0)
                reasons.append("teacher can teach this discipline")

            fuzzy_weight = 0.0
            if best_disc and sim >= 0.45:
                fuzzy_weight = run_idx["skills_disc"].get((teacher, best_disc), 0.0)
                if best_disc == disc and fuzzy_weight > 0:
                    score += 32.0
                    reasons.append("same discipline in RUN")
                elif fuzzy_weight > 0:
                    score += 24.0 + 26.0 * sim
                    reasons.append(f"fuzzy discipline:{best_disc}:{sim:.2f}")

            related_kind_match = False
            for rk in _related_kinds(kind):
                rel_weight = run_idx["skills_kind"].get((teacher, disc, rk), 0.0)
                if rel_weight > 0:
                    score += min(12.0, 5.0 + rel_weight / 10.0)
                    reasons.append(f"related kind:{rk}")
                    related_kind_match = True
                    break

            hint_match = bool(teacher_hint and hint_lastname and hint_lastname == teacher_lastname(teacher))
            admissibility_level = _admissibility_level(
                exact_group_kind=exact_group_kind,
                exact_group_disc=exact_group_disc,
                family_kind=family_kind,
                family_disc=family_disc,
                prefix_kind=prefix_kind,
                prefix_disc=prefix_disc,
                prefix_only_kind=prefix_only_kind,
                prefix_only_disc=prefix_only_disc,
                skill_kind=skill_kind,
                skill_disc=skill_disc,
                fuzzy_similarity=sim if fuzzy_weight > 0 else 0.0,
                has_related_kind=related_kind_match,
                hint_match=hint_match,
                teaches_same_disc_kind=teacher in disc_kind_teachers.get((disc, kind), set()),
                teaches_same_disc=teacher in disc_teachers.get(disc, set()),
            )
            if admissibility_level == 0:
                continue
            scope_penalty, scope_reason = _narrow_disc_kind_guard(
                disc=disc,
                kind=kind,
                teacher=teacher,
                strict_skill_teachers=strict_skill_teachers,
                exact_group_kind=exact_group_kind,
                family_kind=family_kind,
                prefix_kind=prefix_kind,
                prefix_only_kind=prefix_only_kind,
                hint_match=hint_match,
            )
            if scope_penalty:
                score -= scope_penalty
            if scope_reason:
                reasons.append(scope_reason)
            slot_max_level = max(slot_max_level, admissibility_level)

            group_penalty, group_penalty_reasons = _group_pool_penalty(
                exact_group_kind=exact_group_kind,
                exact_group_disc=exact_group_disc,
                family_kind=family_kind,
                family_disc=family_disc,
                prefix_kind=prefix_kind,
                prefix_disc=prefix_disc,
                prefix_only_kind=prefix_only_kind,
                prefix_only_disc=prefix_only_disc,
                exact_kind_pool_active=exact_kind_pool_active,
                exact_disc_pool_active=exact_disc_pool_active,
                family_pool_active=family_pool_active,
            )
            if group_penalty:
                score += group_penalty
                reasons.extend(group_penalty_reasons)

            disc_cap_penalty, disc_cap_reasons, disc_cap_meta = _discipline_capacity_penalty(
                teacher_state,
                run_idx,
                teacher=teacher,
                disc=disc,
                kind=kind,
                group=group,
                group_prefix=group_prefix,
                family_key=family_key,
                admissibility_level=admissibility_level,
                hint_match=hint_match,
            )
            if disc_cap_penalty:
                score -= disc_cap_penalty
                reasons.extend(disc_cap_reasons)

            if teacher in disc_kind_teachers.get((disc, kind), set()):
                score += 24.0
                reasons.append("already teaches same disc+kind")
            elif teacher in disc_teachers.get(disc, set()):
                score += 16.0
                reasons.append("already teaches same discipline")
            if disc in teacher_days.get((teacher, day), set()):
                score += 18.0
                reasons.append("same day same discipline")
            if hint_match:
                score += TEACHER_HINT_STRONG_BONUS
                reasons.append("teacher hint from schedule")

            day_bonus, day_reasons = _teacher_day_bonus(teacher_state, teacher, day, disc, family_key, prefix_year)
            score += day_bonus
            reasons.extend(day_reasons)

            capacity_bonus, capacity_reasons = _capacity_bonus(teacher_state, teacher, kind)
            score += capacity_bonus
            reasons.extend(capacity_reasons)

            dyn_bonus, dyn_reasons = _state_family_bonus(teacher_state, teacher, slot, disc, kind)
            if dyn_bonus:
                score += dyn_bonus
                reasons.extend(dyn_reasons)

            if not teacher_is_available(teacher_state, teacher, day, pair, time, disc=disc, kind=kind, room=room, group=group, week_type=_txt(slot.get("week_type"))):
                continue

            prefer_hit = False
            for rule in prefer_rules:
                when = rule.get("when", {})
                assign = rule.get("assign", {})
                if assign.get("teacher") == teacher and _rule_matches_slot(when, slot):
                    score += PREFER_RULE_BONUS
                    prefer_hit = True
                    reasons.append("prefer rule")
            banned = False
            for rule in ban_rules:
                when = rule.get("when", {})
                assign = rule.get("assign", {})
                if assign.get("teacher") == teacher and _rule_matches_slot(when, slot):
                    banned = True
                    break
            if banned:
                continue

            subject_fit = _bounded_subject_fit(
                exact_group_kind=exact_group_kind,
                exact_group_disc=exact_group_disc,
                family_kind=family_kind,
                family_disc=family_disc,
                prefix_kind=prefix_kind,
                prefix_disc=prefix_disc,
                prefix_only_kind=prefix_only_kind,
                prefix_only_disc=prefix_only_disc,
                skill_kind=skill_kind,
                skill_disc=skill_disc,
                fuzzy_similarity=sim if fuzzy_weight > 0 else 0.0,
                has_related_kind=related_kind_match,
                teaches_same_disc_kind=teacher in disc_kind_teachers.get((disc, kind), set()),
                teaches_same_disc=teacher in disc_teachers.get(disc, set()),
            )
            group_fit = _bounded_group_fit(
                exact_group_kind=exact_group_kind,
                exact_group_disc=exact_group_disc,
                family_kind=family_kind,
                family_disc=family_disc,
                prefix_kind=prefix_kind,
                prefix_disc=prefix_disc,
                prefix_only_kind=prefix_only_kind,
                prefix_only_disc=prefix_only_disc,
                subgroup=subgroup,
            )
            day_fit, new_day_penalty = _bounded_day_fit(teacher_state, teacher, day, disc, family_key, prefix_year)
            load_fit = _bounded_load_fit(teacher_state, teacher, kind)
            peer_day_bonus, peer_day_reasons = _subject_peer_day_bias(
                teacher_state,
                teacher=teacher,
                disc=disc,
                kind=kind,
                day=day,
                current_new_day_penalty=new_day_penalty,
                strict_skill_teachers=strict_skill_teachers,
                hint_match=hint_match,
            )
            for _r in peer_day_reasons:
                if _r not in reasons:
                    reasons.append(_r)
            score += peer_day_bonus
            stream_bonus, stream_reasons = _stream_lecture_bias(
                slot=slot,
                kind=kind,
                teacher=teacher,
                disc=disc,
                exact_group_kind=exact_group_kind,
                exact_group_disc=exact_group_disc,
                skill_kind=skill_kind,
                strict_skill_teachers=strict_skill_teachers,
                teacher_state=teacher_state,
            )
            for _r in stream_reasons:
                if _r not in reasons:
                    reasons.append(_r)
            score += stream_bonus
            disc_series_penalty, disc_series_reasons, disc_series_meta = _discipline_series_growth_penalty(
                teacher_state,
                teacher=teacher,
                disc=disc,
                kind=kind,
                group=group,
                group_prefix=group_prefix,
                family_key=family_key,
                exact_group_kind=exact_group_kind,
                exact_group_disc=exact_group_disc,
                hint_match=hint_match,
            )
            for _r in disc_series_reasons:
                if _r not in reasons:
                    reasons.append(_r)
            score -= disc_series_penalty
            continuity_fit = _bounded_continuity_fit(
                teacher_state,
                teacher,
                day,
                disc,
                kind,
                group,
                family_key,
                prefix_year,
                disc_kind_teachers,
                disc_teachers,
                teacher_days,
            )
            if continuity_fit >= 0.85 and admissibility_level < 3:
                continuity_fit = max(0.40, continuity_fit - 0.18)
            elif continuity_fit >= 0.60 and admissibility_level < 3:
                continuity_fit = max(0.34, continuity_fit - 0.10)
            hint_fit = 1.0 if hint_match else 0.0
            if prefer_hit:
                hint_fit = max(hint_fit, 0.75)

            if score > 0:
                slot_rows.append(
                    {
                        "slot_id": slot.get("slot_id"),
                        "teacher": teacher,
                        "heuristic_score": round(float(score), 3),
                        "admissibility_level": int(admissibility_level),
                        "subject_fit": round(float(subject_fit), 4),
                        "group_fit": round(float(group_fit), 4),
                        "day_fit": round(float(day_fit), 4),
                        "load_fit": round(float(load_fit), 4),
                        "continuity_fit": round(float(continuity_fit), 4),
                        "hint_fit": round(float(hint_fit), 4),
                        "new_day_penalty": round(float(new_day_penalty), 4),
                        "discipline_capacity_penalty": round(float(disc_cap_meta.get("discipline_capacity_penalty", 0.0)), 4),
                        "disc_confirmed_group_count": round(float(disc_cap_meta.get("disc_confirmed_group_count", 0.0)), 2),
                        "disc_current_group_count": round(float(disc_cap_meta.get("disc_current_group_count", 0.0)), 2),
                        "disc_confirmed_prefix_count": round(float(disc_cap_meta.get("disc_confirmed_prefix_count", 0.0)), 2),
                        "disc_current_prefix_count": round(float(disc_cap_meta.get("disc_current_prefix_count", 0.0)), 2),
                        "disc_confirmed_slot_capacity": round(float(disc_cap_meta.get("disc_confirmed_slot_capacity", 0.0)), 2),
                        "disc_current_slot_count": round(float(disc_cap_meta.get("disc_current_slot_count", 0.0)), 2),
                        "disc_series_penalty": round(float(disc_series_meta.get("disc_series_penalty", 0.0)), 4),
                        "disc_known_group_count": round(float(disc_series_meta.get("disc_known_group_count", 0.0)), 2),
                        "disc_known_day_count": round(float(disc_series_meta.get("disc_known_day_count", 0.0)), 2),
                        "reason": " | ".join(dict.fromkeys(reasons)),
                    }
                )

        if not slot_rows:
            continue

        slot_df = pd.DataFrame(slot_rows)
        if slot_max_level >= 3:
            slot_df["heuristic_score"] = slot_df.apply(
                lambda r: float(r["heuristic_score"]) - (55.0 if int(r["admissibility_level"]) < 3 else 0.0),
                axis=1,
            )
            slot_df["reason"] = slot_df.apply(
                lambda r: f"{r['reason']} | strict pool active" if int(r["admissibility_level"]) >= 3 else f"{r['reason']} | penalized non-strict candidate",
                axis=1,
            )
        elif slot_max_level >= 2:
            slot_df["heuristic_score"] = slot_df.apply(
                lambda r: float(r["heuristic_score"]) - (28.0 if int(r["admissibility_level"]) < 2 else 0.0),
                axis=1,
            )
            slot_df["reason"] = slot_df.apply(
                lambda r: f"{r['reason']} | limited pool active" if int(r["admissibility_level"]) >= 2 else f"{r['reason']} | penalized fallback candidate",
                axis=1,
            )
        slot_df["weighted_score"] = slot_df.apply(lambda r: weighted_sum_score(r, CANDIDATE_CRITERIA), axis=1)
        slot_df["topsis_score"] = topsis_scores(slot_df, CANDIDATE_CRITERIA)
        slot_df["math_score"] = (0.65 * slot_df["topsis_score"] + 0.35 * slot_df["weighted_score"]).clip(lower=0.0, upper=1.0)
        weak_gap_penalty = slot_df["admissibility_level"].apply(lambda lvl: 0.0 if int(lvl) >= 3 else 8.0)
        weak_foundation_penalty = slot_df.apply(lambda r: 14.0 if _is_foundation_disc(disc) and float(r.get("new_day_penalty", 0.0) or 0.0) >= 0.99 and int(r.get("admissibility_level", 0) or 0) < 3 else 0.0, axis=1)
        limited_foundation_penalty = slot_df.apply(lambda r: 18.0 if _is_foundation_disc(disc) and int(r.get("admissibility_level", 0) or 0) < 3 else 0.0, axis=1)
        slot_df["score"] = (slot_df["heuristic_score"] + 36.0 * slot_df["math_score"] - weak_gap_penalty - weak_foundation_penalty - limited_foundation_penalty).round(3)
        slot_df["reason"] = slot_df.apply(
            lambda r: (
                f"{r['reason']} | math weighted={float(r['weighted_score']):.3f}"
                f" | math topsis={float(r['topsis_score']):.3f}"
            ),
            axis=1,
        )
        all_rows.extend(slot_df.to_dict(orient="records"))

    cand_df = pd.DataFrame(all_rows)
    if len(cand_df) == 0:
        return cand_df
    cand_df = cand_df.sort_values(["slot_id", "score", "heuristic_score"], ascending=[True, False, False]).drop_duplicates(["slot_id", "teacher"], keep="first")
    return cand_df


def _safe_recovery_candidate(slot: pd.Series, cand: pd.Series) -> bool:
    """Проверяет, можно ли использовать кандидата во втором, более мягком, но безопасном проходе."""
    level = int(cand.get("admissibility_level", 0) or 0)
    if level < 2:
        return False
    math_score = float(cand.get("math_score", 0.0) or 0.0)
    group_fit = float(cand.get("group_fit", 0.0) or 0.0)
    new_day_penalty = float(cand.get("new_day_penalty", 0.0) or 0.0)
    hint_fit = float(cand.get("hint_fit", 0.0) or 0.0)
    continuity_fit = float(cand.get("continuity_fit", 0.0) or 0.0)
    score_gap = float(cand.get("score_gap", 999.0) or 999.0)
    disc = _txt(slot.get("disc_key"))

    lecture_kind = _txt(slot.get("Вид_занятия_норм")) == "лек"
    if math_score < RECOVERY_MIN_MATH_SCORE:
        return False
    if score_gap < 3.0 and hint_fit < 0.99 and continuity_fit < 0.72:
        return False
    if _is_foundation_disc(disc):
        if new_day_penalty >= 0.99 and hint_fit < 0.99 and continuity_fit < 0.86:
            return False
        if group_fit < 0.58 and hint_fit < 0.99:
            # для потоковых/лекционных базовых дисциплин допускаем второй проход,
            # если преподаватель уже присутствует в этот день и имеет уверенную мат. оценку
            if not (lecture_kind and new_day_penalty < 0.5 and math_score >= 0.50 and continuity_fit >= 0.0):
                return False
    else:
        if group_fit < RECOVERY_MIN_GROUP_FIT and hint_fit < 0.99:
            return False
        if new_day_penalty >= 0.99 and hint_fit < 0.99 and continuity_fit < 0.68:
            return False
    return True


def allocate_unmatched_safe_recovery(
    unmatched_slots: pd.DataFrame,
    candidates: pd.DataFrame,
    teacher_state: dict,
) -> pd.DataFrame:
    """Второй проход: осторожно возвращает часть покрытия только для безопасных ограниченных кандидатов."""
    if unmatched_slots is None or len(unmatched_slots) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "score", "reason", "assign_type"])
    if candidates is None or len(candidates) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "score", "reason", "assign_type"])

    chosen = []
    slots = unmatched_slots.copy()
    for _, slot in slots.iterrows():
        slot_id = slot["slot_id"]
        opts = candidates[candidates["slot_id"] == slot_id].sort_values(["score", "math_score", "heuristic_score"], ascending=False)
        picked = None
        for _, cand in opts.iterrows():
            if not _safe_recovery_candidate(slot, cand):
                continue
            teacher = cand["teacher"]
            if not teacher_is_available(teacher_state, teacher, slot.get("День недели"), slot.get("Пара"), slot.get("Время"), disc=_txt(slot.get("disc_key")), kind=_txt(slot.get("Вид_занятия_норм")), room=_txt(slot.get("Аудитория")), group=_txt(slot.get("Учебная группа")), week_type=_txt(slot.get("week_type"))):
                continue
            effective_score = float(cand["score"]) - 4.0
            if effective_score < RECOVERY_MIN_SCORE:
                continue
            picked = cand.copy()
            picked["score"] = effective_score
            picked["reason"] = f"{cand.get('reason','')} | safe recovery pass".strip(' |')
            break

        if picked is None:
            continue

        rec = slot.to_dict()
        rec["Преподаватель"] = picked["teacher"]
        rec["score"] = float(picked["score"])
        rec["reason"] = picked["reason"]
        rec["assign_type"] = "auto_recovered"
        rec["_trusted_propagation"] = False
        rec["confidence"] = max(0.22, min(0.72, float(picked.get("math_score", 0.0) or 0.0) * 0.85))
        chosen.append(rec)
        apply_assignment_to_state(teacher_state, rec)

    return pd.DataFrame(chosen)


def allocate_unmatched_greedy(
    unmatched_slots: pd.DataFrame,
    candidates: pd.DataFrame,
    teacher_state: dict,
) -> pd.DataFrame:
    """Жадно распределяет оставшиеся слоты по лучшим допустимым кандидатам."""
    if unmatched_slots is None or len(unmatched_slots) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "score", "reason", "assign_type"])
    if candidates is None or len(candidates) == 0:
        return pd.DataFrame(columns=["slot_id", "Преподаватель", "score", "reason", "assign_type"])

    cand_counts = candidates.groupby("slot_id", as_index=False).size().rename(columns={"size": "cand_count"})
    slots = unmatched_slots.merge(cand_counts, on="slot_id", how="left")
    slots["cand_count"] = slots["cand_count"].fillna(0).astype(int)

    best_scores = candidates.groupby("slot_id", as_index=False).agg(best_score=("score", "max"))
    slots = slots.merge(best_scores, on="slot_id", how="left")
    slots["best_score"] = slots["best_score"].fillna(-1)
    slots = slots.sort_values(["cand_count", "best_score"], ascending=[True, False])

    chosen = []
    strict_min_score = 62.0
    fallback_min_score = 38.0
    for _, slot in slots.iterrows():
        slot_id = slot["slot_id"]
        opts = candidates[candidates["slot_id"] == slot_id].sort_values(["score", "math_score", "heuristic_score"], ascending=False)
        picked = None

        for threshold in (strict_min_score, fallback_min_score):
            for _, cand in opts.iterrows():
                teacher = cand["teacher"]
                if not teacher_is_available(teacher_state, teacher, slot.get("День недели"), slot.get("Пара"), slot.get("Время"), disc=_txt(slot.get("disc_key")), kind=_txt(slot.get("Вид_занятия_норм")), room=_txt(slot.get("Аудитория")), group=_txt(slot.get("Учебная группа")), week_type=_txt(slot.get("week_type"))):
                    continue
                extra_bonus, extra_reasons = _state_family_bonus(teacher_state, teacher, slot, _txt(slot.get("disc_key")), _txt(slot.get("Вид_занятия_норм")))
                effective_score = float(cand["score"]) + extra_bonus
                if float(cand.get("score_gap", 999.0) or 999.0) < 6.0 and int(cand.get("admissibility_level", 0) or 0) < 3:
                    effective_score -= 10.0
                if _is_foundation_disc(_txt(slot.get("disc_key"))) and float(cand.get("new_day_penalty", 0.0) or 0.0) >= 0.99 and int(cand.get("admissibility_level", 0) or 0) < 3:
                    effective_score -= 12.0
                if effective_score < threshold:
                    continue
                picked = cand.copy()
                picked["score"] = effective_score
                propagation_trusted = (
                    int(cand.get("admissibility_level", 0) or 0) >= 3
                    and float(cand.get("math_score", 0.0) or 0.0) >= TRUST_MIN_MATH_SCORE
                    and float(cand.get("score_gap", 0.0) or 0.0) >= TRUST_MIN_SCORE_GAP
                    and float(cand.get("subject_fit", 0.0) or 0.0) >= TRUST_MIN_SUBJECT_FIT
                )
                picked["_trusted_propagation"] = propagation_trusted
                if extra_reasons:
                    tail = " | ".join(dict.fromkeys(extra_reasons))
                    prev_reason = cand.get("reason", "")
                    picked["reason"] = f"{prev_reason} | {tail}" if prev_reason else tail
                if not propagation_trusted:
                    prev_reason = picked.get("reason", "")
                    picked["reason"] = f"{prev_reason} | weak propagation" if prev_reason else "weak propagation"
                break
            if picked is not None:
                break

        if picked is None:
            continue

        rec = slot.to_dict()
        rec["Преподаватель"] = picked["teacher"]
        rec["score"] = float(picked["score"])
        rec["reason"] = picked["reason"]
        rec["assign_type"] = "auto_allocated"
        rec["_trusted_propagation"] = bool(picked.get("_trusted_propagation", False))
        base_conf = min(0.99, max(0.25, float(picked["score"]) / 180.0))
        if not rec["_trusted_propagation"]:
            base_conf = max(0.25, base_conf - 0.08)
        rec["confidence"] = base_conf
        chosen.append(rec)
        apply_assignment_to_state(teacher_state, rec)

    return pd.DataFrame(chosen)


def merge_locked_and_allocated(base_slots: pd.DataFrame, locked: pd.DataFrame, allocated: pd.DataFrame) -> pd.DataFrame:
    """Объединяет базовые слоты с жёсткими и автоматически найденными назначениями."""
    df = base_slots.copy()
    if "Преподаватель" not in df.columns:
        df["Преподаватель"] = None
    if "assign_type" not in df.columns:
        df["assign_type"] = None
    if "confidence" not in df.columns:
        df["confidence"] = None
    if "reason" not in df.columns:
        df["reason"] = None
    if "score" not in df.columns:
        df["score"] = None

    for part in [locked, allocated]:
        if part is None or len(part) == 0:
            continue
        lookup = part.set_index("slot_id")
        common = [c for c in ["Преподаватель", "assign_type", "confidence", "reason", "score"] if c in lookup.columns]
        mask = df["slot_id"].isin(lookup.index)
        for c in common:
            df.loc[mask, c] = df.loc[mask, "slot_id"].map(lookup[c].to_dict())
    return df
