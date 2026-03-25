"""Эвристическое распределение преподавателей по неразобранным слотам расписания."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .load_model import apply_assignment_to_state, teacher_is_available, teacher_remaining_capacity
from .normalize import _txt, best_disc_match, extract_group_parts, teacher_lastname
from .un_parser import build_teacher_group_links, build_teacher_skills


@dataclass
class CandidateScore:
    """Хранит итоговую оценку кандидата и текстовое объяснение назначения."""
    teacher: str
    score: float
    reason: str


def _slot_group_context(slot: pd.Series) -> dict[str, str]:
    """Возвращает нормализованный контекст учебной группы для текущего слота."""
    group = _txt(slot.get("Учебная группа"))
    return extract_group_parts(group)


def _state_family_bonus(state: dict[str, Any], teacher: str, slot: pd.Series, disc: str, kind: str) -> tuple[float, list[str]]:
    """Рассчитывает бонус за ранее сделанные назначения в той же группе или направлении."""
    bonus = 0.0
    reasons: list[str] = []
    ctx = _slot_group_context(slot)
    family_key = ctx.get("family_key", "")
    prefix_year = ctx.get("prefix_year", "")

    if (disc, kind) in state.get("assigned_disc_kind", {}).get(teacher, set()):
        bonus += 24.0
        reasons.append("already assigned same disc+kind")
    if family_key and (family_key, disc, kind) in state.get("assigned_family_disc_kind", {}).get(teacher, set()):
        bonus += 110.0
        reasons.append("family propagation")
    elif prefix_year and (prefix_year, disc, kind) in state.get("assigned_prefix_year_disc_kind", {}).get(teacher, set()):
        bonus += 70.0
        reasons.append("same direction/year propagation")
    return bonus, reasons



def _prep_assigned_index(assignments: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """Строит индексы уже закреплённых назначений для повторного использования в скоринге."""
    disc_kind_teachers: dict[tuple[str, str], set[str]] = {}
    teacher_days: dict[tuple[str, str], set[str]] = {}
    family_disc_kind_teachers: dict[tuple[str, str, str], set[str]] = {}
    prefix_year_disc_kind_teachers: dict[tuple[str, str, str], set[str]] = {}
    if assignments is None or len(assignments) == 0:
        return disc_kind_teachers, teacher_days, family_disc_kind_teachers, prefix_year_disc_kind_teachers
    for _, r in assignments.iterrows():
        t = _txt(r.get("Преподаватель"))
        d = _txt(r.get("disc_key"))
        k = _txt(r.get("Вид_занятия_норм"))
        day = _txt(r.get("День недели"))
        group = _txt(r.get("Учебная группа"))
        gctx = extract_group_parts(group)
        fam = gctx.get("family_key", "")
        pyear = gctx.get("prefix_year", "")
        if t and d and k:
            disc_kind_teachers.setdefault((d, k), set()).add(t)
            if fam:
                family_disc_kind_teachers.setdefault((fam, d, k), set()).add(t)
            if pyear:
                prefix_year_disc_kind_teachers.setdefault((pyear, d, k), set()).add(t)
        if t and day and d:
            teacher_days.setdefault((t, day), set()).add(d)
    return disc_kind_teachers, teacher_days, family_disc_kind_teachers, prefix_year_disc_kind_teachers


def _related_kinds(kind: str) -> list[str]:
    """Возвращает родственные виды занятий для fallback-поиска кандидатов."""
    if kind == "лаб":
        return ["сем"]
    if kind == "сем":
        return ["лаб"]
    return []


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

    group_links = build_teacher_group_links(run_atoms)
    skills = build_teacher_skills(run_atoms)
    disc_kind_teachers, teacher_days, family_disc_kind_teachers, prefix_year_disc_kind_teachers = _prep_assigned_index(locked_assignments)

    by_kind = {k: v.copy() for k, v in run_atoms.groupby("kind_norm")}
    skills_map = {}
    for _, r in skills.iterrows():
        skills_map[(r["Преподаватель"], r["disc_key"], r["kind_norm"])] = float(r.get("skill_weight", 0) or 0)

    group_map = {}
    family_map = {}
    prefix_year_map = {}
    for _, r in group_links.iterrows():
        teacher = r["Преподаватель"]
        group = r["Учебная группа"]
        disc = r["disc_key"]
        kind_norm = r["kind_norm"]
        units = float(r.get("capacity_units", 0) or 0)
        group_map[(teacher, group, disc, kind_norm)] = units
        gctx = extract_group_parts(group)
        fam = gctx.get("family_key", "")
        pyear = gctx.get("prefix_year", "")
        if fam:
            family_map[(teacher, fam, disc, kind_norm)] = family_map.get((teacher, fam, disc, kind_norm), 0.0) + units
        if pyear:
            prefix_year_map[(teacher, pyear, disc, kind_norm)] = prefix_year_map.get((teacher, pyear, disc, kind_norm), 0.0) + units

    prefer_rules = []
    ban_rules = []
    if isinstance(mappings, dict):
        for rule in mappings.get("rules", []):
            mode = str(rule.get("mode") or "force_teacher")
            if mode == "prefer_teacher":
                prefer_rules.append(rule)
            elif mode == "ban_teacher":
                ban_rules.append(rule)

    rows = []
    for _, slot in unmatched_slots.iterrows():
        kind = _txt(slot.get("Вид_занятия_норм"))
        if not kind or kind not in by_kind:
            continue
        pool = by_kind[kind]
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

        t_scores: dict[str, float] = {}
        t_reasons: dict[str, list[str]] = {}

        for teacher in family_disc_kind_teachers.get((family_key, disc, kind), set()):
            t_scores[teacher] = max(t_scores.get(teacher, 0.0), 118.0)
            t_reasons.setdefault(teacher, []).append("seeded from family assignment")
        for teacher in prefix_year_disc_kind_teachers.get((prefix_year, disc, kind), set()):
            t_scores[teacher] = max(t_scores.get(teacher, 0.0), 96.0)
            t_reasons.setdefault(teacher, []).append("seeded from direction/year assignment")
        for teacher in disc_kind_teachers.get((disc, kind), set()):
            t_scores[teacher] = max(t_scores.get(teacher, 0.0), 88.0)
            t_reasons.setdefault(teacher, []).append("seeded from assigned same disc+kind")

        if teacher_hint and hint_lastname:
            hinted_teachers = sorted(set(run_atoms.loc[run_atoms["teacher_lastname"] == hint_lastname, "Преподаватель"].astype(str).tolist()))
            for teacher in hinted_teachers:
                t_scores[teacher] = max(t_scores.get(teacher, 0.0), 114.0)
                t_reasons.setdefault(teacher, []).append("seeded from teacher hint")

        # exact group+disc+kind
        exact = pool[(pool["Учебная группа"] == group) & (pool["disc_key"] == disc)]
        for teacher, g in exact.groupby("Преподаватель"):
            t_scores[teacher] = max(t_scores.get(teacher, 0.0), 120.0 + float(g["capacity_units"].sum()))
            t_reasons.setdefault(teacher, []).append("exact group+disc+kind")

        # group+kind fallback
        gk = pool[pool["Учебная группа"] == group]
        for teacher, g in gk.groupby("Преподаватель"):
            cur = 75.0 + float(g["capacity_units"].sum())
            if cur > t_scores.get(teacher, 0.0):
                t_scores[teacher] = cur
            t_reasons.setdefault(teacher, []).append("group+kind")

        # disc+kind
        dk = pool[pool["disc_key"] == disc]
        for teacher, g in dk.groupby("Преподаватель"):
            cur = 70.0 + float(g["capacity_units"].sum())
            if cur > t_scores.get(teacher, 0.0):
                t_scores[teacher] = cur
            t_reasons.setdefault(teacher, []).append("disc+kind")

        # related-kind fallback (сем <-> лаб)
        for rk in _related_kinds(kind):
            rpool = by_kind.get(rk)
            if rpool is None or len(rpool) == 0:
                continue
            exact_related = rpool[(rpool["Учебная группа"] == group) & (rpool["disc_key"] == disc)]
            for teacher, g in exact_related.groupby("Преподаватель"):
                cur = 58.0 + float(g["capacity_units"].sum())
                if cur > t_scores.get(teacher, 0.0):
                    t_scores[teacher] = cur
                t_reasons.setdefault(teacher, []).append(f"exact group+disc+related-kind:{rk}")
            disc_related = rpool[rpool["disc_key"] == disc]
            for teacher, g in disc_related.groupby("Преподаватель"):
                cur = 44.0 + float(g["capacity_units"].sum())
                if cur > t_scores.get(teacher, 0.0):
                    t_scores[teacher] = cur
                t_reasons.setdefault(teacher, []).append(f"disc+related-kind:{rk}")

        # fuzzy by same kind
        disc_list = pool["disc_key"].dropna().astype(str).unique().tolist()
        best, sim = best_disc_match(disc, disc_list)
        if best and sim >= 0.45:
            fuzz = pool[pool["disc_key"] == best]
            for teacher, g in fuzz.groupby("Преподаватель"):
                cur = 35.0 + 50.0 * sim + float(g["capacity_units"].sum())
                if cur > t_scores.get(teacher, 0.0):
                    t_scores[teacher] = cur
                t_reasons.setdefault(teacher, []).append(f"fuzzy:{best}:{sim:.2f}")

        # same discipline already assigned somewhere
        for teacher in list(t_scores.keys()):
            if teacher in disc_kind_teachers.get((disc, kind), set()):
                t_scores[teacher] += 28.0
                t_reasons.setdefault(teacher, []).append("already teaches same disc+kind")
            if disc in teacher_days.get((teacher, day), set()):
                t_scores[teacher] += 18.0
                t_reasons.setdefault(teacher, []).append("same day same discipline")
            if teacher_hint and hint_lastname and hint_lastname == teacher_lastname(teacher):
                t_scores[teacher] += 80.0
                t_reasons.setdefault(teacher, []).append("teacher hint from schedule")

            cap_total = teacher_remaining_capacity(teacher_state, teacher)
            cap_kind = teacher_remaining_capacity(teacher_state, teacher, kind)
            if cap_total > 0:
                t_scores[teacher] += min(20.0, cap_total * 4.0)
                t_reasons.setdefault(teacher, []).append("remaining load")
            else:
                t_scores[teacher] -= min(80.0, abs(cap_total) * 10.0)
                t_reasons.setdefault(teacher, []).append("over capacity")
            if cap_kind > 0:
                t_scores[teacher] += min(15.0, cap_kind * 5.0)

            if not teacher_is_available(teacher_state, teacher, day, pair, time, disc=disc, kind=kind, room=room, group=group):
                t_scores[teacher] -= 1000.0
                t_reasons.setdefault(teacher, []).append("time conflict")

            skill_weight = skills_map.get((teacher, disc, kind), 0.0)
            if skill_weight > 0:
                t_scores[teacher] += min(20.0, skill_weight / 6.0)
            group_weight = group_map.get((teacher, group, disc, kind), 0.0)
            if group_weight > 0:
                t_scores[teacher] += min(12.0, group_weight * 3.0)
                t_reasons.setdefault(teacher, []).append("same exact group in RUN")
            family_weight = family_map.get((teacher, family_key, disc, kind), 0.0) if family_key else 0.0
            if family_weight > 0:
                t_scores[teacher] += min(65.0, 40.0 + family_weight * 6.0)
                t_reasons.setdefault(teacher, []).append("same family in RUN")
            prefix_weight = prefix_year_map.get((teacher, prefix_year, disc, kind), 0.0) if prefix_year else 0.0
            if prefix_weight > 0:
                t_scores[teacher] += min(35.0, 18.0 + prefix_weight * 4.0)
                t_reasons.setdefault(teacher, []).append("same direction/year in RUN")
            dyn_bonus, dyn_reasons = _state_family_bonus(teacher_state, teacher, slot, disc, kind)
            if dyn_bonus:
                t_scores[teacher] += dyn_bonus
                for rr in dyn_reasons:
                    t_reasons.setdefault(teacher, []).append(rr)

            for rule in prefer_rules:
                when = rule.get("when", {})
                assign = rule.get("assign", {})
                if assign.get("teacher") != teacher:
                    continue
                if _rule_matches_slot(when, slot):
                    t_scores[teacher] += 30.0
                    t_reasons.setdefault(teacher, []).append("prefer rule")
            for rule in ban_rules:
                when = rule.get("when", {})
                assign = rule.get("assign", {})
                if assign.get("teacher") != teacher:
                    continue
                if _rule_matches_slot(when, slot):
                    t_scores[teacher] -= 1000.0
                    t_reasons.setdefault(teacher, []).append("ban rule")

        for teacher, score in t_scores.items():
            if score <= 0:
                continue
            rows.append(
                {
                    "slot_id": slot.get("slot_id"),
                    "teacher": teacher,
                    "score": round(float(score), 3),
                    "reason": " | ".join(dict.fromkeys(t_reasons.get(teacher, []))),
                }
            )

    cand_df = pd.DataFrame(rows)
    if len(cand_df) == 0:
        return cand_df
    cand_df = cand_df.sort_values(["slot_id", "score"], ascending=[True, False])
    return cand_df



def _rule_matches_slot(when: dict, slot: pd.Series) -> bool:
    """Проверяет, подходит ли пользовательское правило к конкретному слоту."""
    group = _txt(slot.get("Учебная группа"))
    disc = _txt(slot.get("disc_key"))
    kind = _txt(slot.get("Вид_занятия_норм"))
    if when.get("group") and when.get("group") != group:
        return False
    if when.get("disc_key") and when.get("disc_key") != disc:
        return False
    if when.get("kind") and when.get("kind") != kind:
        return False
    return True



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
    min_score = 55.0
    for _, slot in slots.iterrows():
        slot_id = slot["slot_id"]
        opts = candidates[candidates["slot_id"] == slot_id].sort_values("score", ascending=False)
        picked = None
        for _, cand in opts.iterrows():
            teacher = cand["teacher"]
            if not teacher_is_available(teacher_state, teacher, slot.get("День недели"), slot.get("Пара"), slot.get("Время"), disc=_txt(slot.get("disc_key")), kind=_txt(slot.get("Вид_занятия_норм")), room=_txt(slot.get("Аудитория")), group=_txt(slot.get("Учебная группа"))):
                continue
            extra_bonus, extra_reasons = _state_family_bonus(teacher_state, teacher, slot, _txt(slot.get("disc_key")), _txt(slot.get("Вид_занятия_норм")))
            effective_score = float(cand["score"]) + extra_bonus
            threshold = min_score - 10.0 if extra_bonus >= 100.0 else min_score
            if effective_score < threshold:
                continue
            picked = cand.copy()
            picked["score"] = effective_score
            if extra_reasons:
                tail = " | ".join(dict.fromkeys(extra_reasons))
                prev_reason = cand.get("reason", "")
                picked["reason"] = f"{prev_reason} | {tail}" if prev_reason else tail
            break

        # Если уверенного и допустимого кандидата нет, лучше оставить слот unmatched,
        # чем назначить случайного преподавателя и испортить расписание.
        if picked is None:
            continue

        rec = slot.to_dict()
        rec["Преподаватель"] = picked["teacher"]
        rec["score"] = float(picked["score"])
        rec["reason"] = picked["reason"]
        rec["assign_type"] = "auto_allocated"
        rec["confidence"] = min(0.99, max(0.25, float(picked["score"]) / 150.0))
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
