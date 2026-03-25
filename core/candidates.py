"""Подготовка кандидатов для выбора преподавателя в пользовательском интерфейсе."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .allocation import build_slot_candidates
from .load_model import build_teacher_state
from .normalize import disc_tokens, jaccard
from .un_parser import build_teacher_capacity


@dataclass
class Slot:
    """Описывает слот, для которого требуется подобрать преподавателя."""
    group: str
    disc_key: str
    kind: str



def build_candidates(
    un_expanded: pd.DataFrame,
    slot: Slot,
    top_n: int = 8,
    min_score: float = 0.20,
) -> list[dict[str, Any]]:
    """Возвращает верхнюю часть списка кандидатов для одного слота в интерфейсе."""
    if un_expanded is None or len(un_expanded) == 0:
        return []

    # Совместимость: UI раньше читал un_svodnaya_expanded.xlsx, а allocator ожидает run_atoms.
    # Поддерживаем оба формата, чтобы endpoint /api/candidates не падал.
    run_atoms = un_expanded.copy()
    if "kind_norm" not in run_atoms.columns and "Вид_работы_норм" in run_atoms.columns:
        run_atoms = run_atoms.rename(columns={"Вид_работы_норм": "kind_norm"})
    if "discipline_norm" not in run_atoms.columns and "disc_key" in run_atoms.columns:
        run_atoms["discipline_norm"] = run_atoms["disc_key"]
    if "hours_kind" not in run_atoms.columns:
        run_atoms["hours_kind"] = 0.0
    if "capacity_units" not in run_atoms.columns:
        run_atoms["capacity_units"] = 0.0
    if "teacher_norm" not in run_atoms.columns and "Преподаватель" in run_atoms.columns:
        run_atoms["teacher_norm"] = run_atoms["Преподаватель"].astype(str)
    if "teacher_lastname" not in run_atoms.columns and "Преподаватель" in run_atoms.columns:
        run_atoms["teacher_lastname"] = run_atoms["Преподаватель"].astype(str)
    if "Учебная группа" not in run_atoms.columns or "Преподаватель" not in run_atoms.columns or "kind_norm" not in run_atoms.columns:
        return []

    slot_df = pd.DataFrame(
        [
            {
                "slot_id": "preview-slot",
                "Учебная группа": slot.group.strip(),
                "disc_key": slot.disc_key.strip(),
                "Вид_занятия_норм": slot.kind.strip(),
                "День недели": "",
                "Пара": None,
                "Время": "",
                "teacher_hint": "",
            }
        ]
    )
    teacher_state = build_teacher_state(build_teacher_capacity(run_atoms), None)
    cands = build_slot_candidates(slot_df, run_atoms, teacher_state)
    if len(cands) == 0:
        return []
    cands = cands[cands["score"] >= float(min_score)].copy()
    cands = cands.sort_values("score", ascending=False).head(top_n)
    out = []
    for _, r in cands.iterrows():
        out.append(
            {
                "teacher": r["teacher"],
                "score": float(r["score"]),
                "reason": r["reason"],
            }
        )
    return out
