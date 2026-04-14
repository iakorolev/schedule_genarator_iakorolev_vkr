"""Математические методы для ранжирования кандидатов и диагностики ВКР.

Модуль не меняет внешний сценарий работы программы, а формализует внутреннюю
оценку кандидатов через матрицу компетенций и многокритериальное ранжирование.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .normalize import _txt
from .un_parser import build_teacher_skills


CRITERION_BENEFIT = "benefit"
CRITERION_COST = "cost"


def clamp01(value: float | int | None) -> float:
    """Ограничивает число интервалом [0, 1]."""
    try:
        x = float(value or 0.0)
    except Exception:
        return 0.0
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return x


def safe_ratio(numerator: float | int | None, denominator: float | int | None, default: float = 0.0) -> float:
    """Возвращает устойчивое отношение двух величин."""
    try:
        num = float(numerator or 0.0)
        den = float(denominator or 0.0)
    except Exception:
        return float(default)
    if abs(den) < 1e-12:
        return float(default)
    return num / den


def vector_normalize(values: Sequence[float]) -> np.ndarray:
    """Выполняет векторную нормализацию столбца для TOPSIS."""
    arr = np.asarray(list(values), dtype=float)
    denom = np.linalg.norm(arr)
    if denom <= 1e-12:
        return np.zeros_like(arr)
    return arr / denom


def weighted_sum_score(row: pd.Series, criteria: Sequence[dict]) -> float:
    """Считает взвешенную сумму по уже нормированным критериям [0, 1]."""
    score = 0.0
    for c in criteria:
        name = c["name"]
        weight = float(c.get("weight", 0.0) or 0.0)
        mode = c.get("mode", CRITERION_BENEFIT)
        value = clamp01(row.get(name, 0.0))
        score += weight * (value if mode == CRITERION_BENEFIT else (1.0 - value))
    return clamp01(score)


def topsis_scores(df: pd.DataFrame, criteria: Sequence[dict]) -> pd.Series:
    """Вычисляет коэффициент близости TOPSIS для набора альтернатив.

    Все критерии ожидаются в диапазоне [0, 1]. Для одной альтернативы возвращается
    её взвешенная сумма, чтобы не получать вырожденный случай 0/0.
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    if len(df) == 1:
        only = weighted_sum_score(df.iloc[0], criteria)
        return pd.Series([only], index=df.index, dtype=float)

    names = [c["name"] for c in criteria]
    matrix = df[names].fillna(0.0).astype(float).clip(lower=0.0, upper=1.0).to_numpy(copy=True)

    for j in range(matrix.shape[1]):
        matrix[:, j] = vector_normalize(matrix[:, j])

    weights = np.array([float(c.get("weight", 0.0) or 0.0) for c in criteria], dtype=float)
    weighted = matrix * weights

    ideal_best = np.zeros(weighted.shape[1], dtype=float)
    ideal_worst = np.zeros(weighted.shape[1], dtype=float)
    for j, c in enumerate(criteria):
        if c.get("mode", CRITERION_BENEFIT) == CRITERION_COST:
            ideal_best[j] = weighted[:, j].min()
            ideal_worst[j] = weighted[:, j].max()
        else:
            ideal_best[j] = weighted[:, j].max()
            ideal_worst[j] = weighted[:, j].min()

    dist_best = np.linalg.norm(weighted - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    denom = dist_best + dist_worst
    closeness = np.divide(dist_worst, denom, out=np.zeros_like(dist_worst), where=denom > 1e-12)
    return pd.Series(closeness.clip(0.0, 1.0), index=df.index, dtype=float)


def build_teacher_competency_matrix(run_atoms: pd.DataFrame) -> pd.DataFrame:
    """Строит матрицу компетенций преподавателей по дисциплинам и видам занятий.

    Матрица нужна и для диагностического анализа, и как формализованная база ВКР.
    """
    if run_atoms is None or len(run_atoms) == 0:
        return pd.DataFrame(columns=[
            "Преподаватель",
            "disc_key",
            "kind_norm",
            "competency_weight",
            "discipline_share",
            "group_count",
        ])

    skills = build_teacher_skills(run_atoms).copy()
    totals = (
        skills.groupby("Преподаватель", as_index=False)
        .agg(total_skill_weight=("skill_weight", "sum"))
    )
    skills = skills.merge(totals, on="Преподаватель", how="left")
    skills["discipline_share"] = skills.apply(
        lambda r: clamp01(safe_ratio(r.get("skill_weight", 0.0), r.get("total_skill_weight", 0.0), default=0.0)),
        axis=1,
    )
    skills["competency_weight"] = (
        0.75 * skills["discipline_share"].fillna(0.0)
        + 0.25 * skills["skill_groups"].fillna(0.0).clip(upper=4.0) / 4.0
    ).clip(lower=0.0, upper=1.0)
    skills = skills.rename(columns={"skill_groups": "group_count"})
    return skills[[
        "Преподаватель",
        "disc_key",
        "kind_norm",
        "competency_weight",
        "discipline_share",
        "group_count",
        "skill_hours",
        "skill_weight",
    ]].sort_values(["Преподаватель", "disc_key", "kind_norm"]).reset_index(drop=True)


def build_teacher_competency_pivot(run_atoms: pd.DataFrame) -> pd.DataFrame:
    """Возвращает удобную широкую матрицу компетенций teacher x (disc, kind)."""
    matrix = build_teacher_competency_matrix(run_atoms)
    if len(matrix) == 0:
        return pd.DataFrame()
    tmp = matrix.copy()
    tmp["disc_kind"] = tmp["disc_key"].astype(str) + " | " + tmp["kind_norm"].astype(str)
    pivot = tmp.pivot_table(
        index="Преподаватель",
        columns="disc_kind",
        values="competency_weight",
        fill_value=0.0,
        aggfunc="max",
    )
    pivot = pivot.reset_index()
    return pivot
