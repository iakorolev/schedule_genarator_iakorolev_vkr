"""Загрузка, сохранение и применение пользовательских правил назначения преподавателей."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


VALID_MODES = {"force_teacher", "lock_assignment", "prefer_teacher", "ban_teacher", "prefer_teacher_scope", "ban_teacher_scope"}



def load_mappings(path: Path) -> dict:
    """Загружает пользовательские правила из JSON-файла."""
    if not path.exists():
        return {"rules": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"rules": []}
        if "rules" not in data or not isinstance(data["rules"], list):
            data["rules"] = []
        for r in data["rules"]:
            if isinstance(r, dict):
                _normalize_rule(r)
        return data
    except Exception:
        return {"rules": []}



def save_mappings(path: Path, data: dict) -> None:
    """Сохраняет правила назначения в JSON-файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)



def add_rule(path: Path, when: dict, assign: dict, mode: str = "force_teacher") -> dict:
    """Добавляет или заменяет правило назначения преподавателя."""
    data = load_mappings(path)

    rule = {"when": dict(when), "assign": dict(assign), "mode": mode}
    _normalize_rule(rule)

    rules = data.get("rules", [])
    rules = [r for r in rules if not _same_rule(r, rule)]
    rules.append(rule)

    data["rules"] = rules
    save_mappings(path, data)
    return data



def apply_mappings(merged: pd.DataFrame, mappings: dict, override: bool = False) -> pd.DataFrame:
    """Применяет правила к таблице слотов и проставляет ручные назначения."""
    df = merged.copy()

    if "rules" not in mappings or not isinstance(mappings["rules"], list):
        return df

    rules = []
    for r in mappings["rules"]:
        if not isinstance(r, dict):
            continue
        rr = {"when": dict(r.get("when", {})), "assign": dict(r.get("assign", {})), "mode": r.get("mode") or "force_teacher"}
        _normalize_rule(rr)
        if rr["mode"] in {"force_teacher", "lock_assignment"} and not rr["assign"].get("teacher"):
            continue
        rules.append(rr)

    if not rules:
        return df

    if "Учебная группа" not in df.columns or "disc_key" not in df.columns or "Вид_занятия_норм" not in df.columns:
        return df
    if "Преподаватель" not in df.columns:
        df["Преподаватель"] = None
    if "assign_type" not in df.columns:
        df["assign_type"] = None
    if "confidence" not in df.columns:
        df["confidence"] = None
    if "reason" not in df.columns:
        df["reason"] = None

    df["_g"] = df["Учебная группа"].astype(str)
    df["_d"] = df["disc_key"].astype(str)
    df["_k"] = df["Вид_занятия_норм"].astype(str)

    target_mask = pd.Series(True, index=df.index)
    if not override:
        target_mask = df["Преподаватель"].isna()

    for rule in rules:
        if rule.get("mode") not in {"force_teacher", "lock_assignment"}:
            continue
        when = rule.get("when", {})
        teacher = rule.get("assign", {}).get("teacher")
        if not teacher:
            continue
        mask = _match_mask(df, when)
        if not override:
            mask = mask & target_mask
        if not mask.any():
            continue
        df.loc[mask, "Преподаватель"] = teacher
        df.loc[mask, "assign_type"] = "manual_force"
        df.loc[mask, "confidence"] = 1.0
        df.loc[mask, "reason"] = "manual mapping"
        target_mask.loc[mask] = False

    return df.drop(columns=["_g", "_d", "_k"], errors="ignore")



def _match_mask(df: pd.DataFrame, when: dict) -> pd.Series:
    """Строит булеву маску строк, соответствующих условию правила."""
    mask = pd.Series(True, index=df.index)
    if when.get("group"):
        mask &= df["_g"].eq(when["group"])
    if when.get("group_prefix") and "group_prefix" in df.columns:
        mask &= df["group_prefix"].astype(str).eq(when["group_prefix"])
    if when.get("disc_key"):
        mask &= df["_d"].eq(when["disc_key"])
    if when.get("kind"):
        mask &= df["_k"].eq(when["kind"])
    if when.get("day") and "День недели" in df.columns:
        mask &= df["День недели"].astype(str).eq(when["day"])
    if when.get("week_type") and "week_type" in df.columns:
        mask &= df["week_type"].astype(str).eq(when["week_type"])
    return mask



def _normalize_rule(rule: dict) -> None:
    """Нормализует структуру одного пользовательского правила."""
    when = rule.get("when", {})
    assign = rule.get("assign", {})

    if not isinstance(when, dict):
        when = {}
    if not isinstance(assign, dict):
        assign = {}

    when2 = {
        "group": _clean(when.get("group")),
        "group_prefix": _clean(when.get("group_prefix")),
        "disc_key": _clean(when.get("disc_key")),
        "kind": _clean(when.get("kind")),
        "day": _clean(when.get("day")),
        "week_type": _clean(when.get("week_type")),
    }

    assign2 = {
        "teacher": _clean(assign.get("teacher")),
    }

    mode = _clean(rule.get("mode")) or "force_teacher"
    if mode not in VALID_MODES:
        mode = "force_teacher"

    rule["when"] = {k: v for k, v in when2.items() if v}
    rule["assign"] = {k: v for k, v in assign2.items() if v}
    rule["mode"] = mode



def _clean(x: Any) -> str | None:
    """Приводит значение к аккуратной строке или None."""
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None



def _same_rule(a: dict, b: dict) -> bool:
    """Сравнивает два правила на полное совпадение."""
    return (a.get("when") == b.get("when")) and (a.get("assign") == b.get("assign")) and (a.get("mode") == b.get("mode"))
