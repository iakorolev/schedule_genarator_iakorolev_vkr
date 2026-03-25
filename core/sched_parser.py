"""Разбор файла общего расписания и преобразование ячеек в атомарные слоты."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .normalize import (
    _txt,
    clean_text,
    extract_teacher_hints,
    find_rooms,
    normalize_day,
    normalize_disc,
    normalize_group,
    normalize_kind_sched,
    normalize_room,
    normalize_time,
    split_multi_kinds,
)


KIND_RE = re.compile(r"\b(лек/сем|сем/лек|лек/лаб|лаб/лек|сем/лаб|лаб/сем|лек(?:ция)?|сем(?:инар)?|пр|практика|лаб|лр)\b", re.I)
SUBGROUP_RE = re.compile(r"(?:подгруппа|группа)\s*([А-ЯA-ZA-Za-z0-9]+)", re.I)
AB_MARK_RE = re.compile(r"(?=(?:^|\s)[абвгдежз]\))", re.I)
SEGMENT_START_RE = re.compile(
    r"(?=[А-ЯA-ZЁ(][^;\n]{2,120}?\s*;\s*(?:лек(?:ция)?|сем(?:инар)?|пр|практика|лаб|лр|лек/сем|сем/лек|лек/лаб|лаб/лек|сем/лаб|лаб/сем)\b)",
    re.I,
)


def _clean_cell_text(raw: Any) -> str:
    """Очищает содержимое ячейки расписания перед разбором."""
    s = clean_text(raw)
    s = s.replace("\n", " ")
    s = s.replace("%", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*[/％%]\s*$", "", s)
    return s.strip()


def split_cell_into_blocks(text: str) -> list[str]:
    """Разбивает сложную ячейку расписания на отдельные смысловые блоки."""
    s = _clean_cell_text(text)
    if not s:
        return []

    parts: list[str] = []

    # Маркеры а)/б)/в) считаем отдельными блоками только если они стоят
    # в начале строки или после явного разделителя, а не внутри "подгруппа А)".
    if re.search(r"(?:^|[;/])\s*[абвгдежз]\)", s, flags=re.I):
        marked = [p.strip() for p in re.split(r"(?=(?:^|[;/])\s*[абвгдежз]\))", s) if p.strip()]
    else:
        marked = [s]

    multi_start_re = re.compile(
        r"[А-ЯA-ZЁ(][^;\n]{2,120}?\s*;\s*(?i:(?:лек/сем|сем/лек|лек/лаб|лаб/лек|сем/лаб|лаб/сем|лек(?:ция)?|сем(?:инар)?|пр|практика|лаб|лр))\b"
    )

    for chunk in marked:
        chunk = re.sub(r"^\s*[абвгдежз]\)\s*", "", chunk, flags=re.I).strip()
        if not chunk:
            continue

        # В некоторых ячейках подблоки идут подряд как
        # "... ОРД-559 б) (подблок...) ..." без слеша между ними.
        marker_parts = [p.strip(" /") for p in re.split(r"(?=\s+[абвгдежз]\)\s*(?:\(|[А-ЯA-ZЁ]))", chunk, flags=re.I) if p.strip(" /")]
        if len(marker_parts) > 1:
            for seg in marker_parts:
                if seg:
                    parts.append(seg)
            continue

        # Делим по " / " только между полноценными сегментами.
        slash_parts = [p.strip(" /") for p in re.split(r"\s+/\s+(?=[А-ЯA-ZЁ(])", chunk) if p.strip(" /")]
        if len(slash_parts) > 1:
            for seg in slash_parts:
                if seg:
                    parts.append(seg)
            continue

        starts = [m.start() for m in multi_start_re.finditer(chunk)]
        starts = sorted(set(starts))
        if len(starts) > 1:
            for i, start_pos in enumerate(starts):
                end_pos = starts[i + 1] if i + 1 < len(starts) else len(chunk)
                seg = chunk[start_pos:end_pos].strip(" /")
                if seg:
                    parts.append(seg)
            continue

        parts.append(chunk)

    out = []
    for p in parts:
        p = p.strip(" /")
        if p and p not in out:
            out.append(p)
    return out



def _extract_subgroup(text: str) -> str:
    """Извлекает обозначение подгруппы из текстового блока."""
    s = clean_text(text)
    m = SUBGROUP_RE.search(s)
    if m:
        return m.group(1).strip()
    if s[:2].lower() in {"а)", "б)", "в)", "г)"}:
        return s[0].upper()
    return ""



def parse_block(block: str) -> list[dict[str, Any]]:
    """Разбирает один блок ячейки расписания в структурированную запись."""
    block = _clean_cell_text(block)
    if not block:
        return []

    subgroup = _extract_subgroup(block)
    block2 = re.sub(r"\((?:подгруппа|группа)[^)]*\)", " ", block, flags=re.I)
    block2 = re.sub(r"^\s*[абвгдежз]\)\s*", "", block2, flags=re.I)
    block2 = re.sub(r"\s+", " ", block2).strip()

    parts = [p.strip() for p in block2.split(";") if p.strip()]
    if not parts:
        return []

    discipline = parts[0]
    remainder = parts[1:]

    kind_raw = ""
    teacher_hints: list[str] = []
    rooms: list[str] = []
    misc: list[str] = []

    for p in remainder:
        if not kind_raw and KIND_RE.search(p):
            kind_raw = KIND_RE.search(p).group(1)
            more = p[KIND_RE.search(p).end():].strip(" ;,")
            if more:
                misc.append(more)
            continue

        phints = extract_teacher_hints(p)
        if phints:
            for t in phints:
                if t not in teacher_hints:
                    teacher_hints.append(t)
            cleaned = p
            for t in phints:
                cleaned = cleaned.replace(t, " ")
            p = cleaned.strip(" ;,")

        rs = find_rooms(p)
        if rs:
            for r in rs:
                if r not in rooms:
                    rooms.append(r)
            stripped = p
            for r in rs:
                stripped = re.sub(re.escape(r), " ", stripped, flags=re.I)
            p = stripped.strip(" ;,")

        if p and p.lower() != "преподаватель":
            misc.append(p)

    if not kind_raw:
        m = KIND_RE.search(block2)
        if m:
            kind_raw = m.group(1)

    kinds = split_multi_kinds(kind_raw) if kind_raw else []
    if not kinds:
        norm_kind = normalize_kind_sched(kind_raw)
        if norm_kind:
            kinds = [norm_kind]

    room = " / ".join(dict.fromkeys(normalize_room(r) for r in rooms if r))
    teacher_hint = " / ".join(dict.fromkeys(teacher_hints))
    misc_text = " ; ".join([m for m in misc if m])

    if not kinds:
        kinds = [None]

    out = []
    for kind in kinds:
        out.append(
            {
                "discipline_raw": discipline,
                "discipline_norm": normalize_disc(discipline),
                "disc_key": normalize_disc(discipline),
                "kind_raw": kind_raw,
                "kind_norm": kind,
                "room": room,
                "teacher_hint": teacher_hint,
                "subgroup": subgroup,
                "misc": misc_text,
                "source_block": block,
            }
        )
    return out



def parse_subject(raw: Any) -> dict[str, Any]:
    """Возвращает краткое legacy-представление содержимого ячейки."""
    atoms = []
    for block in split_cell_into_blocks(raw):
        atoms.extend(parse_block(block))
    if not atoms:
        return {"Дисциплина": None, "Вид_занятия": None, "Аудитория": None}
    first = atoms[0]
    return {
        "Дисциплина": first.get("discipline_raw"),
        "Вид_занятия": first.get("kind_raw"),
        "Аудитория": first.get("room"),
    }



def read_schedule_atoms(sched_xlsx: Path, sheet_name: str = "ФМиЕН") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает файл расписания и возвращает сырые, нормализованные и атомарные таблицы."""
    xls = pd.ExcelFile(sched_xlsx)
    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

    data = df.iloc[5:].copy()
    rows = []
    atom_rows = []
    slot_seq = 1

    for col_idx in range(4, df.shape[1]):
        group_code = normalize_group(df.iloc[4, col_idx])
        if not group_code:
            continue

        for ridx in range(5, df.shape[0]):
            raw_subject = df.iloc[ridx, col_idx]
            if pd.isna(raw_subject) or str(raw_subject).strip() == "":
                continue

            day = normalize_day(df.iloc[ridx, 1])
            pair_val = df.iloc[ridx, 2]
            try:
                pair_val = int(pair_val) if not pd.isna(pair_val) else None
            except Exception:
                pair_val = None
            time_val = normalize_time(df.iloc[ridx, 3])

            parsed = parse_subject(raw_subject)
            rows.append(
                {
                    "День недели": day,
                    "Пара": pair_val,
                    "Время": time_val,
                    "Учебная группа": group_code,
                    "Дисциплина": _txt(parsed.get("Дисциплина")),
                    "Вид_занятия": _txt(parsed.get("Вид_занятия")),
                    "Аудитория": normalize_room(parsed.get("Аудитория")),
                    "raw_cell": _clean_cell_text(raw_subject),
                }
            )

            blocks = split_cell_into_blocks(raw_subject)
            if not blocks:
                continue
            for part_no, block in enumerate(blocks, start=1):
                parsed_blocks = parse_block(block)
                if not parsed_blocks:
                    continue
                for atom_no, atom in enumerate(parsed_blocks, start=1):
                    atom_rows.append(
                        {
                            "slot_id": f"S{slot_seq:05d}",
                            "block_no": part_no,
                            "atom_no": atom_no,
                            "День недели": day,
                            "Пара": pair_val,
                            "Время": time_val,
                            "Учебная группа": group_code,
                            "subgroup": atom.get("subgroup") or "",
                            "Дисциплина": atom.get("discipline_raw") or "",
                            "discipline_norm": atom.get("discipline_norm") or "",
                            "disc_key": atom.get("disc_key") or "",
                            "Вид_занятия": atom.get("kind_raw") or "",
                            "Вид_занятия_норм": atom.get("kind_norm"),
                            "Аудитория": atom.get("room") or "",
                            "teacher_hint": atom.get("teacher_hint") or "",
                            "misc": atom.get("misc") or "",
                            "source_text": _clean_cell_text(raw_subject),
                            "source_block": atom.get("source_block") or block,
                            "parse_quality": 1 if atom.get("kind_norm") else 0,
                        }
                    )
                slot_seq += 1

    parsed_df = pd.DataFrame(rows)
    atoms_df = pd.DataFrame(atom_rows)
    if len(atoms_df) == 0:
        atoms_df = pd.DataFrame(
            columns=[
                "slot_id",
                "День недели",
                "Пара",
                "Время",
                "Учебная группа",
                "subgroup",
                "Дисциплина",
                "discipline_norm",
                "disc_key",
                "Вид_занятия",
                "Вид_занятия_норм",
                "Аудитория",
                "teacher_hint",
                "source_text",
                "source_block",
                "parse_quality",
            ]
        )
    norm_df = atoms_df[atoms_df["Вид_занятия_норм"].notna()].copy()
    return parsed_df, norm_df, atoms_df



def read_schedule(sched_xlsx: Path, sheet_name: str = "ФМиЕН") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает только parsed- и normalized-версии расписания."""
    parsed_df, norm_df, _atoms_df = read_schedule_atoms(sched_xlsx, sheet_name=sheet_name)
    return parsed_df, norm_df
