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
    normalize_week_type,
    split_multi_kinds,
)


KIND_RE = re.compile(r"\b(лек/сем|сем/лек|лек/лаб|лаб/лек|сем/лаб|лаб/сем|лек(?:ция)?|сем(?:инар)?|пр|практика|лаб|лр)\b", re.I)
SUBGROUP_RE = re.compile(r"(?:подгруппа|группа)\s*([А-ЯA-ZA-Za-z0-9]+)", re.I)
NEW_STYLE_HEADER_RE = re.compile(r"^(Лекция|Лабораторная работа|Практические и другие)\s*:\s*(.+?)\s*$", re.I)
FULLNAME_RE = re.compile(r"\b[А-ЯЁA-Z][а-яёa-z]+(?:-[А-ЯЁA-Z]?[а-яёa-z]+)*(?:\s+[А-ЯЁA-Z][а-яёa-z]+(?:-[А-ЯЁA-Z]?[а-яёa-z]+)*){1,3}\b")

DAY_CODES = {"ПН", "ВТ", "СР", "ЧТ", "ПТ", "СБ", "ВС"}


def _clean_cell_text(raw: Any) -> str:
    """Очищает содержимое ячейки расписания перед разбором."""
    s = clean_text(raw)
    s = s.replace("\n", " ")
    s = s.replace("%", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*[/％%]\s*$", "", s)
    return s.strip()



def _clean_multiline_text(raw: Any) -> str:
    """Очищает текст, сохраняя переносы строк для формата TDSheet."""
    s = clean_text(raw)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()



def _is_new_style_block(text: str) -> bool:
    """Проверяет, похож ли блок на новый формат расписания TDSheet."""
    raw = _clean_multiline_text(text)
    if not raw:
        return False
    first_line = next((ln.strip() for ln in raw.split("\n") if ln.strip()), "")
    return bool(NEW_STYLE_HEADER_RE.match(first_line))



def _extract_fullname_hints(text: str) -> list[str]:
    """Извлекает полные ФИО из строки нового формата расписания."""
    s = _clean_cell_text(text)
    if not s:
        return []
    out: list[str] = []
    for m in FULLNAME_RE.finditer(s):
        name = re.sub(r"\s+", " ", m.group(0)).strip()
        if name and name not in out:
            out.append(name)
    return out



def _parse_new_block(block: str) -> list[dict[str, Any]]:
    """Разбирает блок нового формата: 'Лекция: ... / Практические и другие: ...'."""
    raw = _clean_multiline_text(block)
    if not raw:
        return []

    lines = [re.sub(r"\s+", " ", ln).strip(" :-") for ln in raw.split("\n") if re.sub(r"\s+", " ", ln).strip()]
    if not lines:
        return []

    m = NEW_STYLE_HEADER_RE.match(lines[0])
    if not m:
        return []

    kind_label = m.group(1).strip()
    discipline = m.group(2).strip(" :-")
    kind_norm = normalize_kind_sched(kind_label)

    teacher_hints: list[str] = []
    rooms: list[str] = []
    misc: list[str] = []

    for line in lines[1:]:
        line_rooms = find_rooms(line)
        for room in line_rooms:
            room_norm = normalize_room(room)
            if room_norm and room_norm not in rooms:
                rooms.append(room_norm)

        teacher_source = line
        teacher_source = re.sub(r"(?:Орджоникидзе,\s*3\s*)?ОРД\s*-\s*\d+[а-яa-z]?", " ", teacher_source, flags=re.I)
        teacher_source = re.sub(r"(?:Миклухо-Маклая,\s*6\s*)?ГК\s*-\s*\d+[а-яa-z]?", " ", teacher_source, flags=re.I)
        teacher_source = re.sub(r"ФОК\s*-?\s*\d+", " ", teacher_source, flags=re.I)
        teacher_source = re.sub(r"ДОТ\s+ДОТ\s*-\s*0+", " ", teacher_source, flags=re.I)
        teacher_source = re.sub(r"[()]", " ", teacher_source)
        teacher_source = re.sub(r"\s+", " ", teacher_source).strip()

        line_teachers = _extract_fullname_hints(teacher_source)
        if not line_teachers:
            line_teachers = extract_teacher_hints(teacher_source)
        for teacher in line_teachers:
            if teacher and teacher not in teacher_hints:
                teacher_hints.append(teacher)

        cleaned = teacher_source
        for teacher in line_teachers:
            cleaned = re.sub(re.escape(teacher), " ", cleaned, flags=re.I)
        cleaned = re.sub(r"[()]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,-")
        if cleaned:
            misc.append(cleaned)

    return [{
        "discipline_raw": discipline,
        "discipline_norm": normalize_disc(discipline),
        "disc_key": normalize_disc(discipline),
        "kind_raw": kind_label,
        "kind_norm": kind_norm,
        "room": " / ".join(dict.fromkeys(rooms)),
        "teacher_hint": " / ".join(dict.fromkeys(teacher_hints)),
        "subgroup": "",
        "misc": " ; ".join(dict.fromkeys(misc)),
        "source_block": raw,
    }]



def split_cell_into_blocks(text: str) -> list[str]:
    """Разбивает сложную ячейку расписания на отдельные смысловые блоки."""
    raw = _clean_multiline_text(text)
    if not raw:
        return []

    if _is_new_style_block(raw):
        pieces = [re.sub(r"\n{2,}", "\n", part).strip() for part in re.split(r"\n\s*-{5,}\s*\n", raw) if part and part.strip()]
        if len(pieces) <= 1:
            pieces = [p.strip() for p in re.split(r"(?=\n?(?:Лекция|Лабораторная работа|Практические и другие)\s*:)", raw) if p.strip()]
        out: list[str] = []
        for piece in pieces:
            piece = piece.strip()
            if piece and piece not in out:
                out.append(piece)
        return out

    s = _clean_cell_text(raw)
    parts: list[str] = []

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

        marker_parts = [p.strip(" /") for p in re.split(r"(?=\s+[абвгдежз]\)\s*(?:\(|[А-ЯA-ZЁ]))", chunk, flags=re.I) if p.strip(" /")]
        if len(marker_parts) > 1:
            for seg in marker_parts:
                if seg:
                    parts.append(seg)
            continue

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
    if _is_new_style_block(block):
        return _parse_new_block(block)

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
                room_norm = normalize_room(r)
                if room_norm and room_norm not in rooms:
                    rooms.append(room_norm)
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



def _resolve_sheet_name(xls: pd.ExcelFile, preferred: str | None = None) -> str:
    """Подбирает лист расписания для старого и нового форматов."""
    if preferred and preferred in xls.sheet_names:
        return preferred
    for candidate in ["ФМиЕН", "TDSheet", "Sheet1"]:
        if candidate in xls.sheet_names:
            return candidate
    return xls.sheet_names[0]



def _detect_layout(df: pd.DataFrame, sheet_name: str) -> dict[str, Any]:
    """Определяет схему чтения файла расписания."""
    first_cell = _txt(df.iloc[4, 0]).upper() if df.shape[0] > 5 and df.shape[1] > 0 else ""
    second_row_label = _txt(df.iloc[5, 1]).lower() if df.shape[0] > 6 and df.shape[1] > 1 else ""
    if sheet_name == "TDSheet" or (first_cell == "ДНИ" and second_row_label.startswith("часы")):
        return {
            "format": "tdsheet",
            "group_row": 5,
            "start_row": 6,
            "group_col_start": 4,
            "day_col": 0,
            "pair_col": None,
            "time_col": 1,
            "week_col": 3,
        }
    return {
        "format": "legacy",
        "group_row": 4,
        "start_row": 5,
        "group_col_start": 4,
        "day_col": 1,
        "pair_col": 2,
        "time_col": 3,
        "week_col": None,
    }



def _iter_schedule_rows(df: pd.DataFrame, layout: dict[str, Any]):
    """Итерируется по строкам расписания и восстанавливает день, пару и время."""
    current_day = ""
    current_time = ""
    current_pair: int | None = None

    for ridx in range(layout["start_row"], df.shape[0]):
        day_val = normalize_day(df.iloc[ridx, layout["day_col"]]) if layout.get("day_col") is not None else ""
        if day_val in DAY_CODES:
            if day_val != current_day:
                current_day = day_val
                current_time = ""
                current_pair = 0 if layout["format"] == "tdsheet" else None

        time_val = normalize_time(df.iloc[ridx, layout["time_col"]]) if layout.get("time_col") is not None else ""
        if time_val:
            if layout["format"] == "tdsheet":
                if time_val != current_time:
                    current_time = time_val
                    current_pair = int(current_pair or 0) + 1
            else:
                current_time = time_val

        if layout.get("pair_col") is not None:
            pair_val = df.iloc[ridx, layout["pair_col"]]
            try:
                current_pair = int(pair_val) if not pd.isna(pair_val) else current_pair
            except Exception:
                pass

        week_type = normalize_week_type(df.iloc[ridx, layout["week_col"]]) if layout.get("week_col") is not None else ""
        yield ridx, current_day, current_pair, current_time, week_type



def read_schedule_atoms(sched_xlsx: Path, sheet_name: str = "ФМиЕН") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает файл расписания и возвращает сырые, нормализованные и атомарные таблицы."""
    xls = pd.ExcelFile(sched_xlsx)
    actual_sheet = _resolve_sheet_name(xls, preferred=sheet_name)
    df = pd.read_excel(xls, sheet_name=actual_sheet, header=None)
    layout = _detect_layout(df, actual_sheet)

    rows = []
    atom_rows = []
    slot_seq = 1

    for col_idx in range(layout["group_col_start"], df.shape[1]):
        group_code = normalize_group(df.iloc[layout["group_row"], col_idx])
        if not group_code:
            continue

        for ridx, day, pair_val, time_val, week_type in _iter_schedule_rows(df, layout):
            raw_subject = df.iloc[ridx, col_idx]
            if not _clean_cell_text(raw_subject):
                continue
            if not day or not time_val:
                continue

            parsed = parse_subject(raw_subject)
            rows.append(
                {
                    "День недели": day,
                    "Пара": pair_val,
                    "Время": time_val,
                    "week_type": week_type,
                    "Учебная группа": group_code,
                    "Дисциплина": _txt(parsed.get("Дисциплина")),
                    "Вид_занятия": _txt(parsed.get("Вид_занятия")),
                    "Аудитория": normalize_room(parsed.get("Аудитория")),
                    "raw_cell": _clean_cell_text(raw_subject),
                    "source_sheet": actual_sheet,
                    "schedule_format": layout["format"],
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
                            "week_type": week_type,
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
                            "source_sheet": actual_sheet,
                            "schedule_format": layout["format"],
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
                "week_type",
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
                "source_sheet",
                "schedule_format",
            ]
        )
    norm_df = atoms_df[atoms_df["Вид_занятия_норм"].notna()].copy()
    return parsed_df, norm_df, atoms_df



def read_schedule(sched_xlsx: Path, sheet_name: str = "ФМиЕН") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает только parsed- и normalized-версии расписания."""
    parsed_df, norm_df, _atoms_df = read_schedule_atoms(sched_xlsx, sheet_name=sheet_name)
    return parsed_df, norm_df



def read_schedule_atoms_multi(schedule_paths: list[Path], sheet_name: str = "ФМиЕН") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает несколько файлов расписания и объединяет их в одну таблицу без конфликтов slot_id."""
    parsed_frames: list[pd.DataFrame] = []
    norm_frames: list[pd.DataFrame] = []
    atom_frames: list[pd.DataFrame] = []
    slot_seq = 1

    for sched_path in schedule_paths:
        parsed_df, norm_df, atoms_df = read_schedule_atoms(Path(sched_path), sheet_name=sheet_name)
        if len(atoms_df) > 0:
            atoms_df = atoms_df.copy()
            unique_slots = list(dict.fromkeys(atoms_df["slot_id"].astype(str).tolist()))
            remap = {old: f"S{slot_seq + idx:05d}" for idx, old in enumerate(unique_slots)}
            atoms_df["slot_id"] = atoms_df["slot_id"].astype(str).map(remap)
            if len(norm_df) > 0:
                norm_df = atoms_df[atoms_df["Вид_занятия_норм"].notna()].copy()
            slot_seq += len(unique_slots)
        parsed_frames.append(parsed_df)
        norm_frames.append(norm_df)
        atom_frames.append(atoms_df)

    parsed_all = pd.concat(parsed_frames, ignore_index=True) if parsed_frames else pd.DataFrame()
    norm_all = pd.concat(norm_frames, ignore_index=True) if norm_frames else pd.DataFrame()
    atoms_all = pd.concat(atom_frames, ignore_index=True) if atom_frames else pd.DataFrame()
    return parsed_all, norm_all, atoms_all
