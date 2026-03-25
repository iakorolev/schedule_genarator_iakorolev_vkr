"""Нормализация текстовых сущностей: дней, времени, дисциплин, групп и видов занятий."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"


def _txt(value: Any) -> str:
    """Безопасно приводит значение к строке без NaN и лишних пробелов."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_text(value: Any) -> str:
    """Очищает строку от нестандартных пробелов и выравнивает форматирование."""
    s = _txt(value)
    if not s:
        return ""
    s = s.replace("\xa0", " ")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s.strip()


DAY_ALIASES = {
    "ПОНЕДЕЛЬНИК": "ПН",
    "ВТОРНИК": "ВТ",
    "СРЕДА": "СР",
    "ЧЕТВЕРГ": "ЧТ",
    "ПЯТНИЦА": "ПТ",
    "СУББОТА": "СБ",
    "ВОСКРЕСЕНЬЕ": "ВС",
}


KIND_ALIASES_DEFAULT = {
    "лек": "лек",
    "лекция": "лек",
    "лекц": "лек",
    "сем": "сем",
    "семинар": "сем",
    "пр": "сем",
    "практ": "сем",
    "практика": "сем",
    "практические": "сем",
    "лаб": "лаб",
    "лр": "лаб",
    "лабораторная": "лаб",
    "лабораторные": "лаб",
}

DISC_ALIASES_DEFAULT = {
    "ин яз": "иностранный язык",
    "ин язык": "иностранный язык",
    "иностр язык": "иностранный язык",
    "иностранный язык 1ый язык": "иностранный язык",
    "иностранный язык 1 й язык": "иностранный язык",
    "иностранный язык 2ой язык": "второй иностранный язык",
    "иностранный язык 2 ой язык": "второй иностранный язык",
    "русский язык и культура речи": "русский язык и культура речи",
    "компьютерные науки и технологии программирования": "компьютерные науки и технологии программирования",
    "дискретная математика и математическая логика": "дискретная математика и математическая логика",
    "основы военной подготовки безопасность жизнедеятельности": "основы военной подготовки безопасность жизнедеятельности",
}


@lru_cache(maxsize=1)
def kind_aliases() -> dict[str, str]:
    """Возвращает словарь алиасов для видов занятий."""
    path = CONFIG_DIR / "kind_aliases.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k).strip().lower(): str(v).strip().lower() for k, v in data.items() if str(k).strip() and str(v).strip()}
        except Exception:
            pass
    return KIND_ALIASES_DEFAULT.copy()


@lru_cache(maxsize=1)
def disc_aliases() -> dict[str, str]:
    """Возвращает словарь алиасов для дисциплин."""
    path = CONFIG_DIR / "disc_aliases.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {_normalize_disc_base(str(k)): _normalize_disc_base(str(v)) for k, v in data.items() if str(k).strip() and str(v).strip()}
        except Exception:
            pass
    return {_normalize_disc_base(k): _normalize_disc_base(v) for k, v in DISC_ALIASES_DEFAULT.items()}


def normalize_day(day: str) -> str:
    """Нормализует обозначение дня недели."""
    s = clean_text(day).upper()
    for k, v in DAY_ALIASES.items():
        s = s.replace(k, v)
    return s


TIME_RE = re.compile(r"(\d{1,2})[\.:](\d{2})\s*-\s*(\d{1,2})[\.:](\d{2})")


def normalize_time(t: str) -> str:
    """Нормализует строковое представление времени пары."""
    s = clean_text(t).replace(",", ".")
    m = TIME_RE.search(s)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}-{int(m.group(3)):02d}:{m.group(4)}"
    s = re.sub(r"\s+", "", s)
    return s


ROOM_PATTERNS = [
    re.compile(r"ОРД-\d+[а-яa-z]?", re.I),
    re.compile(r"ФОК-?\d+", re.I),
    re.compile(r"ТУИС", re.I),
    re.compile(r"дистанционно", re.I),
    re.compile(r"см\.\s*расписание[^;]*", re.I),
]


def normalize_room(room: str) -> str:
    """Нормализует обозначение аудитории или дистанционного формата."""
    s = clean_text(room)
    s = s.replace(" ;", ";").replace("; ", "; ")
    s = s.replace("  ", " ")
    return s


def normalize_group(raw: str) -> str:
    """Нормализует обозначение учебной группы."""
    s = clean_text(raw)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", "", s)
    return s


def extract_group_parts(group: str) -> dict[str, str]:
    """Извлекает составные части кода учебной группы."""
    g = normalize_group(group)
    m = re.match(r"(?P<prefix>[A-Za-zА-Яа-яЁё]+(?:бд|мд|зд)?)-(?P<num>\d+)-(?P<year>\d+)$", g)
    if not m:
        return {
            "group_norm": g,
            "group_prefix": g,
            "group_num": "",
            "course_year": "",
            "family_key": g,
            "prefix_year": g,
        }
    prefix = m.group("prefix")
    num = m.group("num")
    year = m.group("year")
    return {
        "group_norm": g,
        "group_prefix": prefix,
        "group_num": num,
        "course_year": year,
        "family_key": f"{prefix}-{year}",
        "prefix_year": f"{prefix}-{year}",
    }


def group_family_key(group: str) -> str:
    """Возвращает семейный ключ группы без номера подгруппы."""
    return extract_group_parts(group).get("family_key", "")


def _normalize_disc_base(s: str) -> str:
    """Выполняет базовую нормализацию названия дисциплины."""
    s = clean_text(s).lower().replace("ё", "е")
    if not s:
        return ""
    s = re.sub(r"\([^)]*подгруппа[^)]*\)", " ", s)
    s = re.sub(r"\((?:подгруппа|группа)[^)]*\)", " ", s)
    s = re.sub(r"^\s*[абвгдежз]\)\s*", "", s)
    s = s.replace("\\", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,:;!?*%№]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_disc(s: str) -> str:
    """Возвращает нормализованный ключ дисциплины."""
    s = _normalize_disc_base(s)
    aliases = disc_aliases()
    return aliases.get(s, s)


def disc_tokens(s: str) -> set[str]:
    """Разбивает дисциплину на токены для нестрогого сравнения."""
    s = normalize_disc(s)
    if not s:
        return set()
    toks = [t for t in re.split(r"\s+", s) if t]
    stop = {"и", "в", "во", "на", "по", "для", "о", "об", "к", "из", "с", "со", "при", "или", "а", "основы"}
    toks = [t for t in toks if t not in stop and len(t) > 1]
    return set(toks)


def jaccard(a: set[str], b: set[str]) -> float:
    """Вычисляет коэффициент Жаккара для двух наборов токенов."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


def best_disc_match(target_disc: str, candidates: list[str]) -> tuple[str | None, float]:
    """Находит лучшее совпадение дисциплины в списке кандидатов."""
    ta = disc_tokens(target_disc)
    if not ta:
        return None, 0.0
    best = None
    best_score = 0.0
    for c in candidates:
        sc = jaccard(ta, disc_tokens(c))
        if sc > best_score:
            best_score = sc
            best = c
    return best, best_score


def normalize_kind(value: str, source: str | None = None) -> str | None:
    """Нормализует вид занятия с опорой на словарь алиасов."""
    s = clean_text(value).lower().replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    aliases = kind_aliases()
    for k, v in aliases.items():
        if s == k or s.startswith(k + " ") or f" {k} " in f" {s} ":
            return v
    if "лек" in s:
        return "лек"
    if "сем" in s or s == "пр" or "практ" in s:
        return "сем"
    if "лаб" in s or "лр" in s:
        return "лаб"
    return None


def normalize_kind_un(kind: str) -> str | None:
    """Нормализует вид работы из файла РУН/УН."""
    return normalize_kind(kind, source="run")


def normalize_kind_sched(kind: str) -> str | None:
    """Нормализует вид занятия из файла общего расписания."""
    return normalize_kind(kind, source="sched")


def split_multi_kinds(kind_raw: str) -> list[str]:
    """Разделяет комбинированный вид занятия на отдельные значения."""
    s = clean_text(kind_raw).lower()
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"\s*/\s*", s) if p.strip()]
    out: list[str] = []
    for p in parts:
        n = normalize_kind(p)
        if n and n not in out:
            out.append(n)
    if not out:
        n = normalize_kind(s)
        if n:
            out.append(n)
    return out


def find_rooms(text: str) -> list[str]:
    """Извлекает из строки возможные обозначения аудиторий."""
    s = clean_text(text)
    found: list[str] = []
    for pat in ROOM_PATTERNS:
        for m in pat.finditer(s):
            room = normalize_room(m.group(0))
            if room and room not in found:
                found.append(room)
    return found


TEACHER_RE = re.compile(r"(?:доц\.|проф\.|ст\.преп\.|асс\.)?\s*[А-ЯЁ][а-яё-]+\s+[А-Я]\.[А-Я]\.", re.U)


def looks_like_teacher(text: str) -> bool:
    """Проверяет, похоже ли значение на ФИО преподавателя."""
    return bool(TEACHER_RE.search(clean_text(text)))


def extract_teacher_hints(text: str) -> list[str]:
    """Извлекает teacher hints из текстового блока расписания."""
    s = clean_text(text)
    out = []
    for m in TEACHER_RE.finditer(s):
        t = re.sub(r"\s+", " ", m.group(0)).strip()
        t = re.sub(r"^(?:доц\.|проф\.|ст\.преп\.|асс\.)\s*", "", t, flags=re.I)
        if t and t.lower() != "преподаватель" and t not in out:
            out.append(t)
    return out


def teacher_lastname(name: str) -> str:
    """Выделяет фамилию преподавателя для сопоставления."""
    s = clean_text(name)
    if not s:
        return ""
    return s.split()[0].lower().replace("ё", "е")
