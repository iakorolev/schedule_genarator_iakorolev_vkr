"""Операции с файловыми сессиями и JSON-метаданными web-приложения."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def now_id() -> str:
    """Возвращает строковый идентификатор сессии на основе текущей даты и времени."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    """Создаёт каталог, если он ещё не существует."""
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Сохраняет словарь в JSON-файл через временный файл."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Читает JSON-файл и возвращает словарь с данными."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def session_root(base: Path) -> Path:
    """Возвращает корневой каталог пользовательских сессий."""
    root = base / "sessions"
    ensure_dir(root)
    return root


def create_session(base: Path) -> Path:
    """Создаёт новую сессию и файл метаданных для неё."""
    sid = now_id()
    sdir = session_root(base) / sid
    ensure_dir(sdir)
    write_json(sdir / "meta.json", {"id": sid})
    return sdir


def set_current(base: Path, session_dir: Path) -> None:
    """Сохраняет путь к текущей активной сессии."""
    write_json(base / "current_session.json", {"path": str(session_dir)})


def get_current(base: Path) -> Path | None:
    """Возвращает путь к текущей сессии, если она существует."""
    cur = read_json(base / "current_session.json")
    p = cur.get("path")
    if not p:
        return None
    pp = Path(p)
    return pp if pp.exists() else None
