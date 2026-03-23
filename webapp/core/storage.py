from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def session_root(base: Path) -> Path:
    root = base / "sessions"
    ensure_dir(root)
    return root


def create_session(base: Path) -> Path:
    sid = now_id()
    sdir = session_root(base) / sid
    ensure_dir(sdir)
    write_json(sdir / "meta.json", {"id": sid})
    return sdir


def set_current(base: Path, session_dir: Path) -> None:
    write_json(base / "current_session.json", {"path": str(session_dir)})


def get_current(base: Path) -> Path | None:
    cur = read_json(base / "current_session.json")
    p = cur.get("path")
    if not p:
        return None
    pp = Path(p)
    return pp if pp.exists() else None
