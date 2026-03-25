"""Flask-приложение с маршрутами загрузки файлов, сборки черновика и скачивания результатов."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request, render_template, send_file

from core.candidates import Slot, build_candidates
from core.mappings import add_rule, load_mappings
from core.pipeline import build_timetable_bundle  # у тебя уже есть/будет

from webapp.core.storage import create_session, set_current, get_current


app = Flask(__name__, static_folder="static", template_folder="templates")

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

MAPPINGS_PATH = DATA / "mappings.json"


def _session_dir() -> Path | None:
    """Возвращает путь к текущей активной сессии web-приложения."""
    return get_current(DATA)


def _require_session() -> tuple[Path | None, Any | None]:
    """Проверяет наличие активной сессии и готовит ответ об ошибке при её отсутствии."""
    sdir = _session_dir()
    if sdir is None:
        return None, (jsonify({"error": "no session, upload files first"}), 400)
    return sdir, None


@app.get("/")
def index() -> Any:
    """Отображает стартовую страницу web-приложения."""
    return render_template("index.html")


@app.post("/api/session/new")
def api_session_new() -> Any:
    """Создаёт новую пользовательскую сессию и делает её текущей."""
    sdir = create_session(DATA)
    set_current(DATA, sdir)
    return jsonify({"ok": True, "session": str(sdir)})


@app.post("/api/upload")
def api_upload() -> Any:
    """Принимает и сохраняет загруженный входной файл."""
    sdir, err = _require_session()
    if err:
        return err

    ftype = (request.form.get("type") or "").strip()
    if ftype not in ("run", "sched"):
        return jsonify({"error": "type must be run or sched"}), 400

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "file is required"}), 400

    out = sdir / ("run.xlsx" if ftype == "run" else "sched.xlsx")
    file.save(out)

    return jsonify({"ok": True, "saved": str(out)})


@app.get("/api/session/status")
def api_session_status() -> Any:
    """Возвращает сводную информацию о состоянии текущей сессии."""
    sdir = _session_dir()
    if sdir is None:
        return jsonify({"has_session": False})

    return jsonify({
        "has_session": True,
        "session": str(sdir),
        "has_run": (sdir / "run.xlsx").exists(),
        "has_sched": (sdir / "sched.xlsx").exists(),
        "has_unmatched": (sdir / "unmatched_slots.xlsx").exists(),
        "has_output": (sdir / "timetable_by_teachers.xlsx").exists(),
        "has_teacher_loads": (sdir / "teacher_load_summary.xlsx").exists(),
        "has_assignments": (sdir / "schedule_with_teachers.xlsx").exists(),
        "has_external_slots": (sdir / "external_non_department_slots.xlsx").exists(),
    })


@app.post("/api/build_draft")
def api_build_draft() -> Any:
    """Запускает построение черновика преподавательского расписания."""
    sdir, err = _require_session()
    if err:
        return err

    run_path = sdir / "run.xlsx"
    sched_path = sdir / "sched.xlsx"
    if not run_path.exists() or not sched_path.exists():
        return jsonify({"error": "upload run and sched first"}), 400

    bundle = build_timetable_bundle(
        run_path=run_path,
        sched_path=sched_path,
        out_dir=sdir,
        mappings_path=MAPPINGS_PATH,
    )

    return jsonify({
        "ok": True,
        "out_dir": str(sdir),
        "stats": bundle.get("stats", {}),
        "files": bundle.get("files", {}),
    })


@app.get("/unmatched")
def unmatched_page() -> Any:
    """Открывает страницу с нераспределёнными слотами."""
    return render_template("unmatched.html")


@app.get("/api/unmatched")
def api_unmatched() -> Any:
    """Возвращает список нераспределённых слотов в JSON-формате."""
    sdir, err = _require_session()
    if err:
        return err

    p = sdir / "unmatched_slots.xlsx"
    if not p.exists():
        return jsonify({"count": 0, "rows": []})

    df = pd.read_excel(p).fillna("").reset_index(drop=True)

    rows = []
    for i, r in df.iterrows():
        rows.append({
            "id": int(i),
            "day": str(r.get("День недели", "")),
            "pair": str(r.get("Пара", "")),
            "time": str(r.get("Время", "")),
            "group": str(r.get("Учебная группа", "")),
            "disc_key": str(r.get("disc_key", "")),
            "kind": str(r.get("Вид_занятия_норм", "")),
            "room": str(r.get("Аудитория", "")),
        })
    return jsonify({"count": len(rows), "rows": rows})


@app.get("/api/candidates")
def api_candidates() -> Any:
    """Возвращает список кандидатов для выбранного проблемного слота."""
    sdir, err = _require_session()
    if err:
        return err

    slot_id = request.args.get("slot_id", None)
    if slot_id is None:
        return jsonify({"error": "slot_id is required"}), 400
    try:
        slot_id = int(slot_id)
    except Exception:
        return jsonify({"error": "slot_id must be int"}), 400

    unmatched_path = sdir / "unmatched_slots.xlsx"
    # Предпочитаем run_atoms.xlsx: он содержит нормализованные поля, нужные allocator/candidates.
    run_atoms_path = sdir / "run_atoms.xlsx"
    un_expanded_path = sdir / "un_svodnaya_expanded.xlsx"

    if not unmatched_path.exists():
        return jsonify({"error": "no unmatched file, build draft first"}), 400
    if not run_atoms_path.exists() and not un_expanded_path.exists():
        return jsonify({"error": "no run_atoms.xlsx or un_svodnaya_expanded.xlsx, build draft first"}), 400

    dfu = pd.read_excel(unmatched_path).fillna("").reset_index(drop=True)
    if slot_id < 0 or slot_id >= len(dfu):
        return jsonify({"error": "slot_id out of range"}), 400

    r = dfu.iloc[slot_id]
    slot = Slot(
        group=str(r.get("Учебная группа", "")).strip(),
        disc_key=str(r.get("disc_key", "")).strip(),
        kind=str(r.get("Вид_занятия_норм", "")).strip(),
    )

    try:
        source_path = run_atoms_path if run_atoms_path.exists() else un_expanded_path
        run_like = pd.read_excel(source_path)
        cands = build_candidates(run_like, slot, top_n=10, min_score=0.15)
        return jsonify({
            "slot_id": slot_id,
            "slot": {"group": slot.group, "disc_key": slot.disc_key, "kind": slot.kind},
            "candidates": cands,
        })
    except Exception as e:
        return jsonify({"error": f"candidates failed: {e}"}), 500


@app.post("/api/mapping")
def api_mapping() -> Any:
    """Сохраняет ручное правило выбора преподавателя."""
    data = request.get_json(silent=True) or {}
    slot = data.get("slot", {})
    teacher = (data.get("teacher") or "").strip()

    if not isinstance(slot, dict):
        return jsonify({"error": "slot must be object"}), 400
    if teacher == "":
        return jsonify({"error": "teacher is required"}), 400

    when = {
        "group": (slot.get("group") or "").strip(),
        "disc_key": (slot.get("disc_key") or "").strip(),
        "kind": (slot.get("kind") or "").strip(),
    }

    if when["group"] == "" or when["disc_key"] == "" or when["kind"] == "":
        return jsonify({"error": "slot.group/slot.disc_key/slot.kind required"}), 400

    add_rule(MAPPINGS_PATH, when=when, assign={"teacher": teacher})
    return jsonify({"ok": True})


@app.get("/api/mappings")
def api_mappings() -> Any:
    """Возвращает текущий набор пользовательских правил."""
    return jsonify(load_mappings(MAPPINGS_PATH))


@app.post("/api/rebuild")
def api_rebuild() -> Any:
    """Пересобирает расписание с учётом пользовательских правил."""
    sdir, err = _require_session()
    if err:
        return err

    run_path = sdir / "run.xlsx"
    sched_path = sdir / "sched.xlsx"
    if not run_path.exists() or not sched_path.exists():
        return jsonify({"error": "upload run and sched first"}), 400

    bundle = build_timetable_bundle(
        run_path=run_path,
        sched_path=sched_path,
        out_dir=sdir,
        mappings_path=MAPPINGS_PATH,
    )

    return jsonify({
        "ok": True,
        "stats": bundle.get("stats", {}),
        "files": bundle.get("files", {}),
    })


@app.get("/download/timetable")
def download_timetable() -> Any:
    """Отдаёт итоговое преподавательское расписание для скачивания."""
    sdir, err = _require_session()
    if err:
        return err

    p = sdir / "timetable_by_teachers.xlsx"
    if not p.exists():
        return jsonify({"error": "no output, build draft first"}), 404
    return send_file(p, as_attachment=True)


@app.get("/download/unmatched")
def download_unmatched() -> Any:
    """Отдаёт файл с нераспределёнными слотами."""
    sdir, err = _require_session()
    if err:
        return err

    p = sdir / "unmatched_slots.xlsx"
    if not p.exists():
        return jsonify({"error": "no unmatched file, build draft first"}), 404
    return send_file(p, as_attachment=True)


@app.get("/download/teacher-loads")
def download_teacher_loads() -> Any:
    """Отдаёт файл со сводкой по нагрузке преподавателей."""
    sdir, err = _require_session()
    if err:
        return err

    p = sdir / "teacher_load_summary.xlsx"
    if not p.exists():
        return jsonify({"error": "no teacher load summary, build draft first"}), 404
    return send_file(p, as_attachment=True)


@app.get("/download/external-slots")
def download_external_slots() -> Any:
    """Отдаёт файл с исключёнными внешними слотами."""
    sdir, err = _require_session()
    if err:
        return err

    p = sdir / "external_non_department_slots.xlsx"
    if not p.exists():
        return jsonify({"error": "no external slots file, build draft first"}), 404
    return send_file(p, as_attachment=True)


@app.get("/download/assignments")
def download_assignments() -> Any:
    """Отдаёт файл со всеми назначениями преподавателей."""
    sdir, err = _require_session()
    if err:
        return err

    p = sdir / "schedule_with_teachers.xlsx"
    if not p.exists():
        return jsonify({"error": "no assignments file, build draft first"}), 404
    return send_file(p, as_attachment=True)


if __name__ == "__main__":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    app.run(host="127.0.0.1", port=5000, debug=True)
