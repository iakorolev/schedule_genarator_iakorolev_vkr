"""Flask-приложение с маршрутами загрузки файлов, сборки черновика и скачивания результатов."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request, render_template, send_file

from core.candidates import Slot, build_candidates
from core.mappings import add_rule, load_mappings
from core.pipeline import build_timetable_bundle
from core.rule_suggestions import apply_rule_suggestions

from webapp.core.run_overlay import (
    EDITABLE_FIELDS,
    NEW_ROW_FIELDS,
    VIEW_FIELDS,
    add_overlay_row,
    delete_overlay_row,
    effective_run_path,
    ensure_run_edited_download,
    filter_run_rows,
    get_overlay_stats,
    get_run_field_suggestions,
    init_overlay_store,
    read_run_rows,
    restore_overlay_row,
    revert_overlay_all,
    revert_overlay_row,
    update_overlay_row,
)
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


def _schedule_upload_targets(sdir: Path) -> list[Path]:
    """Возвращает доступные файлы общего расписания для текущей сессии."""
    candidates = [sdir / "sched.xlsx", sdir / "sched_bak.xlsx", sdir / "sched_mag.xlsx"]
    return [p for p in candidates if p.exists()]



def _resolve_schedule_input(sdir: Path) -> Path | list[Path] | None:
    """Выбирает один или несколько входных файлов расписания."""
    targets = _schedule_upload_targets(sdir)
    if not targets:
        return None
    if len(targets) == 1:
        return targets[0]
    return targets





def _row_id_from_payload(data: dict[str, Any]) -> str:
    """Извлекает row_id из JSON payload, поддерживая старый excel_row."""
    raw = data.get("row_id", data.get("excel_row"))
    if raw in (None, ""):
        raise ValueError("row_id is required")
    return str(raw).strip()

def _validate_new_run_row(values: dict[str, Any]) -> list[str]:
    """Проверяет минимально обязательные поля новой строки РУН."""
    values = values if isinstance(values, dict) else {}

    def txt(name: str) -> str:
        return str(values.get(name, "") or "").strip()

    errors: list[str] = []
    if not txt("Преподаватель"):
        errors.append("укажите преподавателя")
    if not txt("Дисциплина"):
        errors.append("укажите дисциплину")
    if not txt("Вид_работы"):
        errors.append("укажите вид занятия")
    if not txt("Код_группы") and not txt("Номер_группы"):
        errors.append("укажите код группы или номер группы")
    return errors


def _build_bundle_for_session(sdir: Path) -> dict[str, Any]:
    """Пересобирает расписание по актуальному состоянию сессии и overlay-правок."""
    run_path = effective_run_path(sdir)
    sched_input = _resolve_schedule_input(sdir)
    if not run_path.exists() or not sched_input:
        raise FileNotFoundError("upload run and sched first")

    reference_path = sdir / "reference.xlsx"
    return build_timetable_bundle(
        run_path=run_path,
        sched_path=sched_input,
        out_dir=sdir,
        mappings_path=MAPPINGS_PATH,
        reference_path=reference_path if reference_path.exists() else None,
    )


@app.get("/")
def index() -> Any:
    """Отображает стартовую страницу web-приложения."""
    return render_template("index.html")


@app.get("/run-editor")
def run_editor_page() -> Any:
    """Открывает страницу редактирования РУН с overlay-правками."""
    return render_template("run_editor.html")


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
    if ftype not in ("run", "sched", "reference"):
        return jsonify({"error": "type must be run, sched or reference"}), 400

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "file is required"}), 400

    if ftype == "run":
        out = sdir / "run.xlsx"
    elif ftype == "reference":
        out = sdir / "reference.xlsx"
    else:
        fname = (file.filename or "").lower()
        if "бакалавр" in fname:
            out = sdir / "sched_bak.xlsx"
        elif "магистр" in fname:
            out = sdir / "sched_mag.xlsx"
        else:
            out = sdir / "sched.xlsx"
    file.save(out)

    if ftype == "run":
        init_overlay_store(sdir)

    return jsonify({"ok": True, "saved": str(out)})


@app.get("/api/session/status")
def api_session_status() -> Any:
    """Возвращает сводную информацию о состоянии текущей сессии."""
    sdir = _session_dir()
    if sdir is None:
        return jsonify({"has_session": False})

    overlay_stats = get_overlay_stats(sdir)
    return jsonify({
        "has_session": True,
        "session": str(sdir),
        "has_run": (sdir / "run.xlsx").exists(),
        "has_sched": bool(_schedule_upload_targets(sdir)),
        "has_reference": (sdir / "reference.xlsx").exists(),
        "has_unmatched": (sdir / "unmatched_slots.xlsx").exists(),
        "has_output": (sdir / "timetable_by_teachers.xlsx").exists(),
        "has_compare": (sdir / "compare_summary.xlsx").exists(),
        "has_rule_suggestions": (sdir / "rule_suggestions.xlsx").exists(),
        "has_accepted_suggestions": (sdir / "accepted_rule_suggestions.xlsx").exists(),
        "has_teacher_loads": (sdir / "teacher_load_summary.xlsx").exists(),
        "has_assignments": (sdir / "schedule_with_teachers.xlsx").exists(),
        "has_external_slots": (sdir / "external_non_department_slots.xlsx").exists(),
        "overlay": overlay_stats,
    })


@app.post("/api/build_draft")
def api_build_draft() -> Any:
    """Запускает построение черновика преподавательского расписания."""
    sdir, err = _require_session()
    if err:
        return err

    try:
        bundle = _build_bundle_for_session(sdir)
    except FileNotFoundError:
        return jsonify({"error": "upload run and sched first"}), 400

    return jsonify({
        "ok": True,
        "out_dir": str(sdir),
        "stats": bundle.get("stats", {}),
        "files": bundle.get("files", {}),
        "overlay": get_overlay_stats(sdir),
        "effective_run": str(effective_run_path(sdir)),
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


@app.post("/api/apply_suggestions")
def api_apply_suggestions() -> Any:
    """Применяет рекомендации из rule_suggestions.xlsx в mappings.json."""
    sdir, err = _require_session()
    if err:
        return err

    suggestions_path = sdir / "rule_suggestions.xlsx"
    if not suggestions_path.exists():
        return jsonify({"error": "no rule_suggestions.xlsx, build draft first"}), 404

    data = request.get_json(silent=True) or {}
    priority = str(data.get("priority") or "high").strip() or "high"
    modes = data.get("modes") if isinstance(data.get("modes"), list) else None
    sources = data.get("sources") if isinstance(data.get("sources"), list) else None
    limit = data.get("limit")
    try:
        limit = int(limit) if limit not in (None, "") else None
    except Exception:
        limit = None

    result = apply_rule_suggestions(
        suggestions_path=suggestions_path,
        mappings_path=MAPPINGS_PATH,
        out_dir=sdir,
        priority=priority,
        modes=modes,
        sources=sources,
        limit=limit,
    )

    response = {"ok": True, **result.get("summary", {}), "files": result.get("files", {})}

    if data.get("rebuild"):
        try:
            bundle = _build_bundle_for_session(sdir)
        except FileNotFoundError:
            return jsonify({"error": "upload run and sched first"}), 400
        response["rebuild_stats"] = bundle.get("stats", {})
        response["rebuild_files"] = bundle.get("files", {})
    return jsonify(response)


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

    try:
        bundle = _build_bundle_for_session(sdir)
    except FileNotFoundError:
        return jsonify({"error": "upload run and sched first"}), 400

    return jsonify({
        "ok": True,
        "stats": bundle.get("stats", {}),
        "files": bundle.get("files", {}),
        "overlay": get_overlay_stats(sdir),
        "effective_run": str(effective_run_path(sdir)),
    })


@app.get("/api/run_editor/rows")
def api_run_editor_rows() -> Any:
    """Возвращает строки редактора РУН с учётом overlay-правок."""
    sdir, err = _require_session()
    if err:
        return err

    run_path = sdir / "run.xlsx"
    if not run_path.exists():
        return jsonify({"error": "upload run first"}), 400

    query = request.args.get("query", "")
    changed_only = str(request.args.get("changed_only", "")).strip().lower() in {"1", "true", "yes", "on"}
    try:
        limit_raw = request.args.get("limit", "")
        limit = int(limit_raw) if str(limit_raw).strip() else None
    except Exception:
        limit = None

    rows = read_run_rows(run_path, sdir)
    filtered = filter_run_rows(rows, query=query, changed_only=changed_only)
    if limit is not None and limit >= 0:
        filtered = filtered[:limit]

    return jsonify({
        "ok": True,
        "count": len(filtered),
        "total": len(rows),
        "rows": filtered,
        "editable_fields": EDITABLE_FIELDS,
        "view_fields": VIEW_FIELDS,
        "new_row_fields": NEW_ROW_FIELDS,
        "suggestions": get_run_field_suggestions(run_path, sdir),
        "overlay": get_overlay_stats(sdir),
    })


@app.post("/api/run_editor/add_row")
def api_run_editor_add_row() -> Any:
    """Добавляет новую строку РУН в overlay и при необходимости пересобирает расписание."""
    sdir, err = _require_session()
    if err:
        return err

    run_path = sdir / "run.xlsx"
    if not run_path.exists():
        return jsonify({"error": "upload run first"}), 400

    data = request.get_json(silent=True) or {}
    values = data.get("values") if isinstance(data.get("values"), dict) else {}
    rebuild = bool(data.get("rebuild", True))

    validation_errors = _validate_new_run_row(values)
    if validation_errors:
        return jsonify({"error": "; ".join(validation_errors), "validation_errors": validation_errors}), 400

    try:
        _, row_id = add_overlay_row(sdir, run_path, values)
        edited_path = ensure_run_edited_download(sdir)
    except Exception as e:
        return jsonify({"error": f"add row failed: {e}"}), 500

    response: dict[str, Any] = {
        "ok": True,
        "added": True,
        "row_id": row_id,
        "overlay": get_overlay_stats(sdir),
        "effective_run": str(effective_run_path(sdir)),
        "download_run": str(edited_path),
    }

    if rebuild:
        try:
            bundle = _build_bundle_for_session(sdir)
            response["rebuild_ok"] = True
            response["rebuild_stats"] = bundle.get("stats", {})
            response["rebuild_files"] = bundle.get("files", {})
        except Exception as e:
            response["rebuild_ok"] = False
            response["rebuild_error"] = str(e)
    return jsonify(response)


@app.post("/api/run_editor/save_row")
def api_run_editor_save_row() -> Any:
    """Сохраняет overlay-правки одной строки РУН и при необходимости пересобирает расписание."""
    sdir, err = _require_session()
    if err:
        return err

    run_path = sdir / "run.xlsx"
    if not run_path.exists():
        return jsonify({"error": "upload run first"}), 400

    data = request.get_json(silent=True) or {}
    values = data.get("values") if isinstance(data.get("values"), dict) else {}
    try:
        row_id = _row_id_from_payload(data)
    except Exception:
        return jsonify({"error": "row_id is required"}), 400
    rebuild = bool(data.get("rebuild", True))

    try:
        update_overlay_row(sdir, run_path, row_id, values)
        edited_path = ensure_run_edited_download(sdir)
    except Exception as e:
        return jsonify({"error": f"save row failed: {e}"}), 500

    response: dict[str, Any] = {
        "ok": True,
        "saved": True,
        "row_id": row_id,
        "overlay": get_overlay_stats(sdir),
        "effective_run": str(effective_run_path(sdir)),
        "download_run": str(edited_path),
    }

    if rebuild:
        try:
            bundle = _build_bundle_for_session(sdir)
            response["rebuild_ok"] = True
            response["rebuild_stats"] = bundle.get("stats", {})
            response["rebuild_files"] = bundle.get("files", {})
        except Exception as e:
            response["rebuild_ok"] = False
            response["rebuild_error"] = str(e)
    return jsonify(response)


@app.post("/api/run_editor/revert_row")
def api_run_editor_revert_row() -> Any:
    """Отменяет overlay-правки одной строки и при необходимости пересобирает расписание."""
    sdir, err = _require_session()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    try:
        row_id = _row_id_from_payload(data)
    except Exception:
        return jsonify({"error": "row_id is required"}), 400
    rebuild = bool(data.get("rebuild", True))

    revert_overlay_row(sdir, row_id)
    edited_path = ensure_run_edited_download(sdir)
    response: dict[str, Any] = {
        "ok": True,
        "reverted": True,
        "row_id": row_id,
        "overlay": get_overlay_stats(sdir),
        "effective_run": str(effective_run_path(sdir)),
        "download_run": str(edited_path),
    }

    if rebuild:
        try:
            bundle = _build_bundle_for_session(sdir)
            response["rebuild_ok"] = True
            response["rebuild_stats"] = bundle.get("stats", {})
            response["rebuild_files"] = bundle.get("files", {})
        except Exception as e:
            response["rebuild_ok"] = False
            response["rebuild_error"] = str(e)
    return jsonify(response)


@app.post("/api/run_editor/delete_row")
def api_run_editor_delete_row() -> Any:
    """Помечает строку РУН как удалённую и при необходимости пересобирает расписание."""
    sdir, err = _require_session()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    try:
        row_id = _row_id_from_payload(data)
    except Exception:
        return jsonify({"error": "row_id is required"}), 400
    rebuild = bool(data.get("rebuild", True))

    delete_overlay_row(sdir, row_id)
    edited_path = ensure_run_edited_download(sdir)
    response: dict[str, Any] = {
        "ok": True,
        "deleted": True,
        "row_id": row_id,
        "overlay": get_overlay_stats(sdir),
        "effective_run": str(effective_run_path(sdir)),
        "download_run": str(edited_path),
    }

    if rebuild:
        try:
            bundle = _build_bundle_for_session(sdir)
            response["rebuild_ok"] = True
            response["rebuild_stats"] = bundle.get("stats", {})
            response["rebuild_files"] = bundle.get("files", {})
        except Exception as e:
            response["rebuild_ok"] = False
            response["rebuild_error"] = str(e)
    return jsonify(response)


@app.post("/api/run_editor/restore_row")
def api_run_editor_restore_row() -> Any:
    """Снимает пометку удаления со строки РУН и при необходимости пересобирает расписание."""
    sdir, err = _require_session()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    try:
        row_id = _row_id_from_payload(data)
    except Exception:
        return jsonify({"error": "row_id is required"}), 400
    rebuild = bool(data.get("rebuild", True))

    restore_overlay_row(sdir, row_id)
    edited_path = ensure_run_edited_download(sdir)
    response: dict[str, Any] = {
        "ok": True,
        "restored": True,
        "row_id": row_id,
        "overlay": get_overlay_stats(sdir),
        "effective_run": str(effective_run_path(sdir)),
        "download_run": str(edited_path),
    }

    if rebuild:
        try:
            bundle = _build_bundle_for_session(sdir)
            response["rebuild_ok"] = True
            response["rebuild_stats"] = bundle.get("stats", {})
            response["rebuild_files"] = bundle.get("files", {})
        except Exception as e:
            response["rebuild_ok"] = False
            response["rebuild_error"] = str(e)
    return jsonify(response)


@app.post("/api/run_editor/revert_all")
def api_run_editor_revert_all() -> Any:
    """Очищает все overlay-правки и при необходимости пересобирает расписание."""
    sdir, err = _require_session()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    rebuild = bool(data.get("rebuild", True))

    revert_overlay_all(sdir)
    edited_path = ensure_run_edited_download(sdir)
    response: dict[str, Any] = {
        "ok": True,
        "reverted_all": True,
        "overlay": get_overlay_stats(sdir),
        "effective_run": str(effective_run_path(sdir)),
        "download_run": str(edited_path),
    }

    if rebuild:
        try:
            bundle = _build_bundle_for_session(sdir)
            response["rebuild_ok"] = True
            response["rebuild_stats"] = bundle.get("stats", {})
            response["rebuild_files"] = bundle.get("files", {})
        except Exception as e:
            response["rebuild_ok"] = False
            response["rebuild_error"] = str(e)
    return jsonify(response)


@app.get("/download/run-edited")
def download_run_edited() -> Any:
    """Отдаёт файл run_edited.xlsx, собранный из исходного РУН и overlay-правок."""
    sdir, err = _require_session()
    if err:
        return err

    original = sdir / "run.xlsx"
    if not original.exists():
        return jsonify({"error": "no run file, upload run first"}), 404
    p = ensure_run_edited_download(sdir)
    if not p.exists():
        return jsonify({"error": "failed to build run_edited.xlsx"}), 500
    return send_file(p, as_attachment=True)


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


@app.get("/download/compare-summary")
def download_compare_summary() -> Any:
    """Отдаёт сводный файл сравнения с эталоном."""
    sdir, err = _require_session()
    if err:
        return err

    p = sdir / "compare_summary.xlsx"
    if not p.exists():
        return jsonify({"error": "no compare file, upload reference and build draft first"}), 404
    return send_file(p, as_attachment=True)


@app.get("/download/compare-slots")
def download_compare_slots() -> Any:
    """Отдаёт детальный файл сравнения слотов с эталоном."""
    sdir, err = _require_session()
    if err:
        return err

    p = sdir / "compare_slots.xlsx"
    if not p.exists():
        return jsonify({"error": "no compare slots file, upload reference and build draft first"}), 404
    return send_file(p, as_attachment=True)


@app.get("/download/rule-suggestions")
def download_rule_suggestions() -> Any:
    """Отдаёт файл с рекомендациями по правилам."""
    sdir, err = _require_session()
    if err:
        return err
    p = sdir / "rule_suggestions.xlsx"
    if not p.exists():
        return jsonify({"error": "no rule suggestions file, build draft first"}), 404
    return send_file(p, as_attachment=True)


@app.get("/download/accepted-suggestions")
def download_accepted_suggestions() -> Any:
    """Отдаёт отчёт по применённым рекомендациям."""
    sdir, err = _require_session()
    if err:
        return err
    p = sdir / "accepted_rule_suggestions.xlsx"
    if not p.exists():
        return jsonify({"error": "no accepted suggestions file, apply suggestions first"}), 404
    return send_file(p, as_attachment=True)


if __name__ == "__main__":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    app.run(host="127.0.0.1", port=5000, debug=True)
