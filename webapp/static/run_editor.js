const elTbody = document.getElementById("tbody");
const elStatus = document.getElementById("status");
const elLog = document.getElementById("log");
const btnReload = document.getElementById("btnReload");
const btnRevertAll = document.getElementById("btnRevertAll");
const btnApplyFilter = document.getElementById("btnApplyFilter");
const elFilterQuery = document.getElementById("filterQuery");
const elFilterChangedOnly = document.getElementById("filterChangedOnly");
const btnAddRow = document.getElementById("btnAddRow");
const btnResetNewRow = document.getElementById("btnResetNewRow");
const newRowForm = document.getElementById("newRowForm");
const suggestionsHost = document.getElementById("suggestionsHost");
const newRowHint = document.getElementById("newRowHint");

const TABLE_FIELDS = [
  "Преподаватель",
  "Дисциплина",
  "Вид_работы",
  "Код_группы",
  "Номер_группы",
  "Кафедра",
  "Семестр",
  "Лекции_часы",
  "Практика_часы",
  "Лабораторные_часы",
  "Всего_часов",
];

const NEW_ROW_FIELDS = [
  ["Преподаватель", "преподаватель"],
  ["Дисциплина", "дисциплина"],
  ["Вид_работы", "вид занятия"],
  ["Код_группы", "код группы"],
  ["Номер_группы", "номер группы"],
  ["Кафедра", "кафедра"],
  ["Семестр", "семестр"],
  ["Лекции_часы", "лек"],
  ["Практика_часы", "пр"],
  ["Лабораторные_часы", "лаб"],
  ["Всего_часов", "всего"],
  ["Должность", "должность"],
  ["Тип_занятости", "тип занятости"],
  ["Количество_студентов", "кол-во студентов"],
  ["ООП", "ооп"],
  ["Код_ООП", "код ооп"],
];

let FIELD_SUGGESTIONS = {};

function log(s) {
  elLog.textContent = (elLog.textContent + s + "\n").slice(-20000);
}

async function parseJsonSafe(r) {
  const text = await r.text();
  try {
    return JSON.parse(text);
  } catch (_) {
    throw new Error(text || `HTTP ${r.status}`);
  }
}

async function apiGet(url) {
  const r = await fetch(url, {cache: "no-store"});
  const j = await parseJsonSafe(r);
  if (!r.ok) throw new Error(j.error || "request failed");
  return j;
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {})
  });
  const j = await parseJsonSafe(r);
  if (!r.ok) throw new Error(j.error || "request failed");
  return j;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function sanitizeFieldName(field) {
  return String(field || "").replaceAll(/[^a-zA-Z0-9_а-яА-ЯёЁ-]/g, "_");
}

function suggestionId(field) {
  return `suggest-${sanitizeFieldName(field)}`;
}

function suggestionAttr(field) {
  const items = FIELD_SUGGESTIONS[field] || [];
  return items.length ? ` list="${escapeHtml(suggestionId(field))}"` : "";
}

function suggestionMeta(field) {
  const items = FIELD_SUGGESTIONS[field] || [];
  if (!items.length) return "";
  return `<div class="field-tip">подсказок: ${items.length}</div>`;
}

function preserveNewRowValues() {
  return collectRowValues(newRowForm, "input[data-new-field]");
}

function renderSuggestionLists() {
  suggestionsHost.innerHTML = "";
  for (const [field, values] of Object.entries(FIELD_SUGGESTIONS)) {
    if (!Array.isArray(values) || !values.length) continue;
    const datalist = document.createElement("datalist");
    datalist.id = suggestionId(field);
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      datalist.appendChild(option);
    }
    suggestionsHost.appendChild(datalist);
  }
}

function isChanged(row, field) {
  return (row.changed_fields || []).includes(field);
}

function makeInput(row, field) {
  const value = row.current?.[field] ?? "";
  const cls = isChanged(row, field) ? "run-input changed" : "run-input";
  return `<input class="${cls}" data-field="${escapeHtml(field)}" value="${escapeHtml(value)}"${suggestionAttr(field)}>`;
}

function renderNewRowForm(preservedValues = null) {
  const values = preservedValues || {};
  newRowForm.innerHTML = "";
  for (const [field, label] of NEW_ROW_FIELDS) {
    const wrap = document.createElement("label");
    wrap.className = "new-row-field";
    wrap.innerHTML = `
      <span>${escapeHtml(label)}</span>
      <input class="txt" data-new-field="${escapeHtml(field)}" placeholder="${escapeHtml(label)}" value="${escapeHtml(values[field] || "")}"${suggestionAttr(field)}>
      ${suggestionMeta(field)}
    `;
    newRowForm.appendChild(wrap);
  }
}

function collectRowValues(container, selector) {
  const values = {};
  for (const input of container.querySelectorAll(selector)) {
    values[input.dataset.field || input.dataset.newField] = input.value;
  }
  return values;
}

function setBusy(buttons, disabled) {
  for (const button of buttons) {
    if (button) button.disabled = disabled;
  }
}

async function rebuildResponseLog(j, prefix) {
  if (j.rebuild_ok === false) {
    log(`${prefix}, но пересборка завершилась ошибкой: ${j.rebuild_error || "unknown error"}`);
    return;
  }
  log(`${prefix}, пересборка ok`);
  if (j.rebuild_stats) log(JSON.stringify(j.rebuild_stats, null, 2));
}

function makeRow(row) {
  const tr = document.createElement("tr");
  tr.dataset.rowId = String(row.row_id);
  if (row.changed) tr.classList.add("changed-row");
  if (row.deleted) tr.classList.add("deleted-row");
  if (row.added) tr.classList.add("added-row");

  const badgeParts = [];
  if (row.added) badgeParts.push('<div class="row-badge added">новая</div>');
  if (row.deleted) badgeParts.push('<div class="row-badge">удалено</div>');

  tr.innerHTML = `
    <td class="mono">${escapeHtml(row.display_row || row.row_id || row.excel_row || "")}${badgeParts.join("")}</td>
    <td>${makeInput(row, "Преподаватель")}</td>
    <td>${makeInput(row, "Дисциплина")}</td>
    <td>${makeInput(row, "Вид_работы")}</td>
    <td>${makeInput(row, "Код_группы")}</td>
    <td>${makeInput(row, "Номер_группы")}</td>
    <td>${makeInput(row, "Кафедра")}</td>
    <td>${makeInput(row, "Семестр")}</td>
    <td>${makeInput(row, "Лекции_часы")}</td>
    <td>${makeInput(row, "Практика_часы")}</td>
    <td>${makeInput(row, "Лабораторные_часы")}</td>
    <td>${makeInput(row, "Всего_часов")}</td>
    <td>
      <div class="col-actions">
        <button class="btn-save">сохранить</button>
        <button class="btn-revert">${row.added ? "убрать новую строку" : "отменить"}</button>
        <button class="btn-delete">${row.deleted ? "восстановить занятие" : "удалить занятие"}</button>
      </div>
    </td>
  `;

  const saveBtn = tr.querySelector(".btn-save");
  const revertBtn = tr.querySelector(".btn-revert");
  const deleteBtn = tr.querySelector(".btn-delete");

  saveBtn.addEventListener("click", async () => {
    try {
      setBusy([saveBtn, revertBtn, deleteBtn], true);
      const values = collectRowValues(tr, "input[data-field]");
      log(`сохраняю строку ${row.row_id} и пересобираю расписание...`);
      const j = await apiPost("/api/run_editor/save_row", {
        row_id: row.row_id,
        values,
        rebuild: true,
      });
      await rebuildResponseLog(j, `строка ${row.row_id} сохранена`);
      await render();
    } catch (e) {
      log("ошибка: " + (e.message || e));
    } finally {
      setBusy([saveBtn, revertBtn, deleteBtn], false);
    }
  });

  revertBtn.addEventListener("click", async () => {
    try {
      setBusy([saveBtn, revertBtn, deleteBtn], true);
      log(`${row.added ? "убираю новую строку" : "отменяю правки строки"} ${row.row_id} и пересобираю расписание...`);
      const j = await apiPost("/api/run_editor/revert_row", {
        row_id: row.row_id,
        rebuild: true,
      });
      await rebuildResponseLog(j, row.added ? `новая строка ${row.row_id} удалена из overlay` : `правки строки ${row.row_id} отменены`);
      await render();
    } catch (e) {
      log("ошибка: " + (e.message || e));
    } finally {
      setBusy([saveBtn, revertBtn, deleteBtn], false);
    }
  });

  deleteBtn.addEventListener("click", async () => {
    try {
      const action = row.deleted ? "восстанавливаю" : "удаляю";
      const endpoint = row.deleted ? "/api/run_editor/restore_row" : "/api/run_editor/delete_row";
      setBusy([saveBtn, revertBtn, deleteBtn], true);
      log(`${action} занятие в строке ${row.row_id} и пересобираю расписание...`);
      const j = await apiPost(endpoint, {
        row_id: row.row_id,
        rebuild: true,
      });
      await rebuildResponseLog(j, `строка ${row.row_id} обновлена`);
      await render();
    } catch (e) {
      log("ошибка: " + (e.message || e));
    } finally {
      setBusy([saveBtn, revertBtn, deleteBtn], false);
    }
  });

  return tr;
}

async function render() {
  const preservedValues = preserveNewRowValues();
  const query = encodeURIComponent(elFilterQuery.value || "");
  const changedOnly = elFilterChangedOnly.checked ? "1" : "0";
  elStatus.textContent = "загрузка...";
  elTbody.innerHTML = "";

  const j = await apiGet(`/api/run_editor/rows?query=${query}&changed_only=${changedOnly}`);
  FIELD_SUGGESTIONS = j.suggestions || {};
  renderSuggestionLists();
  renderNewRowForm(preservedValues);

  const overlay = j.overlay || {};
  elStatus.textContent = `показано: ${j.count} / ${j.total} | overlay rows: ${overlay.changed_rows || 0} | added: ${overlay.added_rows || 0} | deleted: ${overlay.deleted_rows || 0} | overlay fields: ${overlay.changed_fields || 0}`;

  const teacherCount = (FIELD_SUGGESTIONS["Преподаватель"] || []).length;
  const discCount = (FIELD_SUGGESTIONS["Дисциплина"] || []).length;
  const groupCount = (FIELD_SUGGESTIONS["Код_группы"] || []).length;
  newRowHint.textContent = `можно вводить вручную или выбирать из подсказок: преподаватели ${teacherCount}, дисциплины ${discCount}, коды групп ${groupCount}`;

  for (const row of j.rows || []) {
    elTbody.appendChild(makeRow(row));
  }
}

btnReload.addEventListener("click", () => {
  render().catch(e => log("ошибка: " + (e.message || e)));
});

btnApplyFilter.addEventListener("click", () => {
  render().catch(e => log("ошибка: " + (e.message || e)));
});

elFilterQuery.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    render().catch(err => log("ошибка: " + (err.message || err)));
  }
});

btnRevertAll.addEventListener("click", async () => {
  try {
    log("сбрасываю все правки overlay и пересобираю расписание...");
    const j = await apiPost("/api/run_editor/revert_all", {rebuild: true});
    await rebuildResponseLog(j, "overlay очищен");
    await render();
  } catch (e) {
    log("ошибка: " + (e.message || e));
  }
});

btnResetNewRow.addEventListener("click", () => {
  for (const input of newRowForm.querySelectorAll("input[data-new-field]")) {
    input.value = "";
  }
});

btnAddRow.addEventListener("click", async () => {
  try {
    setBusy([btnAddRow, btnResetNewRow], true);
    const values = collectRowValues(newRowForm, "input[data-new-field]");
    log("добавляю новую строку рун и пересобираю расписание...");
    const j = await apiPost("/api/run_editor/add_row", {
      values,
      rebuild: true,
    });
    await rebuildResponseLog(j, `новая строка ${j.row_id || ""} добавлена`);
    btnResetNewRow.click();
    await render();
  } catch (e) {
    log("ошибка: " + (e.message || e));
  } finally {
    setBusy([btnAddRow, btnResetNewRow], false);
  }
});

renderNewRowForm();
render().catch(e => log("ошибка: " + (e.message || e)));
