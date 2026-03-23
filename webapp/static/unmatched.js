const elTbody = document.getElementById("tbody");
const elStatus = document.getElementById("status");
const elLog = document.getElementById("log");
const btnReload = document.getElementById("btnReload");
const btnRebuild = document.getElementById("btnRebuild");

function log(s) {
  elLog.textContent = (elLog.textContent + s + "\n").slice(-12000);
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

function mkOpt(text, value) {
  const o = document.createElement("option");
  o.value = value ?? text;
  o.textContent = text;
  return o;
}

async function loadCandidates(slotId, select) {
  select.innerHTML = "";
  select.appendChild(mkOpt("загрузка...", ""));

  const j = await apiGet(`/api/candidates?slot_id=${slotId}`);
  select.innerHTML = "";
  select.appendChild(mkOpt("выбрать...", ""));

  for (const c of j.candidates || []) {
    const label = `${c.teacher} (score ${Number(c.score).toFixed(2)})`;
    select.appendChild(mkOpt(label, c.teacher));
  }

  if ((j.candidates || []).length === 0) {
    select.appendChild(mkOpt("кандидатов нет", ""));
  }
}

function makeRow(r) {
  const tr = document.createElement("tr");
  tr.dataset.slotId = String(r.id);

  const td = (x) => {
    const e = document.createElement("td");
    e.textContent = x ?? "";
    return e;
  };

  tr.appendChild(td(r.day));
  tr.appendChild(td(r.pair));
  tr.appendChild(td(r.time));
  tr.appendChild(td(r.group));
  tr.appendChild(td(r.disc_key));
  tr.appendChild(td(r.kind));
  tr.appendChild(td(r.room));

  const tdSel = document.createElement("td");
  const sel = document.createElement("select");
  sel.className = "sel";
  tdSel.appendChild(sel);
  tr.appendChild(tdSel);

  const tdBtn = document.createElement("td");
  const btn = document.createElement("button");
  btn.textContent = "сохранить";
  btn.className = "btn";
  tdBtn.appendChild(btn);
  tr.appendChild(tdBtn);

  tr._slot = {group: r.group, disc_key: r.disc_key, kind: r.kind};
  tr._select = sel;
  tr._button = btn;

  return tr;
}

async function render() {
  elStatus.textContent = "загрузка...";
  elTbody.innerHTML = "";
  elLog.textContent = "";

  const j = await apiGet("/api/unmatched");
  elStatus.textContent = `строк: ${j.count}`;

  for (const r of j.rows || []) {
    const tr = makeRow(r);
    elTbody.appendChild(tr);

    loadCandidates(r.id, tr._select).catch(e => {
      tr._select.innerHTML = "";
      tr._select.appendChild(mkOpt("ошибка кандидатов", ""));
      log("candidates error: " + (e.message || e));
    });

    tr._button.addEventListener("click", async () => {
      const teacher = (tr._select.value || "").trim();
      if (!teacher) {
        log("выбери преподавателя");
        return;
      }

      try {
        await apiPost("/api/mapping", {slot: tr._slot, teacher});
        tr.classList.add("saved");
        log("сохранено правило: " + teacher);
      } catch (e) {
        log("ошибка: " + (e.message || e));
      }
    });
  }
}

btnReload.addEventListener("click", () => {
  render().catch(e => log("ошибка: " + (e.message || e)));
});

btnRebuild.addEventListener("click", async () => {
  try {
    log("пересборка...");
    const j = await apiPost("/api/rebuild", {});
    log("ok");
    log(JSON.stringify(j.stats || {}, null, 2));
    await render();
  } catch (e) {
    log("ошибка: " + (e.message || e));
  }
});

render().catch(e => log("ошибка: " + (e.message || e)));
