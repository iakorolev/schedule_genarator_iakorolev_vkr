const elStatus = document.getElementById("status");
const elLog = document.getElementById("log");

const btnNew = document.getElementById("btnNew");
const btnUploadRun = document.getElementById("btnUploadRun");
const btnUploadSched = document.getElementById("btnUploadSched");
const btnBuild = document.getElementById("btnBuild");

const fileRun = document.getElementById("fileRun");
const fileSched = document.getElementById("fileSched");

function log(s) {
  elLog.textContent = (elLog.textContent + s + "\n").slice(-12000);
}

async function apiGet(url) {
  const r = await fetch(url, {cache: "no-store"});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "request failed");
  return j;
}

async function apiPostJson(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {})
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "request failed");
  return j;
}

async function apiPostFile(type, file) {
  const fd = new FormData();
  fd.append("type", type);
  fd.append("file", file);

  const r = await fetch("/api/upload", {method: "POST", body: fd});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "upload failed");
  return j;
}

async function refresh() {
  const j = await apiGet("/api/session/status");
  if (!j.has_session) {
    elStatus.textContent = "сессии нет";
    return;
  }
  elStatus.textContent = `сессия: ok | run=${j.has_run} | sched=${j.has_sched} | unmatched=${j.has_unmatched} | output=${j.has_output}`;
}

btnNew.addEventListener("click", async () => {
  try {
    const j = await apiPostJson("/api/session/new", {});
    log("создана сессия: " + j.session);
    await refresh();
  } catch (e) {
    log("ошибка: " + e.message);
  }
});

btnUploadRun.addEventListener("click", async () => {
  try {
    if (!fileRun.files.length) { log("выбери файл run"); return; }
    const j = await apiPostFile("run", fileRun.files[0]);
    log("run загружен: " + j.saved);
    await refresh();
  } catch (e) {
    log("ошибка: " + e.message);
  }
});

btnUploadSched.addEventListener("click", async () => {
  try {
    if (!fileSched.files.length) { log("выбери файл sched"); return; }
    const j = await apiPostFile("sched", fileSched.files[0]);
    log("sched загружен: " + j.saved);
    await refresh();
  } catch (e) {
    log("ошибка: " + e.message);
  }
});

btnBuild.addEventListener("click", async () => {
  try {
    log("строю черновик...");
    const j = await apiPostJson("/api/build_draft", {});
    log("ok");
    log(JSON.stringify(j.stats || {}, null, 2));
    await refresh();
  } catch (e) {
    log("ошибка: " + e.message);
  }
});

refresh().catch(() => {});
