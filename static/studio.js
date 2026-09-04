const API = "";
const $ = (s) => document.querySelector(s);
const SYSTEM_PROMPT = "你是径舟书斋的伴读书童「小舟」。仅根据文档片段回答。";
const uploadZone = $("#uploadZone");
const fileInput = $("#fileInput");
const docList = $("#docList");
const docCount = $("#docCount");
const messages = $("#messages");
const chatEmpty = $("#chatEmpty");
const chatInput = $("#chatInput");
const sendBtn = $("#sendBtn");
const headerInfo = $("#headerInfo");
let selectedDocs = new Set();
let allDocs = [];
uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFiles(fileInput.files));
sendBtn.addEventListener("click", send);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
document.querySelectorAll(".feat-card").forEach((btn) => {
  btn.addEventListener("click", () => runFeature(btn.dataset.feature));
});
$("#modalClose")?.addEventListener("click", () => { $("#modalMask").style.display = "none"; });
$("#sampleBtn")?.addEventListener("click", loadSample);
async function handleFiles(files) {
  for (const f of files) {
    const text = await f.text();
    await indexText(f.name.replace(/\.(txt|md|pdf)$/i, ""), text);
  }
}
async function indexText(name, text) {
  const id = "d" + Math.random().toString(36).slice(2, 10);
  const res = await fetch(API + "/api/index", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: id, name, text }),
  });
  if (!res.ok) { toast("收卷未成"); return; }
  toast("已收卷：" + name);
  await refreshDocs();
}
async function loadSample() {
  try {
    const res = await fetch(API + "/api/sample", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    toast("示例教材已入舱");
    await refreshDocs();
    const sample = allDocs.find((d) => d.id === "sample-linalg");
    if (sample) selectedDocs.add(sample.id);
    renderDocs(); updateInput();
  } catch (e) { toast("示例未载入"); }
}
async function refreshDocs() {
  allDocs = await fetch(API + "/api/docs").then((r) => r.json());
  renderDocs(); updateInput();
}
function renderDocs() {
  docCount.textContent = String(allDocs.length);
  if (!allDocs.length) { docList.innerHTML = '<div class="doc-empty">舱中尚无卷册</div>'; return; }
  docList.innerHTML = allDocs.map((d) =>
    `<div class="doc-card${selectedDocs.has(d.id) ? " selected" : ""}" data-id="${d.id}"><div class="dc-name">${esc(d.name)}</div><div class="dc-meta">${d.chunks} 片段</div></div>`
  ).join("");
  docList.querySelectorAll(".doc-card").forEach((el) => {
    el.addEventListener("click", () => {
      const id = el.dataset.id;
      if (selectedDocs.has(id)) selectedDocs.delete(id); else selectedDocs.add(id);
      renderDocs(); updateInput();
    });
  });
}
function updateInput() {
  const on = selectedDocs.size > 0;
  chatInput.disabled = sendBtn.disabled = !on;
  document.querySelectorAll(".feat-card").forEach((b) => { b.disabled = !on; });
  headerInfo.textContent = on ? `已择 ${selectedDocs.size} 卷` : "择卷而问";
}
async function send() {
  const query = chatInput.value.trim();
  if (!query || !selectedDocs.size) return;
  chatInput.value = "";
  if (chatEmpty) chatEmpty.style.display = "none";
  addMsg("user", query);
  const bubble = addMsg("assistant", "…");
  const body = JSON.stringify({ query, doc_ids: [...selectedDocs], system: SYSTEM_PROMPT });
  const res = await fetch(API + "/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  const raw = await res.text();
  let text = ""; let sources = [];
  raw.split("\n").forEach((line) => {
    if (!line.startsWith("data: ") || line.includes("[DONE]")) return;
    try {
      const p = JSON.parse(line.slice(6));
      if (p.text) text += p.text;
      if (p.sources) sources = p.sources;
      if (p.error) text = p.error;
    } catch (e) {}
  });
  bubble.querySelector(".msg-bubble").textContent = text || "未得回答";
  if (sources.length) {
    const box = document.createElement("div");
    box.className = "msg-sources";
    box.innerHTML = `<span class="src-chip" style="cursor:default">检索解释：BGE 向量余弦 Top-${sources.length}，点击笺片查看原文</span>`;
    sources.forEach((s) => {
      const chip = document.createElement("button");
      chip.type = "button"; chip.className = "src-chip";
      chip.textContent = `${s.name} 片段${s.chunk} · ${s.score}`;
      chip.onclick = () => openChunk(s);
      box.appendChild(chip);
    });
    bubble.appendChild(box);
  }
}
function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.innerHTML = `<div class="msg-body"><div class="msg-bubble">${esc(text)}</div></div>`;
  messages.appendChild(el);
  return el;
}
async function runFeature(name) {
  const body = JSON.stringify({ doc_ids: [...selectedDocs], query: "" });
  if (name === "guide") return loadGuide(body);
  $("#modalMask").style.display = "flex";
  $("#modalTitle").textContent = name;
  $("#modalBody").textContent = "生成中…";
  const path = { flashcards: "/api/flashcards", quiz: "/api/quiz", mindmap: "/api/mindmap", report: "/api/report", table: "/api/table", infographic: "/api/infographic" }[name];
  if (!path) return;
  const res = await fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body });
  $("#modalBody").innerHTML = `<pre>${esc(JSON.stringify(await res.json(), null, 2))}</pre>`;
}
async function loadGuide(body) {
  $("#modalMask").style.display = "flex";
  $("#modalTitle").textContent = "指南";
  $("#modalBody").textContent = "生成中…";
  const res = await fetch(API + "/api/study-guide", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  const g = await res.json();
  $("#modalBody").innerHTML = `<h3>${esc(g.title || "学习指南")}</h3><p>${esc(g.overview || "")}</p><pre>${esc(JSON.stringify(g, null, 2))}</pre>`;
}
async function openChunk(s) {
  const id = s.doc_id;
  const idx = s.chunk ?? s.chunk_idx;
  const item = await fetch(API + "/api/docs/" + id + "/chunks/" + idx).then((r) => r.json());
  let drawer = $(".chunk-drawer");
  if (!drawer) { drawer = document.createElement("div"); drawer.className = "chunk-drawer"; document.body.appendChild(drawer); }
  drawer.innerHTML = `<div>${esc(item.doc_name || s.name)} 片段${idx}</div><div>${esc(s.why || "")}</div><pre>${esc(item.text || s.preview || "")}</pre>`;
}
function toast(msg) { headerInfo.textContent = msg; }
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => {
    if (c === "&") return "\u0026amp;";
    if (c === "<") return "\u0026lt;";
    if (c === ">") return "\u0026gt;";
    if (c === '"') return "\u0026quot;";
    return "&#39;";
  });
}
refreshDocs();
window.openChunk = openChunk;
