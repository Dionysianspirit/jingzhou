const API = "";
const $ = (s) => document.querySelector(s);
const SYSTEM_PROMPT = "你是径舟书斋的伴读书童「小舟」。仅根据文档片段回答。";
const ERROR_TYPES = ["概念混淆", "公式遗忘", "审题偏移", "推导断裂"];
const uploadZone = $("#uploadZone");
const fileInput = $("#fileInput");
const docList = $("#docList");
const docCount = $("#docCount");
const messages = $("#messages");
const chatEmpty = $("#chatEmpty");
const chatInput = $("#chatInput");
const sendBtn = $("#sendBtn");
const headerInfo = $("#headerInfo");
const modalMask = $("#modalMask");
const modalTitle = $("#modalTitle");
const modalBody = $("#modalBody");

let selectedDocs = new Set();
let allDocs = [];
let isStreaming = false;
let fcCards = [];
let fcIdx = 0;
let quizQuestions = [];
let quizAnswers = [];
let quizSubmitted = false;
let quizTrace = [];
let quizStartTime = 0;
let quizErrorHistory = [];
let convId = null;
let convs = [];
let toastUntil = 0;
let toastTimer = null;
const TOAST_MS = 2800;

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => handleFiles(fileInput.files));
sendBtn.addEventListener("click", send);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
document.querySelectorAll(".feat-card").forEach((btn) => {
  btn.addEventListener("click", () => runFeature(btn.dataset.feature));
});
$("#modalClose")?.addEventListener("click", closeModal);
modalMask?.addEventListener("click", (e) => { if (e.target === modalMask) closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
$("#sampleBtn")?.addEventListener("click", loadSample);
$("#convNew")?.addEventListener("click", newConversation);

async function handleFiles(files) {
  for (const f of files) {
    const lower = f.name.toLowerCase();
    try {
      if (lower.endsWith(".pdf") || f.type === "application/pdf") {
        await indexPDF(f);
      } else if (lower.endsWith(".docx") || lower.endsWith(".epub")) {
        await indexBinary(f);
      } else if (lower.endsWith(".txt") || lower.endsWith(".md") || (f.type || "").startsWith("text/")) {
        await indexText(f.name.replace(/\.(txt|md)$/i, ""), await readTextSmart(f));
      } else {
        toast("暂只收 PDF / DOCX / EPUB / TXT / MD：" + f.name);
      }
    } catch (err) {
      toast("收卷失败：" + (err.message || f.name));
    }
  }
  if (fileInput) fileInput.value = "";
}

async function indexPDF(file) {
  if (typeof pdfjsLib === "undefined") throw new Error("pdf.js 未加载");
  pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  setPg(8);
  toast("正在拆解 PDF…");
  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  const parts = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    parts.push(content.items.map((it) => it.str).join(" "));
    setPg(8 + (i / pdf.numPages) * 70);
  }
  const text = parts.join("\n\n").replace(/[ \t]+\n/g, "\n").trim();
  if (!text) { setPg(0); throw new Error("该 PDF 没有可提取文字（可能是扫描件）"); }
  await indexText(file.name.replace(/\.pdf$/i, ""), text);
}

function setPg(p) {
  const bar = $("#progress");
  if (!bar) return;
  bar.style.width = Math.max(0, Math.min(100, p)) + "%";
  if (p >= 100) setTimeout(() => { bar.style.width = "0"; }, 400);
}

function newDocId() {
  return "d" + Math.random().toString(36).slice(2, 10);
}

async function errDetail(res, fallback) {
  try {
    const data = await res.json();
    return typeof data.detail === "string" ? data.detail : fallback;
  } catch (e) { return fallback; }
}

// 中文 TXT 仍常见 GBK 编码：先按 UTF-8 严格解码，失败再按 GB18030。
async function readTextSmart(file) {
  const buf = await file.arrayBuffer();
  try { return new TextDecoder("utf-8", { fatal: true }).decode(buf); }
  catch (e) {
    try { return new TextDecoder("gb18030").decode(buf); }
    catch (e2) { return new TextDecoder("utf-8").decode(buf); }
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("文件读取失败"));
    reader.readAsDataURL(file);
  });
}

async function indexText(name, text) {
  const res = await fetch(API + "/api/index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: newDocId(), name, text }),
  });
  if (!res.ok) { setPg(0); throw new Error(await errDetail(res, name)); }
  setPg(100);
  toast("已收卷：" + name);
  await refreshDocs();
}

// DOCX / EPUB 交给本机服务解析（标准库，不出本机），再走同一条索引链路。
async function indexBinary(file) {
  const name = file.name.replace(/\.(docx|epub)$/i, "");
  setPg(10);
  toast("正在解析 " + file.name + "…");
  const data_b64 = await fileToBase64(file);
  setPg(45);
  const res = await fetch(API + "/api/index-file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: newDocId(), name, filename: file.name, data_b64 }),
  });
  if (!res.ok) { setPg(0); throw new Error(await errDetail(res, file.name)); }
  setPg(100);
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
  try { allDocs = await fetch(API + "/api/docs").then((r) => r.json()); }
  catch { return; }
  renderDocs(); updateInput();
}

function renderDocs() {
  docCount.textContent = String(allDocs.length);
  if (!allDocs.length) {
    docList.innerHTML = '<div class="doc-empty">舱中尚无卷册</div>';
    return;
  }
  docList.innerHTML = allDocs.map((d) =>
    `<div class="doc-card${selectedDocs.has(d.id) ? " selected" : ""}" data-id="${esc(d.id)}">
      <div class="dc-main">
        <div class="dc-name" title="${esc(d.name)}">${esc(d.name)}</div>
        <div class="dc-meta">${d.chunks} 片段</div>
      </div>
      <button type="button" class="dc-del" title="移出书斋" aria-label="将《${esc(d.name)}》移出书斋">✕</button>
    </div>`
  ).join("");
  docList.querySelectorAll(".doc-card").forEach((el) => {
    el.addEventListener("click", () => {
      const id = el.dataset.id;
      if (selectedDocs.has(id)) selectedDocs.delete(id); else selectedDocs.add(id);
      renderDocs(); updateInput();
    });
    el.querySelector(".dc-del").addEventListener("click", (e) => {
      e.stopPropagation();
      removeDoc(el.dataset.id);
    });
  });
}

async function removeDoc(id) {
  const doc = allDocs.find((d) => d.id === id);
  if (!doc || isStreaming) return;
  if (!window.confirm(`将《${doc.name}》移出书斋？已建立的索引会一并删除。`)) return;
  try {
    const res = await fetch(API + "/api/remove", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: id }),
    });
    if (!res.ok) throw new Error(await errDetail(res, "移出未成"));
    selectedDocs.delete(id);
    await refreshDocs();
    toast("已移出：" + doc.name);
  } catch (e) { toast(e.message || "移出未成"); }
}

function updateInput() {
  const on = selectedDocs.size > 0 && !isStreaming;
  chatInput.disabled = sendBtn.disabled = !on;
  document.querySelectorAll(".feat-card").forEach((b) => { b.disabled = !on; });
  // 提示语要留给人看清：还在展示期内就不抢它的位置
  if (Date.now() >= toastUntil) headerInfo.textContent = statusLine();
}

function statusLine() {
  return selectedDocs.size ? `已择 ${selectedDocs.size} 卷` : "择卷而问";
}

// 读取 SSE：逐块解析 `data: {...}`，每个事件交给 onPayload。
async function readSSE(res, onPayload) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      for (const line of part.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") continue;
        try { onPayload(JSON.parse(data)); } catch (e) {}
      }
    }
  }
}

async function send() {
  const query = chatInput.value.trim();
  if (!query || !selectedDocs.size || isStreaming) return;
  chatInput.value = "";
  if (chatEmpty) chatEmpty.style.display = "none";
  addMsg("user", query);
  const el = addMsg("assistant", "小舟研墨中…");
  const bubble = el.querySelector(".msg-bubble");
  bubble.classList.add("pending");
  isStreaming = true; updateInput();
  let text = ""; let sources = []; let meta = {}; let failed = "";
  try {
    const body = { query, doc_ids: [...selectedDocs], system: SYSTEM_PROMPT, remember: true };
    if (convId) body.conversation_id = convId;
    const res = await fetch(API + "/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      failed = await errDetail(res, `请求失败（${res.status}）`);
      if (res.status === 404 && convId) convId = null;  // 对话已被删除：下一问另起新对话
    } else {
      await readSSE(res, (p) => {
        if (p.text) {
          text += p.text;
          bubble.classList.remove("pending");
          bubble.textContent = text;
          messages.scrollTop = messages.scrollHeight;
        }
        if (p.sources) sources = p.sources;
        if (p.search_query !== undefined) meta = p;
        if (p.conversation_id) convId = p.conversation_id;
        if (p.error) failed = p.error;
      });
    }
  } catch (e) {
    failed = failed || "舟楫失连，请再试";
  }
  bubble.classList.remove("pending");
  if (text) fillAnswer(el, text, sources, meta.rewritten ? meta.search_query : "");
  if (failed) {
    if (text) addNote(el, failed, "err");
    else { bubble.textContent = failed; bubble.classList.add("err"); }
  } else if (!text) {
    bubble.textContent = "未得回答";
  }
  isStreaming = false; updateInput();
  messages.scrollTop = messages.scrollHeight;
  refreshConversations();
}

function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.innerHTML = `<div class="msg-body"><div class="msg-bubble">${esc(text)}</div></div>`;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function addNote(el, text, cls) {
  const note = document.createElement("div");
  note.className = "msg-note" + (cls ? " " + cls : "");
  note.textContent = text;
  el.appendChild(note);
  return note;
}

const MODE_LABELS = { hybrid: "语义 + 关键词融合", dense: "BGE 向量余弦", bm25: "BM25 关键词" };

// 一轮回答的完整呈现：正文、（若被改写）实际用于检索的问句、可点开的来源笺。
function fillAnswer(el, text, sources, searchQuery) {
  const bubble = el.querySelector(".msg-bubble");
  bubble.classList.add("rendered");
  bubble.innerHTML = renderMd(text);
  if (searchQuery) {
    const note = addNote(el, "");
    note.innerHTML = `<b>检索问句：</b>${esc(searchQuery)}`;
    note.title = "追问先被改写成独立问句，再拿去检索教材";
  }
  if (!sources || !sources.length) return;
  const box = document.createElement("div");
  box.className = "msg-sources";
  const mode = MODE_LABELS[sources[0].mode] || MODE_LABELS.dense;
  box.innerHTML = `<span class="src-chip label">检索解释：${esc(mode)} Top-${sources.length}</span>`;
  sources.forEach((s) => {
    const chip = document.createElement("button");
    chip.type = "button"; chip.className = "src-chip";
    chip.textContent = `${s.name} 片段${s.chunk} · ${s.score}`;
    if (s.why) chip.title = s.why;
    chip.onclick = () => openChunk(s);
    box.appendChild(chip);
  });
  el.appendChild(box);
}

// ---- 对话：列表、恢复、删除 ----
async function refreshConversations() {
  try {
    const res = await fetch(API + "/api/conversations");
    convs = res.ok ? await res.json() : [];
  } catch (e) { convs = []; }
  renderConversations();
}

function renderConversations() {
  const box = $("#convList");
  if (!box) return;
  if (!convs.length) {
    box.innerHTML = '<div class="doc-empty">问过的话会留在这里</div>';
    return;
  }
  box.innerHTML = convs.map((c) =>
    `<div class="conv-item${c.conversation_id === convId ? " active" : ""}" data-id="${esc(c.conversation_id)}">
      <div class="cv-main">
        <div class="cv-title" title="${esc(c.title)}">${esc(c.title || "未命名对话")}</div>
        <div class="cv-meta">${c.exchanges} 问 · ${fmtTime(c.updated_at)}</div>
      </div>
      <button type="button" class="cv-del" title="删除对话" aria-label="删除对话：${esc(c.title)}">✕</button>
    </div>`
  ).join("");
  box.querySelectorAll(".conv-item").forEach((el) => {
    el.addEventListener("click", () => openConversation(el.dataset.id));
    el.querySelector(".cv-del").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConversation(el.dataset.id);
    });
  });
}

function clearMessages() {
  messages.querySelectorAll(".msg").forEach((el) => el.remove());
}

function newConversation() {
  if (isStreaming) return;
  convId = null;
  clearMessages();
  if (chatEmpty) chatEmpty.style.display = "";
  renderConversations();
  if (!chatInput.disabled) chatInput.focus();
}

async function openConversation(id) {
  if (isStreaming) return;
  let conv;
  try {
    const res = await fetch(API + "/api/conversations/" + encodeURIComponent(id));
    if (!res.ok) throw new Error("对话已不存在");
    conv = await res.json();
  } catch (e) {
    toast(e.message || "对话未能载入");
    refreshConversations();
    return;
  }
  convId = conv.conversation_id;
  clearMessages();
  if (chatEmpty) chatEmpty.style.display = "none";
  let lastQuestion = "";
  (conv.turns || []).forEach((t) => {
    if (t.role === "user") { lastQuestion = t.content; addMsg("user", t.content); return; }
    const el = addMsg("assistant", "");
    const rewritten = t.search_query && t.search_query !== lastQuestion ? t.search_query : "";
    fillAnswer(el, t.content, t.sources || [], rewritten);
  });
  // 接着聊时沿用这段对话当时选的卷册（只保留仍在书斋里的）。
  const live = new Set(allDocs.map((d) => d.id));
  const keep = (conv.doc_ids || []).filter((d) => live.has(d));
  if (keep.length) selectedDocs = new Set(keep);
  renderDocs(); updateInput(); renderConversations();
  messages.scrollTop = messages.scrollHeight;
}

async function deleteConversation(id) {
  if (isStreaming) return;
  const conv = convs.find((c) => c.conversation_id === id);
  if (!window.confirm(`删除对话「${conv ? conv.title : id}」？`)) return;
  try {
    await fetch(API + "/api/conversations/" + encodeURIComponent(id), { method: "DELETE" });
  } catch (e) {}
  if (id === convId) newConversation();
  refreshConversations();
}

function openModal(title) {
  modalTitle.textContent = title;
  modalMask.style.display = "flex";
}

function closeModal() {
  modalMask.style.display = "none";
}

function failBox(res, fallback) {
  return res && res.status === 503
    ? "未配置 LLM_API_KEY，此功能需要大模型。可先用「考核」查看识海偏误图（无 Key 时走示例卷）。"
    : fallback;
}

async function runFeature(name) {
  if (!selectedDocs.size || isStreaming) return;
  const titles = {
    agent: "自主学习", guide: "指南", flashcards: "笺卡", quiz: "考核",
    mindmap: "脉络图", report: "析报", table: "簿册", infographic: "览图",
  };
  openModal(titles[name] || name);
  modalBody.innerHTML = `<div class="loading-spin">正在准备${titles[name] || name}</div>`;
  const body = JSON.stringify({ doc_ids: [...selectedDocs], query: chatInput.value.trim(), count: 6 });
  try {
    if (name === "agent") return await runAgent();
    if (name === "quiz") return await loadQuiz(body);
    if (name === "flashcards") return await loadFlashcards(body);
    if (name === "guide") return await loadGuide(body);
    if (name === "mindmap") return await loadMindmap(body);
    if (name === "report") return await loadReport(body);
    if (name === "table") return await loadTable(body);
    if (name === "infographic") return await loadInfographic(body);
  } catch (e) {
    modalBody.innerHTML = `<p class="err-line">${esc(e.message || "装载失败")}</p>`;
  }
}

async function loadFlashcards(body) {
  const res = await fetch(API + "/api/flashcards", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!res.ok) { modalBody.innerHTML = `<p class="err-line">${esc(failBox(res, await res.text()))}</p>`; return; }
  const data = await res.json();
  fcCards = data.cards || [];
  fcIdx = 0;
  if (!fcCards.length) { modalBody.innerHTML = '<p class="empty-line">未能制得笺卡</p>'; return; }
  renderFlashcard();
}

function renderFlashcard() {
  const c = fcCards[fcIdx];
  modalBody.innerHTML = `
    <div class="fc-container">
      <div class="fc-nav">
        <button type="button" id="fcPrev"${fcIdx === 0 ? " disabled" : ""}>‹</button>
        <span class="fc-count">${fcIdx + 1} / ${fcCards.length}</span>
        <button type="button" id="fcNext"${fcIdx === fcCards.length - 1 ? " disabled" : ""}>›</button>
      </div>
      <div class="fc-card" id="fcCard">
        <div class="fc-card-inner">
          <div class="fc-front">${esc(c.q)}</div>
          <div class="fc-back">${esc(c.a)}</div>
        </div>
      </div>
      <span class="fc-hint">点击卡片翻转</span>
      <div class="fc-export">
        <button type="button" id="fcAnki">导出 Anki 卡包</button>
        <button type="button" id="fcTxt" class="ghost">导出文本</button>
      </div>
      <span class="fc-export-note" id="fcExportNote">共 ${fcCards.length} 张 · 重复导入同一批卡片会更新而不是重复添加</span>
    </div>`;
  $("#fcAnki").addEventListener("click", () => exportAnki("apkg"));
  $("#fcTxt").addEventListener("click", () => exportAnki("tsv"));
  $("#fcCard").addEventListener("click", () => $("#fcCard").classList.toggle("flipped"));
  $("#fcPrev").addEventListener("click", () => { if (fcIdx > 0) { fcIdx -= 1; renderFlashcard(); } });
  $("#fcNext").addEventListener("click", () => { if (fcIdx < fcCards.length - 1) { fcIdx += 1; renderFlashcard(); } });
}

function ankiDeckName() {
  if (selectedDocs.size === 1) {
    const doc = allDocs.find((d) => selectedDocs.has(d.id));
    if (doc) return "径舟::" + doc.name.replace(/::/g, " ").slice(0, 60);
  }
  return "径舟";
}

async function exportAnki(format) {
  const note = $("#fcExportNote");
  const buttons = [$("#fcAnki"), $("#fcTxt")];
  const deck = ankiDeckName();
  buttons.forEach((b) => { if (b) b.disabled = true; });
  try {
    const res = await fetch(API + "/api/flashcards/export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cards: fcCards.map((c) => ({ q: String(c.q ?? ""), a: String(c.a ?? "") })),
        deck, format,
      }),
    });
    if (!res.ok) throw new Error(await errDetail(res, "导出未成"));
    const blob = await res.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = deck.replace(/::/g, "-") + (format === "tsv" ? ".txt" : ".apkg");
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 4000);
    const count = res.headers.get("X-Card-Count") || fcCards.length;
    if (note) note.textContent = format === "tsv"
      ? `已导出 ${count} 张为文本：在 Anki 里用「文件 → 导入」选择该文件`
      : `已导出 ${count} 张到牌组「${deck}」：双击 .apkg 即可导入 Anki`;
  } catch (e) {
    if (note) note.textContent = e.message || "导出未成";
  }
  buttons.forEach((b) => { if (b) b.disabled = false; });
}

async function loadQuiz(body) {
  let data = null;
  const res = await fetch(API + "/api/quiz", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (res.ok) data = await res.json();
  else if (res.status === 503) data = await fetch(API + "/api/sample-quiz").then((r) => r.json());
  else { modalBody.innerHTML = `<p class="err-line">${esc(await res.text())}</p>`; return; }
  quizQuestions = data.questions || [];
  quizAnswers = new Array(quizQuestions.length).fill(-1);
  quizSubmitted = false;
  quizTrace = [];
  quizStartTime = Date.now();
  if (data.fallback) toast("未配置大模型，已载入示例考核");
  if (!quizQuestions.length) { modalBody.innerHTML = '<p class="empty-line">未能拟得题目</p>'; return; }
  renderQuiz();
}

function recordTrace(qi, oi, action) {
  quizTrace.push({ qi, oi, action, ts: Date.now() - quizStartTime });
}

function renderWrongAnalysis(q, qi) {
  const userOi = quizAnswers[qi];
  const wa = q.wrong_analysis && q.wrong_analysis[String(userOi)];
  if (!wa) return "";
  return `<div class="wa-block"><span class="wa-type ${esc(wa.type)}">${esc(wa.type)}</span>${esc(wa.reason)}</div>`;
}

function accumulateErrors() {
  quizQuestions.forEach((q, qi) => {
    const userOi = quizAnswers[qi];
    if (userOi !== q.answer && q.wrong_analysis) {
      const wa = q.wrong_analysis[String(userOi)];
      if (wa) quizErrorHistory.push({ type: wa.type, question: q.q });
    }
  });
}

function buildCognitivePanel() {
  const total = quizErrorHistory.length;
  if (total === 0) {
    return `<div class="cog-panel"><h4>识海偏误图</h4>
      <p class="cog-sub">此卷未见偏失。待错题累积，图谱会标出概念混淆、公式遗忘、审题偏移、推导断裂。</p></div>`;
  }
  return `<div class="cog-panel">
    <h4>识海偏误图</h4>
    <p class="cog-sub">已察 ${total} 处学问偏失。谬由分四类：概念混淆、公式遗忘、审题偏移、推导断裂。</p>
    <div class="cog-row"><div class="radar-wrap" id="radarChart"></div></div>
  </div>`;
}

function buildTracePanel() {
  const events = quizTrace.filter((t) => t.action !== "submit");
  if (!events.length) return "";
  const totalTime = quizTrace.find((t) => t.action === "submit")?.ts || events[events.length - 1].ts;
  const byQ = {};
  events.forEach((t) => {
    const k = t.qi >= 0 ? `第${t.qi + 1}题` : "";
    if (!byQ[k]) byQ[k] = [];
    byQ[k].push(t);
  });
  const rows = Object.entries(byQ).map(([qLabel, evts]) => {
    evts.sort((a, b) => a.ts - b.ts);
    const hesitates = [];
    for (let i = 1; i < evts.length; i++) {
      if (evts[i].ts - evts[i - 1].ts > 5000) hesitates.push({ start: evts[i - 1].ts, end: evts[i].ts });
    }
    const clicks = evts.map((e) => {
      const leftPct = totalTime > 0 ? (e.ts / totalTime * 100) : 0;
      const isCorrect = quizSubmitted && quizQuestions[e.qi] && e.oi === quizQuestions[e.qi].answer;
      const isWrong = quizSubmitted && quizQuestions[e.qi] && quizAnswers[e.qi] === e.oi && e.oi !== quizQuestions[e.qi].answer;
      let cls = e.action;
      if (quizSubmitted) cls += isCorrect ? " correct-final" : (isWrong ? " wrong-final" : "");
      return `<div class="trace-click ${cls}" style="left:${leftPct}%"></div>`;
    }).join("");
    const hesi = hesitates.map((h) => {
      const lp = totalTime > 0 ? (h.start / totalTime * 100) : 0;
      const wp = totalTime > 0 ? ((h.end - h.start) / totalTime * 100) : 0;
      return `<div class="trace-hesitate" style="left:${lp}%;width:${Math.max(wp, 2)}%"></div>`;
    }).join("");
    return `<div class="trace-row"><span class="tr-q">${qLabel}</span><div class="trace-bar-wrap">${hesi}<div class="trace-clicks">${clicks}</div></div></div>`;
  }).join("");
  const changeCount = quizTrace.filter((t) => t.action === "change").length;
  return `<div class="trace-panel">
    <h4>研思足迹</h4>
    <p class="cog-sub">总用时 ${(totalTime / 1000).toFixed(1)} 秒 · 点击 ${events.length} 次 · 改选 ${changeCount} 次</p>
    <div class="trace-viz">${rows}</div>
    <div class="trace-legend">
      <span><span class="tl-dot select"></span>首次选择</span>
      <span><span class="tl-dot change"></span>改选</span>
      <span><span class="tl-dot hesitate"></span>犹豫区间 >5s</span>
    </div>
  </div>`;
}

function renderRadarChart() {
  const el = document.getElementById("radarChart");
  if (!el || !window.echarts) return;
  const counts = { "概念混淆": 0, "公式遗忘": 0, "审题偏移": 0, "推导断裂": 0 };
  quizErrorHistory.forEach((e) => { if (counts[e.type] !== undefined) counts[e.type]++; });
  const chart = echarts.init(el);
  chart.setOption({
    radar: {
      center: ["50%", "55%"], radius: "65%",
      indicator: ERROR_TYPES.map((t) => ({ name: t, max: Math.max(3, ...Object.values(counts)) })),
      axisName: { fontSize: 12, color: "#6A7A8A" },
      splitLine: { lineStyle: { color: "rgba(30,41,55,0.04)" } },
      splitArea: { show: false },
      axisLine: { lineStyle: { color: "rgba(30,41,55,0.06)" } },
    },
    series: [{
      type: "radar",
      data: [{
        value: ERROR_TYPES.map((t) => counts[t]), name: "学问偏失分布",
        areaStyle: { color: "rgba(90,155,200,0.10)" },
        lineStyle: { color: "#5A9BC8", width: 2 },
        itemStyle: { color: "#C4A882" },
      }],
      symbol: "circle", symbolSize: 6,
    }],
  });
}

function renderQuiz() {
  const letters = ["甲", "乙", "丙", "丁"];
  const score = quizAnswers.filter((a, i) => a === quizQuestions[i].answer).length;
  let html = quizQuestions.map((q, qi) => `
    <div class="quiz-q">
      <div class="qq-title">${qi + 1}. ${esc(q.q)}</div>
      <div class="qq-options">
        ${(q.options || []).map((opt, oi) => {
          let cls = "";
          if (quizSubmitted) {
            if (oi === q.answer) cls = " qq-correct";
            else if (quizAnswers[qi] === oi && oi !== q.answer) cls = " qq-wrong";
          } else if (quizAnswers[qi] === oi) cls = " qq-chosen";
          return `<div class="qq-opt${cls}" data-qi="${qi}" data-oi="${oi}"><span class="qq-letter">${letters[oi] || oi}</span>${esc(opt)}</div>`;
        }).join("")}
      </div>
      <div class="qq-exp${quizSubmitted ? " show" : ""}">${quizSubmitted ? esc(q.explanation || "") : ""}</div>
      ${quizSubmitted && quizAnswers[qi] !== q.answer && q.wrong_analysis ? renderWrongAnalysis(q, qi) : ""}
    </div>`).join("");
  html += quizSubmitted
    ? `<div class="quiz-score">得第：${score} / ${quizQuestions.length}</div>`
    : `<div style="text-align:center;margin-top:16px"><button type="button" id="quizSubmit">交卷</button></div>`;
  if (quizSubmitted) {
    html += buildCognitivePanel();
    html += buildTracePanel();
  }
  modalBody.innerHTML = html;
  if (!quizSubmitted) {
    modalBody.querySelectorAll(".qq-opt").forEach((opt) => {
      opt.addEventListener("click", () => {
        const qi = parseInt(opt.dataset.qi, 10);
        const oi = parseInt(opt.dataset.oi, 10);
        if (quizAnswers[qi] >= 0 && quizAnswers[qi] !== oi) recordTrace(qi, oi, "change");
        else if (quizAnswers[qi] < 0) recordTrace(qi, oi, "select");
        quizAnswers[qi] = oi;
        renderQuiz();
      });
    });
    $("#quizSubmit")?.addEventListener("click", () => {
      if (quizAnswers.some((a) => a < 0)) { toast("还有题目未作答"); return; }
      recordTrace(-1, -1, "submit");
      quizSubmitted = true;
      accumulateErrors();
      renderQuiz();
      setTimeout(renderRadarChart, 200);
    });
  } else {
    setTimeout(renderRadarChart, 80);
  }
}

async function loadGuide(body) {
  const res = await fetch(API + "/api/study-guide", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!res.ok) { modalBody.innerHTML = `<p class="err-line">${esc(failBox(res, await res.text()))}</p>`; return; }
  const g = await res.json();
  const points = (g.key_points || g.keyPoints || []).map((x) => `<li>${esc(x)}</li>`).join("");
  const pits = (g.pitfalls || []).map((x) => `<li>${esc(x)}</li>`).join("");
  const qs = (g.questions || []).map((x) => `<li>${esc(x)}</li>`).join("");
  modalBody.innerHTML = `
    <div class="report-content">
      <h2>${esc(g.title || "学习指南")}</h2>
      <p>${esc(g.overview || "")}</p>
      ${points ? `<h3>要点</h3><ul>${points}</ul>` : ""}
      ${pits ? `<h3>易错</h3><ul>${pits}</ul>` : ""}
      ${qs ? `<h3>思考题</h3><ul>${qs}</ul>` : ""}
    </div>`;
}

async function loadMindmap(body) {
  const res = await fetch(API + "/api/mindmap", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!res.ok) { modalBody.innerHTML = `<p class="err-line">${esc(failBox(res, await res.text()))}</p>`; return; }
  const data = await res.json();
  const md = data.markdown || "";
  if (!md) { modalBody.innerHTML = '<p class="empty-line">未能绘得脉络</p>'; return; }
  modalBody.innerHTML = `<pre class="mm-md">${esc(md)}</pre>`;
}

async function loadReport(body) {
  const res = await fetch(API + "/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!res.ok) { modalBody.innerHTML = `<p class="err-line">${esc(failBox(res, await res.text()))}</p>`; return; }
  modalBody.innerHTML = '<div class="report-content streaming" id="reportOut"></div>';
  const out = $("#reportOut");
  let text = "";
  await readSSE(res, (p) => {
    if (p.text) text += p.text;
    if (p.error) text += p.error;
    out.textContent = text;
    modalBody.scrollTop = modalBody.scrollHeight;
  });
  out.classList.remove("streaming");
  out.innerHTML = renderMd(text || "未得析报");
}

// 极简 Markdown：标题、无序/有序列表、加粗、行内代码、段落。
// 先整体转义再生成标签，模型输出里的任何 HTML 都只会被当成文字。
function renderMd(t) {
  const inline = (x) => x
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  const out = [];
  let list = null;
  let para = [];
  const flushPara = () => {
    if (para.length) { out.push("<p>" + para.join("<br>") + "</p>"); para = []; }
  };
  const flushList = () => {
    if (!list) return;
    out.push(`<${list.tag}>` + list.items.map((i) => `<li>${i}</li>`).join("") + `</${list.tag}>`);
    list = null;
  };
  const pushItem = (tag, item) => {
    flushPara();
    if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
    list.items.push(inline(item));
  };
  for (const raw of esc(t).split("\n")) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) { flushPara(); flushList(); continue; }
    if ((m = line.match(/^(#{1,4})\s+(.+)$/))) {
      flushPara(); flushList();
      const level = Math.min(m[1].length, 3);
      out.push(`<h${level}>${inline(m[2])}</h${level}>`);
      continue;
    }
    if ((m = line.match(/^\s*[-*•]\s+(.+)$/))) { pushItem("ul", m[1]); continue; }
    if ((m = line.match(/^\s*\d+[.)、]\s+(.+)$/))) { pushItem("ol", m[1]); continue; }
    flushList();
    para.push(inline(line));
  }
  flushPara(); flushList();
  return out.join("");
}

async function loadTable(body) {
  const res = await fetch(API + "/api/table", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!res.ok) { modalBody.innerHTML = `<p class="err-line">${esc(failBox(res, await res.text()))}</p>`; return; }
  const data = await res.json();
  if (data.headers && data.rows) {
    const head = "<tr>" + data.headers.map((h) => `<th>${esc(h)}</th>`).join("") + "</tr>";
    const rows = data.rows.map((r) => "<tr>" + (Array.isArray(r) ? r : Object.values(r)).map((c) => `<td>${esc(c)}</td>`).join("") + "</tr>").join("");
    modalBody.innerHTML = `<div class="table-content"><h3>${esc(data.title || "簿册")}</h3><table>${head}${rows}</table></div>`;
    return;
  }
  modalBody.innerHTML = `<pre>${esc(JSON.stringify(data, null, 2))}</pre>`;
}

async function loadInfographic(body) {
  const res = await fetch(API + "/api/infographic", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!res.ok) { modalBody.innerHTML = `<p class="err-line">${esc(failBox(res, await res.text()))}</p>`; return; }
  const d = await res.json();
  let html = `<div class="info-hero"><h2>${esc(d.title || "卷宗览图")}</h2><div class="ih-insight">${esc(d.keyInsight || d.key_insight || "")}</div></div>`;
  if (d.stats && d.stats.length) {
    html += '<div class="info-stats">' + d.stats.map((s) =>
      `<div class="stat-card"><div class="sv">${esc(s.value)}</div><div class="sl">${esc(s.label)}</div><div class="sd">${esc(s.desc || "")}</div></div>`
    ).join("") + "</div>";
  }
  if (d.comparisons && d.comparisons.length) {
    html += '<div class="info-compare">' + d.comparisons.map((c) =>
      `<div class="compare-card"><div class="ca">${esc(c.aspect)}</div><div class="cr"><div class="ci">${esc(c.left)}</div><span class="vs">VS</span><div class="ci">${esc(c.right)}</div></div></div>`
    ).join("") + "</div>";
  }
  if (d.conceptFlow && d.conceptFlow.length) {
    html += '<div class="info-flow"><h4>义理脉络</h4><div class="flow-steps">' + d.conceptFlow.map((f, i) =>
      `<div class="flow-step"><div class="fs-num">${i + 1}</div><div class="fs-title">${esc(f.step)}</div><div class="fs-desc">${esc(f.description || "")}</div></div>`
    ).join("") + "</div></div>";
  }
  modalBody.innerHTML = html || `<pre>${esc(JSON.stringify(d, null, 2))}</pre>`;
}

async function openChunk(s) {
  const idx = s.chunk ?? s.chunk_idx;
  let item = {};
  try {
    const res = await fetch(API + "/api/docs/" + encodeURIComponent(s.doc_id) + "/chunks/" + idx);
    item = res.ok ? await res.json() : { text: "该片段已不在书斋中（卷册可能已被移出）。" };
  } catch (e) { item = { text: "片段未能载入。" }; }
  let drawer = document.querySelector(".chunk-drawer");
  if (!drawer) { drawer = document.createElement("div"); drawer.className = "chunk-drawer"; document.body.appendChild(drawer); }
  drawer.innerHTML = `
    <div class="cd-head">
      <span class="cd-title">${esc(item.doc_name || s.name || "")} 片段${esc(idx)}</span>
      <button type="button" class="cd-close" aria-label="关闭原文">✕</button>
    </div>
    ${s.why ? `<div class="cd-why">${esc(s.why)}</div>` : ""}
    <pre>${esc(item.text || "")}</pre>`;
  drawer.querySelector(".cd-close").addEventListener("click", () => drawer.remove());
}

// 顶栏状态行兼作提示：停留片刻后回到「已择 N 卷」。
function toast(msg) {
  headerInfo.textContent = msg;
  toastUntil = Date.now() + TOAST_MS;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { headerInfo.textContent = statusLine(); }, TOAST_MS);
}

// ---- 自主学习 Agent：目标 → 检索 → 诊断 → 判分 → 讲解 → 总结 ----
const AG_TOOL_NAMES = {
  search_knowledge: "检索教材",
  read_source: "核对原文",
  create_study_guide: "提炼要点",
  create_quiz: "生成诊断",
  analyze_mistakes: "判分分析",
  answer_question: "针对性讲解",
};
let agSid = null;
let agQuiz = [];
let agAnswers = [];

async function runAgent() {
  agSid = null; agQuiz = []; agAnswers = [];
  modalBody.innerHTML = `
    <div class="ag-wrap">
      <div class="ag-steps" id="agSteps"></div>
      <div class="ag-body" id="agBody"></div>
    </div>`;
  const typed = chatInput.value.trim();
  if (typed) {
    $("#agBody").innerHTML = '<div class="loading-spin">正在分析目标…</div>';
    await streamAgent("/api/agent", { goal: typed, doc_ids: [...selectedDocs] });
    return;
  }
  await renderAgentHome();
}

async function renderAgentHome() {
  let sessions = [];
  try {
    const res = await fetch(API + "/api/agent");
    if (res.ok) sessions = await res.json();
  } catch (e) { sessions = []; }
  const pending = sessions.filter((s) => s.status === "awaiting_answers");
  const done = sessions.filter((s) => s.status === "finished").slice(0, 5);
  let html = "";
  if (pending.length) {
    html += `<div class="ag-sessions"><h4>未完的学习</h4>` + pending.map((s) => `
      <div class="ag-session" data-sid="${esc(s.session_id)}" data-mode="resume">
        <div class="ag-ses-info">
          <b>${esc(s.goal)}</b>
          <span class="ag-ses-meta">${esc(s.stage)} · ${s.step_count} 步 · ${fmtTime(s.updated_at)}</span>
        </div>
        <button type="button" class="ag-resume-btn">继续作答</button>
      </div>`).join("") + `</div>`;
  }
  html += `
    <div class="ag-goal-box">
      <textarea id="agGoal" rows="2" placeholder="例如：帮我复习前三章，重点找出我薄弱的知识"></textarea>
      <button type="button" id="agStart">开始学习</button>
    </div>`;
  if (done.length) {
    html += `<div class="ag-sessions"><h4>已了结的学习</h4>` + done.map((s) => `
      <div class="ag-session done" data-sid="${esc(s.session_id)}" data-mode="review">
        <div class="ag-ses-info">
          <b>${esc(s.goal)}</b>
          <span class="ag-ses-meta">${s.weakness_count} 个薄弱点 · ${fmtTime(s.updated_at)}</span>
        </div>
        <button type="button" class="ag-resume-btn ghost">回看总结</button>
      </div>`).join("") + `</div>`;
  }
  $("#agBody").innerHTML = html;
  $("#agStart").addEventListener("click", async () => {
    const goal = $("#agGoal").value.trim();
    if (goal.length < 4) { toast("学习目标太短了"); return; }
    $("#agBody").innerHTML = '<div class="loading-spin">正在分析目标…</div>';
    await streamAgent("/api/agent", { goal, doc_ids: [...selectedDocs] });
  });
  $("#agBody").querySelectorAll(".ag-session .ag-resume-btn").forEach((btn) => {
    const row = btn.closest(".ag-session");
    btn.addEventListener("click", () => {
      if (row.dataset.mode === "resume") resumeAgentSession(row.dataset.sid);
      else reviewAgentSession(row.dataset.sid);
    });
  });
}

function fmtTime(ts) {
  const d = new Date((ts || 0) * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function agTimeline(steps, goal) {
  const head = agStepEl("分析目标", goal);
  head.classList.remove("pending"); head.classList.add("done");
  (steps || []).forEach((s) => {
    const row = agStepEl(`${s.step}. ${AG_TOOL_NAMES[s.tool] || s.tool}`, s.reason || "");
    row.classList.remove("pending");
    row.classList.add(s.ok ? "done" : "fail");
    if (s.result_summary) {
      const d = document.createElement("div");
      d.className = "ag-sum";
      d.textContent = s.result_summary;
      row.querySelector(".ag-step-body").appendChild(d);
    }
  });
}

async function fetchAgentState(sid) {
  const res = await fetch(API + "/api/agent/" + sid);
  if (!res.ok) throw new Error("会话不存在或已失效");
  return res.json();
}

async function resumeAgentSession(sid) {
  $("#agBody").innerHTML = '<div class="loading-spin">正在恢复会话…</div>';
  let st;
  try { st = await fetchAgentState(sid); }
  catch (e) { $("#agBody").innerHTML = `<p class="err-line">${esc(e.message)}</p>`; return; }
  if (st.status !== "awaiting_answers" || !(st.quiz || []).length) {
    $("#agBody").innerHTML = '<p class="empty-line">该会话已无法继续作答</p>';
    return;
  }
  agSid = sid;
  agTimeline(st.steps, st.goal);
  agQuiz = st.quiz;
  agAnswers = new Array(agQuiz.length).fill(-1);
  renderAgentQuiz();
}

async function reviewAgentSession(sid) {
  $("#agBody").innerHTML = '<div class="loading-spin">正在调出总结…</div>';
  let st;
  try { st = await fetchAgentState(sid); }
  catch (e) { $("#agBody").innerHTML = `<p class="err-line">${esc(e.message)}</p>`; return; }
  agSid = sid;
  agTimeline(st.steps, st.goal);
  renderAgentFinal({ summary: st.final_summary, weaknesses: st.weaknesses, sources: st.sources });
}

function agStepEl(title, reason) {
  const el = document.createElement("div");
  el.className = "ag-step pending";
  el.innerHTML = `<span class="ag-dot"></span><div class="ag-step-body">
    <div class="ag-step-title">${esc(title)}</div>
    ${reason ? `<div class="ag-reason">${esc(reason)}</div>` : ""}</div>`;
  $("#agSteps").appendChild(el);
  return el;
}

function handleAgentEvent(e) {
  const steps = $("#agSteps");
  if (!steps) return;
  const spin = $("#agBody .loading-spin");
  if (spin) spin.remove();
  if (e.type === "planning") {
    agStepEl("分析目标", e.goal).classList.add("done");
  } else if (e.type === "tool_call") {
    const el = agStepEl(`${e.step}. ${AG_TOOL_NAMES[e.tool] || e.tool}`, e.reason);
    el.dataset.step = e.step;
  } else if (e.type === "tool_result") {
    const el = steps.querySelector(`.ag-step[data-step="${e.step}"]`) || steps.lastElementChild;
    if (el) {
      el.classList.remove("pending");
      el.classList.add(e.ok ? "done" : "fail");
      if (e.summary) {
        const s = document.createElement("div");
        s.className = "ag-sum";
        s.textContent = e.summary;
        el.querySelector(".ag-step-body").appendChild(s);
      }
    }
    if (e.tool === "create_study_guide" && e.ok && e.data && e.data.key_points) {
      const box = document.createElement("div");
      box.className = "ag-explain";
      box.innerHTML = `<h4>要点提炼</h4><ul>` +
        e.data.key_points.map((k) => `<li>${esc(k)}</li>`).join("") + `</ul>`;
      $("#agBody").appendChild(box);
    }
    if (e.tool === "answer_question" && e.ok && e.data && e.data.answer) {
      const box = document.createElement("div");
      box.className = "ag-explain";
      let cites = "";
      const cs = e.data.citations || [];
      if (cs.length) {
        cites = `<div class="ag-src-row">` + cs.map((c) => {
          if (!c.verified)
            return `<span class="src-chip warn">✕ ${esc(c.name)} 片段${c.chunk} · 未能在原文核对</span>`;
          if (!c.in_context)
            return `<button type="button" class="src-chip cite warn" data-doc="${esc(c.doc_id)}" data-chunk="${c.chunk}">△ ${esc(c.name)} 片段${c.chunk} · 不在本次依据中</button>`;
          return `<button type="button" class="src-chip cite" data-doc="${esc(c.doc_id)}" data-chunk="${c.chunk}">✓ ${esc(c.name)} 片段${c.chunk}</button>`;
        }).join("") + `</div>`;
      }
      box.innerHTML = `<h4>针对性讲解</h4><div class="report-content">${renderMd(e.data.answer)}</div>${cites}`;
      box.querySelectorAll(".src-chip.cite").forEach((chip) => {
        chip.addEventListener("click", () =>
          openChunk({ doc_id: chip.dataset.doc, chunk: parseInt(chip.dataset.chunk, 10), name: chip.textContent }));
      });
      $("#agBody").appendChild(box);
    }
  } else if (e.type === "awaiting_input") {
    agSid = e.session_id;
    agQuiz = e.quiz || [];
    agAnswers = new Array(agQuiz.length).fill(-1);
    renderAgentQuiz();
  } else if (e.type === "final") {
    renderAgentFinal(e);
  } else if (e.type === "error") {
    agStepEl("出错", e.message).classList.add("fail");
  }
  modalBody.scrollTop = modalBody.scrollHeight;
}

async function streamAgent(url, body) {
  let res;
  try {
    res = await fetch(API + url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    handleAgentEvent({ type: "error", message: "舟楫失连，请再试" });
    return;
  }
  if (!res.ok) {
    let msg = `请求失败（${res.status}）`;
    try { msg = (await res.json()).detail || msg; } catch (e2) {}
    handleAgentEvent({ type: "error", message: msg });
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    parts.forEach((p) => {
      if (!p.startsWith("data: ") || p.includes("[DONE]")) return;
      try { handleAgentEvent(JSON.parse(p.slice(6))); } catch (e) {}
    });
  }
}

function renderAgentQuiz() {
  const letters = ["甲", "乙", "丙", "丁"];
  const all = agAnswers.every((a) => a >= 0);
  let html = `<div class="ag-quiz"><h4>诊断自测</h4>`;
  html += agQuiz.map((q, qi) => `
    <div class="quiz-q">
      <div class="qq-title">${qi + 1}. ${esc(q.q)}</div>
      <div class="qq-options">
        ${(q.options || []).map((opt, oi) =>
          `<div class="qq-opt${agAnswers[qi] === oi ? " qq-chosen" : ""}" data-qi="${qi}" data-oi="${oi}">
            <span class="qq-letter">${letters[oi] || oi}</span>${esc(opt)}</div>`).join("")}
      </div>
    </div>`).join("");
  html += `<div class="ag-submit-row"><button type="button" id="agSubmit"${all ? "" : " disabled"}>交卷，继续复习</button></div></div>`;
  $("#agBody").innerHTML = html;
  $("#agBody").querySelectorAll(".qq-opt").forEach((opt) => {
    opt.addEventListener("click", () => {
      agAnswers[parseInt(opt.dataset.qi, 10)] = parseInt(opt.dataset.oi, 10);
      renderAgentQuiz();
    });
  });
  $("#agSubmit").addEventListener("click", submitAgentAnswers);
}

function submitAgentAnswers() {
  $("#agBody").innerHTML = '<div class="loading-spin">正在判分并安排针对性复习…</div>';
  streamAgent(`/api/agent/${agSid}/answers`, { answers: agAnswers });
}

function renderAgentFinal(e) {
  let html = '<div class="ag-final">';
  if (e.weaknesses && e.weaknesses.length) {
    html += `<h4>薄弱点（${e.weaknesses.length}）</h4>` + e.weaknesses.map((w) =>
      `<div class="wa-block"><span class="wa-type ${esc(w.error_type || "概念混淆")}">${esc(w.error_type || "未分类")}</span>
       <b>${esc(w.topic)}</b><div>${esc(w.reason || "")}</div></div>`).join("");
  }
  html += `<h4>复习总结</h4><div class="report-content">${renderMd(e.summary || "")}</div>`;
  if (e.sources && e.sources.length) {
    html += `<div class="ag-src-row">` + e.sources.map((s) =>
      `<button type="button" class="src-chip" data-doc="${esc(s.doc_id)}" data-chunk="${s.chunk}">${esc(s.name)} 片段${s.chunk}</button>`).join("") + `</div>`;
  }
  html += "</div>";
  $("#agBody").innerHTML = html;
  $("#agBody").querySelectorAll(".ag-src-row .src-chip").forEach((chip) => {
    chip.addEventListener("click", () =>
      openChunk({ doc_id: chip.dataset.doc, chunk: parseInt(chip.dataset.chunk, 10), name: chip.textContent }));
  });
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => {
    if (c === "&") return "\u0026amp;";
    if (c === "<") return "\u0026lt;";
    if (c === ">") return "\u0026gt;";
    if (c === '"') return "\u0026quot;";
    return "&#39;";
  });
}

refreshDocs().then(refreshConversations);
window.openChunk = openChunk;
