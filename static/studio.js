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

async function handleFiles(files) {
  for (const f of files) {
    const lower = f.name.toLowerCase();
    try {
      if (lower.endsWith(".pdf") || f.type === "application/pdf") {
        await indexPDF(f);
      } else if (lower.endsWith(".txt") || lower.endsWith(".md") || (f.type || "").startsWith("text/")) {
        await indexText(f.name.replace(/\.(txt|md)$/i, ""), await f.text());
      } else {
        toast("暂只收 PDF / TXT / MD：" + f.name);
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

async function indexText(name, text) {
  const id = "d" + Math.random().toString(36).slice(2, 10);
  const res = await fetch(API + "/api/index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: id, name, text }),
  });
  if (!res.ok) { toast("收卷未成"); setPg(0); return; }
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
    `<div class="doc-card${selectedDocs.has(d.id) ? " selected" : ""}" data-id="${d.id}">
      <div class="dc-name">${esc(d.name)}</div>
      <div class="dc-meta">${d.chunks} 片段</div>
    </div>`
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
  const on = selectedDocs.size > 0 && !isStreaming;
  chatInput.disabled = sendBtn.disabled = !on;
  document.querySelectorAll(".feat-card").forEach((b) => { b.disabled = !on; });
  headerInfo.textContent = selectedDocs.size ? `已择 ${selectedDocs.size} 卷` : "择卷而问";
}

async function send() {
  const query = chatInput.value.trim();
  if (!query || !selectedDocs.size || isStreaming) return;
  chatInput.value = "";
  if (chatEmpty) chatEmpty.style.display = "none";
  addMsg("user", query);
  const bubble = addMsg("assistant", "小舟研墨中…");
  isStreaming = true; updateInput();
  let text = ""; let sources = [];
  try {
    const res = await fetch(API + "/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, doc_ids: [...selectedDocs], system: SYSTEM_PROMPT }),
    });
    const raw = await res.text();
    raw.split("\n").forEach((line) => {
      if (!line.startsWith("data: ") || line.includes("[DONE]")) return;
      try {
        const p = JSON.parse(line.slice(6));
        if (p.text) text += p.text;
        if (p.sources) sources = p.sources;
        if (p.error) text = p.error;
      } catch (e) {}
    });
  } catch (e) {
    text = "舟楫失连，请再试";
  }
  bubble.querySelector(".msg-bubble").textContent = text || "未得回答";
  if (sources.length) {
    const box = document.createElement("div");
    box.className = "msg-sources";
    box.innerHTML = `<span class="src-chip" style="cursor:default">检索解释：BGE 向量余弦 Top-${sources.length}</span>`;
    sources.forEach((s) => {
      const chip = document.createElement("button");
      chip.type = "button"; chip.className = "src-chip";
      chip.textContent = `${s.name} 片段${s.chunk} · ${s.score}`;
      chip.onclick = () => openChunk(s);
      box.appendChild(chip);
    });
    bubble.appendChild(box);
  }
  isStreaming = false; updateInput();
}

function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.innerHTML = `<div class="msg-body"><div class="msg-bubble">${esc(text)}</div></div>`;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
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
    guide: "指南", flashcards: "笺卡", quiz: "考核",
    mindmap: "脉络图", report: "析报", table: "簿册", infographic: "览图",
  };
  openModal(titles[name] || name);
  modalBody.innerHTML = `<div class="loading-spin">正在准备${titles[name] || name}</div>`;
  const body = JSON.stringify({ doc_ids: [...selectedDocs], query: chatInput.value.trim(), count: 6 });
  try {
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
    </div>`;
  $("#fcCard").addEventListener("click", () => $("#fcCard").classList.toggle("flipped"));
  $("#fcPrev").addEventListener("click", () => { if (fcIdx > 0) { fcIdx -= 1; renderFlashcard(); } });
  $("#fcNext").addEventListener("click", () => { if (fcIdx < fcCards.length - 1) { fcIdx += 1; renderFlashcard(); } });
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
  const raw = await res.text();
  let text = "";
  raw.split("\n").forEach((line) => {
    if (!line.startsWith("data: ") || line.includes("[DONE]")) return;
    try {
      const p = JSON.parse(line.slice(6));
      if (p.text) text += p.text;
      if (p.error) text += p.error;
    } catch (e) {}
  });
  out.classList.remove("streaming");
  out.innerHTML = renderMd(text || "未得析报");
}

function renderMd(t) {
  let h = esc(t);
  h = h.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  h = h.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  h = h.replace(/^# (.+)$/gm, "<h1>$1</h1>");
  h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/^- (.+)$/gm, "<li>$1</li>");
  h = h.replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>");
  return h;
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
  const item = await fetch(API + "/api/docs/" + s.doc_id + "/chunks/" + (s.chunk ?? s.chunk_idx)).then((r) => r.json());
  let drawer = document.querySelector(".chunk-drawer");
  if (!drawer) { drawer = document.createElement("div"); drawer.className = "chunk-drawer"; document.body.appendChild(drawer); }
  drawer.innerHTML = `<div>${esc(item.doc_name || s.name)} 片段${s.chunk ?? s.chunk_idx}</div><pre>${esc(item.text || "")}</pre>`;
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
