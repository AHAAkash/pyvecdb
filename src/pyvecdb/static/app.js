"use strict";
// All text from the server is inserted with textContent / createTextNode, never innerHTML.

const $ = (sel) => document.querySelector(sel);
function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const INK = "#13222e", QUERY = "#c2185b", HIT = "#d99a00";
const state = { points: [], documents: [], hits: new Map(), query: null, hover: -1, rays: 1, readOnly: false };
const hue = (s) => { let x = 0; for (const c of s) x = (x * 31 + c.charCodeAt(0)) >>> 0; return x % 360; };
const colour = (docId) => `hsl(${hue(docId)} 55% 40%)`;

// ---- API -------------------------------------------------------------------
async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json" };
  const key = sessionStorage.getItem("pyvecdb-key");
  if (key) headers["X-API-Key"] = key;
  const res = await fetch("/api" + path, { ...opts, headers });
  if (res.status === 401) { $("#keybar").hidden = false; throw new Error("This server needs an API key."); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : res.statusText);
  }
  return data;
}

// ---- map -------------------------------------------------------------------
const cv = $("#map"), ctx = cv.getContext("2d");
let W = 0, H = 0;
const PAD = 44, SPAN = 1.2;
const sx = (x) => PAD + ((Math.max(-SPAN, Math.min(SPAN, x)) + SPAN) / (2 * SPAN)) * (W - 2 * PAD);
const sy = (y) => H - PAD - ((Math.max(-SPAN, Math.min(SPAN, y)) + SPAN) / (2 * SPAN)) * (H - 2 * PAD);

function resize() {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  W = r.width; H = r.height;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  const hasHits = state.hits.size > 0;
  if (state.query && hasHits) {
    const qx = sx(state.query[0]), qy = sy(state.query[1]);
    ctx.save();
    ctx.setLineDash([5, 5]); ctx.strokeStyle = QUERY; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.75;
    for (const p of state.points) {
      if (!state.hits.has(p.id)) continue;
      ctx.beginPath(); ctx.moveTo(qx, qy);
      ctx.lineTo(qx + (sx(p.x) - qx) * state.rays, qy + (sy(p.y) - qy) * state.rays); ctx.stroke();
    }
    ctx.restore();
  }
  state.points.forEach((p, i) => {
    const hit = state.hits.has(p.id), x = sx(p.x), y = sy(p.y);
    ctx.globalAlpha = hasHits && !hit ? 0.35 : 1;
    ctx.beginPath(); ctx.arc(x, y, hit ? 6.5 : 5, 0, Math.PI * 2);
    ctx.fillStyle = colour(p.doc_id); ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.globalAlpha = 1;
    if (hit) {
      ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.lineWidth = 2.5; ctx.strokeStyle = HIT; ctx.stroke();
      ctx.font = "700 13px system-ui, sans-serif"; ctx.lineWidth = 3; ctx.strokeStyle = "#fff"; ctx.fillStyle = INK;
      const label = String(state.hits.get(p.id));
      ctx.strokeText(label, x + 14, y - 8); ctx.fillText(label, x + 14, y - 8);
    }
    if (i === state.hover) {
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.lineWidth = 2; ctx.strokeStyle = INK; ctx.stroke();
    }
  });
  if (state.query) {
    const qx = sx(state.query[0]), qy = sy(state.query[1]);
    ctx.beginPath();
    ctx.moveTo(qx, qy - 10); ctx.lineTo(qx + 10, qy); ctx.lineTo(qx, qy + 10); ctx.lineTo(qx - 10, qy); ctx.closePath();
    ctx.fillStyle = QUERY; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
  }
}

function animateRays() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { state.rays = 1; draw(); return; }
  const t0 = performance.now();
  const step = (t) => { state.rays = Math.min(1, (t - t0) / 450); draw(); if (state.rays < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

const tip = $("#tip");
cv.addEventListener("pointermove", (e) => {
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  let best = -1, bd = 14 * 14;
  state.points.forEach((p, i) => {
    const dx = sx(p.x) - x, dy = sy(p.y) - y, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  });
  if (best !== state.hover) { state.hover = best; draw(); }
  if (best < 0) { tip.hidden = true; return; }
  const p = state.points[best];
  tip.replaceChildren(h("b", {}, `${p.title} (part ${p.chunk_index + 1})`), p.snippet + (p.snippet.length >= 160 ? "…" : ""));
  tip.hidden = false;
  tip.style.left = Math.max(4, Math.min(x + 14, W - 290)) + "px";
  tip.style.top = Math.max(4, Math.min(y + 14, H - 90)) + "px";
});
cv.addEventListener("pointerleave", () => { state.hover = -1; tip.hidden = true; draw(); });
new ResizeObserver(resize).observe(cv);

// ---- documents ---------------------------------------------------------------
function renderChips(s) {
  $("#chips").replaceChildren(
    h("li", {}, `embed ${s.embedder}`), h("li", {}, `answers ${s.generator}`),
    h("li", {}, `${s.documents} docs`), h("li", {}, `${s.chunks} chunks`));
}

function renderDocs(docs) {
  const ul = $("#docs");
  const filter = $("#docfilter").value.trim().toLocaleLowerCase();
  const visible = docs.filter((d) => d.title.toLocaleLowerCase().includes(filter));
  $("#doc-total").textContent = String(docs.length);
  if (!docs.length) {
    ul.replaceChildren(h("li", { class: "empty-docs" }, "No documents yet. Add one above or load the sample collection."));
    return;
  }
  if (!visible.length) {
    ul.replaceChildren(h("li", { class: "empty-docs" }, "No documents match that filter."));
    return;
  }
  ul.replaceChildren(...visible.map((d) => {
    const dot = h("span", { class: "dot" });
    dot.style.setProperty("background", colour(d.doc_id));
    const actions = [h("small", {}, `${d.chunks} ${d.chunks === 1 ? "chunk" : "chunks"} `)];
    if (!state.readOnly) {
      actions.push(h("button", { type: "button", "aria-label": `Remove ${d.title}`, onclick: () => removeDoc(d.doc_id) }, "Remove"));
    }
    return h("li", {},
      h("span", { class: "name" }, dot, h("span", {}, d.title)),
      h("span", {}, actions));
  }));
}

async function refresh() {
  const [health, stats, docs, map] = await Promise.all([api("/health"), api("/stats"), api("/docs"), api("/map")]);
  state.readOnly = health.read_only;
  $("#instance-mode").textContent = state.readOnly ? "READ-ONLY DEMO" : "SYSTEM READY";
  $("#add-h").textContent = state.readOnly ? "Sample collection" : "Add knowledge";
  $("#add-intro").hidden = state.readOnly;
  $("#readonly-note").hidden = !state.readOnly;
  $("#addform").hidden = state.readOnly;
  renderChips(stats);
  state.documents = docs.documents;
  renderDocs(state.documents);
  state.points = map.points; state.hits = new Map(); state.query = null; state.hover = -1;
  $("#mapempty").hidden = state.points.length > 0;
  $("#point-count").textContent = new Intl.NumberFormat().format(state.points.length);
  $("#map-status").textContent = state.points.length ? "Index up to date" : "Waiting for data";
  draw();
}

function say(msg, isErr = false) { const el = $("#addmsg"); el.textContent = msg; el.className = "msg" + (isErr ? " err" : ""); }

async function removeDoc(id) {
  try { await api("/docs/" + encodeURIComponent(id), { method: "DELETE" }); await refresh(); }
  catch (e) { say(e.message, true); }
}

$("#addform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("#addbtn"); btn.disabled = true; say("Embedding and indexing…");
  try {
    const d = await api("/docs", { method: "POST", body: JSON.stringify({ title: $("#title").value, text: $("#text").value }) });
    say(`Added “${d.title}” as ${d.chunks} ${d.chunks === 1 ? "chunk" : "chunks"}.`);
    $("#title").value = ""; $("#text").value = "";
    $("#text-count").textContent = "0 / 1,000,000";
    await refresh();
  } catch (err) { say(err.message, true); } finally { btn.disabled = false; }
});

$("#text").addEventListener("input", (e) => {
  $("#text-count").textContent = `${new Intl.NumberFormat().format(e.target.value.length)} / 1,000,000`;
});

$("#docfilter").addEventListener("input", () => renderDocs(state.documents));

$("#q").addEventListener("input", (e) => {
  $("#query-count").textContent = `${e.target.value.length.toLocaleString()} / 2,000`;
});

$("#demobtn").addEventListener("click", async () => {
  const btn = $("#demobtn"); btn.disabled = true; say("Loading samples…");
  try { const r = await api("/demo", { method: "POST" }); say(`Loaded ${r.loaded} sample documents.`); await refresh(); }
  catch (err) { say(err.message, true); } finally { btn.disabled = false; }
});

// ---- ask / search -------------------------------------------------------------
const out = $("#out");

function renderAnswer(text, nSources) {
  const nodes = text.split(/(\[\d+\])/).map((part) => {
    const m = /^\[(\d+)\]$/.exec(part);
    if (!m || +m[1] < 1 || +m[1] > nSources) return part;
    return h("button", { type: "button", class: "cite", "aria-label": `Go to source ${m[1]}`, onclick: () => flash(+m[1]) }, part);
  });
  return nodes;
}

function flash(n) {
  const li = document.getElementById("src-" + n);
  if (!li) return;
  li.scrollIntoView({ block: "nearest" });
  li.classList.add("flash"); setTimeout(() => li.classList.remove("flash"), 1200);
}

function renderSources(items) {
  return h("ol", { class: "sources" }, items.map((c, i) => {
    const scores = [];
    if (c.similarity != null) scores.push(`similarity ${c.similarity.toFixed(2)}`);
    if (c.bm25 != null) scores.push(`keyword ${c.bm25.toFixed(1)}`);
    return h("li", { id: "src-" + (i + 1) },
      h("div", { class: "head" }, h("span", { class: "rank" }, i + 1), h("strong", {}, c.title),
        h("span", { class: "scores" }, scores.join(", "))),
      h("details", {}, h("summary", {}, "Show passage"), h("p", {}, c.text)));
  }));
}

function animateOutput() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const elements = out.querySelectorAll(".answer, .sources li");
  elements.forEach((el, index) => {
    if (typeof el.animate === "function") {
      el.animate(
        [{ opacity: 0, transform: "translateY(7px)" }, { opacity: 1, transform: "translateY(0)" }],
        { duration: 260, delay: Math.min(index * 45, 270), easing: "cubic-bezier(.2,.7,.2,1)" }
      );
    }
  });
}

async function run(kind) {
  const query = $("#q").value.trim();
  if (!query) return;
  const body = { query, mode: $("#mode").value, k: kind === "ask" ? 4 : 6 };
  if ($("#mmr").checked) body.mmr = 0.6;
  const btns = [$("#askbtn"), $("#searchbtn")]; btns.forEach((b) => (b.disabled = true));
  out.replaceChildren(h("p", { class: "meta" }, kind === "ask" ? "Retrieving passages and writing an answer…" : "Searching…"));
  try {
    const data = await api(kind === "ask" ? "/ask" : "/search", { method: "POST", body: JSON.stringify(body) });
    const items = kind === "ask" ? data.contexts : data.results;
    state.hits = new Map(items.map((c, i) => [c.chunk_id, i + 1]));
    state.query = data.query_xy; state.rays = 0; animateRays();
    const parts = [];
    if (kind === "ask") {
      parts.push(h("div", { class: "answer" + (data.abstained ? " abstain" : "") }, renderAnswer(data.answer, items.length)));
      if (data.invalid_citations.length)
        parts.push(h("p", { class: "meta err" }, `The answer cites sources that do not exist: ${data.invalid_citations.join(", ")}.`));
      parts.push(h("p", { class: "meta" },
        `Retrieved in ${data.retrieval_ms} ms, answered in ${data.generation_ms} ms by ${data.generator}.`));
    } else {
      parts.push(h("p", { class: "meta" }, items.length ? `${items.length} passages found.` : "No passages matched."));
    }
    if (items.length) parts.push(h("h2", {}, "Sources"), renderSources(items));
    out.replaceChildren(...parts);
    animateOutput();
  } catch (err) {
    out.replaceChildren(h("p", { class: "err" }, err.message));
  } finally { btns.forEach((b) => (b.disabled = false)); }
}

$("#askform").addEventListener("submit", (e) => { e.preventDefault(); run("ask"); });
$("#searchbtn").addEventListener("click", () => run("search"));
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run("ask"); } });
document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    $("#q").value = button.dataset.query;
    $("#query-count").textContent = `${$("#q").value.length} / 2,000`;
    run("ask");
  });
});

$("#keysave").addEventListener("click", () => {
  sessionStorage.setItem("pyvecdb-key", $("#keyinput").value);
  $("#keybar").hidden = true; refresh().catch((e) => say(e.message, true));
});

const guideDialog = $("#guide-dialog");
const guideOpen = $("#guide-open");
guideOpen.addEventListener("click", () => {
  guideDialog.showModal();
  $("#guide-close").focus();
});
$("#guide-close").addEventListener("click", () => guideDialog.close());
guideDialog.addEventListener("click", (e) => {
  if (e.target === guideDialog) guideDialog.close();
});
guideDialog.addEventListener("close", () => guideOpen.focus());

$("#copy-command").addEventListener("click", async () => {
  const status = $("#copy-status");
  status.className = "copy-status";
  try {
    await navigator.clipboard.writeText($("#run-command").textContent.trim());
    status.textContent = "Commands copied. Paste them into PowerShell from the project folder.";
  } catch (error) {
    status.className = "copy-status err";
    status.textContent = "Could not access the clipboard. Select the commands above and copy them manually.";
  }
});

refresh().catch((e) => say(e.message, true));
