"use strict";
/* app.js — the console page.
 *
 * Two deliberate rules:
 *  1. Every number is written into a DOM element with a stable id. Nothing
 *     numeric lives only inside <svg>: the arc is decoration, the text is the
 *     data. That is what lets an agent's browser/monitor tool read the live
 *     numbers by reading the page, and what makes the page's own readout
 *     testable.
 *  2. The page reads the *public* monitor endpoints (`/monitor.json` for state,
 *     `/ws/monitor` for events) rather than some private channel. If those were
 *     not enough to drive this UI, they would not really be agent-readable.
 */

const POLL_MS = 400;
const MAX_LOG_NODES = 4000;      // DOM cap; the server's ring buffer is the real history
const MAX_BUFFER = 5000;
const ARC_START = 0.75 * Math.PI;   // same 270° sweep as ShadRiskMeter's _RiskArcPainter
const ARC_SWEEP = 1.5 * Math.PI;
const GAUGE_CX = 100, GAUGE_CY = 100, GAUGE_R = 84, ARC_LEN = GAUGE_R * ARC_SWEEP;

const STAGES = [
  "model_load", "source_open", "decode", "resample", "run_start", "window", "features",
  "infer", "softmax", "score", "ema", "decision", "verdict", "alert", "backend_call",
  "diff", "window_done", "control", "run_end", "error",
];

const el = (id) => document.getElementById(id);
const state = { events: [], scores: [], snapshot: null, ws: null, link: "starting" };

function fmt(value, digits) {
  const d = digits === undefined ? 3 : digits;
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(d);
}

function bandFor(score) {
  if (score < 0.30) return "verified";
  if (score < 0.70) return "suspicious";
  return "detected";
}
function verdictFor(score) {
  if (score < 0.30) return "VERIFIED HUMAN";
  if (score < 0.70) return "SUSPICIOUS";
  return "AI DETECTED";
}

function polar(radius, angle) {
  return [GAUGE_CX + radius * Math.cos(angle), GAUGE_CY + radius * Math.sin(angle)];
}
function arcPath() {
  const [x0, y0] = polar(GAUGE_R, ARC_START);
  const [x1, y1] = polar(GAUGE_R, ARC_START + ARC_SWEEP);
  return `M ${x0} ${y0} A ${GAUGE_R} ${GAUGE_R} 0 1 1 ${x1} ${y1}`;
}

function initGauge() {
  const track = el("arc-track"), value = el("arc-value");
  const d = arcPath();
  track.setAttribute("d", d);
  value.setAttribute("d", d);
  value.style.strokeDasharray = String(ARC_LEN);
  value.style.strokeDashoffset = String(ARC_LEN);
}

function drawThresholdTick(threshold) {
  const angle = ARC_START + ARC_SWEEP * Math.max(0, Math.min(1, threshold));
  const [x0, y0] = polar(GAUGE_R - 11, angle);
  const [x1, y1] = polar(GAUGE_R + 11, angle);
  const tick = el("arc-threshold");
  tick.setAttribute("x1", x0); tick.setAttribute("y1", y0);
  tick.setAttribute("x2", x1); tick.setAttribute("y2", y1);
}

function drawSpark() {
  const raw = el("spark-raw"), ema = el("spark-ema");
  const points = state.scores.slice(-120);
  if (!points.length) { raw.setAttribute("points", ""); ema.setAttribute("points", ""); return; }
  const step = points.length > 1 ? 300 / (points.length - 1) : 300;
  const y = (v) => (60 - Math.max(0, Math.min(1, v)) * 58 - 1).toFixed(1);
  raw.setAttribute("points", points.map((p, i) => `${(i * step).toFixed(1)},${y(p.raw)}`).join(" "));
  ema.setAttribute("points", points.map((p, i) => `${(i * step).toFixed(1)},${y(p.ema)}`).join(" "));
  const threshold = (state.snapshot && state.snapshot.state && state.snapshot.state.runner
    && state.snapshot.state.runner.config.threshold) || 0.6;
  const line = el("spark-threshold");
  line.setAttribute("y1", y(threshold)); line.setAttribute("y2", y(threshold));
}

function setLink(text, kind) {
  state.link = text;
  el("conn-status").textContent = text;
  el("conn-dot").className = "dot " + (kind || "");
}

/* ------------------------------------------------------------ diagnostic log */
function flattenData(data) {
  if (!data) return "";
  const parts = [];
  Object.keys(data).sort().forEach((key) => {
    const value = data[key];
    if (value === null || value === undefined) return;
    if (Array.isArray(value)) parts.push(`${key}=[${value.map((v) => (typeof v === "number" ? +v.toFixed(4) : v)).join(",")}]`);
    else if (typeof value === "object") parts.push(`${key}=${JSON.stringify(value)}`);
    else parts.push(`${key}=${value}`);
  });
  return parts.join("  ");
}

function matchesFilters(ev) {
  const stage = el("log-stage").value;
  const sev = el("log-sev").value;
  const needle = el("log-search").value.trim().toLowerCase();
  if (stage && ev.stage !== stage) return false;
  if (sev && ev.severity !== sev) return false;
  if (needle) {
    const haystack = `${ev.stage} ${ev.severity} ${ev.message} ${flattenData(ev.data)}`.toLowerCase();
    if (!haystack.includes(needle)) return false;
  }
  return true;
}

function makeLine(ev) {
  const li = document.createElement("li");
  li.dataset.seq = ev.seq;
  li.dataset.stage = ev.stage;
  li.dataset.sev = ev.severity;
  if (ev.t !== undefined && ev.t !== null) li.dataset.t = ev.t;
  li.className = "sev-" + ev.severity;

  const ts = document.createElement("span"); ts.className = "ts";
  ts.textContent = ev.ts.slice(11, 23);
  const sev = document.createElement("span"); sev.className = "sev";
  sev.textContent = ev.severity;
  const stage = document.createElement("span"); stage.className = "stage";
  stage.textContent = ev.stage;
  const msg = document.createElement("span"); msg.className = "msg";
  msg.textContent = (ev.t !== undefined && ev.t !== null ? `t=${ev.t.toFixed(2)}s ` : "") + ev.message;
  const data = document.createElement("span"); data.className = "data";
  data.textContent = flattenData(ev.data);

  li.append(ts, sev, stage, msg, data);
  return li;
}

function appendEvent(ev) {
  const log = el("diag-log");
  if (!matchesFilters(ev)) return;
  log.appendChild(makeLine(ev));
  while (log.childElementCount > MAX_LOG_NODES) log.removeChild(log.firstChild);
  if (el("log-follow").checked) log.scrollTop = log.scrollHeight;
}

function rebuildLog() {
  const log = el("diag-log");
  log.textContent = "";
  const visible = state.events.filter(matchesFilters);
  // Only the newest slice is drawn: the buffer keeps the history, the DOM stays
  // responsive while a filtered view is on screen.
  visible.slice(-1500).forEach((ev) => log.appendChild(makeLine(ev)));
  el("log-count").textContent = `${state.events.length} lines · showing ${Math.min(visible.length, 1500)}`;
  if (el("log-follow").checked) log.scrollTop = log.scrollHeight;
  state.scores = visible.length ? state.scores : state.scores;
}

function pushEvent(ev) {
  state.events.push(ev);
  if (state.events.length > MAX_BUFFER) state.events.splice(0, state.events.length - MAX_BUFFER);
  const seen = el("log-stage");
  if (!seen.querySelector(`option[value="${ev.stage}"]`)) {
    const option = document.createElement("option");
    option.value = ev.stage; option.textContent = ev.stage;
    seen.appendChild(option);
  }
  if (ev.stage === "score" && ev.data && typeof ev.data.raw_score === "number") {
    // The EMA for this window arrives on the next event; keep the pair together.
    state.scores.push({ raw: ev.data.raw_score, ema: null });
  }
  if (ev.stage === "ema" && ev.data && typeof ev.data.ema === "number") {
    const last = state.scores[state.scores.length - 1];
    if (last && last.ema === null) last.ema = ev.data.ema;
    else state.scores.push({ raw: ev.data.ema, ema: ev.data.ema });
  }
  if (state.scores.length > 400) state.scores.splice(0, state.scores.length - 400);
  appendEvent(ev);
  if (!el("log-follow").checked) el("log-count").textContent = `${state.events.length} lines`;
}

/* -------------------------------------------------------------- stage table */
function lastMsFor(stage) {
  for (let i = state.events.length - 1; i >= 0; i -= 1) {
    const ev = state.events[i];
    if (ev.stage !== stage) continue;
    const d = ev.data || {};
    const candidate = d.infer_ms ?? d.features_ms ?? d.e2e_ms ?? d.decode_ms ?? d.http_ms
      ?? d.load_ms ?? d.latency_ms ?? d.backend_latency_ms;
    if (typeof candidate === "number") return candidate;
  }
  return null;
}

function initStageTable() {
  const body = el("stage-table").querySelector("tbody");
  STAGES.forEach((stage) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td id="stage-${stage}">${stage}</td>`
      + `<td id="stage-${stage}-count">0</td><td id="stage-${stage}-ms">—</td>`;
    body.appendChild(tr);
  });
}

function renderStageTable(snapshot) {
  const counts = (snapshot.log && snapshot.log.stage_counts) || {};
  STAGES.forEach((stage) => {
    const count = el(`stage-${stage}-count`);
    const ms = el(`stage-${stage}-ms`);
    if (count) count.textContent = String(counts[stage] || 0);
    if (ms) {
      const value = lastMsFor(stage);
      ms.textContent = value === null ? "—" : fmt(value, 1);
    }
  });
}

/* --------------------------------------------------------------- rendering */
const BAND_COLOURS = { verified: "#21d4b2", suspicious: "#f59e0b", detected: "#ff6268" };

function renderSnapshot(snapshot) {
  state.snapshot = snapshot;
  const st = snapshot.state || {};
  const runner = st.runner || {};
  const cfg = runner.config || {};
  const local = st.local || {};
  const win = st.window || {};
  const diff = st.diff || {};
  const tally = runner.tally || {};
  const model = st.model || {};
  const source = runner.source || {};

  // ---- AI rating: arc and percentage follow the EMA, exactly as the phone's
  // ShadRiskMeter does; the raw window score sits beside it because diagnosing
  // the smoothing needs both numbers, not only the smoothed one.
  const scored = local.scored === true && typeof local.ema === "number";
  const ema = scored ? local.ema : 0;
  const band = bandFor(ema);
  el("ai-rating-pct").textContent = `${Math.round(ema * 100)}%`;
  el("ai-rating-raw").textContent = scored ? fmt(local.raw_score, 4) : "—";
  el("ai-rating-ema").textContent = scored ? fmt(local.ema, 4) : "—";
  el("ai-rating-verdict").textContent = scored ? verdictFor(ema) : "NO DATA";
  el("ai-rating-verdict").className = "pill " + (scored ? band : "");
  el("ai-rating-attack").textContent = scored && local.attack_type
    ? `${local.attack_type} ${fmt(local.attack_confidence, 2)}` : "—";
  el("ai-state").textContent = scored ? local.state : "—";
  el("ai-consecutive").textContent = scored ? local.consecutive_high : "—";
  el("ai-threshold").textContent = fmt(cfg.threshold, 2);
  el("window-index").textContent = win.index === undefined ? "—" : win.index;
  el("window-t").textContent = fmt(win.t_start_s, 2);
  el("window-rms").textContent = fmt(win.rms_dbfs, 1);
  el("window-e2e").textContent = fmt(win.e2e_ms, 1);

  const card = el("rating-card");
  card.className = "rating" + (scored ? " " + band : "");
  card.style.color = BAND_COLOURS[band];
  el("arc-value").style.strokeDashoffset = String(ARC_LEN * (1 - Math.max(0, Math.min(1, ema))));
  drawThresholdTick(cfg.threshold === undefined ? 0.6 : cfg.threshold);
  drawSpark();

  // ---- pipeline identity
  el("cfg-window").textContent = cfg.window_s === undefined ? "—" : `${cfg.window_s}s / ${cfg.window_samples} samples`;
  el("cfg-hop").textContent = cfg.hop_s === undefined ? "—" : `${cfg.hop_s}s / ${cfg.hop_samples}`;
  el("cfg-speed").textContent = cfg.speed === undefined ? "—" : String(cfg.speed);
  el("cfg-model-loaded").textContent = model.loaded
    ? `yes · ${model.sha256_short}` : `NO — ${model.load_error || "not loaded"}`;
  const contract = model.contract || {};
  el("cfg-contract").textContent = contract.n_frames
    ? `${contract.n_frames}x${contract.n_lfcc} + ${contract.n_scalars}` : "—";
  el("cfg-decoder").textContent = source.decoder
    ? `${source.decoder} / ${source.subtype}${source.device_parity === false ? " (not device-parseable)" : ""}`
    : "—";
  el("cfg-source").textContent = source.file || "—";

  // ---- diff strip
  el("diff-local").textContent = fmt(diff.local_raw, 4);
  el("diff-backend").textContent = fmt(diff.backend_raw, 4);
  el("diff-delta").textContent = fmt(diff.delta_raw, 4);
  el("diff-backend-ema").textContent = fmt(diff.backend_ema, 4);
  el("diff-verdict").textContent = `${diff.verdict_local || "—"} / ${diff.verdict_backend || "—"}`;
  el("diff-scorer").textContent = diff.backend_reports || "—";
  el("diff-http").textContent = diff.latency_ms === null || diff.latency_ms === undefined
    ? "—" : `${fmt(diff.latency_ms, 1)} ms`;
  const flags = el("diff-flags");
  flags.textContent = "";
  (diff.flags || []).forEach((flag) => {
    const span = document.createElement("span");
    span.className = "flag " + flag;
    span.textContent = flag;
    flags.appendChild(span);
  });

  // ---- tally
  el("tally-windows").textContent = `${tally.windows_scored || 0} / ${tally.windows_silent || 0}`;
  el("tally-agree").textContent = tally.agree_pct === null || tally.agree_pct === undefined
    ? "—" : `${tally.agree_pct}% (${tally.agree || 0}/${tally.compared || 0})`;
  el("tally-diverge").textContent = String(tally.diverge || 0);
  el("tally-band").textContent = String(tally.band_mismatch || 0);
  el("tally-delta").textContent = fmt(tally.mean_abs_delta_raw, 4);
  el("tally-errors").textContent = String(tally.backend_errors || 0);
  el("tally-alerts").textContent = String(tally.alerts || 0);
  el("tally-latency").textContent = `${fmt(tally.e2e_p50_ms, 1)} / ${fmt(tally.e2e_p95_ms, 1)}`;

  // ---- controls mirror the server's config unless the user is dragging one
  const active = document.activeElement ? document.activeElement.id : "";
  if (active !== "input-threshold" && cfg.threshold !== undefined) {
    el("input-threshold").value = String(cfg.threshold);
    el("lbl-threshold").textContent = fmt(cfg.threshold, 2);
  }
  if (active !== "input-speed" && cfg.speed !== undefined) {
    el("input-speed").value = String(cfg.speed);
    el("lbl-speed").textContent = fmt(cfg.speed, 1);
  }
  if (active !== "input-hop" && cfg.hop_samples !== undefined) {
    el("input-hop").value = String(cfg.hop_samples);
    el("lbl-hop").textContent = String(cfg.hop_samples);
  }
  if (active !== "input-loop") el("input-loop").checked = !!cfg.loop;
  if (active !== "input-backend") el("input-backend").checked = cfg.backend_enabled !== false;
  if (active !== "backend-mode" && cfg.backend_mode) el("backend-mode").value = cfg.backend_mode;

  renderStageTable(snapshot);
  const text = JSON.stringify({ state: st, log: snapshot.log }, null, 1);
  el("monitor-json").textContent = text.length > 4000
    ? `${text.slice(0, 4000)}\n… (truncated — GET /monitor.json for the full object)` : text;
}

/* --------------------------------------------------------------- controls */
async function control(payload) {
  try {
    const res = await fetch("/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json();
    if (!res.ok) {
      setLink(`control failed (${payload.action}): ${body.detail || res.status}`, "bad");
      return null;
    }
    setLink(`control ok: ${payload.action}`, "ok");
    if (body.state) renderSnapshot({ state: body.state, log: (state.snapshot || {}).log });
    return body;
  } catch (error) {
    setLink(`control error (${payload.action}): ${error.message}`, "bad");
    return null;
  }
}

function selectedPath() {
  const typed = el("source-path").value.trim();
  if (typed) return typed;
  return el("source-select").value || "";
}

// Makes a path startable again without retyping/reimporting it: called after
// a successful import or a successful manual-path start so the picker always
// reflects "things this session has already pointed at," not just the fixed
// on-disk source dirs from page load.
function ensureSourceOption(path) {
  if (!path) return;
  const select = el("source-select");
  const existing = Array.from(select.options).find((o) => o.value === path);
  if (existing) { select.value = path; return; }
  const option = document.createElement("option");
  const name = path.split(/[\\/]/).pop() || path;
  option.value = path;
  option.textContent = `${name}  ·  imported`;
  option.title = path;
  select.insertBefore(option, select.firstChild);
  select.value = path;
}

function bindControls() {
  el("btn-start").onclick = async () => {
    const path = selectedPath();
    if (!path) { setLink("pick a source file first", "bad"); return; }
    const body = await control({ action: "start", path });
    // Only list it once the server actually accepted it — a failed/typo'd
    // path (see the raw-path field) shouldn't get remembered as if it worked.
    if (body) ensureSourceOption(path);
  };
  el("btn-stop").onclick = () => control({ action: "stop" });
  // #file-import is a plain, visible native file input (see styles.css) —
  // no click-forwarding or label indirection, both of which a real Chromium
  // anti-abuse check can defeat for a hidden/near-zero-size target.
  el("file-import").onchange = async () => {
    const file = el("file-import").files[0];
    el("file-import").value = "";
    if (!file) return;
    setLink(`uploading ${file.name}…`, "warn");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) { setLink(`upload failed: ${body.detail || res.status}`, "bad"); return; }
      el("source-path").value = body.path;
      setLink(`uploaded ${body.name} (${body.size_bytes} bytes) — starting`, "ok");
      // Refresh from the server (the upload dir is now a source dir — see
      // server.py) so the imported file survives a page reload too, not just
      // this in-memory session; ensureSourceOption is the fallback in case
      // the refresh races the filesystem write.
      await loadSources();
      ensureSourceOption(body.path);
      await control({ action: "start", path: body.path });
    } catch (error) { setLink(`upload error: ${error.message}`, "bad"); }
  };
  el("btn-pause").onclick = () => control({ action: "pause" });
  el("btn-resume").onclick = () => control({ action: "resume" });
  el("btn-step").onclick = () => control({ action: "step" });
  el("btn-reload").onclick = () => control({ action: "reload-model" });

  el("btn-clear").onclick = () => {
    state.events = []; state.scores = [];
    el("diag-log").textContent = "";
    el("log-count").textContent = "0 lines";
    control({ action: "clear-monitor" });
  };

  el("input-threshold").oninput = (e) => { el("lbl-threshold").textContent = fmt(Number(e.target.value), 2); };
  el("input-threshold").onchange = (e) => control({ action: "threshold", value: Number(e.target.value) });
  el("input-speed").oninput = (e) => { el("lbl-speed").textContent = fmt(Number(e.target.value), 1); };
  el("input-speed").onchange = (e) => control({ action: "speed", value: Number(e.target.value) });
  el("input-hop").oninput = (e) => { el("lbl-hop").textContent = String(e.target.value); };
  el("input-hop").onchange = (e) => control({ action: "hop", value: Number(e.target.value) });
  el("input-loop").onchange = (e) => control({ action: "loop", enabled: e.target.checked });
  el("input-backend").onchange = (e) => control({
    action: "backend", enabled: e.target.checked, mode: el("backend-mode").value,
  });
  el("backend-mode").onchange = (e) => control({
    action: "backend", enabled: el("input-backend").checked, mode: e.target.value,
  });

  ["log-stage", "log-sev"].forEach((id) => { el(id).onchange = rebuildLog; });
  el("log-search").oninput = () => {
    clearTimeout(bindControls.filterTimer);
    bindControls.filterTimer = setTimeout(rebuildLog, 200);
  };
  el("log-follow").onchange = () => {
    if (el("log-follow").checked) el("diag-log").scrollTop = el("diag-log").scrollHeight;
  };

  el("btn-copy").onclick = async () => {
    const text = Array.from(el("diag-log").children).map((li) => li.textContent).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setLink(`copied ${el("diag-log").childElementCount} lines`, "ok");
    } catch (error) { setLink(`copy failed: ${error.message}`, "bad"); }
  };

  el("btn-download").onclick = async () => {
    // Pulled from the server's own NDJSON endpoint rather than the DOM, so the
    // download is the complete ring buffer (and proves the endpoint works).
    try {
      const res = await fetch("/logs.ndjson?since=0");
      const blob = new Blob([await res.text()], { type: "application/x-ndjson" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `voiceguard-monitor-${Date.now()}.ndjson`; a.click();
      URL.revokeObjectURL(url);
    } catch (error) { setLink(`download failed: ${error.message}`, "bad"); }
  };
}

async function loadSources() {
  const res = await fetch("/api/sources");
  const body = await res.json();
  const select = el("source-select");
  select.textContent = "";
  (body.items || []).forEach((item) => {
    if (item.truncated) return;
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = `${item.name}  ·  ${item.dir}`;
    option.title = item.path;
    select.appendChild(option);
  });
  return body;
}

/* ------------------------------------------------------------ live updates */
function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws/monitor?replay=300`);
  state.ws = ws;
  ws.onopen = () => setLink("websocket live", "ok");
  ws.onmessage = (event) => {
    const ev = JSON.parse(event.data);
    if (ev.seq <= (state.maxSeq || 0)) return;   // replay after a reconnect: skip what we saw
    state.maxSeq = ev.seq;
    pushEvent(ev);
  };
  ws.onclose = () => {
    setLink("websocket closed — reconnecting", "warn");
    setTimeout(connectSocket, 1500);
  };
  ws.onerror = () => setLink("websocket error", "bad");
}

async function poll() {
  try {
    const res = await fetch("/monitor.json", { cache: "no-store" });
    renderSnapshot(await res.json());
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) setLink("polling (websocket down)", "warn");
  } catch (error) {
    setLink(`console unreachable: ${error.message}`, "bad");
  }
}

async function init() {
  initGauge();
  initStageTable();
  bindControls();
  try {
    await loadSources();
    const health = await (await fetch("/api/health")).json();
    el("model-sha").textContent = health.model_sha256_short || "—";
    el("backend-url").textContent = `${health.backend_url} (${health.backend_mode})`;
    setLink(health.model_loaded ? "ready" : "model NOT loaded", health.model_loaded ? "ok" : "bad");
    const config = await (await fetch("/api/config")).json();
    el("lbl-threshold").textContent = fmt(config.threshold, 2);
    el("lbl-speed").textContent = fmt(config.speed, 1);
    el("lbl-hop").textContent = String(config.hop_samples);
    el("input-hop").max = String(config.max_hop_samples);
  } catch (error) {
    setLink(`startup failed: ${error.message}`, "bad");
  }
  connectSocket();
  setInterval(poll, POLL_MS);
  poll();
}

document.addEventListener("DOMContentLoaded", init);