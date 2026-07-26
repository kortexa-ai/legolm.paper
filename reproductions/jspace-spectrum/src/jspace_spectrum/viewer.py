"""Self-contained interactive replay for token-level spectrum recordings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _rounded(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values]


def summarize_groups(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group"]), []).append(row)
    output = []
    for group, values in sorted(grouped.items()):
        matrix = np.asarray([row["utterance"] for row in values], dtype=float)
        output.append(
            {
                "group": group,
                "count": len(values),
                "case_ids": sorted({str(row["case_id"]) for row in values}),
                "mean": _rounded(matrix.mean(axis=0)),
                "minimum": _rounded(matrix.min(axis=0)),
                "maximum": _rounded(matrix.max(axis=0)),
            }
        )
    return output


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>J-space spectrum replay</title>
<style>
:root {
  color-scheme: dark;
  --bg: #11100f;
  --panel: #1b1917;
  --panel2: #23201d;
  --ink: #f4eee4;
  --muted: #a9a098;
  --line: #3b3631;
  --coral: #ef735f;
  --teal: #55c2b5;
  --gold: #e8b768;
  --blue: #77a8ea;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 14% -4%, #39251f 0, transparent 33rem),
    radial-gradient(circle at 92% 8%, #17302d 0, transparent 34rem),
    var(--bg);
  color: var(--ink);
  font: 15px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(1440px, calc(100% - 32px)); margin: 0 auto; padding: 36px 0 60px; }
header { display: flex; align-items: end; justify-content: space-between; gap: 28px; margin-bottom: 22px; }
.eyebrow { color: var(--gold); text-transform: uppercase; letter-spacing: .16em; font-size: 11px; }
h1 { margin: 5px 0 2px; font: 600 clamp(30px, 5vw, 58px)/.98 Georgia, serif; letter-spacing: -.035em; }
.subtitle { color: var(--muted); max-width: 720px; }
.stats { text-align: right; color: var(--muted); white-space: nowrap; }
.stats strong { color: var(--ink); font-size: 19px; }
.controls {
  display: grid;
  grid-template-columns: 1.05fr 1.5fr 1fr .8fr 1fr;
  gap: 10px;
  padding: 13px;
  border: 1px solid var(--line);
  border-radius: 15px;
  background: color-mix(in srgb, var(--panel) 91%, transparent);
  position: sticky;
  top: 8px;
  z-index: 4;
  backdrop-filter: blur(18px);
}
label { display: grid; gap: 5px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
select, button, input[type="range"] {
  width: 100%;
  color: var(--ink);
  background: var(--panel2);
  border: 1px solid var(--line);
  border-radius: 9px;
  min-height: 38px;
  padding: 7px 9px;
}
button { cursor: pointer; font-weight: 650; }
.grid { display: grid; grid-template-columns: minmax(520px, 1.3fr) minmax(380px, .9fr); gap: 14px; margin-top: 14px; }
.card { background: color-mix(in srgb, var(--panel) 94%, transparent); border: 1px solid var(--line); border-radius: 17px; padding: 18px; min-width: 0; }
.card h2 { margin: 0; font: 600 21px/1.2 Georgia, serif; }
.card-head { display: flex; justify-content: space-between; gap: 20px; align-items: baseline; margin-bottom: 9px; }
.card-note { color: var(--muted); font-size: 12px; }
#radar { display: block; width: 100%; height: min(58vw, 610px); min-height: 460px; }
.radar-grid { fill: none; stroke: var(--line); stroke-width: 1; }
.radar-zero { fill: none; stroke: #d7cabd; stroke-width: 1.2; stroke-dasharray: 4 5; opacity: .75; }
.radar-axis { stroke: var(--line); stroke-width: 1; }
.radar-shape { fill: color-mix(in srgb, var(--coral) 18%, transparent); stroke: var(--coral); stroke-width: 2.6; }
.radar-point { fill: var(--coral); stroke: var(--bg); stroke-width: 1.5; }
.radar-label { fill: var(--ink); font-size: 11px; text-anchor: middle; }
.radar-negative { fill: var(--muted); font-size: 9px; text-anchor: middle; }
.axis-value { fill: var(--gold); font-size: 9px; text-anchor: middle; }
.utterance {
  margin: 8px 0 12px;
  padding: 15px 16px;
  min-height: 58px;
  border-left: 3px solid var(--coral);
  border-radius: 5px 12px 12px 5px;
  background: var(--panel2);
  font: 18px/1.45 Georgia, serif;
}
.token-strip { display: flex; flex-wrap: wrap; gap: 6px; min-height: 38px; }
.token {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 5px 7px;
  color: var(--muted);
  background: #171513;
  cursor: pointer;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}
.token.active { color: var(--ink); border-color: var(--coral); background: #3c211d; }
.transport { display: grid; grid-template-columns: 90px 1fr 58px; gap: 9px; align-items: center; margin: 12px 0; }
#trace { display: block; width: 100%; height: 210px; overflow: visible; }
.trace-axis { stroke: var(--line); stroke-width: 1; }
.trace-zero { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 4; }
.trace-line { fill: none; stroke: var(--teal); stroke-width: 2.3; }
.trace-dot { fill: var(--teal); }
.trace-active { fill: var(--coral); stroke: var(--ink); stroke-width: 1.2; }
.coordinates { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin-top: 10px; }
.coordinate { padding: 8px 9px; border: 1px solid var(--line); border-radius: 9px; background: #171513; }
.coordinate .name { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }
.coordinate .value { margin-top: 2px; font: 600 15px ui-monospace, SFMono-Regular, Menlo, monospace; }
.positive .value { color: var(--teal); }
.negative .value { color: var(--coral); }
.meta { display: grid; grid-template-columns: 1fr 1fr; gap: 7px 20px; margin-top: 14px; color: var(--muted); font-size: 12px; }
.meta span { color: var(--ink); }
footer { color: var(--muted); font-size: 12px; margin-top: 16px; }
@media (max-width: 980px) {
  header { align-items: start; flex-direction: column; }
  .stats { text-align: left; }
  .controls { grid-template-columns: 1fr 1fr; position: static; }
  .grid { grid-template-columns: 1fr; }
}
@media (max-width: 620px) {
  main { width: min(100% - 18px, 1440px); padding-top: 20px; }
  .controls { grid-template-columns: 1fr; }
  .coordinates { grid-template-columns: 1fr 1fr; }
  #radar { min-height: 410px; }
}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="eyebrow">activation recording · no generation</div>
      <h1>Where is “meh”?</h1>
      <div class="subtitle">Replay exact user-token coordinates across twelve fitted social directions and residual depth.</div>
    </div>
    <div class="stats"><strong id="case-count">—</strong> passes<br><span id="model-name">—</span></div>
  </header>
  <section class="controls">
    <label>Family<select id="group"></select></label>
    <label>Utterance<select id="case"></select></label>
    <label>System prompt<select id="system"></select></label>
    <label>Layer<select id="layer"></select></label>
    <label>Trace axis<select id="axis"></select></label>
  </section>
  <section class="grid">
    <article class="card">
      <div class="card-head"><h2>Signed radar</h2><span class="card-note">outward = positive · inward = negative</span></div>
      <svg id="radar" viewBox="0 0 650 610" role="img" aria-label="Twelve-axis activation radar"></svg>
    </article>
    <article class="card">
      <div class="card-head"><h2>Token replay</h2><span class="card-note" id="token-note">—</span></div>
      <div class="utterance" id="utterance"></div>
      <div class="token-strip" id="tokens"></div>
      <div class="transport">
        <button id="play">▶ Play</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
        <output id="position">1 / 1</output>
      </div>
      <svg id="trace" viewBox="0 0 560 210" role="img" aria-label="Selected coordinate across tokens"></svg>
      <div class="coordinates" id="coordinates"></div>
      <div class="meta" id="meta"></div>
    </article>
  </section>
  <footer>Coordinates are centered and scaled by 24 neutral messages within each system prompt and layer. The model is frozen; this page replays saved forward passes.</footer>
</main>
<script id="recording" type="application/json">__JSPACE_RECORDING__</script>
<script>
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("recording").textContent);
  const $ = id => document.getElementById(id);
  const state = { group: "", caseId: "", system: "", layer: "", axis: 0, token: 0, timer: null };
  const controls = { group: $("group"), case: $("case"), system: $("system"), layer: $("layer"), axis: $("axis") };
  const svgNS = "http://www.w3.org/2000/svg";
  const element = (name, attrs = {}, text = "") => {
    const node = document.createElementNS(svgNS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text) node.textContent = text;
    return node;
  };
  const unique = values => [...new Set(values)];
  const casesFor = () => data.cases.filter(row => row.group === state.group && row.system === state.system);
  const current = () => data.cases.find(row => row.id === state.caseId && row.system === state.system) || casesFor()[0] || data.cases[0];
  const vector = (row, tokenIndex = null) => {
    const layer = String(state.layer);
    if (tokenIndex === null) return row.utterance_by_layer[layer];
    return row.tokens[tokenIndex].coordinates_by_layer[layer];
  };
  const option = (value, label = value) => {
    const node = document.createElement("option");
    node.value = value; node.textContent = label; return node;
  };
  function initialize() {
    $("case-count").textContent = data.cases.length;
    $("model-name").textContent = `${data.model.id} · ${data.model.device}`;
    const groups = unique(data.cases.map(row => row.group)).sort();
    groups.forEach(group => controls.group.append(option(group)));
    const systems = unique(data.cases.map(row => row.system));
    systems.forEach(system => controls.system.append(option(system)));
    data.lens.trace_layers.forEach(layer => controls.layer.append(option(String(layer), `L${layer}`)));
    data.axis_names.forEach((name, index) => controls.axis.append(option(String(index), name.replaceAll("_", " "))));
    state.group = groups.includes("meh") ? "meh" : groups[0];
    state.system = systems[0];
    state.layer = String(data.lens.source_layer);
    controls.group.value = state.group;
    controls.system.value = state.system;
    controls.layer.value = state.layer;
    rebuildCases();
    bind();
    render();
  }
  function rebuildCases() {
    controls.case.replaceChildren();
    const rows = casesFor();
    rows.forEach(row => controls.case.append(option(row.id, row.text)));
    state.caseId = rows.some(row => row.id === state.caseId) ? state.caseId : (rows[0]?.id || "");
    controls.case.value = state.caseId;
    state.token = 0;
  }
  function bind() {
    controls.group.addEventListener("change", () => { state.group = controls.group.value; rebuildCases(); render(); });
    controls.case.addEventListener("change", () => { state.caseId = controls.case.value; state.token = 0; render(); });
    controls.system.addEventListener("change", () => { state.system = controls.system.value; rebuildCases(); render(); });
    controls.layer.addEventListener("change", () => { state.layer = controls.layer.value; render(); });
    controls.axis.addEventListener("change", () => { state.axis = Number(controls.axis.value); render(); });
    $("scrub").addEventListener("input", () => { state.token = Number($("scrub").value); render(); });
    $("play").addEventListener("click", togglePlay);
  }
  function togglePlay() {
    if (state.timer) {
      clearInterval(state.timer); state.timer = null; $("play").textContent = "▶ Play"; return;
    }
    $("play").textContent = "Ⅱ Pause";
    state.timer = setInterval(() => {
      const row = current();
      state.token = (state.token + 1) % row.tokens.length;
      render();
    }, 620);
  }
  function radar(row) {
    const root = $("radar"); root.replaceChildren();
    const values = vector(row, state.token);
    const limit = Math.max(4, Math.ceil(Math.max(...values.map(Math.abs))));
    const width = 650, height = 610, cx = width / 2, cy = height / 2 + 2, radius = 220;
    const angle = index => -Math.PI / 2 + index * 2 * Math.PI / values.length;
    const point = (index, value) => {
      const scaled = (value + limit) / (2 * limit);
      const r = scaled * radius;
      return [cx + Math.cos(angle(index)) * r, cy + Math.sin(angle(index)) * r];
    };
    [0, .25, .5, .75, 1].forEach(fraction => root.append(element("circle", { cx, cy, r: radius * fraction, class: fraction === .5 ? "radar-zero" : "radar-grid" })));
    data.axis_poles.forEach((pole, index) => {
      const a = angle(index), outside = radius + 42;
      root.append(element("line", { x1: cx, y1: cy, x2: cx + Math.cos(a) * radius, y2: cy + Math.sin(a) * radius, class: "radar-axis" }));
      root.append(element("text", { x: cx + Math.cos(a) * outside, y: cy + Math.sin(a) * outside - 4, class: "radar-label" }, pole.positive));
      root.append(element("text", { x: cx + Math.cos(a) * outside, y: cy + Math.sin(a) * outside + 10, class: "radar-negative" }, pole.negative));
      root.append(element("text", { x: cx + Math.cos(a) * (radius + 19), y: cy + Math.sin(a) * (radius + 19) + 3, class: "axis-value" }, values[index].toFixed(1)));
    });
    const points = values.map((value, index) => point(index, value));
    root.append(element("polygon", { points: points.map(pair => pair.join(",")).join(" "), class: "radar-shape" }));
    points.forEach(pair => root.append(element("circle", { cx: pair[0], cy: pair[1], r: 4, class: "radar-point" })));
  }
  function trace(row) {
    const root = $("trace"); root.replaceChildren();
    const values = row.tokens.map((_, index) => vector(row, index)[state.axis]);
    const left = 30, right = 548, top = 15, bottom = 185;
    const low = Math.min(-1, ...values), high = Math.max(1, ...values);
    const x = index => values.length === 1 ? (left + right) / 2 : left + index * (right - left) / (values.length - 1);
    const y = value => bottom - (value - low) * (bottom - top) / (high - low);
    root.append(element("line", { x1: left, x2: right, y1: y(0), y2: y(0), class: "trace-zero" }));
    root.append(element("line", { x1: left, x2: left, y1: top, y2: bottom, class: "trace-axis" }));
    root.append(element("polyline", { points: values.map((value, index) => `${x(index)},${y(value)}`).join(" "), class: "trace-line" }));
    values.forEach((value, index) => root.append(element("circle", { cx: x(index), cy: y(value), r: index === state.token ? 5 : 3, class: index === state.token ? "trace-active" : "trace-dot" })));
    root.append(element("text", { x: left + 4, y: top + 2, fill: "#a9a098", "font-size": 10 }, high.toFixed(1)));
    root.append(element("text", { x: left + 4, y: bottom, fill: "#a9a098", "font-size": 10 }, low.toFixed(1)));
  }
  function render() {
    const row = current();
    if (!row) return;
    state.caseId = row.id;
    controls.case.value = row.id;
    state.token = Math.min(state.token, row.tokens.length - 1);
    const token = row.tokens[state.token];
    $("utterance").textContent = row.text;
    $("token-note").textContent = `${token.label || token.text} · L${state.layer}`;
    $("scrub").max = Math.max(0, row.tokens.length - 1);
    $("scrub").value = state.token;
    $("position").textContent = `${state.token + 1} / ${row.tokens.length}`;
    $("tokens").replaceChildren(...row.tokens.map((item, index) => {
      const node = document.createElement("button");
      node.className = `token${index === state.token ? " active" : ""}`;
      node.textContent = item.label || item.text;
      node.addEventListener("click", () => { state.token = index; render(); });
      return node;
    }));
    const values = vector(row, state.token);
    $("coordinates").replaceChildren(...values.map((value, index) => {
      const node = document.createElement("div");
      node.className = `coordinate ${value >= 0 ? "positive" : "negative"}`;
      node.innerHTML = `<div class="name">${data.axis_names[index].replaceAll("_", " ")}</div><div class="value">${value >= 0 ? "+" : ""}${value.toFixed(2)}</div>`;
      return node;
    }));
    $("meta").innerHTML = `
      <div>family <span>${row.group}</span></div>
      <div>system <span>${row.system}</span></div>
      <div>case <span>${row.case_id}</span></div>
      <div>user tokens <span>${row.user_tokens}</span></div>
      <div>kind <span>${row.kind}</span></div>
      <div>context tokens <span>${row.context_tokens}</span></div>`;
    radar(row); trace(row);
  }
  initialize();
})();
</script>
</body>
</html>
"""


def render_viewer(recording: Mapping[str, Any], output: Path) -> Path:
    marker = "__JSPACE_RECORDING__"
    if TEMPLATE.count(marker) != 1:
        raise ValueError("viewer data marker is missing or duplicated")
    serialized = json.dumps(
        recording,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    serialized = (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(TEMPLATE.replace(marker, serialized))
    return output
