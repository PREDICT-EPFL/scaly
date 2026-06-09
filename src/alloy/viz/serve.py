from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ._recording import load_recordings, recording_path

_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alloy IR Viz</title>
  <link rel="stylesheet" href="/assets/style.css">
</head>
<body>
  <div id="topbar">
    <b>Alloy Viz</b>
    <span id="meta"></span>
    <button id="reload">reload</button>
  </div>
  <div id="app">
    <aside>
      <input id="filter" placeholder="filter recordings">
      <div id="recordings"></div>
    </aside>
    <main>
      <div id="steps"></div>
      <div id="tabs">
        <button data-tab="asm" class="active">assembly</button>
        <button data-tab="listing">listing</button>
        <button data-tab="dag">dag</button>
        <button data-tab="code">code</button>
        <button id="fit" class="hidden">fit</button>
      </div>
      <section id="panel"></section>
    </main>
  </div>
  <script src="/assets/app.js"></script>
</body>
</html>
"""

_CSS = r"""
:root { --bg:#111; --fg:#eee; --muted:#999; --line:#333; --sel:#2a4; --panel:#181818; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.35 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; }
button,input { background:#202020; color:var(--fg); border:1px solid var(--line); border-radius:3px; padding:5px 8px; }
button { cursor:pointer; }
button.active,.recording.active,.step.active { background:#263b28; border-color:var(--sel); }
#topbar { height:36px; border-bottom:1px solid var(--line); display:flex; gap:12px; align-items:center; padding:0 10px; }
#topbar #meta { color:var(--muted); flex:1; }
#app { height:calc(100vh - 36px); display:grid; grid-template-columns:280px 1fr; }
aside { border-right:1px solid var(--line); overflow:auto; padding:8px; }
aside input { width:100%; margin-bottom:8px; }
.recording { padding:7px; border:1px solid transparent; border-bottom-color:#242424; cursor:pointer; }
.recording .name { font-weight:600; }
.recording .sub { color:var(--muted); font-size:12px; }
main { min-width:0; display:grid; grid-template-rows:auto auto 1fr; }
#steps { display:flex; gap:8px; align-items:stretch; overflow:auto; padding:8px; border-bottom:1px solid var(--line); }
.phase { display:flex; gap:5px; align-items:center; padding-right:8px; border-right:1px solid #252525; }
.phase-title { color:#71d083; font-size:11px; text-transform:uppercase; letter-spacing:.08em; white-space:nowrap; }
.step { white-space:nowrap; }
#tabs { padding:6px 8px; display:flex; gap:6px; border-bottom:1px solid var(--line); }
#tabs button.hidden { display:none; }
#tabs #fit { margin-left:auto; }
#panel { overflow:auto; background:var(--panel); position:relative; }
pre { margin:0; padding:12px; tab-size:2; font:12px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; white-space:pre; }
.empty { padding:24px; color:var(--muted); }
.graph-wrap { width:100%; height:100%; min-height:720px; overflow:hidden; background:#151515; cursor:grab; }
.graph-wrap.dragging { cursor:grabbing; }
svg.graph { width:100%; height:100%; min-height:720px; }
.edge { stroke:#4a4b57; stroke-width:1.4; fill:none; marker-end:url(#arrow); }
.node rect { stroke:#1f2028; stroke-width:1.4; rx:4; }
.node text { fill:#111; font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; pointer-events:none; }
.node:hover rect { stroke:#e5c07b; stroke-width:2; }
"""

_JS = r"""
let recordings = [], curRecording = null, curStep = 0, curTab = 'asm', graphView = null;
const $ = (id) => document.getElementById(id);
const tabSpec = {asm:'assembly', listing:'listing', dag:'graph', code:'code'};
async function load() {
  recordings = await fetch('/recordings').then(r => r.json());
  $('meta').textContent = `${recordings.length} recordings`;
  renderRecordingList();
  if (recordings.length && curRecording === null) selectRecording(0);
  else renderAll();
}
function renderRecordingList() {
  const q = $('filter').value.toLowerCase();
  $('recordings').innerHTML = '';
  recordings.forEach((t, i) => {
    if (q && !(`${t.name} ${t.function}`).toLowerCase().includes(q)) return;
    const d = document.createElement('div');
    d.className = 'recording' + (i === curRecording ? ' active' : '');
    const dt = new Date((t.started_at || 0) * 1000).toLocaleTimeString();
    const phases = [...new Set((t.steps || []).map(s => s.phase || s.dialect))].join(' → ');
    d.innerHTML = `<div class="name">${esc(t.name)}</div><div class="sub">${esc(t.function)} · ${t.steps.length} steps · ${esc(phases)} · ${dt}</div>`;
    d.onclick = () => selectRecording(i);
    $('recordings').appendChild(d);
  });
}
function selectRecording(i) { curRecording = i; curStep = 0; curTab = 'asm'; renderAll(); }
function selectStep(i) { curStep = i; renderAll(); }
function renderAll() { renderRecordingList(); renderSteps(); renderPanel(); }
function renderSteps() {
  $('steps').innerHTML = '';
  const t = recordings[curRecording];
  if (!t) return;
  let last = null, group = null;
  t.steps.forEach((s, i) => {
    const ph = s.phase || s.dialect || 'step';
    if (ph !== last) {
      group = document.createElement('div'); group.className = 'phase';
      group.innerHTML = `<span class="phase-title">${esc(ph)}</span>`;
      $('steps').appendChild(group); last = ph;
    }
    const b = document.createElement('button');
    b.className = 'step' + (i === curStep ? ' active' : '');
    b.textContent = s.name;
    b.title = `${i}: ${ph}`;
    b.onclick = () => selectStep(i);
    group.appendChild(b);
  });
}
function availableTabs(s) {
  const ret = [];
  if (s?.assembly) ret.push('asm');
  if (s?.listing) ret.push('listing');
  if (s?.graph) ret.push('dag');
  if (s?.code) ret.push('code');
  return ret;
}
function renderTabs(s) {
  const avail = availableTabs(s);
  if (!avail.includes(curTab)) curTab = avail[0] || 'asm';
  document.querySelectorAll('#tabs button[data-tab]').forEach(b => {
    b.classList.toggle('hidden', !avail.includes(b.dataset.tab));
    b.classList.toggle('active', b.dataset.tab === curTab);
  });
  $('fit').classList.toggle('hidden', curTab !== 'dag' || !s?.graph);
}
function renderPanel() {
  const t = recordings[curRecording], s = t && t.steps[curStep], p = $('panel');
  renderTabs(s);
  if (!s) { p.innerHTML = '<div class="empty">no recording selected</div>'; return; }
  if (curTab === 'asm') return pre(s.assembly);
  if (curTab === 'listing') return pre(s.listing);
  if (curTab === 'code') return pre(s.code);
  if (curTab === 'dag') return drawGraph(s.graph);
}
function pre(x) { graphView = null; $('panel').innerHTML = `<pre>${esc(x)}</pre>`; }
function esc(x) { return String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function layoutGraph(g) {
  const nodes = g.nodes || [], edges = g.edges || [];
  const depth = new Map(nodes.map(n => [n.id, 0]));
  for (let changed = true; changed;) {
    changed = false;
    for (const e of edges) {
      const nd = Math.max(depth.get(e.to) || 0, (depth.get(e.from) || 0) + 1);
      if (nd !== (depth.get(e.to) || 0)) { depth.set(e.to, nd); changed = true; }
    }
  }
  const layers = [];
  for (const n of nodes) { const d = depth.get(n.id) || 0; (layers[d] ||= []).push(n); }
  const pos = new Map(), dims = new Map(), lineH = 14, pad = 10, colGap = 90, rowGap = 28;
  let x = 30, maxH = 0;
  for (const layer of layers) {
    let colW = 0, y = 30;
    for (const n of layer) {
      const lines = String(n.label).split('\n').slice(0, 7);
      const w = Math.max(78, ...lines.map(l => l.length * 7)) + pad*2;
      const h = lines.length * lineH + pad*2;
      dims.set(n.id, {w,h,lines}); colW = Math.max(colW, w);
    }
    for (const n of layer) { const d = dims.get(n.id); pos.set(n.id, {x, y}); y += d.h + rowGap; maxH = Math.max(maxH, y); }
    x += colW + colGap;
  }
  return {nodes, edges, pos, dims, W:Math.max(900, x+30), H:Math.max(700, maxH+30)};
}
function drawGraph(g) {
  if (!g) { $('panel').innerHTML = '<div class="empty">no DAG for this step</div>'; return; }
  const L = layoutGraph(g);
  let svg = `<div class="graph-wrap" id="graph-wrap"><svg class="graph" viewBox="0 0 ${L.W} ${L.H}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#4a4b57"/></marker></defs><g id="graph-render">`;
  for (const e of L.edges) {
    const a = L.pos.get(e.from), b = L.pos.get(e.to), ad = L.dims.get(e.from), bd = L.dims.get(e.to); if (!a || !b) continue;
    const x1 = a.x + ad.w, y1 = a.y + ad.h/2, x2 = b.x, y2 = b.y + bd.h/2, mid = (x1+x2)/2;
    svg += `<path class="edge" d="M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}"/>`;
  }
  for (const n of L.nodes) {
    const p = L.pos.get(n.id), d = L.dims.get(n.id), lines = d.lines;
    svg += `<g class="node" transform="translate(${p.x},${p.y})"><rect width="${d.w}" height="${d.h}" fill="${n.color || '#fff'}"/>`;
    lines.forEach((l,i) => svg += `<text x="${10}" y="${18+i*14}">${esc(l)}</text>`); svg += '</g>';
  }
  $('panel').innerHTML = svg + '</g></svg></div>';
  setupPanZoom(L);
}
function setupPanZoom(L) {
  const wrap = $('graph-wrap'), render = document.getElementById('graph-render');
  graphView = {k:1, x:0, y:0};
  const apply = () => render.setAttribute('transform', `translate(${graphView.x},${graphView.y}) scale(${graphView.k})`);
  graphView.fit = () => {
    const r = wrap.getBoundingClientRect();
    const k = Math.min(r.width / L.W, r.height / L.H) * 0.95;
    graphView.k = Math.min(2, Math.max(0.05, k)); graphView.x = (r.width - L.W*graphView.k)/2; graphView.y = 20; apply();
  };
  wrap.onwheel = (e) => {
    e.preventDefault();
    const rect = wrap.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const old = graphView.k, nk = Math.min(8, Math.max(0.05, old * Math.exp(-e.deltaY * 0.001)));
    graphView.x = mx - (mx - graphView.x) * (nk / old); graphView.y = my - (my - graphView.y) * (nk / old); graphView.k = nk; apply();
  };
  let drag = null;
  wrap.onmousedown = (e) => { if (e.button) return; drag = {x:e.clientX, y:e.clientY, ox:graphView.x, oy:graphView.y}; wrap.classList.add('dragging'); };
  window.onmousemove = (e) => { if (!drag) return; graphView.x = drag.ox + e.clientX - drag.x; graphView.y = drag.oy + e.clientY - drag.y; apply(); };
  window.onmouseup = () => { drag = null; wrap.classList.remove('dragging'); };
  graphView.fit();
}
$('reload').onclick = load;
$('filter').oninput = renderRecordingList;
$('fit').onclick = () => graphView?.fit?.();
document.querySelectorAll('#tabs button[data-tab]').forEach(b => b.onclick = () => { curTab = b.dataset.tab; renderPanel(); });
load();
"""


class VizServer(ThreadingHTTPServer):
  allow_reuse_address = True

  def __init__(self, server_address: tuple[str, int], recording_file: Path):
    super().__init__(server_address, Handler)
    self.recording_file = recording_file
    self.started_at = time.perf_counter()


class Handler(BaseHTTPRequestHandler):
  server: VizServer

  def log_message(self, format: str, *args: object) -> None:  # keep output tinygrad-quiet
    return

  def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:
    url = urlparse(self.path)
    if url.path == "/":
      return self._send(_HTML.encode(), "text/html; charset=utf-8")
    if url.path == "/assets/style.css":
      return self._send(_CSS.encode(), "text/css; charset=utf-8")
    if url.path == "/assets/app.js":
      return self._send(_JS.encode(), "application/javascript; charset=utf-8")
    if url.path == "/recordings":
      data = load_recordings(self.server.recording_file)
      qs = parse_qs(url.query)
      if qs.get("summary", [""])[0] in {"1", "true"}:
        data = [
          {k: v for k, v in t.items() if k != "steps"}
          | {"steps": [{"name": s.get("name"), "phase": s.get("phase"), "dialect": s.get("dialect")} for s in t.get("steps", [])]}
          for t in data
        ]
      return self._send(json.dumps(data).encode(), "application/json")
    if url.path.startswith("/assets/"):
      content_type = mimetypes.guess_type(url.path)[0] or "application/octet-stream"
      return self._send(b"", content_type, 404)
    return self._send(b"not found", "text/plain", 404)


def serve(*, host: str = "127.0.0.1", port: int = 8000, path: str | Path | None = None, open_browser: bool = False) -> None:
  recording_file = Path(path) if path is not None else recording_path()
  server = VizServer((host, port), recording_file)
  url = f"http://{host}:{port}"
  print(f"*** alloy viz serving {recording_file} on {url}")
  if open_browser:
    threading.Timer(0.2, lambda: webbrowser.open(url)).start()
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("*** alloy viz shutting down")


def main() -> None:
  parser = argparse.ArgumentParser(description="Serve captured Alloy IR recordings")
  parser.add_argument("--host", default="127.0.0.1", help="host/interface to bind (use 0.0.0.0 to broadcast)")
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--recording-path", type=Path, default=recording_path())
  parser.add_argument("--browser", action="store_true", help="open a browser after startup")
  args = parser.parse_args()
  serve(host=args.host, port=args.port, path=args.recording_path, open_browser=args.browser)


if __name__ == "__main__":
  main()
