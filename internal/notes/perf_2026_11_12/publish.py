"""Build local before/after plot assets from the audited compiler-run comparison."""

import argparse
import csv
import json
from pathlib import Path

LABELS = {
  "chain": "Chain Hessian, M",
  "chain_jac": "Chain Jacobian, M",
  "race_cars": "Race-car Hessian, N",
  "race_cars_jac": "Race-car Jacobian, N",
  "npmpc": "Neural MPC Hessian, N",
  "npmpc_jac": "Neural MPC Jacobian, N",
  "npmpc_decoder": "Decoder Hessian, width",
  "npmpc_decoder_jac": "Decoder Jacobian, width",
  "unbumpercars": "Unbumpercars Hessian, C",
}
METRICS = {
  "runtime_ns": ("Evaluation time, µs", 1000),
  "kernel_compile_ms": ("C compilation, seconds", 1000),
  "source_bytes": ("C source, kB", 1000),
  "dispatch_arithmetic": ("Operations per iteration", 1),
}


def write_csv(path, rows):
  with path.open("w") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main(root, repo):
  root, repo = root.resolve(), repo.resolve()
  assert (root / "audit.json").exists()
  with (root / "comparison.csv").open() as fp:
    rows = [row for row in csv.DictReader(fp) if row["backend"] == "scaly" and row["metric"] in METRICS]
  rows.sort(key=lambda row: (list(LABELS).index(row["workload"]), int(row["size"]), list(METRICS).index(row["metric"])))
  values, ratios = [], []
  for row in rows:
    label, divisor = METRICS[row["metric"]]
    common = dict(problem=LABELS[row["workload"]], size=int(row["size"]), metric=label, status=row["status"])
    for side in ("before", "after"):
      values.append({**common, "revision": side, "value": float(row[side]) / divisor})
    ratios.append(
      {
        **common,
        "before": float(row["before"]) / divisor,
        "after": float(row["after"]) / divisor,
        "ratio": float(row["ratio"]),
        "low95": float(row["low95"]) if row["low95"] else None,
        "high95": float(row["high95"]) if row["high95"] else None,
      }
    )
  assets = repo / "docs/assets/benchmarks"
  write_csv(assets / "compiler_run_values.csv", values)
  write_csv(assets / "compiler_run_ratios.csv", ratios)
  params = [
    {"name": "measure", "value": "Evaluation time, µs", "bind": {"input": "select", "name": "Metric: ", "options": [v[0] for v in METRICS.values()]}}
  ]
  common = {
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "params": params,
    "transform": [{"filter": "datum.metric === measure"}],
    "facet": {"field": "problem", "title": None, "sort": list(LABELS.values())},
    "columns": 3,
    "resolve": {"scale": {"x": "independent", "y": "independent"}},
  }
  x = {
    "field": "size",
    "type": "quantitative",
    "title": "Size",
    "scale": {"type": "log", "nice": False},
    "axis": {"format": "~g", "tickCount": 3, "grid": False},
  }
  absolute = {
    **common,
    "data": {"url": "compiler_run_values.csv", "format": {"type": "csv", "parse": {"size": "number", "value": "number"}}},
    "spec": {
      "width": 180,
      "height": 150,
      "mark": {"type": "line", "point": True},
      "encoding": {
        "x": x,
        "y": {"field": "value", "type": "quantitative", "title": None, "scale": {"type": "log", "zero": False}},
        "color": {"field": "revision", "title": None, "scale": {"domain": ["before", "after"], "range": ["#2a78d6", "#eb6834"]}},
        "tooltip": [
          {"field": "problem"},
          {"field": "size"},
          {"field": "revision"},
          {"field": "metric"},
          {"field": "value", "format": ".5~g"},
          {"field": "status"},
        ],
      },
    },
  }
  relative = {
    **common,
    "data": {
      "url": "compiler_run_ratios.csv",
      "format": {"type": "csv", "parse": {k: "number" for k in ("size", "ratio", "low95", "high95", "before", "after")}},
    },
    "spec": {
      "width": 180,
      "height": 150,
      "encoding": {"x": x},
      "layer": [
        {"mark": {"type": "rule", "color": "#c3c2b7"}, "encoding": {"y": {"datum": 1}}},
        {
          "mark": {"type": "rule", "color": "#52514e"},
          "encoding": {
            "y": {"field": "low95", "type": "quantitative", "title": "After / before", "scale": {"type": "log"}},
            "y2": {"field": "high95"},
          },
        },
        {
          "mark": {"type": "line", "point": True, "color": "#2a78d6"},
          "encoding": {
            "y": {"field": "ratio", "type": "quantitative", "title": "After / before", "scale": {"type": "log"}},
            "opacity": {"condition": {"test": "datum.status === 'unresolved' || datum.status === 'unchanged'", "value": 0.35}, "value": 1},
            "tooltip": [
              {"field": "problem"},
              {"field": "size"},
              {"field": "metric"},
              {"field": "before", "format": ".5~g"},
              {"field": "after", "format": ".5~g"},
              {"field": "ratio", "format": ".4f"},
              {"field": "low95", "format": ".4f"},
              {"field": "high95", "format": ".4f"},
              {"field": "status"},
            ],
          },
        },
      ],
    },
  }
  relative["spec"].pop("encoding")
  for layer in relative["spec"]["layer"][1:]:
    layer["encoding"]["x"] = x
  point = relative["spec"]["layer"][2]
  point["mark"] = {"type": "point", "filled": True, "color": "#2a78d6"}
  relative["spec"]["layer"].insert(
    2,
    {
      "mark": {"type": "line", "color": "#2a78d6", "opacity": 0.55},
      "encoding": {"x": x, "y": point["encoding"]["y"]},
    },
  )
  band = json.loads((root / "audit.json").read_text())["observed_control_ratio_band"]
  relative["spec"]["layer"].insert(
    0,
    {
      "transform": [{"filter": "datum.metric === 'Evaluation time, µs'"}, {"aggregate": [], "groupby": ["problem"]}],
      "mark": {"type": "rect", "color": "#c3c2b7", "opacity": 0.35},
      "encoding": {"y": {"datum": band[0], "type": "quantitative", "scale": {"type": "log"}}, "y2": {"datum": band[1]}},
    },
  )
  for layer in relative["spec"]["layer"]:
    y = layer["encoding"]["y"]
    y["scale"] = {"type": "log", "nice": False, "zero": False}
    y["axis"] = {"format": ".4~g", "grid": False, "tickCount": 4}
  for name, spec in (("compiler_run_values", absolute), ("compiler_run_ratios", relative)):
    (assets / f"{name}.vl.json").write_text(json.dumps(spec, indent=2) + "\n")
  html = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:var(--font-sans);color:var(--foreground);margin:0}label{display:flex;gap:12px;align-items:center;margin-bottom:8px}
select{font:inherit;color:var(--foreground);background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:5px}
p{font-size:13px;color:var(--muted-foreground);margin:8px 0}svg{width:100%;height:690px}text{fill:var(--foreground);font-family:var(--font-sans);font-size:11px}
</style></head><body><label>Metric <select id="metric"></select></label><p>After / before. Smaller values are left of 1. Lines show 95% timing intervals. Gray points mark equal values or unresolved changes.</p>
<svg id="chart" viewBox="0 0 710 690" role="img" aria-label="Compiler comparison by workload and size"></svg>
<p>Before: e7f80e1f. After: 5335c46d. Each interval uses three build means, with four or eight alternating retimings per binary. Evaluation-time changes must also clear the shaded 1.6% reference range. C compilation excludes the wrapper and linker. Operation counts cover the most expensive equally repeated blocks.</p>
<script>const data=DATA;const band=BAND;const menu=document.getElementById('metric');
for(const metric of [...new Set(data.map(r=>r.metric))])menu.add(new Option(metric,metric));
const svg=document.getElementById('chart'),ns='http://www.w3.org/2000/svg';
function add(name,attrs,text){const e=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;}
function draw(){svg.replaceChildren();const rows=data.filter(r=>r.metric===menu.value),low=Math.min(.5,...rows.map(r=>r.low95||r.ratio)),high=Math.max(2,...rows.map(r=>r.high95||r.ratio));
const minimum=Math.log(low)-.08,maximum=Math.log(high)+.08,x=v=>235+365*(Math.log(v)-minimum)/(maximum-minimum);
if(menu.value==='Evaluation time, µs')add('rect',{x:x(band[0]),y:25,width:x(band[1])-x(band[0]),height:623,fill:'var(--border)',opacity:.6});
for(const tick of [.125,.25,.5,1,2,4,8].filter(v=>v>=low&&v<=high)){add('line',{x1:x(tick),x2:x(tick),y1:25,y2:648,stroke:'var(--border)','stroke-dasharray':tick===1?'':'3 4'});add('text',{x:x(tick),y:17,'text-anchor':'middle'},tick+'×');}
rows.forEach((r,i)=>{const y=39+i*23,color=['unresolved','unchanged'].includes(r.status)?'var(--muted-foreground)':r.ratio>1?'var(--destructive)':'var(--success)';
add('text',{x:0,y:y+4},r.problem.replace(', width',' W').replace(', M',' M').replace(', N',' N').replace(', C',' C')+'='+r.size);
if(r.low95&&r.high95)add('line',{x1:x(r.low95),x2:x(r.high95),y1:y,y2:y,stroke:color,'stroke-width':2});
const point=add('circle',{cx:x(r.ratio),cy:y,r:4,fill:color});const title=document.createElementNS(ns,'title');title.textContent=`Before ${r.before.toPrecision(5)}, after ${r.after.toPrecision(5)}. ${r.status}.`;point.append(title);
add('text',{x:615,y:y+4},r.ratio.toFixed(3)+'×');});}
menu.onchange=draw;draw();</script></body></html>"""
  html = html.replace("DATA", json.dumps(ratios)).replace("BAND", json.dumps(band))
  (root / "comparison.html").write_text(html)
  print(f"Wrote {len(values)} before/after points and {len(ratios)} metric comparisons")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("root", type=Path)
  parser.add_argument("repo", type=Path)
  args = parser.parse_args()
  main(args.root, args.repo)
