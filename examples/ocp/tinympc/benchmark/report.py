# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Turn the benchmark's JSON results into an HTML report.

    uv run python examples/ocp/tinympc/benchmark/report.py [--o3 results/results-O3.json] [--o2 results/results-O2.json]
                                                [--out internal/notes/tinympc_benchmark_report.html]
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1f22;--muted:#5c6168;--rule:#e2dfd8;--card:#fff;--acc:#3b5fa0;--acc2:#b0613a;--ok:#2f6f5e;--warn:#8a5a1f;--bad:#9b3b4a;--code:#f1eee8}
@media (prefers-color-scheme: dark){:root{--bg:#16181b;--fg:#e6e4df;--muted:#9aa0a8;--rule:#2c2f34;--card:#1d2024;--acc:#86a8e8;--acc2:#e39a74;--ok:#6fc2a8;--warn:#e0ad6a;--bad:#e7899a;--code:#25282d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:36px 16px 72px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;padding-bottom:6px;border-bottom:2px solid var(--rule)}
h3{font-size:16px;margin:22px 0 6px}.sub{color:var(--muted);margin:0 0 18px}code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:1px 4px;border-radius:4px}
.box{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:12px 18px;margin:12px 0}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--card);margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;vertical-align:top;padding:6px 10px;border-bottom:1px solid var(--rule)}
th{font-size:12px;letter-spacing:.02em;color:var(--muted)}tr:last-child td{border-bottom:none}td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--ok);font-weight:600}.warn{color:var(--warn);font-weight:600}.bad{color:var(--bad);font-weight:600}ul{padding-left:22px}li{margin:3px 0}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.chart{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:8px 10px}
.chart h4{margin:0 0 4px;font-size:13px;color:var(--muted);font-weight:600}svg text{fill:var(--muted);font:11px -apple-system,Segoe UI,sans-serif}
.legend{font-size:12.5px;color:var(--muted)}.sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin:0 4px 0 10px}
"""

FAMILIES = {
  "random_mpc": ("Random QP-MPC", [("nx", "nx", lambda r: r["nu"] == 4 and r["N"] == 10), ("nu", "nu", lambda r: r["nx"] == 10 and r["N"] == 10), ("N", "N", lambda r: r["nx"] == 10 and r["nu"] == 4)]),
  "safety_filter": ("Safety filter", [("nx", "nx", lambda r: r["N"] == 10), ("N", "N", lambda r: r["nx"] == 10)]),
  "rocket_landing": ("Rocket landing (SOC)", [("N", "N", lambda r: True)]),
}


def geomean(xs: list[float]) -> float:
  return math.exp(sum(math.log(x) for x in xs) / len(xs))


def speedup(r: dict) -> float:
  return r["tinympc"]["mean_solve_us"] / r["scaly"]["mean_solve_us"]


def chart(title: str, series: list[tuple[str, str, list[tuple[float, float]]]], ylabel: str, ref: float | None = 1.0, logy: bool = False) -> str:
  """A small line chart: ``series`` is ``(label, css colour, [(x, y)])``."""
  w, h, ml, mr, mt, mb = 320, 190, 44, 10, 10, 30
  xs = sorted({x for _, _, pts in series for x, _ in pts})
  ys = [y for _, _, pts in series for _, y in pts] + ([ref] if ref else [])
  f = math.log10 if logy else (lambda v: v)
  lo, hi = min(map(f, ys)), max(map(f, ys))
  if not logy:
    lo = min(lo, 0.0)
  pad = 0.06 * (hi - lo or 1.0)
  lo, hi = lo - (pad if logy else 0.0), hi + pad

  def px(x: float) -> float:
    i = xs.index(x)
    return ml + (w - ml - mr) * (i / max(len(xs) - 1, 1))

  def py(y: float) -> float:
    return mt + (h - mt - mb) * (1 - (f(y) - lo) / (hi - lo))

  out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(title)}">']
  if logy:
    ticks = [10**k for k in range(math.floor(lo), math.ceil(hi) + 1)]
  else:
    raw = (hi - lo) / 4
    step = min((m * 10 ** math.floor(math.log10(raw)) for m in (1, 2, 2.5, 5, 10) if m * 10 ** math.floor(math.log10(raw)) >= raw), default=raw)
    hi = math.ceil(hi / step) * step
    ticks = [lo + step * k for k in range(int(round((hi - lo) / step)) + 1)]
  for t in ticks:
    if lo <= f(t) <= hi:
      y = py(t)
      out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--rule)"/>')
      label = f"{t:g}".replace("e+0", "e")
      out.append(f'<text x="{ml - 4}" y="{y + 3.5:.1f}" text-anchor="end">{label}</text>')
  if ref is not None:
    out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{py(ref):.1f}" y2="{py(ref):.1f}" stroke="var(--muted)" stroke-dasharray="3 3"/>')
  for x in xs:
    out.append(f'<text x="{px(x):.1f}" y="{h - mb + 14}" text-anchor="middle">{x:g}</text>')
  out.append(f'<text x="{(ml + w - mr) / 2:.1f}" y="{h - 3}" text-anchor="middle">{html.escape(ylabel)}</text>')
  for label, colour, pts in series:
    d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(sorted(pts)))
    dash = ' stroke-dasharray="5 3"' if "O2" in label else ""
    out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2"{dash}/>')
    for x, y in pts:
      out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.6" fill="{colour}"><title>{html.escape(label)} {x:g}: {y:.3g}</title></circle>')
  out.append("</svg>")
  return f'<div class="chart"><h4>{html.escape(title)}</h4>{"".join(out)}</div>'


def table(rows: list[dict], lib: dict) -> str:
  head = (
    "<tr><th>Instance</th><th class=r>Steps</th><th class=r>ADMM iters<br>(both)</th><th class=r>TinyMPC<br>µs/solve</th><th class=r>Scaly<br>µs/solve</th>"
    "<th class=r>Speed-up</th><th class=r>ns/iter<br>TinyMPC · Scaly</th><th class=r>Scaly .text<br>bytes</th><th class=r>Scaly gen + cc<br>s</th>"
    "<th class=r>max |Δu₀|</th></tr>"
  )
  body = []
  for r in rows:
    t, g, a = r["tinympc"], r["scaly"], r["agreement"]
    same = a["same_iterations"] == r["steps"]
    sp = speedup(r)
    cls = "ok" if sp >= 1.0 else "bad"
    body.append(
      f"<tr><td><code>{r['name']}</code></td><td class=r>{r['steps']}</td>"
      f"<td class=r>{t['iterations']:.0f}{'' if same else ' <span class=bad>≠</span>'}</td>"
      f"<td class=r>{t['mean_solve_us']:.2f}</td><td class=r>{g['mean_solve_us']:.2f}</td><td class=r><span class={cls}>{sp:.2f}×</span></td>"
      f"<td class=r>{t['per_iteration_ns']:.0f} · {g['per_iteration_ns']:.0f}</td><td class=r>{g['size']['text']:,}</td>"
      f"<td class=r>{g['generate_s']:.2f} + {g['compile_s']:.2f}</td><td class=r>{a['max_u0_diff']:.1e}</td></tr>"
    )
  return f'<div class="scroll"><table>{head}{"".join(body)}</table></div>'


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--o3", type=Path, default=HERE / "results" / "results-O3.json")
  ap.add_argument("--o2", type=Path, default=HERE / "results" / "results-O2.json")
  ap.add_argument("--out", type=Path, default=ROOT / "internal" / "notes" / "tinympc_benchmark_report.html")
  args = ap.parse_args()
  o3, o2 = json.loads(args.o3.read_text()), json.loads(args.o2.read_text())
  lib3 = {"text": 205129, "admm_text": 41075}  # from lib.json / size, see the summary
  by = {opt: {f: [r for r in d["rows"] if r["family"] == f] for f in FAMILIES} for opt, d in (("O3", o3), ("O2", o2))}
  all3 = o3["rows"]
  same = all(r["agreement"]["same_iterations"] == r["steps"] for d in (o3, o2) for r in d["rows"])
  qp_du = max(r["agreement"]["max_u0_diff"] for r in all3 if r["family"] != "rocket_landing")
  soc_du = max(r["agreement"]["max_u0_diff"] for r in all3 if r["family"] == "rocket_landing")
  gm = {f: geomean([speedup(r) for r in by["O3"][f]]) for f in FAMILIES}
  gm2 = {f: geomean([speedup(r) for r in by["O2"][f]]) for f in FAMILIES}
  slower = [r for r in all3 if speedup(r) < 1.0]
  text_lo, text_hi = min(r["scaly"]["size"]["text"] for r in all3), max(r["scaly"]["size"]["text"] for r in all3)
  cc_lo, cc_hi = min(r["scaly"]["generate_s"] + r["scaly"]["compile_s"] for r in all3), max(r["scaly"]["generate_s"] + r["scaly"]["compile_s"] for r in all3)
  lib_cc3 = all3[0]["tinympc"]["compile_s"]
  lib_text3 = all3[0]["tinympc"]["size"]["text"]

  parts = [
    f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
    f"<title>TinyMPC in Scaly</title><style>{CSS}</style></head><body><main>",
    "<h1>TinyMPC in Scaly · benchmark report</h1>",
    f'<p class="sub">examples/ocp/tinympc · the three problem families of the TinyMPC microcontroller benchmarks · {len(all3)} instances · '
    f"{html.escape(o3['compiler'])} on {o3['machine']} ({html.escape(o3['system'])}) · 26 Sep 2026</p>",
    '<div class="box"><p style="margin-top:0"><b>Summary.</b> TinyMPC’s ADMM is written once in Python (<code>examples/ocp/tinympc/solver.py</code>, under 300 lines with its comments) and '
    "Scaly generates C specialized to each problem. On every instance of the three published sweeps it replays the TinyMPC library’s solve "
    "call for call:</p><ul>",
    f"<li><b>Same iterates.</b> Identical ADMM iteration counts at every MPC step of every instance ({'all' if same else 'not all'} {len(all3)}, at -O3 and -O2); "
    f"applied inputs agree to {qp_du:.0e} on the QPs and to {soc_du:.0e} on the rocket (the library computes part of its cone projection in <code>float</code>; "
    "with that made <code>double</code> the difference falls to 9e-14).</li>",
    f"<li><b>Solve time</b> (same compiler, same flags, -O3): geometric-mean speed-up over the library of <b>{gm['random_mpc']:.2f}×</b> on the random QP-MPC sweeps, "
    f"<b>{gm['safety_filter']:.2f}×</b> on the safety filter and <b>{gm['rocket_landing']:.2f}×</b> on the rocket landing "
    f"(at -O2: {gm2['random_mpc']:.2f}×, {gm2['safety_filter']:.2f}×, {gm2['rocket_landing']:.2f}×). "
    f"Slower on {len(slower)} of {len(all3)}: the random QP-MPC with 16–32 inputs, down to {min(speedup(r) for r in all3):.2f}×.</li>",
    f"<li><b>Code size.</b> The generated solver is {text_lo / 1024:.1f}–{text_hi / 1024:.1f} KiB of object code (<code>.text</code>, constants included) against "
    f"{lib_text3 / 1024:.0f} KiB for the library’s three objects ({lib3['admm_text'] / 1024:.0f} KiB for <code>admm.o</code> alone), with no heap, no Eigen and no C++ runtime.</li>",
    f"<li><b>Compile time.</b> {cc_lo:.2f}–{cc_hi:.2f} s per instance for Python generation plus <code>gcc -c</code>, against {lib_cc3:.1f} s to compile the library "
    "(once for all problems, since its sizes are run-time values).</li>",
    "</ul><p style=\"margin-bottom:0\"><b>Found on the way:</b> a Scaly lowering bug that returns wrong numbers when an output is named like an input (todo, below); "
    "the cost of the while-loop carrying its loop invariants (C-112); and five places where the library departs from textbook ADMM, all reproduced so that the iterates compare.</p></div>",
    "<h2>1. What was built</h2>",
    '<div class="scroll"><table><tr><th>File</th><th>Content</th></tr>',
    "<tr><td><code>problem.py</code></td><td>The problem description (<code>Problem</code>, <code>Settings</code>, <code>Cone</code>), the Riccati cache as "
    "<code>tiny_precompute_and_set_cache</code> computes it, and a line-by-line NumPy port of the library’s <code>solve</code>, the test oracle.</td></tr>",
    "<tr><td><code>solver.py</code></td><td><code>build_solver(problem)</code>: one Scaly <code>Function</code> <code>(state, x0, xref, uref, bounds…) → (state_next, iterations, solved, u0)</code>. "
    "The ADMM loop is a <code>sc.while_loop</code>; the Riccati backward pass and the rollout are two <code>sc.scan</code>s over the horizon; box clipping, cone projections "
    "(<code>where</code>, <code>sqrt</code>), residuals (<code>norm_inf</code>) and the stopping test are plain expressions. Dynamics and cache are constants of the generated code; "
    "the warm-start state goes in and out. Options: stage-step lowering, bounds as constants.</td></tr>",
    "<tr><td><code>problems.py</code></td><td>The three families of <code>mcu-solver-benchmarks</code> as closed-loop scenarios, with their published data, settings and sweeps.</td></tr>",
    "<tr><td><code>random_mpc.py</code>, <code>safety_filter.py</code>, <code>rocket_landing.py</code></td><td>Runnable closed loops, each checked against the NumPy reference.</td></tr>",
    "<tr><td><code>benchmark/</code></td><td><code>run_benchmark.py</code> (build, generate, record, replay, measure), a C++ driver for the library, a C driver for the generated code, "
    "a shared episode format, and this report’s generator.</td></tr>",
    "<tr><td><code>tests/integration/test_tinympc.py</code></td><td>10 tests: generated solver against the NumPy reference on each family, state cones with bounds off, the iteration limit, "
    "bounds as constants, the cone projection, the cache, the published random draws, and the three example scripts. 12 mutants, 11 killed; the survivor "
    "(<code>Pinf</code> for <code>Pinfᵀ</code>) is equivalent up to rounding.</td></tr></table></div>",
    "<h2>2. Method</h2><ul>",
    f"<li><b>Library:</b> TinyMPC <code>{o3['tinympc_commit'][:10]}</code> (main, 2 Sep 2026), built from source; one patch, the <code>std::cout</code> that <code>solve</code> prints on "
    "convergence, removed because it would be timed. Adaptive rho off.</li>",
    f"<li><b>Problems:</b> <code>mcu-solver-benchmarks</code> <code>{o3['benchmarks_commit'][:10]}</code>: random QP-MPC (published <code>npz</code> data; sweeps nx 4–32, nu 4–32, N 4–50), "
    "safety filter (double integrator; nx 2–32 at N = 10, N 4–100 at nx = 10), rocket landing with thrust cone (N 2–256).</li>",
    "<li><b>Same numbers on both sides.</b> The library is set up on each instance, and its Riccati cache is written out and built into the generated solver. "
    "The closed loop runs once in Python with the generated solver; each C driver then replays the recorded sequence of warm-started solves, so both solve exactly the same problems.</li>",
    f"<li><b>Timing:</b> only the solve call, <code>clock_gettime</code>; {o3['reps']} runs of each episode from a zero workspace; the median episode total divided by the steps. "
    "Bounds are run-time inputs on both sides.</li>",
    f"<li><b>Build:</b> both sides with <code>{html.escape(' '.join(o3['flags']))}</code> (and -O2), Eigen with <code>-DNDEBUG</code>. Sizes from <code>size</code> on the objects; "
    "compile times best of three.</li>",
    "<li><b>Caveats:</b> the machine is an Apple-silicon Mac through a Linux VM (4 cores), not a microcontroller, and both sides run in double precision; "
    "the microcontroller builds of TinyMPC use <code>float</code>, which this comparison does not cover.</li></ul>",
    "<h2>3. Results</h2>",
    '<p class="legend">Speed-up of the generated solver over the library (library time / Scaly time, per MPC solve):'
    '<span class="sw" style="background:var(--acc)"></span>-O3<span class="sw" style="background:var(--acc2)"></span>-O2 (dashed)</p>',
  ]
  charts = []
  for fam, (label, axes) in FAMILIES.items():
    for key, name, pick in axes:
      series = []
      for opt, colour in (("O3", "var(--acc)"), ("O2", "var(--acc2)")):
        pts = [(r[key], speedup(r)) for r in by[opt][fam] if pick(r)]
        series.append((f"-{opt}", colour, pts))
      charts.append(chart(f"{label}: speed-up vs {name}", series, name))
  parts.append(f'<div class="charts">{"".join(charts)}</div>')
  size_charts = []
  for fam, (label, axes) in FAMILIES.items():
    key, name, pick = axes[0]
    pts = [(r[key], r["scaly"]["size"]["text"] / 1024) for r in by["O3"][fam] if pick(r)]
    size_charts.append(chart(f"{label}: Scaly .text (KiB) vs {name}", [("-O3", "var(--acc)", pts)], name, ref=None))
  parts.append('<h3>Code size of the generated solver (-O3)</h3><p class="legend">The library: 200 KiB for its three objects, whatever the problem.</p>')
  parts.append(f'<div class="charts">{"".join(size_charts)}</div>')
  for fam, (label, _) in FAMILIES.items():
    parts.append(f"<h3>{label} (-O3)</h3>")
    parts.append(table(by["O3"][fam], lib3))
  parts += [
    "<h2>4. Reading the numbers</h2><ul>",
    "<li><b>Where the generated code wins.</b> The stage steps are matrix–vector products with constant matrices. Scaly expands small ones into straight-line code with the "
    "entries as literals, so zeros cost nothing: the double integrator’s <code>A</code>, <code>B</code>, <code>K</code> and <code>Quu⁻¹</code> are block-structured, and the safety "
    "filter gains grow with the dimension (5.8× at nx = 32). With fixed sizes the loops over the horizon have no bookkeeping, and the library’s dynamic Eigen matrices "
    "are gone, and so are the copies of <code>vnew</code> and <code>znew</code> the library makes at every iteration for adaptive rho, even with it off.</li>",
    "<li><b>Where it loses.</b> The random QP-MPC with 16–32 inputs converges in 4–7 iterations per solve, and each iteration is dominated by dense 16–32-wide products "
    "(<code>Quu⁻¹</code>, <code>Kᵀ</code>). Eigen’s vectorized kernels beat the scalar sums of the expanded code there (C-117), and the per-solve cost of packing the carry "
    "weighs more with so few iterations.</li>",
    "<li><b>The carry.</b> A <code>while_loop</code> body reads only its carry, so the references’ linear cost and the bounds travel in it and are copied at every iteration, "
    "and every field is computed into scratch and copied into the next carry. Making the bounds constants (<code>--fixed-bounds</code>) removes two thirds of the invariants and 2–6 % of the solve time (2.73 → 2.63 ms at nx = 32, "
    "18.4 → 17.9 µs at nu = 32). Loop-invariant inputs (C-112, in progress on the Tier 3 branch) and writing concatenated parts straight into the next carry would remove the rest.</li>",
    "<li><b>Code size and compile time are not monotone</b> in the problem size: below Scaly’s automatic expansion limits a whole stage step (for small problems the "
    "whole iteration) becomes straight-line code, above them it stays a loop. That gives the 40–65 KiB peaks at nx = 4, nx = 32 and nu = 32 and the 2–5 s compiles at "
    "nu ≥ 24; <code>lowering=\"block\"</code> keeps loops (35 KiB, 0.3 s at nu = 32) at some cost in speed.</li>",
    "<li><b>Iteration counts</b> are those of today’s library, which reorders the ADMM steps relative to the version that produced the published sheets "
    "(the published sheet has 18 iterations per solve for the safety filter at nx = 4, N = 10; today’s library takes 232 on average, and so does the generated code).</li></ul>",
    "<h2>5. What the port reproduces from the library</h2><ul>",
    "<li>The linear cost uses <code>(Q + ρI)·xref</code>, because <code>work-&gt;Q</code> stores the penalized diagonal: a bias towards the reference that textbook ADMM does not have.</li>",
    "<li>One ρ is folded into the Riccati cache however many constraint sets act on a variable (box and cone on the rocket’s thrust), so the primal step is not the exact "
    "minimizer of the augmented Lagrangian there.</li>",
    "<li><code>project_soc</code> takes <code>float mu</code> and a <code>float</code> norm, and it projects in the metric that scales the cone axis by μ (Euclidean only for μ = 1).</li>",
    "<li>Termination looks at the box slacks only, so cone residuals are not tested (the applied thrust can leave the cone by up to 0.09 at tolerance 1e-2).</li>",
    "<li><code>tiny_api.hpp</code> names the arguments of <code>tiny_set_cone_constraints</code> input-first while the definition takes the state cones first; the library’s own "
    "rocket example passes them the wrong way round (with the cones then disabled by default).</li></ul>",
    "<h2>6. Scaly issues found</h2><ul>",
    "<li><b class=bad>Wrong numbers:</b> when a <code>Function</code> has an output named like one of its inputs, the generated C reads that input from the output buffer "
    "(<code>LowerCtx.register_outputs</code> overwrites <code>self.buffers[name]</code> in <code>passes/lowering.py</code>). A four-line reproduction is in the todo item. "
    "The TinyMPC solver avoids it by naming its output <code>state_next</code>.</li>",
    "<li>Loop invariants in the <code>while_loop</code> carry (C-112), measured above.</li>",
    "<li>Dense constant matrix–vector products in expanded code are scalar sums; a <code>double2</code> form (C-117) would close the gap at nu ≥ 16.</li></ul>",
    "<h2>7. Reproduce</h2>",
    "<p><code>uv run python examples/ocp/tinympc/run_benchmark.py --workdir /tmp/tinympc-bench</code> (add <code>--opt=-O2</code>, <code>--budget 150</code> to run in pieces, "
    "<code>--quick</code> for a toolchain check), then <code>uv run python examples/ocp/tinympc/benchmark/report.py</code>. It clones the pinned TinyMPC and benchmark commits and needs "
    "<code>gcc</code>, <code>g++</code> and <code>size</code>.</p>",
    "<h2>References</h2><ul>",
    '<li>K. Nguyen, S. Schoedel, A. Alavilli, B. Plancher, Z. Manchester, “TinyMPC: Model-Predictive Control on Resource-Constrained Microcontrollers,” ICRA 2024. '
    '<a href="https://arxiv.org/abs/2310.16985">arXiv:2310.16985</a></li>',
    "<li>S. Schoedel, K. Nguyen, E. Nedumaran, B. Plancher, Z. Manchester, “Code Generation and Conic Constraints for Model-Predictive Control on Microcontrollers "
    'with Conic-TinyMPC,” 2024. <a href="https://arxiv.org/abs/2403.18149">arXiv:2403.18149</a></li>',
    '<li>TinyMPC library: <a href="https://github.com/TinyMPC/TinyMPC">github.com/TinyMPC/TinyMPC</a>; benchmarks: '
    '<a href="https://github.com/RoboticExplorationLab/mcu-solver-benchmarks">github.com/RoboticExplorationLab/mcu-solver-benchmarks</a></li></ul>',
    "</main></body></html>",
  ]
  args.out.write_text("\n".join(parts))
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
