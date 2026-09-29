"""Build the plan (``internal/notes/codegen_speed_plan_2026_09_30.html``) and each step's report from their templates and the recorded timings.

    uv run internal/notes/perf_2026_09_30_codegen/plan.py

The checklist is the one place a step's status changes; each finished step links its own report.
"""

from __future__ import annotations

import json
from pathlib import Path

from tables import table

HERE = Path(__file__).resolve().parent
NOTES = HERE.parent
HEAD = (
  (NOTES / "ipm_speed_report.html")
  .read_text()
  .split("<main>")[0]
  .replace("<title>Generated PIQP Speed Pass</title>", "<title>Codegen Speed Plan</title>")
)

# rank, step, todo, change, evidence, status (None while pending, else the result and the report)
CHECKLIST: list[tuple[int, str, str, str, str, tuple[str, str] | None]] = [
  (
    1,
    "O2",
    "C-196",
    "Dot products and sums as independent partial sums in a fixed order: matrix products, matvecs, <code>sum</code>",
    "<code>assoc</code>: mlp_big_fwd 0.43, unbumpercars 0.61, chol_solve_200 0.69, npmpc 0.71, ipm_cvxqp1_dense 0.84, mlp_small_jac 0.86",
    None,
  ),
  (
    2,
    "O1",
    "C-197",
    "Scalarize an entry point that calls nothing (under a callee's op budget, a 4 096-unit work cap)",
    "<code>fast</code>: quat_jac 0.44, rosen_hess 0.66, cartpole_jac 0.75: the identity-seed tables C may not fold",
    ("quat_jac 0.51, cartpole_jac 0.78, rosen_hess 0.83; every other kernel's C unchanged", "codegen_speed_o1_report.html"),
  ),
  (
    3,
    "O3",
    "C-198",
    "FMA contraction across statements on clang, as GCC does by default (<code>-ffp-contract=fast</code>)",
    "<code>contract</code>: mlp_big_fwd 0.80, unbumpercars 0.86, rosen_hess 0.91, riccati_50 0.93; geomean 0.97. Contraction off: 1.09",
    None,
  ),
  (
    4,
    "O4",
    "C-199",
    "NaN-propagating max and min without a separate NaN test",
    "<code>finite</code>: ipm_qafiro_sparse 0.85, ipm_hs118_dense 0.90, ipm_mpc_12_4_20_sparse 0.95",
    None,
  ),
  (
    5,
    "O5",
    "C-200",
    "A concat of one value repeated (a tile) as one loop that fusion can inline",
    "one copy loop per seed in every multi-seed forward-mode kernel (mlp_small_jac, the race cars, the <code>jac</code> snapshot)",
    None,
  ),
  (
    6,
    "O6",
    "C-77",
    "The sparse-derivative assembly fused into the mapped loop (stretch)",
    "35% of the race-car Hessian in the 2026-09-22 study; see the CasADi comparison",
    None,
  ),
]


def checklist() -> str:
  rows = []
  for rank, step, todo, change, evidence, status in CHECKLIST:
    if status is None:
      mark = '<span class="warn">☐ pending</span>'
    else:
      result, report = status
      mark = f'<span class="ok">☑ done</span>: {result} (<a href="{report}">report</a>)'
    rows.append(f"<tr><td>{rank}</td><td>{step}</td><td>{todo}</td><td>{change}</td><td>{evidence}</td><td>{mark}</td></tr>")
  return "\n".join(rows)


def steps() -> str:
  done = sorted((step, todo, status) for _, step, todo, _, _, status in CHECKLIST if status is not None)
  if not done:
    return "<p>None finished yet.</p>"
  items = "".join(f'<li><b>{step}</b> ({todo}): {status[0]}. <a href="{status[1]}">Report</a>.</li>' for step, todo, status in done)
  return f"<ul>{items}</ul>"


# step report: template, output, timing result, column labels
REPORTS: list[tuple[str, str, str, dict[str, str]]] = [
  ("report_o1_template.html", "codegen_speed_o1_report.html", "timing_o1_all.json", {"base/jit": "before", "o1/jit": "after"}),
]


def reports() -> None:
  for template, output, result, labels in REPORTS:
    data = json.loads((HERE / "results" / result).read_text())
    title = template.removeprefix("report_").removesuffix("_template.html").upper()
    head = HEAD.replace("<title>Codegen Speed Plan</title>", f"<title>Codegen Speed {title}</title>")
    body = (HERE / template).read_text().replace("{{TABLE}}", table(data, labels=labels))
    (NOTES / output).write_text(head + body)


def main() -> None:
  body = (HERE / "plan_template.html").read_text()
  diag0 = json.loads((HERE / "results" / "timing_diag0.json").read_text())
  diag1 = json.loads((HERE / "results" / "timing_diag1.json").read_text())
  body = body.replace("{{CHECKLIST}}", checklist()).replace("{{STEPS}}", steps())
  body = body.replace("{{DIAG0}}", table(diag0)).replace("{{DIAG1}}", table(diag1))
  (NOTES / "codegen_speed_plan_2026_09_30.html").write_text(HEAD + body)
  reports()


if __name__ == "__main__":
  main()
