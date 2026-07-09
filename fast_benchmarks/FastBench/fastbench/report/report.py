"""Turn a results.json into a CSV, a markdown report, plots and an HTML page.

Reported metric families:
* control quality - closed-loop cost, tracking RMSE, terminal error,
  settling time, control effort, suboptimality vs the IPOPT reference;
* solve quality - step success rate, OCP/closed-loop constraint violation,
  KKT residual;
* reliability - episode success rate, deadline-miss rate;
* timing - median / p95 / p99 / worst solve time, iterations;
* code - build time, generated-code size & LOC, compiled size, generation time
  (for code-generating backends such as acados).
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd

# headline columns pulled from the aggregated metrics
COLS = [
    ("episode_success_rate", "succ", "{:.2f}"),
    ("closed_loop_cost_mean", "cl_cost", "{:.4g}"),
    ("suboptimality_mean", "subopt", "{:.2%}"),
    ("tracking_rmse_mean", "rmse", "{:.3g}"),
    ("terminal_error_mean", "x_term", "{:.3g}"),
    ("settling_time_s_mean", "t_settle", "{:.2f}"),
    ("control_effort_mean", "effort", "{:.3g}"),
    ("state_constraint_violation_worst", "cviol", "{:.1e}"),
    ("deadline_miss_rate_mean", "miss", "{:.2f}"),
    ("solve_time_median_s_mean", "t_med_ms", "{:.3f}"),
    ("solve_time_p99_s_worst", "t_p99_ms", "{:.3f}"),
    ("iter_median_mean", "iters", "{:.0f}"),
]
BUILD_COLS = [
    ("build_time_s", "build_s", "{:.2f}"),
    ("generation_time_s", "gen_s", "{:.2f}"),
    ("generated_code_bytes", "gen_kB", "{:.0f}"),
    ("generated_code_loc", "gen_loc", "{:.0f}"),
    ("compiled_bytes", "comp_kB", "{:.0f}"),
]


def _flatten(results: dict) -> pd.DataFrame:
    rows = []
    for rec in results["records"]:
        row = {"problem": rec["problem"], "solver": rec["solver"],
               "status": rec.get("status", "?"), "reason": rec.get("reason", "")}
        m = rec.get("metrics", {})
        for key, short, _ in COLS:
            row[short] = m.get(key, np.nan)
        b = rec.get("build", {})
        for key, short, _ in BUILD_COLS:
            v = b.get(key, np.nan)
            if short.endswith("kB") and v == v:
                v = v / 1024.0
            if short == "t_med_ms" or short == "t_p99_ms":
                pass
            row[short] = v
        # convert solve times to ms
        for k in ("t_med_ms", "t_p99_ms"):
            if k in row and row[k] == row[k]:
                row[k] *= 1e3
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    show = ["solver", "status", "succ", "cl_cost", "subopt", "rmse", "x_term",
            "t_settle", "cviol", "miss", "t_med_ms", "t_p99_ms", "iters",
            "build_s", "gen_kB", "gen_loc", "comp_kB"]
    show = [c for c in show if c in df.columns]
    sub = df[show].copy()
    for c in sub.columns:
        if sub[c].dtype.kind in "fc":
            sub[c] = sub[c].map(lambda v: "" if (isinstance(v, float) and np.isnan(v))
                                else (f"{v:.4g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"))
    return sub.to_markdown(index=False)


def _plots(df: pd.DataFrame, outdir: str) -> List[str]:
    imgs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return imgs
    ok = df[df["status"] == "ok"]
    # solve-time per problem
    for prob in ok["problem"].unique():
        d = ok[ok["problem"] == prob]
        if d["t_med_ms"].notna().any():
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.bar(d["solver"], d["t_med_ms"], color="#1f77b4")
            ax.errorbar(d["solver"], d["t_med_ms"],
                        yerr=np.maximum(d["t_p99_ms"] - d["t_med_ms"], 0),
                        fmt="none", ecolor="#444", capsize=4)
            ax.set_ylabel("solve time [ms]"); ax.set_title(f"{prob}: median (bar) / p99 (whisker)")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fn = os.path.join(outdir, f"timing_{prob}.png")
            fig.savefig(fn, dpi=110); plt.close(fig); imgs.append(os.path.basename(fn))
    # codegen size where available
    cg = ok[ok["gen_kB"].notna() & (ok["gen_kB"] > 0)]
    if len(cg):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(cg["problem"] + "/" + cg["solver"], cg["gen_kB"], color="#2ca02c")
        ax.set_ylabel("generated code [kB]"); ax.set_title("Generated C-code size")
        plt.xticks(rotation=30, ha="right"); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fn = os.path.join(outdir, "codegen_size.png")
        fig.savefig(fn, dpi=110); plt.close(fig); imgs.append(os.path.basename(fn))
    return imgs


def _perf_profile(df: pd.DataFrame, outdir: str):
    """Dolan-More performance profile on median solve time across tasks."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    ok = df[(df["status"] == "ok") & df["t_med_ms"].notna()]
    if ok["problem"].nunique() < 2 or ok["solver"].nunique() < 2:
        return None
    pivot = ok.pivot_table(index="problem", columns="solver", values="t_med_ms")
    best = pivot.min(axis=1)
    ratios = pivot.div(best, axis=0)
    taus = np.logspace(0, np.log10(np.nanmax(ratios.values) + 1e-9), 100)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for solver in ratios.columns:
        r = ratios[solver].dropna().values
        rho = [np.mean(r <= t) for t in taus]
        ax.step(taus, rho, where="post", label=solver)
    ax.set_xscale("log"); ax.set_xlabel(r"$\tau$ (x best solve time)")
    ax.set_ylabel(r"fraction of problems"); ax.set_ylim(0, 1.05)
    ax.set_title("Performance profile (median solve time)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fn = os.path.join(outdir, "perf_profile.png")
    fig.savefig(fn, dpi=110); plt.close(fig)
    return os.path.basename(fn)


def build_report(results_path: str, outdir: str = None) -> str:
    with open(results_path) as fh:
        results = json.load(fh)
    if outdir is None:
        outdir = os.path.join(os.path.dirname(results_path), "report")
    os.makedirs(outdir, exist_ok=True)

    df = _flatten(results)
    df.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    imgs = _plots(df, outdir)
    pp = _perf_profile(df, outdir)

    env = results.get("env", {})
    cfg = results.get("config", {})
    md = ["# FastBench report", "",
          f"*{env.get('timestamp_utc','')}* - host `{env.get('cpu','?')}` - "
          f"git `{env.get('git_hash','?')}` - "
          f"{cfg.get('n_episodes','?')} episodes/seed {cfg.get('seed','?')}", "",
          "Solve times in **ms**; sizes in **kB**. `subopt` is relative to the "
          "high-accuracy IPOPT closed-loop cost. `miss` is the deadline-miss "
          "rate (solve time > dt). Empty cells = not measured for that backend.", ""]
    for prob in df["problem"].unique():
        d = df[df["problem"] == prob]
        meta = next((r["problem_meta"] for r in results["records"]
                     if r["problem"] == prob), {})
        md.append(f"## {prob}")
        md.append(f"*{meta.get('description','')}*  "
                  f"(nx={meta.get('nx')}, nu={meta.get('nu')}, N={meta.get('N')}, "
                  f"dt={meta.get('dt')}s, {'LTI/QP' if meta.get('is_lti') else 'nonlinear'})")
        md.append("")
        md.append(_markdown_table(d))
        md.append("")
        for im in imgs:
            if im == f"timing_{prob}.png":
                md.append(f"![timing]({im})\n")
    if pp:
        md.append("## Performance profile\n")
        md.append(f"![perf]({pp})\n")
    if any(i == "codegen_size.png" for i in imgs):
        md.append("## Code generation\n")
        md.append("![codegen](codegen_size.png)\n")
    md_text = "\n".join(md)
    with open(os.path.join(outdir, "REPORT.md"), "w") as fh:
        fh.write(md_text)

    _write_html(md_text, df, imgs, pp, outdir, env)
    print(f"Report written to {outdir} (REPORT.md, index.html, summary.csv)")
    return outdir


def _write_html(md_text, df, imgs, pp, outdir, env):
    try:
        import markdown  # optional
        body = markdown.markdown(md_text, extensions=["tables"])
    except Exception:
        body = "<pre>" + md_text.replace("<", "&lt;") + "</pre>"
        for im in imgs + ([pp] if pp else []):
            body += f'<img src="{im}" style="max-width:640px">'
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>FastBench report</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:3px 7px}}
th{{background:#1F3864;color:#fff}}img{{margin:8px 0}}code{{background:#f3f3f3;padding:1px 4px}}</style>
</head><body>{body}</body></html>"""
    with open(os.path.join(outdir, "index.html"), "w") as fh:
        fh.write(html)
