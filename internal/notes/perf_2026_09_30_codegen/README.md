# The generated-code speed review, 2026-09-30

Studies behind `../codegen_speed_plan_2026_09_30.html` (todo C-196 … C-202) and the per-step
reports `../codegen_speed_o<N>_report.html`. Run each from the repository root. Timings come from an
Apple M3 Max with Apple clang 21 at the JIT's flags (`-O2 -mcpu=native -fno-math-errno`); they are
indicative, not reference-machine results, and no page under `docs/` cites them.

| File | What it does |
| --- | --- |
| `corpus.py` | builds the 26 kernels (tiny, medium, large; dense, sparse, mapped, scanned, the generated IPM, the four benchmark problems' oracles) in a fresh process each, with an empty JIT cache, and writes their C, input blob, the JIT's own outputs and `meta.json` to `build/<variant>/<kernel>/`. Build another checkout's variant with `PYTHONPATH=<worktree>/src uv run --no-sync .../corpus.py --variant base` |
| `timing.py` | compiles each kernel under a set of compile variants (the JIT's flags, `-O3`, `-ffast-math` and each of its parts, `restrict` on callees or on the entry's ABI pointers, every callee inline) and times the cells from C, interleaved round by round, fastest sample; checks every cell's outputs against the JIT's; writes `build/timing_<tag>.json` |
| `time_entry.c` | the C driver: the IPM study's, with a batch of calls per sample (the Mac's clock ticks every 41.7 ns) |
| `tables.py` | a timing result as an HTML table, times over the first cell |
| `plan.py` | builds the plan and the step reports from `plan_template.html`, `report_o<N>_template.html` and `results/`; the checklist in it is where a step's status changes |
| `results/` | the timing results the reports cite: `timing_diag0.json` (compile variants), `timing_diag1.json` (the parts of `-ffast-math`), `timing_o<N>*.json` (each step against the one before), `timing_final.json` (the base commit against the branch), `sweep_base/` and `sweep_final/` (the benchmark sweep against CasADi SX, `bench/run.py sweep`, before and after) |
| `result_template.html` | the plan's closing section, filled with `timing_final.json` |
