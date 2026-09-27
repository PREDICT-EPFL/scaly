# Speeding up the generated PIQP, 2026-09-27

Studies behind `../ipm_speed_report.html` (todo C-135 onward). Run each from the repository root.
Timings come from an Apple M3 Max with Apple clang 21 at the JIT's flags (`-O2 -mcpu=native
-fno-math-errno`); they are indicative, not reference-machine results, and no page under `docs/`
cites them.

| Script | Measures |
| --- | --- |
| `gen.py` | builds the generated solver (both backends) per problem in a fresh process with an empty JIT cache and writes its C, shared library, input blob and `meta.json` (status, iterations, `x`, build, generation and compile time, C lines) to `build/<variant>/`. `--problems quick` (18 problems: Maros–Mészáros, MPC and the `examples/qp_solvers` families) or `all` (55). Build another checkout's variant with `PYTHONPATH=<worktree>/src uv run --no-sync ...gen.py --variant base` |
| `timing.py` | times the built solvers from C (`time_entry.c`, the universal entry in a loop, no Python), variants interleaved round by round, fastest solve per variant; `--piqp` adds vendored PIQP 0.6.2's own warmed solve timer. Prints a Markdown table, geometric means, and whether the variants' `x` agree |
| `prof.py` | self samples per generated procedure (macOS `sample`), with every procedure but the per-step loop bodies kept out of line: where a solve spends its time |
| `time_entry.c` | the C driver: `examples/qp_solvers/time_entry.c` with a nanosecond clock on macOS (`CLOCK_MONOTONIC` ticks in microseconds there) |
