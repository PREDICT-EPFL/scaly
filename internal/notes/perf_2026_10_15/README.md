# October 15 study

This study replaces the race-car figure placeholders with measured N=5,10,25,50,100,200,400
Hessians and N=50 controllers. It measured the clean pushed branch for pull request #181 after
"Sweep the race car from N = 5 to 400 and close its loop at N = 50". The provenance files retain
the measured revision. The subsequent figure and prose changes were pulled after the study ended.

The reference machine, compiler versions, solver libraries and numerical policy match October 14.
The full study took 76.7 minutes. All 77 timed kernels passed both correctness gates, all 21 Scaly
cells completed, and all 6,938 steps in 15 episodes succeeded. Eleven CasADi compiles timed out and
18 larger cells were skipped. All four N=50 race-car episodes completed the same 1,367-step lap as
before. The audit checks each saved horizon and lap rather than assuming the old seed was used.

Artifacts and the sibling log are retained under `benchmarks/results/study-2026-10-15` in the
study worktree and the maintained repository on la015. The latter copy is verified by checksum.
The complete measurement command was:

```bash
SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/study-2026-10-15 --repetitions 1 --order-seed 0 --headline --boost off
uv run benchmarks/run.py report benchmarks/results/study-2026-10-15
```

`publish.py` audits every command's outcome, the full sweep grid, all saved revision/CPU/math
provenance, controller completion, per-step provider agreement and the N=50 race-car configuration.
It writes the five CSV assets and [the complete comparison tables](analysis.md). It preserves the
single study's data, including noisy points, and does not mix in the diagnostic below.

Two unchanged neural-process model predictive control (NPMPC) cells moved by more than 5% since
October 14. `recheck.py` reruns each study's
retained binary three times in fresh processes, without compiling it. It checks the same CPU
policy and the compiled correctness gate. Both days' MX N=25 binaries show the same 85 to 99 µs
spread, and both Scaly N=200 binaries run at about 254 to 263 µs. The movement is measurement
variation, not evidence of a compiler change. Three retimings of an existing binary do not
estimate variability from a fresh compilation.

```bash
uv run internal/notes/perf_2026_10_15/recheck.py benchmarks/results/study-2026-10-14 benchmarks/results/study-2026-10-15 --out benchmarks/results/study-2026-10-15-recheck-2
uv run internal/notes/perf_2026_10_15/publish.py benchmarks/results/study-2026-10-15 benchmarks/results/study-2026-10-14
```

The diagnostic directory must be unused. The first diagnostic attempt is preserved in
`benchmarks/results/study-2026-10-15-recheck`. Its reader wrongly expected the correctness message
on standard error, but the retained wrapper writes it on standard output before its JSON.
The corrected reader accepts that output and the second attempt completed all twelve runs.

The public pages retain the latest origin figures, including their data-derived lead annotations
and clipped bars, and change only their data and the prose those measurements require. No
compiler or benchmark implementation changed during this task. The N=40 and N=500 workloads in
the previous study do not provide cross-study performance comparisons for the new sizes.

## Validation

The full suite passed all 1,531 tests. Ruff lint and formatting, strict ty checks, and the
documentation build passed. The publication script ran successfully against the completed study.
The data refresh changes all four benchmark pages and no example code or reproduction fence.

The T3 preview returned "Preview automation open failed: This host blocks the sandbox T3's browser
runs in (AppArmor on Ubuntu 23.10+)." `verify_figures.py` uses the reference machine's cached Firefox
instead. It rendered all fourteen figures across the overview, scalability and closed-loop pages
in both light and dark mode, without chart errors, and checked the updated runtime lead labels.
Screenshots and its JSON record are in `benchmarks/results/study-2026-10-15-figures`.

With the built site served on port 8781, reproduce that check with:

```bash
uv run --script internal/notes/perf_2026_10_15/verify_figures.py --out benchmarks/results/study-2026-10-15-figures
```

Independent review by Claude Fable 5.1 at low effort found one stale runtime ratio, corrected from
18× to 19×, and two wording issues, also corrected. The seed statement remains a description of
the harness defaults, and LF normalization of the summary asset preserves its parsed values.
The focused docs build passed. GPT-6 Astra at low effort checked those fixes alone and confirmed
them resolved, with no adjacent issue.
