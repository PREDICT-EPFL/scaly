# October 20 study

This study remeasures pull request #181's branch after merging main, which added "Store 0/1 seed
selector tables as bits" (#184) and "Fold a constant cube and check lane strides with NumPy"
(#182). It measured the clean pushed merge on la015 with October 15's grid, toolchain, solver
libraries and CPU and numerical policies. The full study took 82.9 minutes.

All 79 timed kernels passed both correctness gates, all 21 Scaly cells completed, and all 6,938
steps in 15 episodes succeeded. Eleven CasADi compiles timed out and 16 larger cells were skipped.
October 15 had 77 timings and 18 skipped cells. The difference is CasADi mapped SX on npmpc, which
compiled at N=25 and N=50 in 179.2 and 179.4 seconds, just under the 180-second limit, and then
timed out at N=100.

A first launch was killed after about 15 minutes, when the agent process that had started its tmux
server was restarted. Its partial output is kept in `benchmarks/results/study-2026-10-20-killed`
and is not used. The published study is the second, complete launch:

```bash
SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/study-2026-10-20 --repetitions 1 --order-seed 0 --headline --boost off
uv run internal/notes/perf_2026_10_20/publish.py benchmarks/results/study-2026-10-20 benchmarks/results/study-2026-10-15
```

`publish.py` is October 15's audit with every cell compared against October 15, since the grids
now match, and without that study's retained-binary diagnostic. The unbumpercars SQP pair now
differs in iterations on two steps instead of three. It writes the five CSV assets and
[the complete comparison tables](analysis.md).

#184 changes only the npmpc and unbumpercars kernels and controllers. The race-car and chain sweep
cells keep identical source and object sizes, and their Scaly controllers generate byte-identical
C and shared libraries. Race-car SQP's paired function-evaluation median nevertheless moved from
3.12× to 2.81×, because Scaly's identical library took 59 rather than 54 µs per step. That is
variation between processes, which one process per provider cannot estimate.

With the site built and served on port 8781, `verify_figures.py` renders all fourteen figures on
the three chart pages in both color schemes with the reference machine's cached Firefox. It also
checks the runtime lead labels (7.1×, 18×, 19× and 8.7×). Screenshots and the JSON record are in
`benchmarks/results/study-2026-10-20-figures`.

```bash
uv run --script internal/notes/perf_2026_10_20/verify_figures.py --out benchmarks/results/study-2026-10-20-figures
```
