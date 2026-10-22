# Windows Zig SQP validation

Local measurements on October 17, 2026, while implementing Windows builds of
PIQP and SQP with Zig. These are development measurements, separate from the
published reference-machine results. No published benchmark assets were changed.
IPOPT was neither installed nor run.

## Machine and procedure

- Intel Core i5-14600KF, 14 cores and 20 logical processors, 32 GiB installed RAM.
- Windows 11 Pro, build 26200, High performance power plan.
- Zig 0.17.0, whose C frontend reports Clang 22.1.8.
- Python 3.14.8, NumPy 2.4.6, SciPy 1.18.0, CasADi 3.8.0, PIQP 0.6.4.
- JIT flags `-O2 -march=native -fno-math-errno`, scalar libm on both sides.
- One fresh process and empty Scaly compilation cache per provider and problem.
  Zig's own toolchain cache was already warm from setup and smoke runs.
- One benchmark process at a time, in problem order chain, npmpc, race_cars,
  unbumpercars. Seed 0 selected Scaly then CasADi for each provider pair.
- One episode per provider. Boost and CPU affinity were not pinned, and no
  Linux frequency-policy controls were available. These are not headline timings.

Both providers used the same SQP implementation, PIQP subsolver, problem
configuration, warm-start policy and scalar math policy. Each benchmark's own
episode and recording code ran unchanged. Eight short smoke episodes completed
before the eight full episodes.

## Results

All times below are native solver milliseconds per control step, averaged over
the full episode. Function evaluation includes the values and derivatives the
solver requests. Compilation, Python dispatch, plant simulation and recording
are outside this native timer. `summary.csv` also retains construction and wall
solve-call timings under their separate scopes.

| Problem | Size | Steps per provider | Scaly total | CasADi total | Scaly function evaluation | CasADi function evaluation | Total ratio, CasADi/Scaly |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chain | 5 masses, N=12 | 90 | 4.997 | 4.957 | 0.140 | 0.136 | 0.99 |
| Neural-process MPC | N=12, 32/32 decoder | 100 | 0.879 | 1.142 | 0.283 | 0.594 | 1.30 |
| Race-car MPC | N=40, one lap | 1367 | 1.619 | 1.647 | 0.045 | 0.079 | 1.02 |
| Unbumpercars | 8 cars | 200 | 6.605 | 38.387 | 5.743 | 37.431 | 5.81 |

The chain and race-car differences are small relative to ordinary local timing
variation. A single run does not establish a reliable ranking for those cases.
The paired per-step ratios and their 10th and 90th percentiles are in
`paired.csv`, alongside the ratio of mean times shown here.

Every recorded step succeeded. Configurations and episode lengths match within
each provider pair. Chain, neural-process MPC and race-car have identical
per-step iteration and oracle-call counts. Their maximum state/control
differences are respectively 1.17e-8, 1.26e-9 and 4.88e-9.

Unbumpercars has two iteration-count mismatches over 200 paired steps, so its
ratio includes slightly different solver work. That recorder does not retain
oracle-call counts. The largest state difference is 2.62e-7 and the largest
applied-control difference is 2.02e-7. Both runs report 100% success and zero
collision steps. `validation.json` retains the full array comparisons.

## Reproduction and artifacts

These runs used a standalone Zig 0.17.0, with `CC` and `SCALY_CC` set to it
followed by `cc`. Since the `toolchain` extra landed on main, `uv sync --extra
toolchain` installs `zig-bin` 0.17, which the JIT finds without `SCALY_CC`, and
the PIQP build requirements bring the same `zig-bin` on Windows, so `CC` and
`CXX` no longer need to be set. The closed-loop harness still records `CXX` as
its compiler identity and has no `c++` to fall back on, so set `CXX` to an
executable launcher for `zig c++`. Set `SCALY_VECTOR_LIBM=none`.

The full runs used the following command once for each problem, replacing
`<problem>` with `chain`, `npmpc`, `race_cars` and `unbumpercars`:

```bash
uv run --no-sync benchmarks/run.py closed-loop --problem <problem> --solver sqp --oracle scaly,casadi --repetitions 1 --out-dir benchmarks/results/windows-zig-sqp/canonical/<problem>
```

The smoke command added `--smoke`, selected all four problems in one invocation,
and wrote to `benchmarks/results/windows-zig-sqp/smoke` instead. Each saved
provider directory contains its configuration, provenance, telemetry, rollout,
mode timings and MCAP recording. Raw output and compilation caches remain
ignored under `benchmarks/results/windows-zig-sqp/`; only compact summaries are
tracked here.

Regenerate these summaries from the saved local artifacts:

```bash
uv run --no-sync internal/notes/windows_zig_sqp_2026_10_17/summarize.py
```

## Implementation validation

The PIQP/SQP plugin tests passed 73 checks. Their two skips were the Unix-only
malloc preload test and the comparison against IPOPT. Formatting, lint and type
checks excluding the uninstalled IPOPT sources passed. The broad Windows run
collected the available modules and finished with 1382 passed, 117 skipped and
11 failed. Eight failures construct the deliberately excluded IPOPT backend.
The other three were not prose violations: the prose check read the pages in
the Windows code page, where the UTF-8 bytes of `×` decode to an em dash. It
now reads them as UTF-8.

## Loaded DLLs and the JIT cache

Windows keeps a loaded DLL locked, so it can be neither replaced nor deleted.
Publication tolerates losing the rename to another process that already
published and loaded the same content. `recompile()` gives the function a new
cache key, salted with the process id and a per-name counter, and leaves the
previous entry in place for any process still using it. Those entries are never
removed, so the cache grows with every forced recompilation until it is
cleared by hand after the processes exit.

The test-ID baseline was refreshed from the available collection while keeping
the 28 existing IDs in excluded IPOPT modules. Benchmark path parameter IDs
were normalized to forward slashes so Windows preserves the same baseline.
