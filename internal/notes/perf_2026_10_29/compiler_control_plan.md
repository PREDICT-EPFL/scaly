# Compiler controls and regression search

The maintainer authorized this investigation and the agent's launch on October 29. No commits,
pull requests or published pages are part of it. The investigation scripts and specifications were not kept.

The October 20 provenance identifies the October 20 study revision and the environment
`SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++`. GCC is Ubuntu 13.3.0. The sweep already
uses Clang 20.1.8 for C and C++, regardless of those GCC controller variables. The original release
study uses `SCALY_VECTOR_LIBM=glibc SCALY_CC=clang CC=clang CXX=clang++`.

The reference machine is la015. The initial check at 12:10 CET found no tmux session, benchmark
runner or measurement lock, load 0.02/0.04/0.08, all sixteen governors on performance, and boost off.
Main was at `435410e175a7b26213d3973c594276e5904bb0e0` and is left untouched. The release revision
is measured in a clean detached scratch checkout for the release revision.

- [x] Sync the scratch checkout, smoke both compiler environments, and check N=5 outside result directories.
- [x] Run Scaly race-car N=5,10,25,50,100,200,400 and chain M=9 with the historical environment.
- [x] Run canonical Scaly unbumpercars C=8 and chain M=5, each with IPOPT and SQP.
- [x] Repeat both environments in three independent blocks to measure process dispersion. Each
  command retains the original `--repetitions 1 --order-seed 0 --headline --boost off`; sweep
  timing retains `--benchmark-min-time 0.5s`. Rotate which environment starts a block.
- [x] Directly recompile the saved chain M=9 and race-car C kernels with GCC as an additional
  compiler check. Keep the unchanged correctness-gated Google Benchmark wrapper and samples.
  This is separate from reproducing October 20's mixed compiler environment.
- [x] For each slowdown that remains, measure the historical baseline and bisect the first-parent
  main sequence, at most about ten points. Use race-car N=400, chain M=5, or the cheapest
  reproducing cell. Use the unbumpercars C=8 closed loop if its sweep cannot reproduce the loss.
- [x] Measure each isolated culprit and its parent twice, remove scratch worktrees, write the
  findings and comment on #85.

The investigation runner ran each specification sequentially, owned `~/.scaly-measuring`, verified the CPU
policy before and after commands, and wrote every command, environment, timestamp, exit status,
duration and log path to `measurement.json`. It stopped on the first failed command. The launch is a
named tmux session under a persistent systemd user service, with an absolute uv executable.
Outputs go under `benchmarks/results/control-2026-10-29-gcc/` and the corresponding `-clang/`
directory, outside the original study. Only this research job may build or measure on la015.

The protocol is `internal/notes/benchmark_protocol.md`. All runs keep the problem, inputs, native
solver build, numerical policy, native timing boundary, CPU policy and fresh cache/process rules
fixed. Step spread is reported separately from coefficients of variation across process means.
The initial paired controls should take roughly 15 to 25 minutes, followed by adaptive bisection.

Sources read include issue #85's plan, failed launch, confirmation and results, the existing
`analysis.md`, comparison history, October 20 provenance and README, the benchmark procedure,
`benchmarks/README.md`, `run.py`, and the study, sweep, closed-loop, provenance and Google Benchmark
helpers. Repository architecture, codebase, conventions and contributing guidance also apply.

The launch specification was not kept. The session was `control-2026-10-29`, its service was
`scaly-control-20261029`, and its launcher log is `benchmarks/results/control-2026-10-29-job.log`.

## Launch and setup record

The primary job ran from 12:13:48 to 12:28:05 CET. Its 41 commands returned zero.
The additional fixed-source job completed eight current-kernel checks, then stopped because an
October 20 archived link command referred to its removed worktree's Google Benchmark library.
No correctness gate failed. The corrected compiler-check script mapped archived paths into the new output and used
the existing, identically pinned Google Benchmark library. The refinement job reran only the three
pending checks, added a paired Clang M=9 check, and repeated noisy controller and sweep cells.
The refinement job finished at 12:34:00 CET, with all thirteen commands returning zero.

The first bisection setup was stopped before any timing. Applying `CXX=clang++` to `uv sync`
would select a different native-solver cache entry and start a cold build. The corrected setup
clears the measurement compiler variables during `uv sync`, then applies them to benchmark
commands. It also pins the original control checkout's exact existing IPOPT and PIQP libraries
through `SCALY_IPOPT_LIB` and `SCALY_PIQP_LIB`, so all revision comparisons use one solver build.
The bisection provenance and SHA-256 hashes verify this control.

The corrected bisection started at 12:37:24 CET in session
`control-bisect-fixed-2026-10-29`, service `scaly-control-bisect-fixed-20261029`.
Its log is `benchmarks/results/control-2026-10-29-bisect-fixed.log`, outputs are under
`benchmarks/results/control-2026-10-29-bisect-fixed/`. The separate bisection specification was not kept.
It measures the exact October 20 SHA, searches the 25 first-parent main commits to the release
SHA, and repeats each isolated parent and commit twice. Controller confirmations include both
SQP and IPOPT. Only the research job may build or measure on la015.

The first completed search isolates race-car N=400 at `71f8188d4702995d89722d35018347224fc7b966`,
PR #212. Its two-process parent and commit means are 25.50665 and 28.1678 microseconds,
a 10.4% increase. That measured increase is smaller than the full historical one-process gap.
The chain and unbumpercars searches follow sequentially, with an expected end around 12:53 CET.

All corrected bisections finished at 12:51:30 CET. Chain and unbumpercars both isolate
`c99ba3c563a29c8348f8d1b2e3a9d2c43f8d43c7`, PR #166. Both controller solvers have two
parent and two commit confirmations. All research scratch worktrees are removed, no benchmark
process or measurement lock remains, and main is still at its initial SHA. All five scripts pass
formatting/lint checks and both analysis scripts reproduce after cleanup.

The measured conclusions and their limits are in `regression_findings.md`. The findings were posted on
[issue #85](https://github.com/PREDICT-EPFL/scaly/issues/85#issuecomment-6496017593).
