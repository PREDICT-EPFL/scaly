---
name: benchmark-study
description: Plan, run and report a Scaly benchmark study or headline measurement on the reference machine, la015, from sweeps and closed loops to refreshing the published benchmark pages. Use only when the user or another skill asks for it by name, never on your own initiative.
---

# Run a Scaly benchmark study

Read [the benchmark protocol](../../../internal/notes/benchmark_protocol.md), which owns what a
comparison holds constant and why, and [`benchmarks/README.md`](../../../benchmarks/README.md) for the
commands. Read the benchmark pages under `docs/benchmarks/` that the results will feed.

## Plan before running

Post the plan on the issue or pull request before starting anything longer than a smoke run:

- the question, and what result would change a decision
- the comparison and what it holds constant, citing the protocol
- the grid: problems, sizes, backends, solvers and repetitions
- the exact commands and environment, the output directory and the expected duration (the full
  September 23 study took five and a half hours)
- who launches it: the agent, or the maintainer from their own terminal

Wait for the maintainer's answer on the last point. If they launch it, give them the command block
to paste and stop there.

## Prepare

1. Measure only a pushed revision, in a worktree no other thread is using, with no uncommitted
   changes. Copy the solver artifacts and sync: `wt step copy-ignored && uv sync`.
2. Headline numbers come only from la015. Check the `performance` governor and the boost state the
   plan names. `--headline` verifies them but does not set them.
3. Check that nothing else is measuring or compiling: `tmux ls`, `pgrep -af benchmarks/run.py` and
   the load average. If `~/.scaly-measuring` exists, another study owns the machine. Stop and report.
4. Run `uv run benchmarks/run.py smoke`, then check the toolchain with one repetition of the
   smallest size, written outside the study directory:
   `uv run benchmarks/run.py sweep --workloads <problem> --sizes <smallest> --repetitions 1 --out benchmarks/results/<study-name>-check.csv`.

## Run as a persistent job

A study outlives the thread that starts it. Start it in a named tmux session, never as a child of
the agent's shell:

```bash
echo "<study-name> tmux:study-<study-name> eta:<time>" > ~/.scaly-measuring
tmux new-session -d -s study-<study-name> \
  '<environment> uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name> <options> \
   2>&1 | tee benchmarks/results/<study-name>.log; rm -f ~/.scaly-measuring'
```

The log sits beside the output directory because `study` refuses a directory that already has
content. Post the session name, log path, output directory and expected end on the issue, so any
thread on la015 can check progress with `tmux has-session` and the log. While it runs, start no
tests, builds or other measurements on la015. To follow it up, ask the maintainer or schedule a
check through T3 rather than holding the thread open.

If the study fails early, stop it, report the failure and the log lines, and propose the smallest
rerun. Do not restart the whole grid on your own.

## Read the results

1. Run `uv run benchmarks/run.py report benchmarks/results/<study-name>` and read `report.md`.
2. Confirm every timed cell passed its correctness gate. Count timeouts, failures and cells skipped
   after a failure, and compare them with the previous study in
   [the comparison history](../../../internal/notes/benchmark_comparison_history.md).
3. Explain every number you act on. A large coefficient of variation is a measurement problem to
   rerun, not a result. Say which conclusions were measured and which are inferred.
4. Keep any analysis beyond `report` as a tracked script in `internal/notes/perf_<date>/` with a
   short README, as the earlier investigations do.

## Publish

Update the published pages only from one complete study. Never mix machines, numerical policies
or studies in one table. Copy the study's summary files to `docs/assets/benchmarks/` as the
protocol describes, prefer a figure to a table, and write the page text with the `write-docs`
skill. Full tables and cross-study comparisons go to the comparison history. A change of protocol
or of the reference machine's software goes to the protocol note. Python-level timings are smoke
evidence and never a result.

If a study exposes a compiler path that only a benchmark exercises, copy a small reproduction into
`tests/` before changing the benchmark, as `AGENTS.md` requires.

Report the revision, commands, study directory, gate and failure counts, the headline changes with
their dispersion, and the pages changed.
