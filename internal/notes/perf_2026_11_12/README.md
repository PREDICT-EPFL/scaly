# Compiler-run comparison

Compare the pushed compiler-run start, `e7f80e1f`, with the pushed end, `5335c46d`, on la015.
The plan is in the study thread. All results remain local.

`prepare.py` archives the end revision's benchmark package and adds boundary adapters for the
old and new sparse results, mapped expressions, external libraries and solver compilation.
It also copies the current worktree's native solver libraries and records every frozen file's
SHA-256. Explicit library paths keep both revisions on these same binaries, since a new checkout's
cache lookup can otherwise select an older release build. Models, samples, reference calculations, compilation
flags and timed calls stay identical. Each sweep process receives an empty compilation cache.

`run.py` uses one detached worktree, switching it between the pushed revisions. It checks that
the worktree is clean, the benchmark files are unchanged and the native solver libraries match.
The study takes the machine's measurement lock, runs all nine workloads at three sizes with
three Scaly builds and one CasADi control, then all 15 canonical closed loops per revision.
Finally it retimes matching binaries in alternating order, four times per compiled binary.
Commands, elapsed times and failures remain in the manifest and logs. It stops on a failed
command. An expected compile timeout remains a recorded result.

From the study worktree:

```sh
uv run internal/notes/perf_2026_11_12/prepare.py . benchmarks/results/post-run-2026-11-12/frozen
git worktree add --detach benchmarks/results/post-run-2026-11-12/source e7f80e1f
uv run internal/notes/perf_2026_11_12/run.py benchmarks/results/post-run-2026-11-12 benchmarks/results/post-run-2026-11-12/source --preflight
uv run internal/notes/perf_2026_11_12/run.py benchmarks/results/post-run-2026-11-12 benchmarks/results/post-run-2026-11-12/source
```

Run the last command in tmux with its output saved to the sibling study log. Do no other builds,
tests or measurements on la015 while the study owns the lock. Remove the detached worktree after
analysis. Never rerun into these directories, because they retain failed attempts too.

The controls check measurement stability. They do not establish the fastest CasADi encoding.
Three selected sizes do not establish scaling between them. Three fresh Scaly builds estimate
between-build spread; repeated timing of a binary separately checks scheduling variation.
Closed-loop step dispersion is variation along one trajectory, not a confidence interval.

The maintainer also requested local documentation edits showing before/after plots. Prepare
those from the completed comparison, without replacing the older provider-ranking data with
this reduced control grid. No push, publication, tracker change or compiler fix is authorized.

## Audit and local figures

The main study completed at 18:11 UTC, with 105 timed attempts and three CasADi compilation
timeouts per revision. All 81 Scaly attempts, all 15 canonical episodes and all 6,938 solver
steps per revision completed. No correctness or execution gate failed.

`analyze.py` checks pushed clean revisions, CPU policy, frozen-file and native-library hashes,
correctness markers and input bytes, then writes `comparison.csv`, `closed-loop-comparison.csv`,
`artifact-audit.json` and `audit.json`. The chain Jacobian's unrolled Scaly reference differs by
roundoff between revisions; it is checked to 1e-12 rather than claimed to be independent.
All 24 completed CasADi kernel objects match byte for byte.

`recheck.py` repeats saved binaries with large variation, and every unchanged CasADi control.
It ran in tmux from 18:30 to 18:34 UTC with the same CPU policy and no rebuilding. Its additional
288 calls supplement the original 840; the original data stays intact. The combined control
ratios span about 1.6% on either side of one. The main M=9 baseline remains noisy, with identical
source and object bytes across all three builds; the reported interval retains that uncertainty.

Runtime ratios use geometric means of each build's four or eight alternating timings, with a
paired log t interval across three build groups. A change is marked beyond noise only when that
entire 95% interval also clears the observed range of control ratios. C compilation uses a Welch
log interval over three fresh builds per revision. Source sizes and operation counts are exact.
These intervals are cellwise, with no correction for comparing many cells. Operation counts are
the dominant mapped family's per-iteration work, not a total count for the whole kernel.

`publish.py` writes separate local docs assets and an interactive `comparison.html` in the results
directory. It preserves the older provider-ranking plots. Run the analysis and asset generation:

```sh
uv run internal/notes/perf_2026_11_12/analyze.py benchmarks/results/post-run-2026-11-12
uv run internal/notes/perf_2026_11_12/publish.py benchmarks/results/post-run-2026-11-12 .
```

The compact report is `benchmarks/results/post-run-2026-11-12/report.md`. Native timings and static
metrics establish the endpoint changes. Review leads in the report are inferences; this study
does not assign a regression to a single commit or a section to revert. In particular, the
reported 58,541 M=9 operations compare with 270,299 under the baseline's automatic lowering, not
the earlier 46,700 scalar floor named by #427. M=5's dominant-family count rises from 17,315 to
21,679, with a change from 39 to 40 trips. The separate lowering-fusion study explains only about
0.9% of its runtime change, so that change alone does not explain the measured 71% regression.
