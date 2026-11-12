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
