# Safe JIT cache invalidation

Implements #193 under section 6 (#295) of
`internal/notes/compiler_run_2026_11_10/plan.md`. The issue body and #154 decide the design.
Base and pull request target are `origin/main` and `main`. This branch is not a stack.

Invalidation will leave artifact directories untouched and always advance the existing
process-local epoch. A later build in this process gets a fresh path. Other processes keep
their source, temporary output and published library throughout compilation and loading.
Old artifacts need manual cleanup when no process uses the cache.
After one process recompiles, a fresh process still reuses the original shared artifact.
Atomic publication ensures that artifact is complete. The cross-process reuse test pins this.

## Work

- [x] Read the run plan, issue, Windows contract, project rules and surrounding code/tests.
- [x] Add deterministic two-process coverage at linking, publication and loading. See it fail
      on the old implementation. Both processes compare numerical results with NumPy.
- [x] Retain artifact directories and advance the epoch unconditionally. Update recompilation
      tests and affected documentation. Preserve cache reuse and missing-entry invalidation.
- [x] Run focused tests, the full pre-merge hook and byte-identical snapshot checks.
- [x] Commit, push and open draft pull request #316.
- [ ] Get the cross-review and any delta check after the checks pass.
- [ ] Remove this temporary plan, mark ready, and queue once with `gh pr merge 316 --squash --auto`
      after the review verdict. Report evidence and cleanup limitations to the coordinator.

The phases depend on the preceding phase. #64 and #269 share `codegen/jit.py`; #64 must
preserve unconditional epoch advancement without deleting artifact directories. The sparse
stack also edits `function/concrete.py` and `codegen/jit.py`; rebase as necessary. No changes
to the graph digest, solver fingerprints or hash-seed determinism belong here.

## Evidence so far

- All three two-process regressions failed on the original implementation for the intended
  missing-file/directory errors. With the fix, 36 JIT tests and 8 C snapshot tests passed.
- The strengthened JIT and snapshot selection passed again, 44 tests. Disabling epoch advancement
  fails the distinct-path assertion. Restoring directory deletion makes the fresh third process
  compile once instead of reusing the original artifact.
- Ruff and ty passed. The first in-turn full run was interrupted by the provider stream closing.
  A detached 16-worker run then disappeared without writing an exit file while the machine ran
  out of memory. Its log is `/tmp/scaly-193-pre-merge-16-workers-interrupted.log`.
- A detached rerun uses `PYTEST_XDIST_AUTO_NUM_WORKERS=2`. Inspect
  `/tmp/scaly-193-pre-merge.log` and `/tmp/scaly-193-pre-merge.exit`. Poll the exit file.
  It ran in its own session, PID 3458416, launched with Python's
  `subprocess.Popen(..., start_new_session=True)` so detachment completes before launch returns.
- That rerun finished with exit 0: 2606 passed, 52 skipped. Types, Ruff formatting/lint,
  benchmark smoke, docs build and private-name check passed. C snapshots remain unchanged.
- Draft: https://github.com/PREDICT-EPFL/scaly/pull/316. Implementation and initial plan are pushed.
  Lint and docs CI passed. Opus 5.5 xhigh requested three small changes, finding no correctness
  bug: remove obsolete rendering from invalidation, clarify lifetime prose, and update the public
  Function.recompile docstring. Fixes are applied and need checks and one delta check.
  The correctness-only sol xhigh
  review approved with no findings. Its nine targeted tests passed with two workers. In-memory
  mutations confirmed deletion breaks all three process-boundary tests and fresh-process reuse,
  and disabling epoch advancement breaks forced recompilation. Windows was inspected but not
  executed locally. No reviewer edits. The orchestration capabilities were already read successfully.
  Both reviewers should read `.pytest_cache/193-review-brief.md` and the committed diff.
  Task IDs:
  - `node:delegated-task:command%3Amcp%3A89e429d7-e60c-488f-8ea0-276f04e07fb0%3Adelegate-task%3Ascaly-193-opus-review-1-20261110`
  - `node:delegated-task:command%3Amcp%3A89e429d7-e60c-488f-8ea0-276f04e07fb0%3Adelegate-task%3Ascaly-193-sol-review-1-20261110`
  Both reviews are complete. Do not start another full review. The delta checks cover only the fixes.
  No full check is running now.

## Review fixes

- Invalidation now only advances the existing epoch under the lock. No compiler lookup, render,
  fingerprint or unreachable-cache-entry removal remains. No graph-digest/index changes.
- Removed the obsolete scalar-math invalidation test half and renamed tests for the behavior they
  exercise. Before-first-call invalidation checks that it does not probe the compiler.
- Guide and both public recompile docstrings describe another cache path without promising one
  never used by an earlier process with a reused PID. Cache retention, complete libraries and
  manual cleanup are still explicit. PID reuse is accepted under that lifetime.
- The isolated-cache fixture also clears epochs, and the two-process regression uses it.
  Check the load variant twice in one interpreter to confirm independence.
- Platform proof still comes from macOS and Windows CI, which the auto-merge request waits for.

## Run-wide rules

`/home/ted/.scaly-suite/RULES.md` replaces all earlier suite and GitHub relay messages. Read it
again when told it changed. Every new full suite runs detached through:

```sh
~/.scaly-suite/run
```

The wrapper permits two full runs at once with eight workers each. With no argument it runs the
pre-merge hook. A bare full pytest run also goes through it. Run detached with logs and an exit
file under `.pytest_cache/193-checks/`, then poll the exit file inside this turn with short commands.
Python `Popen(..., start_new_session=True)` reliably detaches before the launcher returns.
Focused tests stay outside the slots with at most two workers, using `-n 2` or no xdist. Never
run `-n=auto` outside the wrapper. A full pre-merge checks only a state intended to be pushed.
Rerun killed-worker or timeout failures once before treating them as real failures.

Use GitHub REST for reads and description PATCH updates. Check PR checks at most once every
five minutes and use `watch_pull_request` for wakes. Queue once after the verdict with
`gh pr merge 316 --squash --auto`; it enters the merge queue once checks pass. A rate-limit error
is a wait until the reset shown by `gh api rate_limit`, not a failure. Do not bypass the queue.
