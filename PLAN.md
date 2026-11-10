# Restore nightly CI for issue #279

The compiler run plan at `internal/notes/compiler_run_2026_11_10/plan.md` authorizes
queuing this branch's pull request with `gh pr merge --squash` after review and green
checks. Base and pull request target are main. Tags, releases, deployment workflows
and direct pushes to main remain forbidden.

Scope is the failures in nightly run 42782804912 at main's e7f80e1f:
`benchmarks / macos-arm64`, `python / 3.12` and `python / 3.13`.
Inspect earlier recovered Clang 22 and Windows build failures, but fix them only if
they share a cause. Compiler roadmap sections belong to other threads.

- [x] Read the run plan, implementation procedure, repository rules and live issue.
- [x] Verify the isolated worktree on la015 and check dependencies and open PRs.
- [x] Diagnose all current and recovered failures from their run logs.
- [x] Reproduce the current failures locally where la015 permits, add useful
      regression coverage, then make focused fixes and prove the checks can fail.
- [ ] Run the required merge checks and changed nightly commands locally.
- [ ] Open a draft PR, link it in T3 and trigger the branch's dispatchable nightly.
- [ ] Obtain one correctness review and address findings with a focused delta check.
- [ ] Rebase when necessary, obtain green PR and nightly checks, then queue squash merge.
- [ ] Confirm merge and report evidence, platform limits and any follow-up tasks.

Investigation, fixes and local checks are sequential. PR checks and branch nightly
can run independently after pushing. Review follows passing local checks. Delete
this temporary plan in the final commit before marking the pull request ready.

Initial evidence: both older Python jobs fail when importing `scaly_fatrop` because
`_Backend.render_wrapper` references a type imported only for type checking.
The macOS full smoke fails in the race-cars solver check with Fatrop native status 4.

The Python import was reproduced on both supported older interpreters, and adding
postponed annotations fixes both. Existing backend registration tests also pass.
The macOS failure follows `fatrop_stage_functions`, in `fatrop_matches_ipopt`.
Native status 4 means line search failed. Local round-to-positive-infinity stress
reproduces that status at the original 1e-12 tolerance. At 1e-10 all three directed
rounding modes converge and retain the original 1e-8 solution agreement. A 1e-6
solution mutation is rejected. No solver status or comparison threshold changed.

Earlier Clang 22 failure has no logs and a check annotation saying the hosted
runner lost communication. The Windows build logs show repeated MSYS2 download
HTTP 504s. Both jobs passed in run 42782804912. Neither shares the current causes.

Running: all pre-merge hooks, full required-solver suites on Python 3.12 and 3.13,
and `uv run benchmarks/run.py smoke --full`. Logs are in `/tmp/scaly-279-*`.
