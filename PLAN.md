# Carried ranges, issue #29

Implements the program-dialect contract in `internal/notes/core_compiler_roadmap.md`,
*Loops, conditionals and printing*, under the unattended compiler run plan. Base is
`t3code/28-invocation-model`, PR #313. Keep its original tip outside tracked files for the squash rebase.

The brief limits this task to carried ranges, BREAK_IF, exit variables, verification,
rendering and existing pass guards. Frontend LOOP and AD, fusion policy for loop bodies,
trajectories and frame budgeting belong to #31–#33, #69 and #76.

- [x] Read the run plan, current resource rules, task, prerequisite, design and compiler context.
- [x] Add hand-built dialect, verifier, compiled differential and pass-guard tests; see them fail.
- [x] Implement local vocabulary, renderer and pass guards; prove each guard with a mutation.
- [x] Run focused checks and byte-identical C snapshot checks.
- [x] Commit the push candidate and run detached full pre-merge through `~/.scaly-suite/run`.
- [x] Push a draft PR against #28 and link it in T3.
- [ ] One Opus 5.5 high cross-review, fix findings, one short delta check.
- [ ] After #313 merges, rebase only this layer onto origin/main, retarget, rerun checks.
- [ ] Delete this plan, record evidence and verdict, mark ready and queue once with auto squash.

Full-suite execution follows `~/.scaly-suite/RULES.md`: two slots, six workers per suite,
first come first served. Focused pytest runs use at most two workers. GitHub reads use
REST. No runtime timing claims. The original full-suite wait started at epoch 1794335166;
its old runner and wrapper have been stopped. Restart with `SCALY_SUITE_SINCE=1794335166`.
The coordinator now authorizes Opus review in parallel with the full suite, after rebasing
onto the refreshed #28 branch and rerunning focused checks. Rebase conflict was only the
temporary plan, retained here for this task.

Evidence so far: the first guard run failed 12 cases, and seven added reproductions
failed in the shared producer matcher, index splitting, scatter combination and mask packing.
All 18 isolated guard removals fail their targeted tests. Logs and the original base tip are
under `/tmp/scaly-carried29-zyX7cL/`. The adjacent pass, import-boundary/layering, assembly
and snapshot run passed 252 tests; the dialect/compiled/guard run passed 114 before the
last scalar-carried hoisting regression, which passes in its 20-test focused run.
The rebase onto the refreshed #28 head passes all 373 focused tests, Ruff and strict ty.
The node-ID baseline now includes the 65 added cases; all 3,064 tests collect successfully.
The old CI failures came from that stale baseline. No roadmap departure. Project metadata is unavailable through
the attempted REST endpoint, so the task's identifying issue comment records the branch.

PR #331 is linked and watched; prerequisite PR #313 is watched too. The pre-review candidate passed
the full pre-merge hook with exit zero in `pre-merge-2.log`: 3,012 passed, 52 skipped, plus
benchmark smoke, documentation build and private-name check. Opus high found one act-on issue:
the test wrapper used pointer types differing from the generated C, and the production bool
header existed only for that wrapper. Corrected the wrapper and ctypes, removed the header,
and made incompatible-pointer warnings errors. Also reject carried VECTOR ranges and cover
both pipeline lane orderings with reciprocals enabled and disabled, using a real division.

The fixes pass 198 focused tests, strict GCC compilation, two representative Zig cases,
Ruff and strict ty. Removing the VECTOR guard fails two tests; restoring the wrong wrapper
types fails strict compilation. Both are restored and pass. All 3,138 tests collect after
refreshing the baseline. Next: push these fixes, run a fresh full hook and request the single
Opus delta check on the fixes alone. Notes about unique exit-variable names belong to #31;
conservative optimization inside exit loops and BLOCK is accepted. No further review round.
Keep the refreshed #28 base tip in `base-tip` for the later squash rebase onto main.
Remove this temporary plan before marking the PR ready.
