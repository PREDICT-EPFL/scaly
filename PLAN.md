# Shape-hole numerical output types

Issue #334 follows #288, merged through PR #328. The agreed design is in
`internal/notes/core_compiler_roadmap.md`, Signatures and templates. The compiler-run
plan on main and `~/.scaly-suite/RULES.md` govern review, checks and queuing.

The branch starts at `origin/main`. PRs #348 and #335 touch nearby declarations.
Rebase only for a conflict. Runtime binding, layouts and output reconstruction stay
unchanged. Sparse names remain unpublished.

- [x] Read the issue, prerequisite, nearby changes, project rules and declaration typing.
- [x] Add static regressions for both shape-hole spellings, inferred outputs, nested
  groups, concrete calls and maps, and observe the reported failure.
- [x] Correct the hole annotation. Preserve NumPy typing for declarations and derivative
  results that guarantee dense values. Migrate affected dense consumers without ignores.
- [x] Remove the temporary roadmap departure. Run strict ty, focused tests, mutation
  checks and the detached shared-slot pre-merge hook. Keep C snapshots unchanged.
- [ ] Commit and push the reviewed state, open and link a draft PR, obtain one Opus
  review, fix findings together, and request a delta check if fixes are needed.
- [ ] Remove this temporary plan, update the PR and issue status, wait for CI, and
  queue once with `gh pr merge --squash --auto` after the verdict and green checks.

The main typing choice is the existing NumPy/SciPy union for unconstrained holes.
Inferring density from input declarations would be unsound because a body can
return a sparse constant. Adding sparsity parameters to Expr is outside this task.

Strict ty and both Ruff checks pass. The static regression fails with 23 diagnostics
when the old overload is restored, and removing one expected-error assertion fails
with the expected argument error. The focused dense recheck passes all 152 cases.
The migration adds concrete shapes to known-dense test declarations and narrows five
runtime results with isinstance. No runtime implementation changed.

The detached shared-slot pre-merge hook passed with exit 0, including 2,968 tests
passed and 52 skipped, benchmark smoke, documentation build and the private-name
check. The Python 3.14 and Windows strict ty variants pass. The additional boundary,
snapshot and import run passed all 234 cases. C snapshots remain byte-identical.
The README solver example and three function examples pass. The first full run
caught one incorrect shape added during the dense migration; correcting the solver
host output from length 1 to length 3 fixed it, with both local cases passing before
the full successful rerun.
