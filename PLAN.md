# Typing improvements

Implement the maintainer-approved follow-ups to the Ruff/ty audit. Existing upgrade changes belong to this task. Keep larger GitHub dtype/coercion/template issues out of scope. Commit, push, and open a pull request for the main improvements before starting the separately requested array-like input/output typing change, which must remain uncommitted.

Context: `internal/notes/typing_audit_2026_10_21.md`, `pyproject.toml`, the architecture/codebase/conventions/contributing guides, `internal/notes/compiler_maintenance_context.md`, and the `implement` and `cross-review` skills.

## Main changes

- [x] Confirm this is a linked worktree, read the skills, and state completion criteria.
- [x] Add `py.typed` and verify the built wheel and a consumer outside the source import path. The marker test failed before adding the file; installed-wheel calls type-check, and removing the expected-error directive rejects a mixed call.
- [x] Correct dynamic `group(*parts)` inference while preserving exact fixed groups and the runtime width limit. Added positive and negative static checks first.
- [x] Enable benchmark checking with a benchmark-only CasADi import replacement. Fixed all 15 diagnostics, typed timing records and recorder helpers, and proved ordinary benchmark mistakes still fail the check.
- [x] Improve known statistics, compiled-handle, and derivative-cache types and the playground skeleton code.
- [x] Annotate test fixtures and preserve exact function-builder types. Targeted tests and invalid-call/callback probes pass.
- [x] Integrate changes, add/check static assertions, and regenerate the test collection baseline. There are 1,571 collected tests.
- [x] Run `wt hook pre-merge --yes`, verify C snapshots unchanged, and run supported-version type checks. The hook command was attempted but `wt` is absent. All configured commands passed directly, including benchmark smoke. The full suite passed with 1,558 tests and 13 skipped. C snapshots are unchanged; Python 3.12 and 3.13 type targets pass.
- [ ] Commit and push the main changes, open/link a draft pull request, run `cross-review`, fix findings, and obtain its focused delta check.
- [ ] Remove this temporary plan before marking the pull request ready. Record validation and review outcome, then watch the pull request through T3.

The delegated edits run in parallel with disjoint ownership; integration, checks, review, and publication are sequential. No agent commits or changes the tracker independently.

## Separate final change

- [ ] After the main branch is pushed, separate array-like input types from precise NumPy output types without taking on the deferred coercion policy.
- [ ] Verify this final change and leave it uncommitted for the maintainer. Report any design blocker instead of silently changing the agreed API.
