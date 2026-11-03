# Automatic release reuse for issue #249

The maintainer approved replacing decision 15's unconditional release build with
reuse of completed validation and the exact files it tested. The inputs are
`.github/workflows/release.yml`, `nightly.yml`, `ci-build.yml`, the current release
record, contributor guide, and the GitHub workflow-run and artifact APIs.

- [x] Inspect the existing workflows, issue, PR, Project and environment rules.
- [x] Revise decision 15 and issue #249 with the approved reuse criteria.
- [x] Write failing tests for release selection, complete validation records,
  commit/branch matching, expired or missing files, and skipped nightlies.
- [x] Add the selection/record script and the conditional workflow paths.
- [x] Update release documentation and test the helper's CLI against mocked GitHub.
- [x] Prove perturbations fail and run actionlint and the full pre-merge hook.
- [ ] Push the updated draft PR, exercise the nightly record on Actions, and review
  only this new change with one independent reviewer and one focused fix check.
- [ ] Remove this plan, mark ready, update the Project and watch final CI.

The selection script and workflow wiring are sequential. Documentation follows the
implemented behavior. Independent review follows passing checks. No compiler or
snapshot changes, manual publication gates, or deployment workflow runs are in scope.
The original nightly/source-build/cache handoff evidence remains on PR #253.

The first 37 release tests pass. Live read-only GitHub calls exercised selection
and recorded the six artifacts of the earlier manual nightly. The new TestPyPI
record also requires matching file hashes, so skip-existing cannot bless different
files with the same version. Branch and expiry perturbations fail their tests.

The full pre-merge hook passed with 2,809 tests passing and 50 skipped, plus
benchmark smoke, docs, types, Ruff and private-name checks. Actionlint passes.
The published-file hash perturbation also fails the intended negative cases.
