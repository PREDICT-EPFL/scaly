# Automatic release reuse for issue #249

The maintainer approved replacing decision 15's unconditional release build with
reuse of completed validation and the exact files it tested. The inputs are
`.github/workflows/release.yml`, `nightly.yml`, `ci-build.yml`, the current release
record, contributor guide, and the GitHub workflow-run and artifact APIs.

- [x] Inspect the existing workflows, issue, PR, Project and environment rules.
- [ ] Revise decision 15 and issue #249 with the approved reuse criteria.
- [ ] Write failing tests for release selection, complete validation records,
  commit/branch matching, expired or missing files, and skipped nightlies.
- [ ] Add the selection/record script and the conditional workflow paths.
- [ ] Update release documentation and test the helper's CLI against mocked GitHub.
- [ ] Prove perturbations fail and run actionlint and the full pre-merge hook.
- [ ] Push the updated draft PR, exercise the nightly record on Actions, and review
  only this new change with one independent reviewer and one focused fix check.
- [ ] Remove this plan, mark ready, update the Project and watch final CI.

The selection script and workflow wiring are sequential. Documentation follows the
implemented behavior. Independent review follows passing checks. No compiler or
snapshot changes, manual publication gates, or deployment workflow runs are in scope.
The original nightly/source-build/cache handoff evidence remains on PR #253.
