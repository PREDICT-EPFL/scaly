# Reduce pull-request CI

The maintainer's brief is the task; there is no tracker issue. The branch starts at
current origin/main. Leave the pull request open for a manual merge.

Read `.github/workflows/{ci,ci-platform,ci-test,nightly,release}.yml`, the build and
test actions, `.github/scripts/{nightly,release}.py`, pytest configuration and baseline
hooks, contributor guidance, and the CI/native-build notes. Full local suites on la015
must use the detached `~/.scaly-suite/run` wrapper.

- [x] Confirm the worktree and base, inspect workflow, cache and release dependencies.
- [x] Reduce CI to lint, docs, Linux x86 tests, smoke and the required ci summary.
- [x] Add Platforms for main pushes and manual branch runs, full tests and verify,
  independent concurrency and a separate failure issue using shared report logic.
- [x] Add the slow marker with a configuration default and explicit full-suite selection
  for Platforms, Nightly and the node-ID baseline. Prove it with throwaway checks.
- [x] Update contributor and internal CI guidance. Propose agent guidance in the handoff.
- [x] Validate workflows, all repository checks and release/report paths locally.
- [ ] Commit, push, open and link a draft PR, dispatch Platforms and record any refusal.
- [ ] Run one cross-review, fix findings and confirm only the fixes.
- [ ] Delete this plan, mark ready and finish measuring the PR's own CI. Report results
  per platform and any proof that remains unavailable. Do not merge or queue.

CI's main push trigger and queue probe can go. Platforms retains the existing push-to-main
uv cache saving and native-library saving on all platforms. Release selects Nightly or
Release validation records and does not consume CI runs, so its artifact path is unchanged.

Do not modify test files, compiler code or repository rules. Nightly edits are limited to
full-suite selection and sharing the report. No test receives the slow marker in this PR.

Use the attached empty marker expression `-m=` for full selection. A separate empty argument
looks like the current directory to the existing baseline hook and bypasses its check.
