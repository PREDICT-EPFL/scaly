# Run each CI platform in one job

The maintainer's brief fixes the design. Start from fetched `origin/main`, retain Linux x86_64,
Linux arm64, macOS arm64 and Windows x86_64, and combine each platform's build, install and test
work. Nightly and release retain their existing build-once, test-many structure and artifact and
cache contracts. The parent reviews the pull request, so skip cross-review and do not watch it.

Read `AGENTS.md`, `AGENTS.local.md`, the architecture, codebase, conventions and contributing
guides, the CI/build/test/install/nightly/release workflows, the install scripts and release tests.

- [x] Fetch and reset this clean worktree branch onto `origin/main`.
- [x] Exercise the current summary with combined-job fixtures and confirm the mismatch.
- [x] Extract build, test and install composite actions, preserving comments, suite behavior,
      shell selection, outputs, environment ordering, cache keys and artifact contracts.
- [x] Add the combined platform workflow and update CI orchestration and summary.
- [ ] Run actionlint, fixture checks and all pre-merge hooks.
- [ ] Commit, push, open and link a pull request. Trigger nightly on the branch.
- [ ] Confirm CI and nightly results and retrieve the CI summary. Fix failures, update the pull
      request evidence and remove this temporary plan before marking the pull request ready.

Extraction precedes orchestration. Local checks precede the push. The CI and nightly runs can run
independently after the pull request opens.

Smoke uploads use `dist-smoke-linux-x86_64` and `dist-smoke-pure`, since its independent build runs
alongside the platform matrix and upload-artifact does not allow duplicate names within a run.
The original artifact names and retention stay unchanged in the reusable build workflow and the
default CI suite. The summary also goes to the job log so `gh` can retrieve its exact contents.

Local actionlint and extraction checks pass. The extraction check compares the original step
bodies after the required input and shell changes, reusable workflow inputs, cache keys and
artifact settings. Summary fixtures failed on the original implementation and pass for the
combined jobs, including missing platforms and an install failure with successful tests. An
inverted built/cached result fails the fixtures. The pre-merge hook is still running.
