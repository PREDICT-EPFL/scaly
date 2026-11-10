# Keep distinct dtype descriptors apart

Issue #318 requires expression interning to distinguish every dtype field that affects generated C, including two descriptors with the same name and different C types. Work is based on main and stays separate from section 4's dtype features.

The compiler-run plan and `~/.scaly-suite/RULES.md` override the normal implementation workflow. Full checks run detached through the suite runner. After independent review and green checks, queue this pull request without waiting for maintainer review. No tags, releases, deployments, or direct pushes to main.

- [x] Read the issue, compiler design, project conventions, and overlapping pull requests. No open pull request edits the dtype definitions. Pull requests #320, #312, and #313 edit neighboring expression definitions.
- [x] Add and run the regression before fixing equality. Twelve cases fail on the original implementation. Restoring name-only equality in a fresh process after the fix reproduces the same twelve failures.
- [x] Compare all dtype fields while preserving string-name comparison and its hash contract.
- [ ] Refresh the collection baseline and run focused tests, then all pre-merge checks through the detached runner. Focused type, expression, and C snapshot tests pass, 64 total. All 14 regression cases pass. Ruff and ty pass. C snapshots are unchanged.
- [ ] Open a draft pull request, obtain one correctness review, and address findings. Use a delta check if a fix is needed.
- [ ] Remove this temporary plan, mark the pull request ready, and queue it after green checks. Report criterion evidence, review findings, open points, and follow-ups.

The regression precedes the fix. Checks precede review. Any rebase preserves this scope and reruns checks appropriate to the changed base.

Validation logs are under `/tmp/scaly-318.acJ91o/`. Project field updates are waiting for the shared GraphQL quota to reset; progress is recorded through REST issue comments instead.
