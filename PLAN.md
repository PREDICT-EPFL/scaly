# Shared Windows CI

The issue #202 contract and parent PR #200 preserve installed-wheel acceptance, supported Python/compiler matrices and Windows isolation. Sources: `.github/workflows/{ci,nightly,tests,windows,release}.yml`, `.github/scripts/windows_install.py`, the required project docs and the Windows decisions note.

- [x] Inspect issue, parent and workflow callers. Parent is the requested committed base. Project updates lack read:project.
- [x] Share pure artifacts and test orchestration, retaining platform setup and benchmark boundaries.
- [x] Fold release verification into its existing platform matrix.
- [x] Matrix comparison, actionlint and pre-merge hooks pass. Shared Windows Bash commands are being checked against installed wheels, after the hook's editable suite passed 2489/147skip/2xfail. Fresh Python 3.12 installation passed.
- [ ] Commit/push and open a draft against the parent. Notify coordinator for stack extension.
- [ ] Cross-provider review, fixes-only check if needed, final evidence and ready PR.

Edits precede validation; the review follows passing checks. Remote CI belongs to the coordinator. No snapshot, compiler, backend or benchmark implementation changes.
