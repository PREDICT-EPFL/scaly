# Windows installed wheels and CI

Issue #156 is layer 2 on `t3code/issue-198-windows-checks`, with that branch as PR base.
The issue's maintainer update moves public documentation to #197 and keeps IPOPT in #196.
Inputs: architecture, codebase, conventions, contributing and tracking guides; Windows decisions
in `internal/notes/windows_toolchain_options_2026_10_09.md`; prior wheel verification in
`internal/notes/windows_native_2026_10_22/verify_wheels.py`; CI, wheels and release workflows.

- [x] Read scope, prerequisites and existing package/workflow conventions. Parent head is present.
- [x] Add acceptance checks and demonstrate the missing automatic Zig dependency.
- [x] Add required dependency, installed-wheel Windows Python 3.12–3.14 CI and release verification.
- [ ] Build wheels and verify fresh native installations, then run checks and pre-merge hook.
- [ ] Push draft PR, notify coordinator, obtain cross-provider review and fix/check findings.
- [ ] Remove this temporary plan, mark ready and hand off evidence and pending CI.

Implementation and validation are sequential. The independent review follows passing local checks.
No snapshot, benchmark portability, public documentation or Windows IPOPT changes are in scope.
Project updates are blocked by missing `read:project` access.

Fresh installed-wheel checks pass on native Python 3.12.11, 3.13.1 and 3.14.8.
Initial hook passes with 2478 passed, 147 skipped, 2 xfailed. Final installed-wheel hook is running.
Changed workflows pass actionlint. Latest parent update included without conflict.
