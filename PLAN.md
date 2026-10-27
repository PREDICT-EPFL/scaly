# Issue 228 native bundle builder

Base: t3/ipopt-windows-build-196, directly above PR #218. Scope and decisions are in issue #228 and its fixed-design comment. Read the hook loader/escaping/renaming comments, plugin config/tests, architecture, codebase, conventions, contributing guide and vendored solver notes.

- [x] Inspect issue, base and checkout. Project access lacks read:project.
- [x] Add failing archive-contract and override tests for five layouts. Corrupt-file mutation is detected.
- [x] Share the host-native recipe through a producer command in hatch_build.py; retain work and diagnostics.
- [x] Define repaired native payload, manifest, recipe identity, safe extraction and content-keyed install.
- [ ] Build native Windows archive early; reinstall editable, build wheel, run solver-required and isolated wheel/Zig probes.
- [x] Inventory five control wheels independently under internal/notes/.
- [x] Document producer/local override and contracts. Keep ordinary source wheel jobs unchanged.
- [ ] Run pre-merge checks, one cross-family review and focused fix check.
- [ ] Commit, push and open PR against t3/ipopt-windows-build-196, then report evidence and downstream contracts.

Producer implementation and tests are sequential. Control inventory can run independently and owns only its evidence note. #229 owns producer workflows; #230 owns default consumption, pins and wheel-job setup.

Validation so far: 31 archive tests and seven original hook tests pass; type/lint/format, benchmark smoke and docs build pass. Full pre-merge at four workers has 2740 passed, 107 skipped and only the maintainer-noted hard-coded cc failure. A 20-worker attempt also exhausted Zig thread resources; four workers eliminates those failures. Windows producer is rebuilding the copied retained OpenBLAS tree under build/native-bundle, with MAKE_NB_JOBS=1 and PYTHON_CPU_COUNT=1 after parallel ar writes corrupted its temporary aggregate archive. These affect build scheduling, not compiled OpenBLAS NUM_CORES=20 or the thread policy. Original #218 files were only read. Native source logs and all failed attempts are retained under build/.
