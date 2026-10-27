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

Validation so far: 40 focused tests pass; type/lint/format, benchmark smoke and docs build pass. Full pre-merge at four workers has 2740 passed, 107 skipped and only the maintainer-noted hard-coded cc failure. A 20-worker attempt also exhausted Zig thread resources; four workers eliminates those failures. Fable's one review found a missing repaired sibling in wheel selection; fixed and approved by its focused delta check. Small review notes were addressed, with full diagnostic text retained deliberately as durable effective-flag evidence. No more review rounds.

Coordinator clarified that hand-reassembled aggregates cannot count as the Windows demonstration. The failed serial builds under build/native-bundle do not count. The recipe now locks only ar/ranlib writers, without changing compilation parallelism or OpenBLAS NUM_CORES. A controlled native ar reproduction produced 10 failed writers and only 4/48 members without locking; the real wrapper produced 48/48 and no failures. The regression starts six subprocess writers on one lock. A fresh documented producer invocation runs at ordinary 20-way compilation under build/native-locked, using a fresh copy of #218's retained OpenBLAS tree as its incremental input. There is no native hand repair in this work directory. The first validation-only attempt exposed MSYS brace expansion in HEAD^{commit}; HEAD^0 peels tags without that expansion and is verified with MSYS git. Original #218 files were only read. All failed attempts and diagnostics remain under build/.
