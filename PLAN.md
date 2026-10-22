# Windows core, cache and PIQP wheel (#152, #153, #154, #155)

Branch `codex/windows-zig-sqp`, PR #187, rebased on main after #158 (zig-bin in the JIT). The
maintainer asked to close #152 to #155 here and leave CI for #156. Follow-ups: an issue for IPOPT on
Windows and one for the Windows documentation.

Context: `internal/notes/windows_toolchain_options_2026_10_09.md` (decisions),
`internal/notes/windows_zig_sqp_2026_10_17/` (the SQP closed loops), issue bodies of #152 to #156.

## Steps

- [x] Rebase on main, resolve `find_c_compiler` against zig-bin discovery
- [x] Build PIQP with the zig-bin build requirement on Windows; path-free compiler identity
- [x] #152: standalone probe script and evidence note (cold and warm zig cache, imports, numerics)
- [x] #153: tests compile generated C with the JIT's compiler, so Windows runs them
  (`cc`/`cxx` fixtures in the root conftest, a zig entry in the gcc/clang matrices);
  a cache path with spaces; record the remaining Windows skips
- [x] #154: one `_publish` helper; invalidation removes what it can on every platform;
  tests for two cold processes, reuse by a second process, an old handle after recompilation,
  publication onto a loaded DLL
- [x] #155: build the wheel, install it in a clean venv outside the checkout, direct PIQP solve,
  generated PIQP and SQP solves with status and statistics; script and results in the note
- [x] New evidence note for #152 to #155; move the cache section out of the Oct 17 note
- [ ] Node-ID baseline, `wt hook pre-merge --yes`, cross-review, PR description, ready
- [x] File the IPOPT-on-Windows and Windows-docs issues; comment on #156 about the docs criterion

Independent: #152 probe and #155 wheel check. #154 tests follow the #153 fixture changes because
both edit `tests/codegen/test_jit.py`.
