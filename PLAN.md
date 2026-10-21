# Isolate cache-invalidating tests

The macOS main run failed while linking bicycle_eq_N3. The lowering and integration tests share its key, and lowering calls recompile, which removes the cache directory.

Sources: src/scaly/codegen/jit.py, src/scaly/function/concrete.py, tests/codegen/test_jit.py, tests/passes/lowering/test_lowering.py, tests/passes/test_program.py, conftest.py, docs/dev/{codebase,conventions,contributing,issue_tracking}.md.

- [x] Reproduce the link failure deterministically before editing.
- [x] Share isolated_cache and apply it to every test that invalidates cache entries. Keep C snapshots unchanged.
- [ ] Re-run the reproduction and required checks, including wt pre-merge.
- [ ] File the product-race follow-up, open the PR, obtain one Claude cross-review, and mark ready after resolving findings.

These phases are sequential. Only the independent required checks can run in parallel.
