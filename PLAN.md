# Empty constant-mask loops

Issue #188, dispatched by the AD lane coordinator. Read the issue, compiler architecture,
codebase and conventions, compiler maintenance context, and the One AD engine roadmap.
The fix belongs in `src/scaly/passes/program/pack_masks.py`; coverage belongs beside
`tests/passes/test_pack_masks.py`. AD stack branches remain untouched.

- [x] Inspect issue, worktree, prerequisites and tracking metadata.
- [x] Reproduce with pass-level empty-mask cases and the issue's end-to-end example.
- [x] Skip empty loops before mask layout analysis; verify the regression fails without it.
- [ ] Run `wt hook pre-merge --yes`; stop if any existing C snapshot changes.
- [ ] Open a draft PR, get one Claude cross-review, address findings and check fixes if needed.
- [ ] Remove this temporary plan, mark ready and hand PR monitoring to the coordinator.

These phases are sequential. No parallel implementation is needed. #160 owns removal of its
69 expected-failure marks after this fix merges.
