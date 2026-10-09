# Issue #18: move call and map derivative rules

Base: `t3code/issue-17-ad-helpers`, above #160 and #162. The decided scope is
`internal/notes/core_compiler_roadmap.md`, "One AD engine", #18. Architecture,
`docs/dev/codebase.md`, conventions, contributing, compiler maintenance context,
and the existing forward/reverse, helper, sparse and differential tests informed this plan.

- [x] Inspect issue, prerequisites, checkout, design and callers; claim issue.
- [x] Add a failing ownership check; keep #15 numerical checks unchanged.
- [x] Move call/map rules and helper construction into `ad/calls.py`, passing existing
  traversals as functions to preserve acyclic imports. Update forced callers and package map.
- [ ] Run focused tests, byte-identical C snapshot checks, unchanged differential harness,
  collection baseline, and `wt hook pre-merge --yes`; inspect color-moved diff.
- [ ] Commit, push, open draft PR against #162, link stack, run cross-review and any delta check.
- [ ] Record evidence, remove temporary plan, mark ready, update tracker and watch PR.

Extraction and caller edits are sequential. Independent validation can run in parallel.
No AD logic, traversal, seed-axis, elementwise rules or `_jvp_many_*` deletion is authorized.
