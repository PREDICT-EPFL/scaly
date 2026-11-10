# Pattern holes for sparse leaves

Implements PREDICT-EPFL/scaly#44 on `origin/main`, following decision 12 and
Signatures and templates / Sparse tensors in `internal/notes/core_compiler_roadmap.md`.
The compiler run plan and `~/.scaly-suite/RULES.md` override the normal implement workflow.

- [x] Read the task, prerequisites #42/#43, project instructions and compiler design.
- [x] Confirm #42 already provides concrete values-vector calls in the declared layout.
- [x] Add failing tests for per-pattern instances, repeated reuse, symbolic/numerical agreement,
      copied canonicalization, explicit zeros, errors and the concrete values path.
- [x] Add `arg(..., pattern=...)`, pattern-aware bindings, layout resolution and deterministic names.
- [x] Verify lifted declarations preserve holes; inspect frozen devrush tests for useful cases.
- [ ] Run focused tests, demonstrate a perturbation fails, then the slotted full pre-merge checks.
- [ ] Commit, push once, open and link the draft PR, and run cross-review plus any delta check.
- [ ] Apply review fixes, record evidence and hard-to-undo choices, remove this plan, finish checks,
      mark ready and queue once checks are green.

Independent reads/checks may run in parallel. Implementation, full validation, review and queuing
are sequential. Nearby PRs #328, #335 and #341 may conflict; rebase only when forced.
Do not publish sparse API documentation before #46 and #47.

Hard-to-undo choices to report: `pattern=...` spelling versus a new public sentinel/type, and
bound whole TensorTypes plus resolved leaf layouts as instance keys versus nnz or argument identity.

Focused validation: 134 tests, Ruff and ty pass. The new binding tests failed before the change;
replacing pattern-name tokens with a constant fails both reuse/name cases. Devrush explicitly
deferred pattern holes and contains no cases to port. Full collection baseline now has 3001 tests.
