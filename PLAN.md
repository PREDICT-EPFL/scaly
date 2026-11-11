# Keep masks through sampled sparse-product adjoints

Issue #420, section 2 of `internal/notes/compiler_run_2026_11_10/plan.md`.
The branch was rebased onto origin/main after #408, resolving only the reverse-mode
imports. Run-wide rules are in `~/.scaly-suite/RULES.md`.
The roadmap's decision 1, masked partials under #30, and sparse items #46 and #47
fix the design. Masks through called Functions belong to #339.

- [x] Read the run plan, rules, issue, prerequisite, architecture and compiler context.
- [x] Add failing regression gates for scalar and coordinate masks on either operand,
  singular partials, infinite coefficients, cancellation, negative coefficients and
  both derivative orders, with NumPy finite differences.
- [x] Preserve mask activity independently of coefficients and guard individual
  inactive products, keeping sampled patterns and bounded workspace.
- [x] Drop the mask deliberately and record failing gates. Restore the fix.
  All 18 masked numerical gates fail without adjoint activity propagation.
  Four unmasked value/gradient/Hessian renderings and headers match the base byte for byte.
  The nonlinear infinity regression covers seed-axis views in forward-over-reverse,
  and reverse-over-reverse too. The focused suite passes 189 tests. Collection passes
  with 5,271 tests after refreshing its required baseline.
  After rebasing, 219 focused tests, strict types, Ruff and all four unmasked C/header
  comparisons pass. Full collection passes with 6,172 tests. The old Opus attempt was
  stopped before rebasing; a new review must read the validated, rebased head.
- [ ] Run focused checks, unmasked C comparisons and the full pre-merge hook through
  the detached suite runner. Preserve all snapshot bytes.
- [ ] Push once for review, open and link the PR, obtain Opus cross-review, fix findings
  and perform a delta check if needed. Remove this temporary plan before ready.
- [ ] Update tracker evidence, queue once with squash auto-merge, and watch checks.

Independent reads and checks may run in parallel. Implementation, perturbation,
full validation and review must run in that order. No runtime performance claims.
