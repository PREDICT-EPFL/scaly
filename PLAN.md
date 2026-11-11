# Keep masks through sampled sparse-product adjoints

Issue #420, section 2 of `internal/notes/compiler_run_2026_11_10/plan.md`.
Base is origin/main after #369. Run-wide rules are in `~/.scaly-suite/RULES.md`.
The roadmap's decision 1, masked partials under #30, and sparse items #46 and #47
fix the design. Masks through called Functions belong to #339.

- [x] Read the run plan, rules, issue, prerequisite, architecture and compiler context.
- [ ] Add failing regression gates for scalar and coordinate masks on either operand,
  singular partials, infinite coefficients, cancellation, negative coefficients and
  both derivative orders, with NumPy finite differences.
- [ ] Preserve mask activity independently of coefficients and guard individual
  inactive products, keeping sampled patterns and bounded workspace.
- [ ] Drop the mask deliberately and record failing gates. Restore the fix.
- [ ] Run focused checks, unmasked C comparisons and the full pre-merge hook through
  the detached suite runner. Preserve all snapshot bytes.
- [ ] Push once for review, open and link the PR, obtain Opus cross-review, fix findings
  and perform a delta check if needed. Remove this temporary plan before ready.
- [ ] Update tracker evidence, queue once with squash auto-merge, and watch checks.

Independent reads and checks may run in parallel. Implementation, perturbation,
full validation and review must run in that order. No runtime performance claims.
