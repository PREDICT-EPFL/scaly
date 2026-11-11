# Issue #65: nonsmooth derivatives and extremum reductions

Branch: `t3code/65-nonsmooth-extrema`. Base: `t3code/66-axis-reductions`.
Follow `internal/notes/compiler_run_2026_11_10/plan.md` and `~/.scaly-suite/RULES.md`.
The user brief in this thread owns scope, overlap restrictions and completion criteria.

- [x] Read run plan, suite rules, implement skill and local context. Confirm la015 and clean checkout.
- [x] Confirm #401 remains open at the requested base. Locate partials table and REDUCE AD gaps.
- [ ] Finish reading roadmap rules, issue #65 discussion and section #296, relevant implementation and test harness.
- [ ] Record issue activity through REST. Inspect dependencies and overlapping pull requests through REST.
- [ ] Add failing NumPy differential and AD tests covering axes, keepdims, sparse zeros, ties, empty shapes and NaNs.
- [ ] Implement masked nonsmooth partials and extremum reduction derivatives, support and sparsity.
- [ ] Add public reduce_max, reduce_min, argmax, argmin, norm_inf and norm_1 builders. Preserve int64 index semantics.
- [ ] Make extremum lowering NaN-propagating, including elementwise extrema. Do not edit MATMUL or ir/einsum.py.
- [ ] Update public documentation through write-docs; regenerate snapshots if needed and gather operation counts, source size and C compile time versus main. Runtime measurements wait for Phase 4.
- [ ] Run focused tests with at most two workers; run merge checks detached through ~/.scaly-suite/run.
- [ ] Commit and push reviewable state, open PR and immediately link it in this thread.
- [ ] Delegate exactly one cross-review and its delta check. Post verdict in PR description.
- [ ] If #401 merged, rebase using the user-specified --onto command and retarget main. Otherwise retain stacked base.
- [ ] Queue once with gh pr merge --squash --auto after verdict and green checks; watch and repair any ejection.
- [ ] Remove this temporary plan before readiness. Report criterion evidence, verdict, follow-ups and merge state.

No implementation edits or tests have run yet. No PR exists. Existing source inspection found:
`ad/rules.py` uses x/abs(x), no floor/ceil partials, and fmin/fmax without partials;
`ad/forward.py` and `ad/reverse.py` explicitly reject max/min REDUCE derivatives.
