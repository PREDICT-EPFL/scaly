# Select masks in mapped adjoints and forward products

Automatic differentiation (AD) work for #364 and its folded-in #414, under
`internal/notes/compiler_run_2026_11_10/plan.md`, section 4 (#24). PR #434 is draft,
based on main after #358. Its original pushed state passed the full merge hook.
Review fixes are local and must be pushed together after their checks.

The scope includes forward helper inputs and outputs, also recorded in #432.
Derive activity from select conditions, never numerical tangent or cotangent values.
Keep C without selects byte-identical. Do not change #65's nonsmooth rules or LOOP masks.

## Remaining work

- [x] Reproduce both reports and the correctness review's nine failures.
- [x] Keep full nested conditions in sums and simplify mapped adjoints before extracting activity.
- [x] Preserve sparse masks through packing, movement and assembly.
- [x] Keep the stable single-division contraction for masked quotients.
- [x] Carry selected activity through matrix products using constant coefficient support.
- [x] Carry masks through ordinary and mapped forward helpers, including baked/periodic/colored
      seeds, shared helper packing, unused formals and second derivatives at a zero tangent.
- [x] Update the guide and where docstring for forward calls. #432 belongs in this PR's closing refs.
- [ ] Integrate #65 after PR #408 merges. The two maximum regressions are already added but
      currently fail because this base has no maximum derivative. Remove the obsolete forward
      maximum limitation that #408 adds to docs/guide/derivatives.md. Preserve #65's conventions.
- [ ] Run the final full pre-merge hook detached through ~/.scaly-suite/run, on the state to push.
- [ ] Get the required fresh full Opus review at xhigh, fix its findings and confirm only its delta.
      Its first attempt never reviewed due to quota, resets 12:30 Europe/Zurich on Nov 11.
- [ ] Confirm the Sol correctness findings with one fresh low-effort delta task against the
      originally reviewed PR head, covering the fixes only. Do not request a second full Sol review.
- [ ] Delete this temporary plan, push the combined fixes, update the PR's evidence and verdict
      line, mark ready, update issue status, and queue with gh pr merge 434 --squash --auto.
- [ ] Watch to merge, report an ejection immediately, and settle completed review threads.

## Evidence and review state

- First pushed state: full hook exit 0, 5,310 passed and 56 skipped; benchmark smoke, docs,
  private-name check, Ruff and ty passed. Log /tmp/scaly-364-suite-pBCJnx/recheck.log.
- Local review fixes: 2,164 AD/coverage/snapshot checks passed, excluding the two maximum cases
  that need #65. Log /tmp/scaly-364-ad-review-fixes-final.log. All 116 available select cases,
  including second-forward derivatives, pass. Ruff, ty and the documentation build pass.
- Six C/header files for inline, ordinary-call and mapped gradients without selects still match
  the base byte for byte. Proof directory /tmp/scaly-364-proof-0cqs9ig2/.
- Restoring the original AD modules failed 26 early regressions; all nine review reproductions
  were observed failing and now pass. New sparse, quotient, boundary, cache and zero-formal
  cases are in tests/ad/test_select_derivatives.py.
- Replacing condition-based activity with numerical nonzero activity made both higher-order
  regressions fail: the mapped zero-cotangent Hessian and the called zero-tangent second-forward
  derivative. Restoring condition-based activity passed both. The perturbation was in memory
  only and left no source changes.
- Sol's full correctness review returned changes needed. Its seven findings are fixed except
  maximum integration. Its thread was settled. The original Opus quota-failed attempt was
  cancelled and settled because its brief was obsolete; a new task must review the final fixes.
- GitHub reads use REST. Push rarely. PR #434 is already linked and watched in this thread.

## Follow-up

#435 is a separate reverse matrix-product bug, also reproducible without a select:
sqrt(x @ [[0, 1], [0, 1]]).sum() at x=[1, 4] has a finite derivative but NaN gradients.
Changing that unmasked contraction would violate this task's C-preservation scope. It is
labelled source:agent and documented in the derivatives guide. Forward selected-activity tests
for zero coefficients compare against finite differences; nonzero coefficients retain duality checks.
