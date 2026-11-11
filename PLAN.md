# Select masks in mapped adjoints and forward partials

Automatic differentiation (AD) work for #364 and folded-in #414, under compiler run
section 4 (#24). PR #434 is draft and already linked/watched. Base is main; the
conflict-driven rebase includes #408, #361 and #405. No push since the original draft.

## Scope and rule

The coordinator narrowed scope on Nov 11. #432, #420 and #435 remain open and
unstarted. Forward call-helper changes were removed locally before pushing.
#364 returns select conditions from reverse mapped input adjoints. #414 preserves
selected tangent conditions through downstream partials within a body. Do not change
unmasked singular products, #65 conventions or LOOP masks. Activity comes from select
conditions, never numerical tangent or cotangent values. C without selects stays identical.

The 34-line rule and boundary inventory is in PR #434 and the identical comment on #24:
https://github.com/PREDICT-EPFL/scaly/issues/24#issuecomment-6794728245
The broader zero-coefficient rule is a maintainer decision. Dense reverse contractions,
sampled sparse adjoints and forward call helpers remain documented open boundaries.

## Remaining work

- [x] Reproduce both issues and all in-scope correctness findings; keep full nested conditions,
      simplify mapped adjoints, preserve sparse movement and stable masked division.
- [x] Integrate #65 after #408 merged and pass both required maximum regressions.
- [x] Remove the forward-call helper extension and its tests, restore its documented limit,
      and leave #432 open. Record the rule and every known open boundary as requested.
- [x] Run the final full pre-merge hook detached through ~/.scaly-suite/run on this scoped state.
- [ ] Get the main review from claude-fable-5-2 at high effort, per the updated RULES.md Models
      section. No Opus review ran; its obsolete quota-failed task was cancelled and settled.
- [ ] Confirm Sol's correctness findings in one fresh low-effort delta task against the rebased
      equivalent of its reviewed head, covering only fixes and the explicit scope revision.
- [ ] Fix review findings; confirm only the main review's fixes with a low-effort delta if needed.
- [ ] Delete this temporary plan, push combined review fixes/rebase, rewrite PR evidence and
      verdict, mark ready, update issue status and queue gh pr merge 434 --squash --auto.
- [ ] Watch until merge, report an ejection immediately, and settle completed review threads.

## Evidence

- Original pushed state: full hook exit 0, 5,310 passed and 56 skipped; all checks passed.
  Log /tmp/scaly-364-suite-pBCJnx/recheck.log. This is not final validation.
- Scoped state: 109 select/snapshot/coverage checks pass, including all 99 select cases.
  Log /tmp/scaly-364-scoped-focused.log. Ruff and ty pass.
- Scoped full hook: exit 0, 6,164 passed and 56 skipped; type checking, both Ruff checks,
  benchmark smoke, documentation build and private-name check pass.
  Log /tmp/scaly-364-scoped-suite-u9zsiv/final.log.
- C/header comparisons against current main for inline, called and mapped gradients without
  selects match byte for byte. Proof path is in /tmp/scaly-364-main-proof.path.
- All in-scope reviewer failures were reproduced before fixing. A fresh numerical-nonzero
  activity perturbation failed both the mapped zero-cotangent Hessian and the inline
  second-forward zero-tangent check. Restoring conditions passed both. Logs:
  /tmp/scaly-364-scoped-perturb.log and /tmp/scaly-364-scoped-restored.log.
- Sol's full review returned changes needed; the thread was settled. All in-scope findings are
  fixed locally. Its demand to include #432 is superseded by the coordinator's scope revision.
