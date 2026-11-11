# Select-mask implementation and final review

PR #434 closes #364 and folded-in #414 under compiler run section 4 (#24).
Scope revision: #420, #432 and #435 remain open and unstarted. Forward call-helper
changes were removed. No LOOP or nonsmooth convention changes. Never use numerical
zero tangents/cotangents as activity. Keep select-free gradient C unchanged.

## State

- The original combined fixes were pushed and green. Sol's xhigh full correctness
  review and its one low-effort delta are complete; delta approved. Both threads settled.
- Fable high main review at the pushed head returned changes needed. Its three act-on
  findings are fixed locally: packed nested-mask conjunctions on stored coordinates,
  recursive common-condition accumulation, and reverse-only documentation qualifiers.
- Cheap notes addressed: mask-presence predicates, move_mask reuse, local variable naming,
  and the zero-column example. The rule's structural-zero caveat is added. #443 tracks
  redundant boolean-mask simplification; no new semantic boundary was implemented.
- Conflict-driven rebase includes new conditional calls from main. Only forward imports
  conflicted; upstream conditional dispatch/protocol is retained. COND's open transport
  boundary is listed in the semantic note.
- The known one-ULP assertion in test_called_masks_use_stored_coordinates_for_sparse_outputs
  is owned by another thread. Leave it alone; expect a one-line rebase when its fix lands.

## Evidence

- Previous full hook: 6,164 passed, 56 skipped, all checks pass. Log
  /tmp/scaly-364-scoped-suite-u9zsiv/final.log. This predates the final Fable fixes/rebase.
- Fable fixes: all 10 new sparse-product variants failed before fixing; then all 11 new
  regressions passed, including the shared-outer-condition higher-derivative check.
  /tmp/scaly-434-fable-before.log and /tmp/scaly-434-fable-after.log.
- Focused 120 select/snapshot/coverage checks pass, including all 110 select cases.
  /tmp/scaly-434-fable-focused.log.
- Ten C/header files against current main are byte-identical after the fixes: original six
  smooth inline/called/mapped gradients plus maximum penalty and shared-condition where
  gradients. The reviewed head differed in the latter four files, reproducing finding 2.
  Proof path /tmp/scaly-434-c-proof.path.
- Condition-vs-value perturbation proof remains valid: both mapped zero-cotangent Hessian
  and inline second-forward zero-tangent checks failed with numerical masking and passed
  after restoring condition masks. /tmp/scaly-364-scoped-perturb.log and restored.log.
- The 36-line rule and open-boundary inventory are identical in #434 and comment on #24:
  https://github.com/PREDICT-EPFL/scaly/issues/24#issuecomment-6794728245

## Remaining

- [x] Run the final full hook detached via ~/.scaly-suite/run: exit 0, 6,317 passed,
      56 skipped; all lint/type/snapshot/smoke/docs/private-name checks pass.
      Running log /tmp/scaly-434-fable-suite-t53Kwr/final.log; completion in final.exit.
      The runner path is also in /tmp/scaly-434-fable-suite.path.
- [ ] The fixed state is committed. The one Fable LOW delta hit the five-hour limit before
      reviewing and is snoozed until Nov 11 17:20 UTC / 18:20 Europe/Zurich. Leave it
      pending for automatic retry; do not settle, duplicate or replace it. Its title is
      Confirm Fable select-mask fixes, launched 13:34 UTC; task ids are in the tool record.
      No third
      Sol review and no second full Fable review. Delta base is the rebased equivalent
      of the reviewed head; exclude unrelated upstream changes.
- [ ] Settle Fable's completed main thread once its result has been used; settle the delta
      after using it. The earlier quota-failed attempts are already cancelled/settled.
- [ ] If delta requires another fix, apply it, check it and expose any open point; no third round.
- [ ] Delete this temporary plan in the last commit before ready; push once with all final
      fixes/rebase, link as the next call, update final evidence/verdict, mark ready and
      set the issue statuses to In review. Queue gh pr merge 434 --squash --auto once.
- [ ] Keep the existing PR watch, handle CI/ejection/conflicts, and report when actually merged.
