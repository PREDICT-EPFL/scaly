# Select masks: forward call extension under the push hold

PR #434 covers #364 and #414. The maintainer's latest instruction also includes
forward call boundaries, so #432 is now in this PR rather than a follow-up.
The Nov 12 scope update supersedes the earlier deferral of #432 on #24.
Reverse dense-product transport and structural-zero semantics from #435 remain
separate. COND, LOOP and custom-rule protocols remain outside this change.

## Standing hold

Read ~/.scaly-suite/super/flush-common.md and ~/.scaly-suite/RULES.md. Push nothing,
mark nothing ready, queue nothing, publish no PR change and start no Claude review.
The coordinator schedules reviews and authorizes the push order. Keep this plan.
The live PR still points at the older pushed head; its green checks do not cover
this local extension. Do not claim the earlier reviews cover this new protocol.

## Implemented locally

- Forward helpers accept boolean activity arguments for masked seeds and return
  boolean activity results alongside numerical tangents when needed. The existing
  helper key distinguishes the masked signature; no second AD engine was added.
- Ordinary calls and mapped generic/periodic/colored assembly reconstruct SELECT
  guards from those flags. Packing keeps boolean and numerical result buffers separate.
- Higher forward helpers reserve names to avoid collisions with existing seed inputs.
- An unused structurally-zero formal does not erase another formal's selected activity.
- Guides and where docstring describe both call directions, with the custom-rule limit.

## Evidence

- Exact original scalar input/output reproductions returned NaN before the change.
  /tmp/scaly-434-forward-boundaries-repro-before.log.
- Five representative regression cases pass: ordinary/mapped/nested, runtime/baked
  multiple seeds, both call directions plus caller seed assembly, sparse stored
  coordinates, unused formals, masked/unmasked cache reuse, Hessians and second-forward
  derivatives at a zero but changing tangent. Finite differences agree on finite domains.
  /tmp/scaly-434-forward-boundaries-after.log.
- All AD, C snapshot and operation coverage checks passed: 2,760 tests, with two workers.
  /tmp/scaly-434-forward-boundaries-ad.log. Ruff and ty pass.
- Removing the cache's mask bit fails a constructor/signature gate. Replacing condition
  activity with numerical nonzero activity loses curvature 1/8 at a zero tangent.
  /tmp/scaly-434-forward-cache-perturb.log and forward-activity-perturb.log.
  Restoring the implementation passes: /tmp/scaly-434-forward-restored.log.
- Fourteen extra C/header files match main byte for byte, covering smooth gradients,
  maximum/shared-condition gradients and ordinary/mapped forward products without selects.
  /tmp/scaly-434-forward-proof.path. No snapshots regenerated.
- Prior Sol/Fable full reviews and deltas approved the previous #364/#414 scope only.
  Their threads are settled. Coordinator review of this extension is still needed.

## Finish

- [ ] Refresh the base as required by the flush, preserve upstream work and rerun any
      checks invalidated by a rebase. Regenerate only the node-ID baseline for added tests.
- [ ] Update local PR-body draft/table for the Nov 12 scope change, keeping the rule itself
      and all other boundary decisions. Do not publish or change #24 while held.
- [ ] Full hook detached through ~/.scaly-suite/run; keep the turn alive until its exit file.
- [ ] Commit the evidence here, retain this plan and report branch/head, ahead/behind main,
      hook counts, snapshot differences and the remaining coordinator review/push hold.
- Stop after the report; do not wait, poll, watch, push, mark ready or queue.
