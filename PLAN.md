# Select masks: final review fixes and merge

PR #434 covers #364 and #414. The maintainer's latest instruction also includes
forward call boundaries, so #432 is now in this PR rather than a follow-up.
The Nov 12 scope update supersedes the earlier deferral of #432 on #24.
Reverse dense-product transport and structural-zero semantics from #435 remain
separate. COND, LOOP and custom-rule protocols remain outside this change.

## Latest authorization

The maintainer lifted the push hold after Fable reviewed the forward-call extension
without correctness findings. Fix the two small points, run the full hook, delete
this plan in the last commit, force-push with lease, publish the prepared description,
mark ready and queue once. Keep #435 open. Watch for failed checks or an ejection.
No further review is requested.

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
- Rebased cleanly onto main's CI-cache fix. The detached full hook passed on this state:
  6,407 passed, 56 skipped, exit 0. Type, format, lint, snapshots, benchmark smoke,
  documentation build and private-name checks passed.
  /tmp/scaly-434-forward-suite-sMDjth/final.log and final.exit.
- The local PR-body draft and semantic note contain identical rule text. Compared with
  the section 4 decision on #24, only the two forward-call rows change under the Nov 12
  instruction. Every other boundary row and the rule paragraph stay unchanged.
  /tmp/scaly-434-forward-semantic-check.log. Neither draft has been published.
- Prior Sol/Fable full reviews and deltas approved the previous #364/#414 scope only.
  Their threads are settled. The maintainer reports Fable high reviewed the forward-call
  extension with no correctness findings. Its two small fixes are applied below.

## Final review fixes

- Only masked tangent formals get boolean inputs. The cache key records that tuple;
  baked and unmasked unread formals get no dead flags.
- Restored the original two-line stored-result reconstruction; the added mask movement
  could never receive a masked value.
- The requested helper-input-name regression failed with extra flags before the fix,
  and all 89 select tests pass afterward. Ruff and ty pass. One node ID added.

## Finish

- [x] Apply both Fable points and the single signature regression.
- [ ] Full hook detached through ~/.scaly-suite/run; keep the turn alive to its exit file.
- [ ] Confirm the fourteen C/header comparisons and the prepared semantic text.
- [ ] Delete this plan in the last commit, then force-push with lease and update the PR body.
- [ ] Mark ready, queue once and watch. Fix only an ejection or failed check.
