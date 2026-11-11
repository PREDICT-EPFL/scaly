# Integrate externs after accumulators and predicates

The coordinator set merge order #315 (#45), then #320 (#30), then #355 (#83).
Do not push until the coordinator explicitly says #30 has closed. Auto-merge was reported by
the coordinator as enabled, but the last REST read returned auto_merge=null; verify before handoff.

The task branch is rebased locally onto current origin/main. The old published branch is saved
as scaly-83-before-integration. The only conflicts against main were the sparsity traversal
context and AOT imports. Both sides are retained. Focused tests: 205 passed with -n 2;
Ruff format/lint, ty and diff whitespace checks passed. C snapshots are unchanged against main.

- [x] Rebase onto current main and check the extern/call/AD/cache/snapshot paths.
- [x] Preview the accumulators branch locally and run focused tests with -n 2.
- [x] Preview the predicates branch locally and run focused tests with -n 2.
- [x] Return to the prepared main-based task branch and record any integration work still needed.
- [ ] When the coordinator confirms #30 closed, rebase onto final main, preserving both changes.
- [ ] If resolution changes behavior beyond mechanical conflict fixes, one Opus delta check of
      the resolution only; settle its delegate after using the result.
- [ ] Run the final full hook only through ~/.scaly-suite/run, detached; remove PLAN.md and push
      the final state once. Link #355 immediately after the push. Do not requeue speculatively.

Full suites remain forbidden before the coordinator's word. All GitHub reads use REST.
The two earlier review delegates have already been settled. Namespace follow-up is #357.

Local previews are saved separately as scaly-83-preview-45 and scaly-83-preview-30.
The accumulator preview passed 275 focused tests, including scheduling/sparse plans.
The predicate preview passed 995 focused tests, including select/cast and differential AD.
Both passed Ruff format/lint, ty and whitespace checks. Snapshot files match each preview base.
Each preview used that PR's published head, which is older than current main. They are separate
previews, not a replacement for the final main-based integration once both PRs merge.

All resolution so far is mechanical: preserve the base's sparsity traversal context, preserve
EXTERN's explicit exclusion from body traversal, combine AOT imports, and union the exhaustive
op classifications with the new predicate cases. No extern behavior or tests were changed.
No extra review was needed for these mechanical resolutions. The task branch is restored to its
main-based prepared state. Nothing has been pushed and no full suite has run in this phase.

## JIT index correctness follow-up

The coordinator added the cross-process native-input criterion to #83 after #321 merged.
The branch is now rebased onto main including #321. New rebase resolutions combine module-map
and import-layer entries, and retain external objects/Bsymbolic under the factored compile
command. External tree declarations are encoded by the digest, and index artifacts retain their
namespace plus extern printing metadata during verification.

- [x] Prove include-header edits and linked-binary swaps fail after an unchanged warm hit, in both
      index and verify modes. Four cases failed before the native fingerprints entered the key.
- [x] Reuse prepare_library's keys for both index and rendered keys. Walk graph libraries without
      lowering; hash readable include trees in that shared preparation, using the same native-file
      traversal as the solver fingerprint. Keep linked identities/content/version hashing there.
- [x] Focused index/digest/extern/import/coverage/snapshot tests: 195 passed with -n 2.
- [x] Finish focused checks after the final test assertions, update the collection baseline and
      the section note, commit locally, and wait for the coordinator's #30 signal.
- [ ] One Opus delta check must cover this behavior and the final rebase resolution before push.
      Settle the delegate after using its result. No push or full suite before the signal.

Final focused run after the index fix: 272 passed, 1 skipped with -n 2. The cases cover both
index/verify modes, included and extra readable headers, linked-binary replacements, index
namespace metadata, normal JIT/solver fingerprints, graph digests and C snapshots. Unchanged
external index hits are pinned to zero lowering calls, while every edited input lowers again.
The full collection baseline contains 3665 tests. Ruff, ty and diff whitespace checks pass.
The first index test run immediately after editing source was conservatively unkeyed by the
source-digest clock guard; the subsequent fresh run passes without overriding that guard.

The requested Opus delta is deferred to the final integrated state, after the coordinator's #30
signal, so it covers this new behavior and all final rebase resolutions once. No full suite or
push has been performed during this preparation. The published PR head remains unchanged.
