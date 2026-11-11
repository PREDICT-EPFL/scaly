# Integrate externs after accumulators and predicates

The coordinator set merge order #315 (#45), then #320 (#30), then #355 (#83).
Do not push until the coordinator explicitly says #30 has closed. Auto-merge was reported by
the coordinator as enabled, but the last REST read returned auto_merge=null; verify before handoff.

The task branch is rebased locally onto current origin/main. The old published branch is saved
as scaly-83-before-integration. The only conflicts against main were the sparsity traversal
context and AOT imports. Both sides are retained. Focused tests: 205 passed with -n 2;
Ruff format/lint, ty and diff whitespace checks passed. C snapshots are unchanged against main.

- [x] Rebase onto current main and check the extern/call/AD/cache/snapshot paths.
- [ ] Preview the accumulators branch locally and run focused tests with -n 2.
- [ ] Preview the predicates branch locally and run focused tests with -n 2.
- [ ] Return to the prepared main-based task branch and record any integration work still needed.
- [ ] When the coordinator confirms #30 closed, rebase onto final main, preserving both changes.
- [ ] If resolution changes behavior beyond mechanical conflict fixes, one Opus delta check of
      the resolution only; settle its delegate after using the result.
- [ ] Run the final full hook only through ~/.scaly-suite/run, detached; remove PLAN.md and push
      the final state once. Link #355 immediately after the push. Do not requeue speculatively.

Full suites remain forbidden before the coordinator's word. All GitHub reads use REST.
The two earlier review delegates have already been settled. Namespace follow-up is #357.
