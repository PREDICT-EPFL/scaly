# Refresh #166 on current main

The coordinator authorized rebasing onto updated #18, snapshot regeneration, four-kernel native measurements against current main, and measured fixes from the prior diagnosis. Keep the decided one-traversal AD design and all gates. Main now has bit-packed seed selectors and compact lane helpers. The roadmap, benchmark protocol, prior PR diagnosis and main's seed-selector study guide this work.

- [x] Fetch and rebase onto updated #18, preserving upstream #188 marker and taking main's conflicted snapshot.
- [ ] Regenerate and review snapshots, run the required hook, push with lease.
- [ ] In isolated clean checkouts, measure chain_jac M33, npmpc_jac N100, chain M3, unbumpercars 2 against main, one fresh native process per cell under the revised protocol, repeating uncertain comparisons. Before every timing invocation, check for competing pytest/hook/benchmark jobs and wait.
- [ ] If clear regressions remain, reproduce the diagnosed lowering/scheduling causes in small NumPy tests, implement focused fixes, perturb the checks, and retain only measured improvements.
- [ ] Run checks, one short Claude delta review restricted to fix commits, final hook and push.
- [ ] Refresh PR evidence, snapshot explanation, timing table, chain-check/workspace open points, and verify CI.

Measurements and analysis go under internal/notes/perf_2026_10_21_forward/. Keep every failed or repeated attempt. Do not run tests/builds alongside native timings. Delete this temporary plan in the final commit.
