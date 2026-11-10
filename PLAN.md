# Call support from argument support, #221

Implements the fixed Sparse tensors design in `internal/notes/core_compiler_roadmap.md`.
The compiler run plan and `/home/ted/.scaly-suite/RULES.md` override the usual workflow.
Base is main after #40, branch `t3code/221-call-support`. #167 stacks on this work.

- [x] Read the run plan, suite rules, tracker, architecture and support implementation.
- [x] Add failing CALL/VMAP soundness, cache-bound and self-contained chain M=9 tests.
- [x] Evaluate existing support rules under argument supports, with a bounded callee cache.
      Keep node supports reachable for #167. Reuse repeated stage patterns in maps.
- [x] Prove tests detect removed specialisation and removed bounds. Refresh collection baseline.
- [ ] Run all pre-merge checks detached through `~/.scaly-suite/run`.
      Keep generated C snapshots byte-identical unless a criterion requires a reviewed change.
- [x] Commit, push, open draft PR #333 against main and link it in T3.
- [ ] Cross-review, fix findings, check the fixes, then remove this plan and mark ready.
- [ ] Queue once after the verdict and green checks; report evidence and open points.

Nearby work: #320, #313 and #312 touch expression passes and derivative code.
Only support analysis/cache and its tests are owned here. No Jacobian call rules or lowering changes.
Use REST for GitHub reads, at most two workers for focused tests, and no runtime measurements.

Focused support, chain and C snapshot checks: 61 passed. Both deliberate mutations failed the intended tests.
Bound is 32 per callee: room for RK4 contexts and repeated mapped patterns while capping retained node-support maps.
The issue comment records ownership. A REST status update returned In progress, but subsequent reads still omit the field.
Full pre-merge check is detached at `/tmp/scaly-221/pre-merge.log`, with exit in the adjacent `.exit` file.

Opus review: changes needed. Fixed interrupted-walk cache publication and duplicated constant supports.
Added failing regressions for both and for the cached-root fast path, then confirmed they pass.
Replaced the stand-in chain seed table with the actual Hessian colouring in a self-contained fixture.
Exact RK4 argument support and every seed-axis helper value now match Jacobian images; output counts 564 and 813.
The fallback's precision remains history-dependent, as required by the fixed bounded-cache design.
The restarted full runner must use `SCALY_SUITE_SINCE=1794335160`; the old queued run was cancelled for the code fixes.
