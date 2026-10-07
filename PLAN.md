# Issue 79: vendor PIQP 0.6.4

The issue body is the decided design. Base main; first layer of the #79/#80/#112 stack.
Scope is the PIQP pin, build-cache invalidation, generated notices, and the dual-recovery regression.
Option handling, README changes and other devrush work stay out of scope.

Sources read: issue #79 and its lane comment; architecture, codebase, conventions, contributing,
issue tracking; current hatch_build.py and notices test; devrush's S-207 vendoring change and
plugins/scaly-piqp/tests/test_piqp_dual_recovery.py.

- [x] Port the malloc regression and prove failure on 0.6.2, independently check its geometric reference.
- [x] Bump the pin, invalidate stale builds from notices, isolate sources by tag, rebuild and verify notices.
- [x] Run plugin tests and pre-merge hook including full suite; refresh collection baseline.
- [x] Commit, push draft PR in gh stack, obtain one cross-provider review and check any fixes.
- [ ] Remove this temporary plan, finalize criterion evidence, mark ready, update tracker and watch PR.

The regression proof precedes the bump. Validation precedes review; fixes and final PR follow review.

Old-library direct C probe: fill 25; dense and sparse status -1, NaN coordinates.
Environment setup initially blocked Python tests while IPOPT builds; direct probe uses the exact ported C source.

Validation so far: old pytest regression fails with status -1 and NaNs; stale notices fail.
0.6.4 wheel built, plugin tests 22 passed; independent SLSQP reference agrees.
Build cache rejects each stale dependency version and missing headers.
Full suite 1290 passed, 3 skipped; pre-merge smoke and docs still running.

All seven pre-merge hooks passed. Wheel contents checked. Draft PR #122 created through gh stack and linked in T3.
Independent review is running with Claude Fable 5.1, medium effort.
Task ID: node:delegated-task:command%3Amcp%3A23957f7c-337e-411a-af02-6838aee82271%3Adelegate-task%3Ascaly-79-review-1

Review verdict: approve. Minor internal license note corrected to include verified 0.6.4 result while retaining the historical 0.6.2 result.
Optional public documentation wording deferred to keep the published docs out of the bump scope.
PR will mention obsolete untagged build trees and the allocator-specific regression limit.
Delta check pending for the one-line internal note correction.
