# Static solver interfaces and smaller runtime wrappers

The maintainer revised #112 after reviewing generated code size. PIQP `sparse`
and SQP `qp` select a compiled interface. Runtime parameters still share the
wrapper and oracle build when the compiled interface is the same. The public
construction syntax stays unchanged. Generate only the selected interface and
keep defaults and validation in the backend's Python preparation hook.

Sources: the #112 discussion, solver descriptors and builders, plugin option
preparation and wrappers, generated C interface, reuse tests and exported-call
tests. The earlier independent review and its fixes remain applicable.

- [x] Record the approved decision on #112 and return #125 to draft.
- [x] Add a failing static-interface/cache test and adjust the runtime reuse cases.
- [x] Split validated compilation choices from runtime parameters in the plugin
      contract and carry them through core builders and the external SQP adapter.
- [x] Emit one PIQP/SQP interface and remove obsolete staging, allocation and
      runtime-mode machinery. Restore the fixed PIQP oracle layouts.
- [x] Update the guides and deliberate solver snapshots. Measure source and
      executable-code size, generation time and native solve time.
- [x] Run focused checks, the whole suite and pre-merge checks, and verify the
      numerical references and full harvested globalization gate.
- [ ] Ask Fable for the requested generated-code simplification review, apply
      justified findings, run checks and one delta check if needed.
- [ ] Remove this plan, update the PR evidence, push the stack and monitor CI.

Implementation is sequential. Review follows passing checks. No parallel
implementation agents are needed. Interface specialization does not authorize
changing the selected solver libraries or the extern migration.

Current verification: pre-merge passes with 1316 tests and no skips. Nine complete
numerical records exactly match main, the three affected full benchmark gates
pass, and the full harvested gate records one CasADi compile. All eight snapshot
checks pass. Only solver.c changed among C snapshots, as the maintainer approved
alongside the disjoint vector snapshot work. README snippets and ten examples run.
The new structural gate fails for both backends under a forced-mode perturbation.

Fable 5.1 high-effort review found an unused SQP-only helper in native solver
exports and repetitive PIQP settings dispatch. The helper now belongs to each
SQP wrapper, with distinct symbols. PIQP uses one typed field table for dispatch
and comparison, preserving workspace rebuilds without PIQP-internal assumptions.
SQP checkpoint saves, watchdog trials and the merit fallback share their logic.
The C/C++ warning gate failed before the fix and now passes. A two-SQP host test
covers helper isolation, and the reuse cases exercise the native enum field.

All pre-merge checks pass again with 1317 tests and no skips. The nine backend
records plus four globalization/watchdog records exactly match main. The full
harvested gate still records one CasADi compile. PIQP/IPOPT C and C++ exports
compile with -Wall -Werror. Fable's one low-effort delta check is next.
Source/.text bytes after simplification: SQP 27011/12817, PIQP 11477/2433,
IPOPT 16919/3793. Previous PR: 30318/15745, 16349/4865, 17154/3793.
Logs are `/tmp/scaly-112-static-*.log`; these are local diagnostics, not benchmarks.
