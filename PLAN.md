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

Fable 5.1 high-effort review is running as delegated task
`scaly-112-static-fable-review-1`. Read its terminal result before finalizing.
Source/.text bytes after specialization: SQP 29164/14641, PIQP 14307/4081,
IPOPT 17134/3793. Previous PR: 30318/15745, 16349/4865, 17154/3793.
Logs are `/tmp/scaly-112-static-*.log`; these are local diagnostics, not benchmarks.
