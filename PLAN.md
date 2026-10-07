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
- [ ] Update the guides and deliberate solver snapshots. Measure source and
      executable-code size, generation time and native solve time.
- [ ] Run focused checks, the whole suite and pre-merge checks, and verify the
      numerical references and full harvested globalization gate.
- [ ] Ask Fable for the requested generated-code simplification review, apply
      justified findings, run checks and one delta check if needed.
- [ ] Remove this plan, update the PR evidence, push the stack and monitor CI.

Implementation is sequential. Review follows passing checks. No parallel
implementation agents are needed. Interface specialization does not authorize
changing the selected solver libraries or the extern migration.
