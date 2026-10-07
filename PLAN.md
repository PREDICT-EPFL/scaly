# Runtime solver options, issue #112

The maintainer selected call-time options on #112. This branch is layer 3 above
`t3code/issue-80-piqp-option-names`. PIQP vendoring and the extern migration are
outside this change. The existing PIQP name validation must run at construction.

Sources read: architecture, codebase, conventions, contributing and solver-plugin
guides; solver descriptors/builders; JIT and C framing; all three plugin wrappers;
race-car harvested globalization check; #112 and #80 discussions.

- [x] Inspect prerequisites, checkout, stack, and fixed design.
- [ ] Add failing compilation-reuse tests and capture existing numerical results.
- [ ] Introduce call-time option data and pass it through root and nested solves.
- [ ] Move plugin defaults and validation to Python construction; update wrappers.
- [ ] Restore both CasADi harvested globalizations and update affected guides.
- [ ] Prove source/key reuse, option behavior, and gate sensitivity.
- [ ] Run the whole suite and `wt hook pre-merge`, then one cross-review and delta check.
- [ ] Submit the stacked pull request, record criterion evidence, remove this plan,
      update the tracker and register monitoring.

The option transport precedes plugin conversion. Wrapper changes and behavioral
checks must be complete before documentation and review. No parallel implementation
agents are needed; the independent review follows passing checks.
