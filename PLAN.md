# Runtime solver options, issue #112

The maintainer selected call-time options on #112. This branch is layer 3 above
`t3code/issue-80-piqp-option-names`. PIQP vendoring and the extern migration are
outside this change. The existing PIQP name validation must run at construction.

Sources read: architecture, codebase, conventions, contributing and solver-plugin
guides; solver descriptors/builders; JIT and C framing; all three plugin wrappers;
race-car harvested globalization check; #112 and #80 discussions.

- [x] Inspect prerequisites, checkout, stack, and fixed design.
- [x] Add failing compilation-reuse tests and capture existing numerical results.
- [x] Introduce call-time option data and pass it through root and nested solves.
- [x] Move plugin defaults and validation to Python construction; update wrappers.
- [x] Restore both CasADi harvested globalizations and update affected guides.
- [x] Prove source/key reuse, option behavior, and gate sensitivity.
- [ ] Run the whole suite and `wt hook pre-merge`, then one cross-review and delta check.
- [ ] Submit the stacked pull request, record criterion evidence, remove this plan,
      update the tracker and register monitoring.

The option transport precedes plugin conversion. Wrapper changes and behavioral
checks must be complete before documentation and review. No parallel implementation
agents are needed; the independent review follows passing checks.

Evidence: six cold-cache direct/nested cases compile once and alternate option
sets; nine result records exactly match main with the lane's native libraries
held fixed. A source perturbation makes the reuse gate fail. The full N=40
harvested gate passes both globalizations with one CasADi compilation. The C/C++
runtime entry and default entry both pass. All ten examples and both README
blocks run. The full suite passes (1307 passed, three skipped); the remaining
pre-merge smoke/docs checks are running before review. Only the solver C/header
snapshots changed; diagnostic source-size, generation and native timing records
are in /tmp/scaly-112-measure-{main,after}.log.
