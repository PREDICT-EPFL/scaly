# Issue #21: independent differentiation inputs and stop_gradient

The maintainer revised the planned intermediate-expression design during PR review. Any output
expression remains valid, but each differentiation variable must be an independent input from
`sc.sym` or a Function declaration. Computed expressions and constants must raise a clear error,
including empty shapes. `stop_gradient` and its rules remain in scope.

Read: compiler roadmap One AD engine, AD modes/sparsity/sparse builders, the Function request
layer, existing devrush ports, documentation brief and tracking conventions. The Newton case
will differentiate an energy Function with separate state and parameter inputs, called from the
carry body. No new loop constructs or custom rules are part of this change.

- [x] Add failing invalid-wrt tests across all public expression derivative builders.
- [x] Remove stand-in substitution/restoration and validate independent inputs consistently.
- [x] Rewrite numerical cases against explicit inputs and NumPy, preserving stop-gradient coverage.
- [x] Revise the roadmap, issue #21, guide/API entries and dependent-work notifications.
- [x] Run focused checks, baseline collection, unchanged C snapshots and the full pre-merge hook.
- [ ] Get one short independent check of the maintainer-review changes and prose; fix its findings.
- [ ] Restate PR #171's objective/evidence, remove this plan, push and confirm green CI.

Tests precede the production change. Documentation follows the working contract. Validation
precedes the review check. Keep all lower layers and #172's branch unchanged.

Focused results: 984 AD, input-contract, stop-gradient, QP and C snapshot checks pass. The
new rejection tests failed before the production change. Issue #21 is renamed with revised
criteria; #172 has the contract update, and #170 has a superseded-scope note for maintainer triage.

The complete hook passed with 2,656 tests passed and one optional-toolchain skip, no errors or
xfails. Types, formatting, lint, benchmark smoke, docs and private names pass. The collection
baseline contains 2,657 node IDs; C snapshots have no changes. All revised guide examples and
the validation-disabled mutation were executed. The approved #169 base is unchanged.
