# Detect stage layouts for #238

The task is [#238](https://github.com/PREDICT-EPFL/scaly/issues/238), part of
[#236](https://github.com/PREDICT-EPFL/scaly/issues/236). #88 constructs the same
layout directly. Keep the layout a plain frozen record and the detector separate.

Read the solver descriptors, the stacked problem form, CasADi's
`casadi/interfaces/fatrop/fatrop_interface.cpp`, the chain and race-car
transcriptions, the compiler roadmap and the contributor conventions.

## Work

- [x] Inspect dependencies, current ownership and open pull requests. No blockers or overlaps.
- [x] Add failing tests for stage sizes, original constraint row groups, explicit layouts,
  invalid Jacobian dependencies, shared variables and cross-stage Hessian entries.
- [x] Port the rightmost-nonzero walk into `solvers/stages.py`. Account for initial-state
  equalities and Scaly's separate equality and inequality stacks. Expose lazy detection on
  the descriptor and pass explicitly supplied problem layouts through unchanged.
- [x] Add benchmark correctness gates for the chain and race-car layouts. Register the
  new module and regenerate the pytest collection baseline. Show the new gates can fail.
- [x] Run all pre-merge checks, keeping generated C snapshots unchanged.
- [ ] Commit, push and open a draft pull request. Get one cross-review, apply findings and
  confirm fixes. Remove this plan, mark ready, update the tracker and watch the pull request.

The phases are sequential. The layout carries state and control sizes and tuples of original
row indices for dynamics, local equalities and inequalities. Counts follow from these tuples.
Detection happens only when a consumer requests a descriptor's stage layout, so existing
general-purpose solvers retain support for arbitrary nonlinear programs.

## Evidence

- The new test module failed to collect before implementation because `solvers.stages` did not exist.
- All 27 focused tests pass. Import-layer and public-name checks passed with the focused tests.
- Both benchmark layout gates pass. Corrupting a detected control count makes both fail.
- Disabling the Jacobian and Hessian block checks makes all ten rejection cases fail. The checks
  were restored before validation.
- Type checking, lint and focused formatting passed before the benchmark gates were added.
- `wt hook pre-merge --yes` passed. The full suite reported 2,439 passed and 89 skipped.
  Benchmark smoke passed with IPOPT-dependent checks skipped because the native plugin is
  unavailable. Documentation built without warnings. Generated C snapshots are unchanged.

## Cross-review

- Opus found that reversed, partial and combined initial-state rows were mistaken for dynamics.
  Four added regressions failed before the fix, including a partial first-component condition
  that exposed an incorrect state/control split.
- The detector now removes that false transition and infers unfixed initial-state components
  from the first real transition. All 31 focused tests and both benchmark layout gates pass.
- Clarified equality ordering and the structural-only diagonal guarantee. Supplied layouts remain
  trusted as agreed. The layout type needs no additions.
- Repeat merge checks for the fix, then ask Opus for one low-effort delta check. Remove this plan
  before marking the pull request ready.
- Merge checks passed again after the fix: 2,443 tests passed and 89 skipped. Smoke, strict
  types, lint, formatting, docs and the private-name check passed. No C snapshots changed.
