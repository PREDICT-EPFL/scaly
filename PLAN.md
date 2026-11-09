# Fatrop whole-problem backend

Implement issue #239 against main, stack position 2 of 3. Do not rebase onto
`t3code/88-ocp-builder` or retarget until Ted requests it. The builder module,
its tests and guide belong to #88. Stage Functions belong to #240.

The contract is the issue body and `docs/dev/solver_plugins.md`. Stage layouts
already exist in `solvers/model.py` and `solvers/stages.py` from #238. Native
Fatrop 1.2.0 and BLASFEO packaging are recorded in
`internal/notes/vendored_solvers.md`. Port the paper chain from
`origin/devrush:examples/case_studies/fatrop_chain/scaly_impl.py` using main's API.

- [x] Read the issue, prerequisites, parent, repository rules and plugin contract.
- [x] Add failing small-problem reference tests for stage scattering, duals,
  bounds, options, nested calls and statistics.
- [x] Register Fatrop and generate its OCPCInterface wrapper using static stage
  tables and whole-horizon evaluations returning 1. Remove marker discovery.
- [x] Port the paper's 2D chain. Verify iteration parity with CasADi/Fatrop and
  1e-8 solution agreement with IPOPT for the hand-written chain and race car.
- [x] Update solver guide, backend and API pages with write-docs. Document the
  optional descriptor layout and any necessary protocol changes.
- [x] Prove test sensitivity, run the full pre-merge checks and preserve C snapshots.
- [ ] Commit, push and open a draft PR against main. Run cross-review once, fix
  findings and check fixes, then mark ready and update the tracker.

Steps are sequential except independent checks. Keep experimental scripts and
results out of published docs. Delete this temporary plan before marking ready.

Evidence: 24 Fatrop tests pass, whole suite 2520 passed / 52 skipped.
The sign perturbation fails as intended. Paper chain: 14 iterations on both
Fatrop paths at tol=1e-10; IPOPT max difference 2.498e-15. Existing chain
M3,N10: max difference 6.051e-15; race N3: 2.140e-9. Both solvers return
relaxed variables by default. Full hooks passed.
Zensical's cached build emitted false missing-page warnings. A clean strict
build passed with no warnings. The paper's default tolerance also gives 13
iterations on both Fatrop paths.

Technical review found that fixed bounds can create more stage equalities than
variables and abort Fatrop's uncaught native constructor. Added a pre-constructor
dimension check and a subprocess regression for both bounded expressions and
variable bounds, including recovery on a subsequent call. The two regressions failed with SIGABRT before the guard and pass after it.
Validation after the fix: 26 Fatrop tests pass, whole suite 2522 passed /
52 skipped, Ruff, ty and clean strict docs pass. The technical delta check
approved the constructor guard with no remaining findings.

Main review changes: pin Fatrop's status enum order in a native-package test,
remove redundant IPOPT options and the race factory argument they needed,
explain native bound relaxation, remove internal implementation and branch
wording from user pages, and cite Vanroye et al., FATROP, IROS 2023 in the chain
example. Also remove redundant descriptor checks and the unused get_nx branch,
preserve ERROR when an oracle is nonfinite, and visit bounds through per-stage
offsets. The #88 reference in core stays untouched because the user will stack
this PR on #88 before merging. Main-review delta check pending.

Main-review verification: 27 Fatrop tests pass, whole suite 2523 passed /
52 skipped, Ruff, ty and clean strict docs pass. Both benchmark agreement
gates, the solver-guide fence and both paper-chain scripts pass. The status
enum test fails when a synthetic flag is inserted. The published ub=100
relaxation example was reproduced for Fatrop and IPOPT. Benchmark smoke passed.
