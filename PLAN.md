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
- [ ] Add failing small-problem reference tests for stage scattering, duals,
  bounds, options, nested calls and statistics.
- [ ] Register Fatrop and generate its OCPCInterface wrapper using static stage
  tables and whole-horizon evaluations returning 1. Remove marker discovery.
- [ ] Port the paper's 2D chain. Verify iteration parity with CasADi/Fatrop and
  1e-8 solution agreement with IPOPT for the hand-written chain and race car.
- [ ] Update solver guide, backend and API pages with write-docs. Document the
  optional descriptor layout and any necessary protocol changes.
- [ ] Prove test sensitivity, run the full pre-merge checks and preserve C snapshots.
- [ ] Commit, push and open a draft PR against main. Run cross-review once, fix
  findings and check fixes, then mark ready and update the tracker.

Steps are sequential except independent checks. Keep experimental scripts and
results out of published docs. Delete this temporary plan before marking ready.
