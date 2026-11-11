# Seed-pair lowering for #225

Implement the fixed Sparse tensors design in `internal/notes/core_compiler_roadmap.md`, under
`internal/notes/compiler_run_2026_11_10/plan.md` and `~/.scaly-suite/RULES.md`. Start from
origin/main; branch `t3code/225-seed-pair-lowering`, pull request base main. Also Part of #214.
Do not change scalarization budgets, support rules, sparse outputs or Hessian assembly.

- [x] Read the run plan, rules, #225/#214/#220, prerequisite outcomes and overlapping pull requests.
- [ ] Capture the chain M=9 baseline with default budgets: operations per stage, source bytes,
      generation seconds and generated-C compile seconds. Runtime waits for Phase 4.
- [ ] Add failing differential and structure tests for partly zero mapped tangents and nested calls.
- [ ] Thread argument support through lowering and emit per-entry seed lists beside iteration plans,
      folding known-zero operands and retaining primal work outside seed loops.
- [ ] Cover contractions and movement, and check sparse-data Jacobian code growth requested by #225.
- [ ] Perturb a seed list and demonstrate the differential test fails; restore and pass focused tests.
- [ ] Regenerate C snapshots and node-ID baseline; measure the resulting build.
- [ ] Run all pre-merge checks detached through ~/.scaly-suite/run, polling the exit file in this turn.
- [ ] Commit, push once, open and link draft PR. Review with Opus and high-risk correctness reviewer,
      fix findings and obtain one delta check. Settle used review threads.
- [ ] Remove PLAN.md, publish review fixes together, mark ready, post numbers on #214, queue once
      with gh pr merge --squash --auto and watch checks. Report evidence and any departure.

Known overlap: #314, #355, #358, #360, #361, #368, #369, #382, #404, #406, #408, #411,
#415, #416, #417. Rebase only for a conflict or merge queue ejection, regenerate snapshots then
have the reviewer read that diff. #225 comment also asks to bound sparse matmul/gather/scatter
seed-group source growth; assess what lowering can combine without changing AD support rules.
