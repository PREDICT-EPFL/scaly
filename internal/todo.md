# What to do next

The single actionable list. Rationale lives elsewhere and is linked, never restated:

- **Why we are writing what we write** — [`internal/paper.md`](paper.md): thesis, scope, narrative,
  outline, claim gates, objections.
- **Why a number is or is not admissible** — [`docs/results/fairness.md`](../docs/results/fairness.md):
  what the comparisons hold constant, the measurement protocol, the reference machine.
- **How the suite got here** — [`internal/notes/benchmark-buildout.md`](notes/benchmark-buildout.md):
  the completed B-track and L-track, formulation history, retired workloads.
- **Library-internal phases** — [`internal/roadmap.md`](roadmap.md).

Last reorganized 2026-08-25, after the fairness audit. Ordered by dependency, not by size: the
groups below have to happen roughly in sequence, and items inside a group are independent.

## A. Harness work, before any headline run

The audit's measurements came from scratch scripts under `benchmarks/results/fairness/`. Everything
the paper quotes has to come from the benchmark harness instead, so the runs are reproducible.

- [ ] **A1. Add `casadi_call_mx` and `casadi_map_sx` as sweep backends, and make the chain labels
      literal in the same change.** `casadi_call_mx` is an `MX` elemental `Function` called once per
      repetition; `casadi_map_sx` is an `SX` elemental `Function` through
      `Function.map(..., "serial")` in an `MX` outer graph. These are one change: `harness/sweep.py`
      passes `map_stages=True` for chain unconditionally, and setting it to `False` alone makes the
      label honest *and breaks the smoke tier*, because unrolled `SX` does not compile at those
      sizes. The comment at that line says so. Rationale: paper.md §5.2, fairness.md "Does CasADi
      have loop-preserving codegen?".
- [ ] **A2. Take every swept kernel from the solver descriptor.** `descriptor.hess` rather than a
      hand-written `factory(..., al.sphess(...))` request, so the timed function *is* what the
      optimizer calls. Checked on npmpc N=12: the Hessian already agrees at nnz 269, the Jacobian
      does not (48 of 78 constraint rows). Rationale: fairness.md, and the check is in this file.
- [ ] **A3. Stop building the race-car correctness reference as a dense Alloy Jacobian.** `_samples`
      requests `al.jac("eq", "z")`, which at N=100 is a dense 404x606 Jacobian rendered as scalar C
      that gcc does not finish in twenty minutes. **This blocks the race-car half of Fig 2** and it
      is why the same-machine timer anchor is still missing. Compute the reference in NumPy, CasADi,
      or by finite differences.
- [ ] **A4. Make the exact sparse Lagrangian Hessian the published kernel on every axis**, keeping
      the Jacobian as a long-paper row. Consequence: claim gate 2 has to be re-derived on the
      Hessian. Rationale: paper.md §4 (contribution 2).
- [ ] **A5. Compile the CasADi oracles and choose `expand` per problem.** Keep `True` on race_cars,
      switch to `False` on npmpc and unbumpercars. Add a gate per problem that the timed CasADi
      column is actually compiled, shaped like npmpc's `exact_hessian` check. Rationale:
      fairness.md "`expand` and `jit`, per problem".
- [ ] **A6. Route both IPOPT columns through one `libipopt.so`** and record which in the provenance,
      along with the MUMPS, METIS and BLAS configuration. Needs the code-generated CasADi column,
      since the reverse pin is impossible. Rationale: paper.md §5.4.
- [ ] **A7. Emit the dispatch properties into the sweep CSV**: trip count, per-iteration workspace
      and arithmetic work, coloring width. Table 2 cannot be built without them.
- [ ] **A8. Split executable generated code from static metadata in the reported artifact size**, and
      record integer workspace and argument/result pointer counts. `casadi_map_sx` has fewer lines
      than `casadi_call_mx` at race_cars N=500 but more bytes, so "smaller" is ambiguous without the
      split, and total C bytes cannot support the fixed-executable-body claim.
- [ ] **A9. Derive the mode table from the runs.** Time-to-first-solve and per-step cost, two modes
      for Alloy and three for CasADi. No separate script: the build cost and steady-state mean are
      already recorded. Rationale: paper.md §6, Table 2.
- [ ] **A10. Keep the machine honest.** Pin the `performance` governor for headline runs and test
      whether disabling boost reduces dispersion. Add repeated fresh processes, varied backend order,
      and reported dispersion. Rationale: fairness.md "The reference machine".

## B. Formulation work that gates a claim

- [ ] **B1. Map the unbumpercars pair rows instead of unrolling them.** Highest risk-adjusted item in
      this file and a **submission gate**: the `C(C-1)/2` pair barriers are built by a Python loop, so
      Alloy's own generated source grows quadratically on exactly the axis of the figure it feeds
      (`spjac:g:z` goes 845 -> 1403 -> 3455 lines for C = 2 -> 4 -> 8). This is the one place where
      *we* write the code-size growth the paper argues against.

      **This is a port, not a design.** The predecessor workload did it:
      `git show 1b03820^:benchmarks/problems/unbumpercars/__init__.py` lines 195-224 map
      `pair_c3bf_fn` over the strict upper triangle, with two constant `al.gather` tables built as
      `concatenate([arange(NSTATE) + k * NSTATE for k in bodies])` feeding both the parameter states
      and the first map's output. The pattern survives as
      `tests/integration/test_map.py::test_gather_fed_chained_maps_spjac_and_sphess_match_dense`.
      Retain the exact-Hessian correctness gates through the port.
- [ ] **B2. Move the chain and unbumpercars correctness checks onto the problem side**, as
      `race_cars` does: problem gates into `benchmarks/problems/*/checks.py` behind
      `run.py smoke --select problems`, and a self-contained minimal reproduction of whatever
      IR/AD/codegen shape they covered into `tests/`. Afterwards nothing under `tests/` imports
      `benchmarks.problems`.

## C. The pilot run, then ratification

Nothing below A and B is worth doing until they land, and nothing in the paper is ratified until this
group completes.

- [ ] **C1. Run the pilot on the reference machine under the frozen protocol.** Use it to ratify or
      rewrite the thesis and the claim gates, so that later optimization has a fixed target.
- [ ] **C2. Full sweeps at range** with the mapped pair rows, to C=32.
- [ ] **C3. One closed-loop headline per problem** at the canonical point, with the function
      evaluation, QP solve and globalization split from the statistics ABI.
- [ ] **C4. Re-measure the chain on the Lagrangian Hessian.** The pilot has Alloy losing the chain
      *equality Jacobian* by 1.5x to 2.4x, and that sentence is published. Internal validation only;
      chain is cut from the short paper.

## D. API and release, before the paper freezes

- [ ] **D1. Land the derivative API redesign and the `MAP` to `VMAP` rename.** Paper examples freeze
      only after both. Rationale: paper.md §10.
- [ ] **D2. Add an immutable publication mode**: clean release candidate, every raw run retained, and
      an archive of source, lockfile, inputs, generated code, logs, statistics and manifest.
- [ ] **D3. Validation additions**: one public `fwd` and `adj` test on the same nontrivial `VMAP`
      fixture compared against the unrolled form with a forward/reverse duality check, and a
      finite-difference check of the Lagrangian gradient in the pairwise-map sparse-Hessian test.
- [ ] **D4. Freeze measurements on `0.1.0rc1`, publish `alloy-v0.1.0`** and a durable archive.

## E. Optional, cut first if the schedule slips

- [ ] **E1. l4casadi CPU column** on unbumpercars, with batching and exact second-order generation.
- [ ] **E2. Network-width regime sweep**, and a trained wide decoder from the neural-process-MPC
      authors so the width study becomes a measured closed-loop column instead of an extrapolation.
      Also worth telling them their released episode's reported cost metric cannot be reproduced from
      the trajectory it ships with.

## F. Backlog, not scheduled

Kept because the reasoning is still good, not because anything depends on them.

- **Separate the IPOPT gap into version against build configuration.** Rebuild 3.14.11 with our
  hook's flags, or 3.14.19 against the wheel's OpenBLAS. "We ship a better-tuned linear algebra
  stack" is defensible; "our IPOPT is newer" is not. Rationale: paper.md §5.4.
- **Confirm or drop the reverse / row-coloured sparse-Jacobian hypothesis.** `sparse_jacobian`
  colours columns only, and npmpc's per-stage block is wider than it is tall, which is consistent
  with running more forward sweeps than a row-coloured or reverse pass would need. Prize is bounded
  and knowable, roughly 0.85 -> 1.1 at the shipped decoder width, and it does not change the
  width-axis result Alloy already wins.
- **Make the Program IR passes iterative instead of recursive.** `passes._transform` and
  `_expand_inlinables` recurse per node, so an expression deeper than ~200 chained elementwise ops
  dies with a bare `RecursionError` during lowering. Two witnesses: the race-car objective as a left
  fold, and the neural-process-MPC objective as a *flat* reduction over per-stage slices, which also
  unrolled and died at N=100. See `docs/how_it_works/lowering.md` and the `xfail` in
  `tests/passes/test_program.py`.
- **Replace the race-car tracking NMPC with the MPFC distillation**, in the same `race_cars` package,
  and **laopt as an external baseline** for it once laopt is published.
- **CasADi `sqpmethod` as a secondary reference column.** Opt-in and record-only; its globalization,
  regularization and QP path differ from `alloy-sqp`. Add only if review asks for it.
- **Specialized OCP problem/solver tier** in alloy (structured staged OCP lowering to general form),
  then **fatrop** as its consumer plus a casadi-fatrop baseline. osqp / proxqp / acados as claims
  demand.
- **Wheel building** for the workspace packages (cibuildwheel, per-package native builds).
- **Embedded hardware benchmarks** (Raspberry Pi / Jetson).
- **Diffusion- and GP-based problems; hovercraft MBD/DIAL workload.**
- **GPU backend milestone definition** in `internal/roadmap.md`, the prerequisite for the paper's
  outlook becoming a claim in any later paper.
