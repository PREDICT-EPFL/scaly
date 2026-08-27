# What to do next

The single actionable list. Rationale lives elsewhere and is linked, never restated:

- **Why we are writing what we write** — [`internal/paper.md`](paper.md): thesis, scope, narrative,
  outline, claim gates, objections.
- **Why a number is or is not admissible** — [`docs/results/fairness.md`](../docs/results/fairness.md):
  what the comparisons hold constant, the measurement protocol, the reference machine.
- **How the suite got here** — [`internal/notes/benchmark-buildout.md`](notes/benchmark-buildout.md):
  the completed B-track and L-track, formulation history, retired workloads.
- **What shape a refactoring should take** — [`internal/notes/refactorings.md`](notes/refactorings.md):
  one `#` section per refactoring, kept until that refactoring lands.
- **Library-internal phases** — [`internal/roadmap.md`](roadmap.md).

Last reorganized 2026-08-25, after the fairness audit. Ordered by dependency, not by size: the
groups below have to happen roughly in sequence, and items inside a group are independent.

Agreed order for the coming sessions (2026-08-26): D2, then D1, then D1.5, then A11, each landed on
its own so a failure can be attributed.

## A. Harness work, before any headline run

The audit's measurements came from scratch scripts under `benchmarks/results/fairness/`. Everything
the paper quotes has to come from the benchmark harness instead, so the runs are reproducible.

- [x] **A1. Add `casadi_call_mx` and `casadi_map_sx` as sweep backends, and make the chain labels
      literal in the same change.** `casadi_call_mx` is an `MX` elemental `Function` called once per
      repetition; `casadi_map_sx` is an `SX` elemental `Function` through
      `Function.map(..., "serial")` in an `MX` outer graph. The smoke tier runs mapped SX at M=5
      and literal SX at M=3 because unrolled SX does not compile at M=5. Rationale: paper.md §5.2
      and fairness.md "Does CasADi have loop-preserving codegen?".
- [x] **A2. Take every swept kernel from the solver descriptor.** `descriptor.hess` rather than a
      hand-written `factory(..., al.factory.SpHess(...))` request, so the timed function *is* what the
      optimizer calls. Checked on npmpc N=12: the Hessian already agrees at nnz 269, the Jacobian
      does not (48 of 78 constraint rows). Rationale: fairness.md. Landed for the Alloy side; the
      CasADi kernel now comes from `nlpsol`, and both timed sides use one solver-selected Hessian
      triangle. Rationale: fairness.md; the A11 triangle contract closes the remaining gap.
- [x] **A3. Stop building the race-car correctness reference as a dense Alloy Jacobian.** `_samples`
      requests `al.factory.Jac("eq", "z")`, which at N=100 is a dense 404x606 Jacobian rendered as scalar C
      that gcc does not finish in twenty minutes. **This blocks the race-car half of Fig 2** and it
      is why the same-machine timer anchor is still missing. Compute the reference in NumPy, CasADi,
      or by finite differences.
- [x] **A4. Make the exact sparse Lagrangian Hessian the published kernel on every axis**, keeping
      the Jacobian as a long-paper row. Consequence: claim gate 2 has to be re-derived on the
      Hessian. Rationale: paper.md §4 (contribution 2).
- [x] **A5. Compile the CasADi oracles and choose `expand` per problem.** Keep `True` on race_cars,
      switch to `False` on npmpc and unbumpercars. Add a gate per problem that the timed CasADi
      column is actually compiled, shaped like npmpc's `exact_hessian` check. Rationale:
      fairness.md "`expand` and `jit`, per problem".
- [x] **A6. Route both IPOPT columns through one `libipopt.so`** and record which in the provenance,
      along with the MUMPS, METIS and BLAS configuration. Needs the code-generated CasADi column,
      since the reverse pin is impossible. Rationale: paper.md §5.4.
- [x] **A7. Emit the dispatch properties into the sweep CSV**: trip count, per-iteration workspace
      and arithmetic work, coloring width. Table 2 cannot be built without them.
- [x] **A8. Split executable generated code from static metadata in the reported artifact size**, and
      record integer workspace and argument/result pointer counts. `casadi_map_sx` has fewer lines
      than `casadi_call_mx` at race_cars N=500 but more bytes, so "smaller" is ambiguous without the
      split, and total C bytes cannot support the fixed-executable-body claim.
- [x] **A11. One Hessian triangle per solver, and the CasADi kernel from the `nlpsol`.** Give
      `sphess` a `triangle` option, let the solver descriptor carry the layout its backend wants
      (IPOPT lower, PIQP upper) instead of `hess_lower_mask`, and take the sweep's CasADi kernels
      from `nlpsol.get_function("nlp_hess_l" | "nlp_jac_g")` with `expand` following the encoding.
      Touches the NLP plugin contract, so it lands before C1 and after D2, D1 and D1.5.
      Rationale: A2's sentence is only true once the timed kernel and the oracle have the same
      entries; fairness.md "The measurement protocol".
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
      `tests/integration/test_vmap.py::test_gather_fed_chained_vmaps_spjac_and_sphess_match_dense`.
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

Every refactoring in `internal/notes/refactorings.md` lands before submission, D1 to D4 below. Each
undecided refactoring has one `#` section in that file; completed sections are removed when they
land. The design is that note's; only the lifecycle is here. Paper examples freeze after D1 and D2,
which is what fixes the spellings they use.

- [ ] **D0. Move `internal/paper.md` out of this repository before merging to main.** Blocking, and
      enforced: `.config/wt.toml` has a `pre-merge` check that fails while the file is tracked.

      Why it is urgent rather than tidy: the note contains the "sell only if the reruns establish
      it" list, the "do not sell" list and the objections rehearsal, which are the three things a
      reviewer should least find in our own words. Deleting it at release time does nothing, because
      the content stays in every clone's history, and excising it afterwards means
      `git filter-repo --path internal/paper.md --invert-paths`, which rewrites every SHA from its
      first appearance onward and breaks any archive link or tag that references an old one.

      **The window is open now and closing it is cheap.** The file has never been on main. It exists
      only in unpushed commits on this branch, and `wt merge` squashes, so moving it out before the
      merge means main never sees it and no history is rewritten.

      The design, agreed 2026-08-25:

      - `~/dev/alloy-notes/` as its own git repo, with its own private remote for backup, holding
        `paper.md` and any later private notes. Keeping it under git matters: the note is a dated
        decision log and a plain untracked file would lose its history.
      - `notes -> /home/ted/dev/alloy-notes` as a gitignored symlink in every worktree, with an
        **absolute** target so the link keeps pointing at the one source of truth even if a tool
        copies rather than links it. `/notes/` goes in `.gitignore`.
      - A `[post-start]` step in `.config/wt.toml` that recreates the symlink, so the behaviour does
        not depend on what `wt step copy-ignored` does with symlinks.
      - A line in `AGENTS.md`: what `notes/` is, that nothing public may depend on it, and never
        `git add -f` under it.

      Properties this buys. One source of truth across every worktree and branch, which is correct
      for a planning document since the plan is not per-branch. The worst possible accident commits a
      path string, never content. And the public rationale stays public, because
      `docs/results/fairness.md` carries the methodology and contains no strategy.

      Rejected: a submodule leaks its existence and URL in a committed `.gitmodules` and is unpleasant
      with worktrees; an orphan branch leaves the objects in the same store, so a full clone still
      exposes them; `git-crypt` or `age` puts ciphertext in public history permanently, a poor risk
      profile for a document whose value is candour; an external tool loses grep-ability and
      proximity to the code, which is the whole reason the note works.

- [x] **D1. Land the derivative API redesign.** One name per concept, specs moved under
      `al.factory` and capitalized, `Function.factory` demoted. The five derivative names are
      overloaded across expression and Function inputs, Function-level requests use `(of, wrt)`,
      and the Hessian specs retain doubled `wrt` output names without a second request input.
      Completed 2026-08-26.
- [x] **D1.5. Star colouring for sparse Hessians**, the symmetric colouring CasADi's `hess` uses and
      Alloy lacks. Constant colour count on the shared-variable arrow Hessian that one-sided
      colouring makes grow with the map length. The `SUM` rule of `jvp_many` becomes structural in
      its own small change first. Gate: the colour count must not grow with the map length on the
      `shared_fill` fixture, and `coloring_width` in the sweep CSV on every Hessian workload.
      Completed 2026-08-26.
- [x] **D2. Land the `MAP` to `VMAP` rename.** Operation, builder, exports and internal dispatch in
      one change, so no tree carries both spellings; tests and benchmarks, including the pinned
      pytest node-ID baseline; and the public docs are all on `VMAP`/`vmap`. `al.scan` and
      `al.map_` are gone. Completed 2026-08-26.
- [ ] **D3. Land the solver-problem construction API.** Decided 2026-08-27: inputs and outputs
      are declared pytrees (`al.L` leaves, `al.G` groups) carrying both the symbolic and numeric
      structure, `Function` generic in its two trees, derivatives keeping the source's inputs, one
      `ProblemSpec` of expressions, one `al.solver` returning a plain `Function` whose QP backends
      are gated by a structural quadratic proof, `qp_problem(n, n_eq, n_ineq)` as the typed data
      form, and a `tests/typing/` harness under `ty check --error-on-warning`. Deletes `al.nlp`,
      `al.qp` and `SolverFunction`. D2, D1, D1.5 and A11 have landed; the `ty>=0.0.75` bump it
      needs is done. Design:
      refactorings.md "Solver problem construction".
- [ ] **D4. One matcher: op-indexed tables and one walk-rebuild.** Conditional by design — it lands
      only if the result is smaller than the 78 + 71 lines of `ir/match.py` and `ir/spec.py`, and
      closing the section unlanded is a permitted outcome that still has to be written down. After
      D2, so the byte-for-byte C corpus regenerates once rather than twice. Repointing
      `passes/program.py`'s three `_transform` call sites is where the memo lands, which also closes
      the recursive-Program-IR-passes item in F. Design: refactorings.md "One matcher".
- [ ] **D5. Add an immutable publication mode**: clean release candidate, every raw run retained, and
      an archive of source, lockfile, inputs, generated code, logs, statistics and manifest.
- [ ] **D6. Validation additions**: one public `fwd` and `adj` test on the same nontrivial `VMAP`
      fixture compared against the unrolled form with a forward/reverse duality check, and a
      finite-difference check of the Lagrangian gradient in the pairwise-map sparse-Hessian test.
- [ ] **D7. Freeze measurements on `0.1.0rc1`, publish `alloy-v0.1.0`** and a durable archive.

## D'. Documentation rework

Separate from D because it is not release-blocking, but it is the same category of debt the fairness
audit found in the results pages: prose that outran what the code does.

- [ ] **D'1. Rework `docs/how_it_works/comparison.md`.** A scoped fix landed on 2026-08-25: the
      CasADi section's "one graph with a per-node hint" paragraph presented an inert mechanism as a
      departure, and it now states the repetition claim that is actually true and measured, with the
      lowering hints demoted to an explicit "aspiration, not a feature". The rest of the page still
      predates a lot. Check every "Taken / Changed" row against what the code does today, and check
      the tinygrad, MLIR and JAX sections the same way.
- [ ] **D'2. Audit the whole of `docs/` for claims that outran the implementation**, the way the
      results pages were audited. The pattern to look for is a stated departure or capability whose
      supporting mechanism is recorded but not wired up, and a number with no machine attached. Two
      known instances beyond D'1: `docs/guide/` on the lowering hints, and any surviving timing that
      predates the reference-machine rule in `AGENTS.md`.
- [ ] **D'3. Reconcile the problem READMEs with the audit.** `benchmarks/problems/*/README.md` still
      describe the CasADi columns as "same NLP, same IPOPT, same options, only the oracle provider
      differs", which the audit disproved on two counts.

## E. Optional, cut first if the schedule slips

- [ ] **E1. l4casadi CPU column** on unbumpercars, with batching and exact second-order generation.
- [ ] **E2. Network-width regime sweep**, and a trained wide decoder from the neural-process-MPC
      authors so the width study becomes a measured closed-loop column instead of an extrapolation.
      Also worth telling them their released episode's reported cost metric cannot be reproduced from
      the trajectory it ships with.

## F. Backlog, not scheduled

Kept because the reasoning is still good, not because anything depends on them.

- **Decide `ca.cse` per problem.** The chain CasADi cells call it and the others do not; measured
  at 7% of function evaluation on race_cars and unmeasured on npmpc and unbumpercars. Rationale:
  fairness.md "The measurement protocol".
- **Separate the IPOPT gap into version against build configuration.** Rebuild 3.14.11 with our
  hook's flags, or 3.14.19 against the wheel's OpenBLAS. "We ship a better-tuned linear algebra
  stack" is defensible; "our IPOPT is newer" is not. Rationale: paper.md §5.4.
- **Confirm or drop the reverse / row-coloured sparse-Jacobian hypothesis.** `sparse_jacobian`
  colours columns only, and npmpc's per-stage block is wider than it is tall, which is consistent
  with running more forward sweeps than a row-coloured or reverse pass would need. Prize is bounded
  and knowable, roughly 0.85 -> 1.1 at the shipped decoder width, and it does not change the
  width-axis result Alloy already wins.
- **Make the Program IR passes iterative instead of recursive**, unless D4 gets there first.
  `passes._transform` and
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
