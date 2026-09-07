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

D2, D1, D1.5, and A11 are complete. D3 completes the solver API work on 2026-09-05. Continue API
iteration on dev through D3.1 to D3.3 below. Tracks A and B are complete as of 2026-09-05;
C1–C4 measurements are complete as of 2026-09-06 and leave the paper claim unratified (paper.md
§8). Track C' is the compiler and formulation work those measurements demand; it is the current
focus. Track W is the wrap-up and comes last.

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
- [x] **A9. Derive the mode table from the runs.** Time-to-first-solve and per-step cost, two modes
      for Alloy and three for CasADi. No separate script: the build cost and steady-state mean are
      already recorded. Completed 2026-09-05: `closed-loop` records construction and solve wall times;
      `run.py modes` derives the five rows and dispersion from saved episodes. Rationale: paper.md
      §6, Table 2.
- [x] **A12. Take every CasADi kernel after `transform({})`.** Make `--casadi-transform` the sweep
      default and apply the same flow to the compiled closed-loop `nlpsol` oracles, so both CasADi
      columns are CasADi's best configuration. Rationale: fairness.md "CasADi 3.8".
- [x] **A10. Keep the machine honest.** Pin the `performance` governor for headline runs and test
      whether disabling boost reduces dispersion. Add repeated fresh processes, varied backend order,
      and reported dispersion. Completed 2026-09-05 with fresh-process sweeps and episodes, varied
      order, separate compilation phases, recorded machine controls, and Linux headline checks.
      Boost comparison and selected settings: fairness.md "The reference machine".

## B. Formulation work that gates a claim

- [x] **B1. Map the unbumpercars pair rows instead of unrolling them.** Highest risk-adjusted item in
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
- [x] **B3. Vmap the race-car cost.** `_race_car_nlp` builds the cost as a Python loop over stage
      slices, which makes the Hessian's workspace and runtime quadratic in the horizon; one `vmap`
      over a six-residual stage function with per-stage weights in a constant vector is exact and
      3× faster at N=100. Delete the left-fold workaround comment with it, and re-run the race-car
      smoke gates. Rationale: fairness.md "CasADi 3.8".
- [x] **B4. Lower a sum of pads as one zero-fill and N scatter-adds** (alloy core). The adjoint of a
      slice is a pad, and N pads accumulated into one long vector currently zero-fill N full-length
      buffers; this is what B3 works around and what any user loop over slices will hit. A
      program-dialect rewrite in `passes/program.py`, with a `tests/` reproduction that pins the
      buffer count. Rationale: fairness.md "CasADi 3.8".
- [x] **B2. Move the chain and unbumpercars correctness checks onto the problem side**, as
      `race_cars` does: problem gates into `benchmarks/problems/*/checks.py` behind
      `run.py smoke --select problems`, and a self-contained minimal reproduction of whatever
      IR/AD/codegen shape they covered into `tests/`. Completed 2026-09-05: the remaining problem
      gates moved, independent compiler fixtures remain, and an import-boundary test enforces that
      nothing under `tests/` imports `benchmarks.problems`.

## C. The pilot run, then ratification

Nothing below A and B is worth doing until they land, and nothing in the paper is ratified until this
group completes.

- [x] **C1. Run the pilot on the reference machine under the frozen protocol.** Kernel pilot completed
      2026-09-05 with performance selected, boost off, and five fresh processes per cell. Initial
      Hessian microbenchmark results with synthetic inputs cover race_cars N=10,40,100, unbumpercars C=2,4,8, and npmpc
      N=6,12. [Results](../docs/results/scalability.md#initial-frozen-protocol-pilot-2026-09-05) and
      gate decisions (paper.md §8). The 20% runtime target remains
      failed. C1 contains no closed-loop episodes; C3 records those separately. C2 extends the
      kernel measurements to the full ranges.
- [x] **C2. Full sweeps at range** with the mapped pair rows, to C=32. Completed 2026-09-06
      with five fresh processes through race-car N=500, neural-process N=200, and unbumpercars C=32.
      [Results](../docs/results/scalability.md#extended-hessian-sweeps-2026-09-06) retain timeouts,
      the C=32 source cap, and the separate full-grid unbumpercars sampling rerun. Decisions: paper.md §8.
- [x] **C3. One closed-loop headline per problem** at the canonical point, with the function
      evaluation, QP solve and globalization split from the statistics ABI. Completed 2026-09-05
      with five fresh processes per SQP oracle provider on all four problems, after correcting
      CasADi SQP transformation and Hessian-triangle handling. All 40 episodes succeed.
      [Results](../docs/results/index.md#canonical-closed-loop-sqp-runs-2026-09-05) include the
      failed equal-work check on unbumpercars. Claim limits: paper.md §8, C3 decision.
- [x] **C4. Re-measure the chain on the Lagrangian Hessian.** Completed 2026-09-06 at M=3,5,9
      and N=40 with five fresh processes. [Results](../docs/results/scalability.md#chain-hessian-2026-09-06) and the
      decision in paper.md §8 record the larger Hessian deficit and partial SX compilation at M=5. The published Jacobian
      section is labeled historical; chain remains outside the short paper.

## C'. Compiler and formulation work the measurements demand

The 2026-09-06 [sweeps](../docs/results/scalability.md#extended-hessian-sweeps-2026-09-06) fail claim
gates 2 and 3 on race-car, cap unbumpercars at C=32 on static metadata, and win npmpc and
unbumpercars at range only because the faster CasADi encodings stop compiling inside the budget.
Reading the generated kernels gives one structural cause for the runtime gap, the workspace growth
and the metadata growth together. Rationale and gate status: paper.md §8. Every item here is a
general compiler change with a `tests/` reproduction, per the 2026-08-25 decision in paper.md §10;
after they land, C'6 reruns the whole study.

Start each compiler item by reading how the tools that shaped Alloy solve the same problem, before
designing anything. tinygrad, whose IR and pattern-rewrite infrastructure Alloy's are modelled on,
has a scheduler that fuses elementwise producers into their consumers and a symbolic index
arithmetic that turns strided views into closed-form index expressions: C'1 and C'2 in one place.
MLIR's affine dialect and its loop-fusion, affine-map and memref-normalization passes are the
standard treatment of exactly the loops we emit, and their design notes state the legality
conditions we would otherwise rediscover. JAX's `vmap` batching rules are the reference for what a
mapped derivative rule should produce without materializing per-trip index tables. The goal is to
port the smallest idea that fits Alloy's two dialects, not to adopt a framework; write down what
was read and what was rejected in `internal/notes/refactorings.md` before the implementation.

- [ ] **C'1. A loop-fusion pass on the program dialect.** The race-car Hessian lowers to about 120
      consecutive loops over the mapped axis: a stage-call loop into a full-length buffer, then a
      full-length zero fill, a full-length gather-scatter, a full-length transpose, and so on for the
      next mapped op. Every intermediate is materialized at `N × width`, which is the caller
      workspace that grows with N on race_cars and npmpc and the extra memory passes `SX` never
      makes. Fuse producer and consumer loops over the same trip count into one stage loop whose
      body is per-stage scalar code and whose intermediates are per-stage scratch. Gate: race-car
      `workspace` does not grow with N in the sweep CSV, and the runtime ratio to `SX` narrows. This
      is the single change that attacks gates 2 and 3 at once.
- [ ] **C'2. Affine index maps instead of materialized tables.** The VMAP multi-seed forward rule in
      `ad/forward.py` builds `gather` index arrays of size `nseed × length × slice` (`flat_idx`,
      `tile_indices`), and the reverse rule in `ad/reverse.py` builds per-iteration index lists; the
      renderer emits each as a `static const int64_t` table. Their contents are affine in the trip
      index (`start + it * stride + j`), and the coloring seed tile repeats one stage-invariant 0/1
      pattern per stage. These tables are the static metadata that grows with N on race_cars and
      reaches 54 MB at unbumpercars C=32. Represent affine gathers and scatters structurally
      (a strided window or an index expression) and broadcast stage-invariant constants instead of
      tiling them. Gate: `static_metadata_bytes` fixed across N on race_cars, and the unbumpercars
      C=32 cell compiles under the 50 MiB cap.
- [ ] **C'3. Fold the identities the AD rules introduce, at the expression level.** The SUM rule
      emits `d0 @ ones`, the stride-0 reverse rule emits `ones @ segments`, and the seed tiles are
      multiplied in as dense 0/1 masks; `passes/expr.py` folds `x * 1` only when the constant has the
      result's shape and knows nothing about matmul-with-ones or gathers with identity indices. Add
      those rewrites in `passes/expr.py` so the program dialect never sees them. Pin each with a
      small fixture. Second-order after C'1 and C'2, but cheap.
- [ ] **B5. Vmap the unbumpercars wall rows.** `filters.py` still builds the four wall barriers per
      car in a Python loop after the pair rows were mapped in B1, which is why Alloy's executable
      source still grows with C (145 KB at C=2 to 835 KB at C=32) and why gate 1 fails on that
      problem. Same port shape as B1, with the existing unbumpercars gates retained; extend
      `pair_jac_codegen_growth` so it fails while any row family still unrolls.
- [ ] **C'4. Chain: exploit the stage-block structure.** The coloring width grows with M (12, 24,
      42 at M=3,5,9), so the per-stage Hessian pays that many forward-over-reverse sweeps where `SX`
      computes one symbolic Hessian. Investigate a scalar-level second-order pass inside the stage
      body, then a scatter. Internal workload, so lower priority than C'1 to C'3; it decides whether
      chain can ever enter the long paper's tables.
- [ ] **C'5. Harness gaps.** `dispatch_trip_count`, `dispatch_workspace` and `dispatch_arithmetic`
      are empty for the race_cars and unbumpercars Alloy cells and filled only for npmpc and chain,
      so Table 2 cannot be built from the CSV yet. Gate 4's causal claim needs a same-protocol
      pre-port control: either measure the unrolled pair rows behind a flag or drop the causal
      wording and keep the descriptive one.
- [ ] **C'6. Rerun the study after C'1 to C'3 and B5.** `uv run benchmarks/run.py study --out-dir
      benchmarks/results/followup/<date>`, then paste `report.md` into the results pages and
      re-decide every gate in paper.md §8. The npmpc and unbumpercars range wins are compile-budget
      wins today; after the rerun they are either real wins against a completed encoding or they
      are labeled as budget wins in the paper.

## D. API and release, before the paper freezes

D1 to D4 are the refactorings required before submission. The remaining designs live in
`internal/notes/refactorings.md`; completed sections are removed when they land. D3.1 to D3.3 are
follow-ups on dev, not prerequisites for merging D3. Paper examples freeze after D1 and D2. The
release and archive steps that used to sit here are in track W.

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
- [x] **D3. Land the solver-problem construction API.** Completed 2026-09-05: inputs and outputs
      are declared pytrees (`al.L` leaves, `al.G` groups) carrying both the symbolic and numeric
      structure, `Function` generic in its two trees, derivatives keeping the source's inputs, one
      `ProblemSpec` of expressions, one `al.solver` returning a plain `Function` whose QP backends
      are gated by a structural quadratic proof, `qp_problem(n, n_eq, n_ineq)` as the typed data
      form, and a `tests/typing/` harness under `ty check --error-on-warning`. Deletes `al.nlp`,
      `al.qp` and `SolverFunction`. The design record remains in `typing_playground/`; the public
      behavior is documented in `docs/guide/functions.md` and `docs/guide/solvers.md`.
- [ ] **D3.1. Implement `FunctionTemplate` over concrete Functions**, including specialization,
      deterministic C names, one trace per instance, and lifted derivatives. Design:
      refactorings.md "Function templates" and `typing_playground/README.md`.
- [ ] **D3.2. Preserve declared trees through `vmap` and Function-level differentiation.** Settle
      the mapped input convention and retain runtime and static acceptance tests. Rationale:
      refactorings.md "`vmap` and the AD entry points erase the callee's declared trees".
- [ ] **D3.3. Decide the zero-input Function contract and reduce the private flat call path.**
      Preserve legitimate parameterless solver oracles. Rationale: refactorings.md
      "Zero-input `Function`s, and the flat call seam that survives because of them".
- [ ] **D4. One matcher: op-indexed tables and one walk-rebuild.** Conditional by design — it lands
      only if the result is smaller than the 78 + 71 lines of `ir/match.py` and `ir/spec.py`, and
      closing the section unlanded is a permitted outcome that still has to be written down. After
      D2, so the byte-for-byte C corpus regenerates once rather than twice. Repointing
      `passes/program.py`'s three `_transform` call sites is where the memo lands, which also closes
      the recursive-Program-IR-passes item in F. Design: refactorings.md "One matcher".
- [ ] **D6. Validation additions**: one public `fwd` and `adj` test on the same nontrivial `VMAP`
      fixture compared against the unrolled form with a forward/reverse duality check, and a
      finite-difference check of the Lagrangian gradient in the pairwise-map sparse-Hessian test.

## D'. Documentation rework

Separate from D because it is not release-blocking, but it is the same category of debt the fairness
audit found in the results pages: prose that outran what the code does.

- [ ] **D'1. Rework `docs/how_it_works/comparison.md`.** A scoped fix landed on 2026-08-25: the
      CasADi section's "one graph with a per-node hint" paragraph presented an inert mechanism as a
      departure, and it now states the repetition claim that is actually true and measured, with the
      lowering-hint discussion subsequently removed from the published docs. The rest of the page still
      predates a lot. Check every "Taken / Changed" row against what the code does today, and check
      the tinygrad, MLIR and JAX sections the same way.
- [ ] **D'2. Audit the whole of `docs/` for claims that outran the implementation**, the way the
      results pages were audited. The pattern to look for is a stated departure or capability whose
      supporting mechanism is recorded but not wired up, and a number with no machine attached.
      The lowering-hint claims have been removed. Check any surviving timing that predates the
      reference-machine rule in `AGENTS.md`.
- [ ] **D'3. Reconcile the problem READMEs with the audit.** `benchmarks/problems/*/README.md` still
      describe the CasADi columns as "same NLP, same IPOPT, same options, only the oracle provider
      differs", which the audit disproved on two counts.

## E. Optional, cut first if the schedule slips

- [ ] **E1. l4casadi CPU column** on unbumpercars, with batching and exact second-order generation.
- [ ] **E2. Network-width regime sweep**, and a trained wide decoder from the neural-process-MPC
      authors so the width study becomes a measured closed-loop column instead of an extrapolation.
      Also worth telling them their released episode's reported cost metric cannot be reproduced from
      the trajectory it ships with.

## L. Licensing, before any wheel or tag is public

Alloy and the three plugins are BSD-2-Clause. The plugin wheels also ship other people's binaries,
so each wheel carries its dependencies' license texts the way CasADi does
(`casadi/include/licenses/<dep>/LICENSE`), except that CasADi's `mumps-external` and
`metis-external` entries are the COIN-OR wrapper's EPL text rather than the real MUMPS and METIS
licenses, which we do not copy. Surveyed 2026-09-07. What we ship and what it asks of us:

| Package | Component | License | Obligation |
|---|---|---|---|
| alloy, alloy-sqp | our code | BSD-2 | none |
| alloy-piqp | PIQP, BLASFEO | BSD-2 | notice |
| | Eigen | MPL-2.0 | notice; the build must define `EIGEN_MPL2_ONLY` |
| | LDL inside PIQP (`piqp/sparse/LDL_License.txt`) | LGPL-2.1 | notice; check how it is used |
| alloy-ipopt | IPOPT | EPL-2.0 | notice, upstream source of the pinned version; a separate dynamically loaded module, so our BSD-2 is unaffected |
| | MUMPS | CeCILL-C | notice |
| | OpenBLAS (static, Linux) | BSD-3 | notice |
| | libgfortran, libquadmath | GPL-3 + GCC runtime exception | notice; the exception covers this use |
| | **METIS 4.0.3** | UMN research license | **"may not be sold or redistributed without prior approval." Not shippable.** |

- [ ] **L1. Replace METIS 4 with METIS 5 (Apache-2.0).** The pinned COIN-OR `ThirdParty-Metis`
      branch fetches 4.0.3 via `get.Metis`, whose license forbids redistribution; CasADi ships it
      anyway and we will not. `ThirdParty-Mumps` accepts METIS 5, so build METIS 5.x directly with
      CMake in `plugins/alloy-ipopt/hatch_build.py`, point MUMPS at it, update `build_config.json`,
      and rewrite the METIS section of `internal/notes/vendored_solvers.md`, whose legacy-C warning
      flags become unnecessary. Fallback if METIS 5 misbehaves: MUMPS with its built-in PORD
      ordering and no METIS at all, at a cost on large sparse problems. Blocks every other item
      here from mattering; must land before any wheel reaches even test PyPI.
- [ ] **L2. Root `LICENSE` (BSD-2-Clause, Tudor Oancea, 2026)**, `license = "BSD-2-Clause"` and
      `license-files = ["LICENSE"]` in the root `pyproject.toml`, and the README "License" section
      replaces "TBD".
- [ ] **L3. Copy the same `LICENSE` into each of `plugins/alloy-{sqp,piqp,ipopt}/`** with the same
      two pyproject fields. Each plugin is its own sdist and wheel, so each needs the file in its
      own tree; a copy, not a symlink, so sdists stay correct.
- [ ] **L4. Third-party notices generated by the build hooks.** `hatch_build.py` in `alloy-piqp`
      and `alloy-ipopt` copies each dependency's license text from the already-cloned
      `third_party/` sources into `src/alloy_{piqp,ipopt}/licenses/<dep>/` and writes a short
      `THIRD_PARTY_NOTICES.md` listing name, pinned version from `build_config.json`, license and
      upstream URL; the directory joins the wheel `artifacts`. Generating at build time keeps the
      notices from drifting from the pins. libgfortran and libquadmath are not cloned, so their
      GPL-plus-runtime-exception text is vendored once or taken from the GCC install.
- [ ] **L5. Close the two PIQP questions** in the table: confirm `EIGEN_MPL2_ONLY`, and how the
      LDL code is linked.
- [ ] **L6. A test that the license directory exists for every dependency named in
      `build_config.json`** in each plugin wheel's file list, shown to fail when an entry is
      removed. Plus one paragraph in `docs/dev/contributing.md`: a new vendored dependency needs a
      `build_config.json` entry and a license copied by the hook.

## W. Wrap-up, the very last step

Only after every track above works and the documentation is in good shape: these steps make the
tree public and permanent, and each is cheap to do once and expensive to redo.

- [ ] **W1. Move `internal/paper.md` out of this repository before merging to main.** Blocking, and
      enforced: `.config/wt.toml` has a `pre-merge` check that fails while the file is tracked.

      Why it is urgent rather than tidy: the note contains the "sell only if the reruns establish
      it" list, the "do not sell" list and the objections rehearsal, which are the three things a
      reviewer should least find in our own words. Deleting it at release time does nothing, because
      the content stays in every clone's history, and excising it afterwards means
      `git filter-repo --path internal/paper.md --invert-paths`, which rewrites every SHA from its
      first appearance onward and breaks any archive link or tag that references an old one.

      The file has never been on main, but it is tracked in dev's history. Fast-forwarding this
      API branch to dev preserves that history. Before merging dev into main, move the note out
      and squash the public changes, or remove the private path from the history being published.
      Deleting the file alone does not make a fast-forward to main safe.

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

- [ ] **W2. Add an immutable publication mode**: clean release candidate, every raw run retained, and
      an archive of source, lockfile, inputs, generated code, logs, statistics and manifest.
- [ ] **W3. Freeze measurements on `0.1.0rc1`, publish `alloy-v0.1.0`** and a durable archive. Track L complete first.

## F. Backlog, not scheduled

Kept because the reasoning is still good, not because anything depends on them.

- **Make the compiled CasADi IPOPT cache survive worktree removal.** Include the library search
  paths in the cache identity or make cached artifacts independent of them. Rationale:
  refactorings.md "Compiled CasADi artifacts across worktrees".

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
