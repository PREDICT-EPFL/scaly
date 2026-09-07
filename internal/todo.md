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

Reorganized 2026-09-07. Sections are themes that outlive the first release. Inside each section,
**Now** holds what is actively worked on or next in line, and **Deferred** holds what is
intentionally low priority: the reasoning is still good, nothing depends on it yet. Completed items
are deleted, not archived: git history and the frozen notes hold the record.

### Identifiers

Every item has an identifier `<PREFIX>-<n>`. The prefix names the section the item sits in; the
number comes from one counter shared by the whole file, which only ever grows.

**Next id: 43**

| Prefix | Section |
|---|---|
| API | API |
| C | Compiler internals |
| S | Solvers |
| BH | Benchmark harness |
| BP | Benchmark problems |
| L | Licensing |
| D | Documentation |
| R | Release |

Rules:

- A new item takes the next id and bumps the counter. A deleted item never frees its number.
- Moving an item to another section changes its prefix and keeps its number. Grepping the number
  alone finds the item, or proves it is gone.
- Other documents cite the full id and the title, `S-15 Replace METIS 4 with METIS 5`, so the
  reference survives both a move and a retitle.
- A new section adds a row with a prefix that is not in the table and never was.

Why identifiers at all: they give the short stable handle that Linear or GitHub issues give, while
the list stays git-tracked, lives next to the code, and changes per branch, so a worktree can add,
close and reorder its own items and the merge carries them. A global counter rather than one per
section because items move between sections more often than expected, because eight counters are
eight places to get wrong once completed items are deleted, and because the letters then carry
only the theme and nothing else has to stay stable.

Ordering constraints across sections, the only sequencing that matters:

- C-8 to C-10 and BP-23 come before BH-20, which re-decides every claim gate in paper.md §8.
- S-15 and L-28 to L-31 come before any wheel or tag is public, even on test PyPI.
- R-37 comes before any merge of dev into main.

## API

### Now

- [ ] **API-1. Implement `FunctionTemplate` over concrete Functions**, including specialization,
      deterministic C names, one trace per instance, and lifted derivatives. Design:
      refactorings.md "Function templates" and `typing_playground/README.md`.
- [ ] **API-2. Preserve declared trees through `vmap` and Function-level differentiation.** Settle
      the mapped input convention and retain runtime and static acceptance tests. Rationale:
      refactorings.md "`vmap` and the AD entry points erase the callee's declared trees".
- [ ] **API-3. Decide the zero-input Function contract and reduce the private flat call path.**
      Preserve legitimate parameterless solver oracles. Rationale: refactorings.md
      "Zero-input `Function`s, and the flat call seam that survives because of them".
- [ ] **API-4. Finish the npmpc `FunctionTemplate` example** after API-1. The public
      typed decorators, exact `Function` annotations, and shared Alloy/CasADi runtime parameters
      landed first. Replace the remaining decoder-architecture builders with `FunctionTemplate`.
      This benchmark may use the packed parameter length as its specialization key because it does
      not add more MLP layouts; a general template must distinguish individual layer shapes because
      equal parameter counts do not prove equal architectures.
- [ ] **API-5. Validation additions**: one public `fwd` and `adj` test on the same nontrivial `VMAP`
      fixture compared against the unrolled form with a forward/reverse duality check, and a
      finite-difference check of the Lagrangian gradient in the pairwise-map sparse-Hessian test.

### Deferred

- **API-6. A QP-subproblem contract so alloy-sqp can use other QP plugins.** Today `alloy-sqp`
  imports only `include_dir`/`lib_dir` from `alloy_piqp` and its C template calls
  `piqp_setup/update/solve` and reads `qp->result` directly, so a future OSQP, ProxQP or HPIPM
  plugin would be a standalone solver but not an SQP backend. The contract is narrower than
  `render_wrapper`: set up a QP with fixed sparsity, refill values, solve, read the step and
  multipliers in one sign convention, report status and iteration count, clean up. The hard part is
  form reconciliation (two-sided rows and box bounds versus OSQP's single `l <= Ax <= u`, and
  stage-structured solvers) the way CasADi's `conic` layer does it. Documented as a limitation in
  `docs/guide/solver_backends.md`.
- **API-7. Specialized OCP problem/solver tier** in alloy (structured staged OCP lowering to general
  form), then **fatrop** as its consumer plus a casadi-fatrop baseline. osqp / proxqp / acados as
  claims demand.

## Compiler internals

The 2026-09-06 [sweeps](../docs/results/scalability.md#extended-hessian-sweeps-2026-09-06) fail claim
gates 2 and 3 on race-car, cap unbumpercars at C=32 on static metadata, and win npmpc and
unbumpercars at range only because the faster CasADi encodings stop compiling inside the budget.
Reading the generated kernels gives one structural cause for the runtime gap, the workspace growth
and the metadata growth together. Rationale and gate status: paper.md §8. Every item under **Now**
is a general compiler change with a `tests/` reproduction, per the 2026-08-25 decision in paper.md
§10; after they land, BH-20 re-decides the gates.

Start each compiler item by reading how the tools that shaped Alloy solve the same problem, before
designing anything. tinygrad, whose IR and pattern-rewrite infrastructure Alloy's are modelled on,
has a scheduler that fuses elementwise producers into their consumers and a symbolic index
arithmetic that turns strided views into closed-form index expressions: C-8 and C-9 in one
place. MLIR's affine dialect and its loop-fusion, affine-map and
memref-normalization passes are the standard treatment of exactly the loops we emit, and their
design notes state the legality conditions we would otherwise rediscover. JAX's `vmap` batching
rules are the reference for what a mapped derivative rule should produce without materializing
per-trip index tables. The goal is to port the smallest idea that fits Alloy's two dialects, not to
adopt a framework; write down what was read and what was rejected in
`internal/notes/refactorings.md` before the implementation.

### Now

- [ ] **C-8. A loop-fusion pass on the program dialect.** The race-car Hessian lowers to about 120
      consecutive loops over the mapped axis: a stage-call loop into a full-length buffer, then a
      full-length zero fill, a full-length gather-scatter, a full-length transpose, and so on for the
      next mapped op. Every intermediate is materialized at `N × width`, which is the caller
      workspace that grows with N on race_cars and npmpc and the extra memory passes `SX` never
      makes. Fuse producer and consumer loops over the same trip count into one stage loop whose
      body is per-stage scalar code and whose intermediates are per-stage scratch. Gate: race-car
      `workspace` does not grow with N in the sweep CSV, and the runtime ratio to `SX` narrows. This
      is the single change that attacks gates 2 and 3 at once.
- [ ] **C-9. Affine index maps instead of materialized tables.** The VMAP multi-seed forward rule in
      `ad/forward.py` builds `gather` index arrays of size `nseed × length × slice` (`flat_idx`,
      `tile_indices`), and the reverse rule in `ad/reverse.py` builds per-iteration index lists; the
      renderer emits each as a `static const int64_t` table. Their contents are affine in the trip
      index (`start + it * stride + j`), and the coloring seed tile repeats one stage-invariant 0/1
      pattern per stage. These tables are the static metadata that grows with N on race_cars and
      reaches 54 MB at unbumpercars C=32. Represent affine gathers and scatters structurally
      (a strided window or an index expression) and broadcast stage-invariant constants instead of
      tiling them. Gate: `static_metadata_bytes` fixed across N on race_cars, and the unbumpercars
      C=32 cell compiles under the 50 MiB cap.
- [ ] **C-10. Fold the identities the AD rules introduce, at the expression level.** The SUM rule
      emits `d0 @ ones`, the stride-0 reverse rule emits `ones @ segments`, and the seed tiles are
      multiplied in as dense 0/1 masks; `passes/expr.py` folds `x * 1` only when the constant has the
      result's shape and knows nothing about matmul-with-ones or gathers with identity indices. Add
      those rewrites in `passes/expr.py` so the program dialect never sees them. Pin each with a
      small fixture. Second-order after C-8 and C-9, but cheap.
- [ ] **C-11. Chain: exploit the stage-block structure.** The coloring width grows with M (12, 24,
      42 at M=3,5,9), so the per-stage Hessian pays that many forward-over-reverse sweeps where `SX`
      computes one symbolic Hessian. Investigate a scalar-level second-order pass inside the stage
      body, then a scatter. Internal workload, so lower priority than C-8 to C-10; it
      decides whether chain can ever enter the long paper's tables.
- [ ] **C-12. One matcher: op-indexed tables and one walk-rebuild.** Conditional by design — it
      lands only if the result is smaller than the 78 + 71 lines of `ir/match.py` and `ir/spec.py`,
      and closing the section unlanded is a permitted outcome that still has to be written down.
      Repointing `passes/program.py`'s three `_transform` call sites is where the memo lands, which
      also closes C-13. Design: refactorings.md "One matcher".

### Deferred

- **C-13. Make the Program IR passes iterative instead of recursive**, unless C-12 gets there
  first. `passes._transform` and `_expand_inlinables` recurse per node, so an expression
  deeper than ~200 chained elementwise ops dies with a bare `RecursionError` during lowering. Two
  witnesses: the race-car objective as a left fold, and the neural-process-MPC objective as a *flat*
  reduction over per-stage slices, which also unrolled and died at N=100. See
  `docs/how_it_works/lowering.md` and the `xfail` in `tests/passes/test_program.py`.
- **C-14. Confirm or drop the reverse / row-coloured sparse-Jacobian hypothesis.** `sparse_jacobian`
  colours columns only, and npmpc's per-stage block is wider than it is tall, which is consistent
  with running more forward sweeps than a row-coloured or reverse pass would need. Prize is bounded
  and knowable, roughly 0.85 -> 1.1 at the shipped decoder width, and it does not change the
  width-axis result Alloy already wins.

## Solvers

### Now

- [ ] **S-15. Replace METIS 4 with METIS 5 (Apache-2.0).** The pinned COIN-OR `ThirdParty-Metis`
      branch fetches 4.0.3 via `get.Metis`, whose license forbids redistribution; CasADi ships it
      anyway and we will not. `ThirdParty-Mumps` accepts METIS 5, so build METIS 5.x directly with
      CMake in `plugins/alloy-ipopt/hatch_build.py`, point MUMPS at it, update `build_config.json`,
      and rewrite the METIS section of `internal/notes/vendored_solvers.md`, whose legacy-C warning
      flags become unnecessary. Fallback if METIS 5 misbehaves: MUMPS with its built-in PORD
      ordering and no METIS at all, at a cost on large sparse problems. Blocks L-28 to L-31 from
      mattering; must land before any wheel reaches even test PyPI.

### Deferred

- **S-16. Separate the IPOPT gap into version against build configuration.** Rebuild 3.14.11 with
  our hook's flags, or 3.14.19 against the wheel's OpenBLAS. "We ship a better-tuned linear algebra
  stack" is defensible; "our IPOPT is newer" is not. Rationale: paper.md §5.4.
- **S-17. Make the compiled CasADi IPOPT cache survive worktree removal.** Include the library
  search paths in the cache identity or make cached artifacts independent of them. Rationale:
  refactorings.md "Compiled CasADi artifacts across worktrees".
- **S-18. CasADi `sqpmethod` as a secondary reference column.** Opt-in and record-only; its
  globalization, regularization and QP path differ from `alloy-sqp`. Add only if review asks for it.

## Benchmark harness

### Now

- [ ] **BH-19. Harness gaps.** `dispatch_trip_count`, `dispatch_workspace` and `dispatch_arithmetic`
      are empty for the race_cars and unbumpercars Alloy cells and filled only for npmpc and chain,
      so Table 2 cannot be built from the CSV yet. Gate 4's causal claim needs a same-protocol
      pre-port control: either measure the unrolled pair rows behind a flag or drop the causal
      wording and keep the descriptive one.
- [ ] **BH-20. Rerun the study** after C-8 to C-10 and BP-23. `uv run benchmarks/run.py study
      --out-dir benchmarks/results/followup/<date>`, then paste `report.md` into the results pages
      and re-decide every gate in paper.md §8. The npmpc and unbumpercars range wins are
      compile-budget wins today; after the rerun they are either real wins against a completed
      encoding or they are labeled as budget wins in the paper.
- [ ] **BH-21. Add an immutable publication mode**: clean release candidate, every raw run retained,
      and an archive of source, lockfile, inputs, generated code, logs, statistics and manifest.

### Deferred

- **BH-22. Embedded hardware benchmarks** (Raspberry Pi / Jetson).

## Benchmark problems

### Now

- [ ] **BP-23. Vmap the unbumpercars wall rows.** `filters.py` still builds the four wall barriers
      per car in a Python loop after the pair rows were mapped, which is why Alloy's executable
      source still grows with C (145 KB at C=2 to 835 KB at C=32) and why gate 1 fails on that
      problem. Same port shape as the pair-row port, with the existing unbumpercars gates retained;
      extend `pair_jac_codegen_growth` so it fails while any row family still unrolls.

### Deferred

- **BP-24. Replace the race-car tracking NMPC with the MPFC distillation**, in the same `race_cars`
  package, and **laopt as an external baseline** for it once laopt is published.
- **BP-25. l4casadi CPU column** on unbumpercars, with batching and exact second-order generation.
- **BP-26. Network-width regime sweep**, and a trained wide decoder from the neural-process-MPC
  authors so the width study becomes a measured closed-loop column instead of an extrapolation.
  Also worth telling them their released episode's reported cost metric cannot be reproduced from
  the trajectory it ships with.
- **BP-27. Diffusion- and GP-based problems; hovercraft MBD/DIAL workload.**

## Licensing

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

The METIS 4 row is closed by S-15.

### Now

- [ ] **L-28. Root `LICENSE` (BSD-2-Clause, Tudor Oancea, 2026)**, `license = "BSD-2-Clause"` and
      `license-files = ["LICENSE"]` in the root `pyproject.toml`, and the README "License" section
      replaces "TBD".
- [ ] **L-29. Copy the same `LICENSE` into each of `plugins/alloy-{sqp,piqp,ipopt}/`** with the same
      two pyproject fields. Each plugin is its own sdist and wheel, so each needs the file in its
      own tree; a copy, not a symlink, so sdists stay correct.
- [ ] **L-30. Third-party notices generated by the build hooks.** `hatch_build.py` in `alloy-piqp`
      and `alloy-ipopt` copies each dependency's license text from the already-cloned
      `third_party/` sources into `src/alloy_{piqp,ipopt}/licenses/<dep>/` and writes a short
      `THIRD_PARTY_NOTICES.md` listing name, pinned version from `build_config.json`, license and
      upstream URL; the directory joins the wheel `artifacts`. Generating at build time keeps the
      notices from drifting from the pins. libgfortran and libquadmath are not cloned, so their
      GPL-plus-runtime-exception text is vendored once or taken from the GCC install.
- [ ] **L-31. Close the two PIQP questions** in the table: confirm `EIGEN_MPL2_ONLY`, and how the
      LDL code is linked.
- [ ] **A test that the license directory exists for every dependency named in
      `build_config.json`** in each plugin wheel's file list, shown to fail when an entry is
      removed. Plus one paragraph in `docs/dev/contributing.md`: a new vendored dependency needs a
      `build_config.json` entry and a license copied by the hook.

## Documentation

The same category of debt the fairness audit found in the results pages: prose that outran what
the code does.

### Now

- [ ] **D-32. Rewrite the user-facing documentation**, the index and the guide first. Scoped to what
      is stable today: install, core concepts, the derivative API, Functions and solvers. Leave a
      marked gap where `FunctionTemplate` (API-1) will go, and quote no numbers until BH-20.
- [ ] **D-33. Rework `docs/how_it_works/comparison.md`.** A scoped fix landed on 2026-08-25: the
      CasADi section's "one graph with a per-node hint" paragraph presented an inert mechanism as a
      departure, and it now states the repetition claim that is actually true and measured, with the
      lowering-hint discussion subsequently removed from the published docs. The rest of the page still
      predates a lot. Check every "Taken / Changed" row against what the code does today, and check
      the tinygrad, MLIR and JAX sections the same way.
- [ ] **D-34. Audit the whole of `docs/` for claims that outran the implementation**, the way the
      results pages were audited. The pattern to look for is a stated departure or capability whose
      supporting mechanism is recorded but not wired up, and a number with no machine attached.
      The lowering-hint claims have been removed. Check any surviving timing that predates the
      reference-machine rule in `AGENTS.md`.
- [ ] **D-35. Reconcile the problem READMEs with the audit.** `benchmarks/problems/*/README.md`
      still describe the CasADi columns as "same NLP, same IPOPT, same options, only the oracle
      provider differs", which the audit disproved on two counts.

### Deferred

- **D-36. GPU backend milestone definition** in `internal/roadmap.md`, the prerequisite for the
  paper's outlook becoming a claim in any later paper.

## Release

These steps make the tree public and permanent, and each is cheap to do once and expensive to redo.

### Now

- [ ] **R-37. Move `internal/paper.md` out of this repository before merging to main.** Blocking,
      and enforced: `.config/wt.toml` has a `pre-merge` check that fails while the file is tracked.

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

- [ ] **R-38. Windows support.** Decide the toolchain (MSVC or clang) and the target: the core JIT
      plus `alloy-sqp` and `alloy-piqp` first; `alloy-ipopt` on Windows is a separate later item
      because it drags in Fortran and its own licensing survey.
- [ ] **R-39. Finalize the name.** Decide whether `alloy` ships under that name; see
      `internal/notes/naming.md`.
- [ ] **R-40. Versioning policy.** What a minor bump promises about the generated C symbols, the
      sparsity-table prefixes and the plugin ABI; written into `docs/dev/contributing.md`.
- [ ] **R-41. Wheel building and publishing** for the workspace packages (cibuildwheel, per-package
      native builds), test PyPI first. After L-28 to L-31.
- [ ] **R-42. Freeze measurements on `0.1.0rc1`, publish `alloy-v0.1.0`** and a durable archive.
      After R-41.
