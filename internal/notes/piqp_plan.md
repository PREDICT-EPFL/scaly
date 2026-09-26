# PIQP in Scaly — living plan

**Status:** Tier 1 done (git tag `tier1-complete`) · Tier 2 done on `claude/tier2-sparse` (git tag `tier2-complete`) · Tier 3 next · last updated 2026-09-26 (v4.3)
**Copies:** claude.ai project doc `claude/piqp-plan.md` and repo `internal/notes/piqp_plan.md` hold the same text; update both together. Background and review evidence: `internal/notes/piqp_plan_2026_09_26.html` (v3; superseded where they differ). Tier 1 detail: `claude/tier1-implementation-status.md`.

## Goal and principles
Implement the PIQP QP solver in Scaly, written once in Python and emitted as C specialised to one fixed sparsity pattern, as a drop-in `backend="scaly"` next to the vendored PIQP plugin.
- **One deployment path: generated code only** (decided 2026-09-26). No external BLAS/LAPACK/sparse libraries in shipped code. External kernels are benchmark baselines only, and a possible later tier.
- **Loops stay loops.** Scaly's automatic unrolling applies only to tiny kernels, below its budget.
- **Generic before structured** (decided 2026-09-26). General sparse QPs come first; multistage/Riccati structure is a later tier.
- **Tier 2 is general sparse linear algebra for Scaly**, not a PIQP-private component. PIQP is its first consumer.
- **Every milestone has a gate and a kill criterion.** Every PR gets a report (capability, changes, tests with mutation checks, benchmarks) and each tier ends with an agent review round.

## Tier 1 — core IR primitives (done: C-82 … C-93, PRs 1–9, tag `tier1-complete`)
| # | Item | Status |
|---|---|---|
| 1 | Accumulating scatter / `segment_sum` | done |
| 2 | Compare → bool, `where`, logical ops | done |
| 3 | `reduce_max`/`reduce_min`, `norm_inf`/`norm_1` | done |
| 4 | `segment_max`/`segment_min` | done |
| 5 | `cast`, `isfinite`, `copysign`, integer values | done (int64 inputs now converted at the entry) |
| 6 | `sc.scan` | done |
| 7 | `sc.while_loop` | done |
| 8 | In-place carries | done for fixed-index update chains; runtime-index proof is Tier 2 |
| 9 | Integer/bool outputs across the ABI | bool and int carried as double |
| 10 | Custom derivatives | done |
| — | Multi-seed AD through loops, per-step Jacobian tangents, one backward scan per scan, division-free index maps, raw-address JIT | done (PRs 8–9) |

Results: MPC Hessian N=100 205 → 58 µs; RK4 Hessian 1376 → 93 µs; `scan` builds about 270× faster than unrolling. 1057 tests pass.
Open Tier 1 items: C-89, C-90, C-92, C-94–C-100 in `internal/todo.md`; the `segment` merge question.

## Tier 2 — sparse and dense linear algebra for Scaly (general purpose; done: C-101 … C-113, tag `tier2-complete`)
Aim: sparse matrices and factorizations as first-class, reusable Scaly values, for Newton/SQP KKT steps, Kalman updates, implicit integrators, sparse Jacobian/Hessian workflows and PIQP. It builds on what exists: `SparsityType`, input sparsity declarations, compact outputs of `sparse_jacobian`/`sparse_hessian`.

| # | Item | Content |
|---|---|---|
| 11 | `SparseMatrix` value | Static pattern (CSC in NumPy) plus a compact-values Expr. Construction from dense Exprs, patterns, `sparse_jacobian`/`sparse_hessian` results and COO triplets. Algebra: add (pattern union), scale, sparse×dense, sparse×sparse (symbolic product at generation time), transpose, diagonal ops, block assembly (KKT), `to_dense`. Compact values across the ABI for inputs and outputs. AD through the values. A library-level type: no IR type change |
| 12 | Symbolic analysis (NumPy, generation time) | Orderings (natural, RCM, SciPy SuperLU MMD; AMD port if fill demands it); permutation; elimination tree; postorder; column counts; L pattern; left-looking column tables; contiguous segments chosen by DP; supernode and level statistics |
| 13 | Numeric sparse LDLᵀ/Cholesky | Left-looking column scan with runtime-indexed tables. Quasi-definite with a static sign vector and regularisation; no pivoting. In place on the L carry |
| 14 | Triangular solves, `ldl_solve` | Factor/solve split, so one factorization serves several solves; implicit derivative through `custom_derivative` |
| 15 | Factorization health check | Non-finite or zero pivot → flag; non-finite solution → flag |
| 16 | Iterative refinement | Fixed-k and adaptive, via `while_loop` |
| 17 | Dense kernels (generated) | Cholesky, no-pivot LDLᵀ, trsv/trsm, gemm/syrk as static-shape ops lowered to plain loops; unrolled when tiny |
| 18 | Schedule choice (generated variants only) | Unroll, column scan, or segments, chosen at generation time from the symbolic statistics. Knobs consolidated into one lowering-options object |

Compiler work Tier 2 needs:
- **T2.a** Scan/while step index as a body input.
- **T2.b** `take` / `put_add` / `put` with runtime int64 indices and a dump slot per lane.
- **T2.c** In-place proof at the scan node from the constant tables.
- **T2.d** Index-aware fusion (a minimal C-54).
- **T2.e** Ragged (runtime-length) inner loops.
- **T2.f** Sparsity override on `custom_derivative` and a trajectory-memory guard.
- **T2.g** Multi-tensor carries or a packing helper; int64 workspace spill.

Gates:
- Residual and forward error against SciPy on random quasi-definite KKTs at δ ∈ {1e-4, 1e-8, 1e-10, 1e-13}, on MPC KKTs and on the small Maros–Mészáros subset.
- Generated factorization ≤2× QDLDL-class C on the same permuted matrix (QDLDL runs as a benchmark baseline only).
- Generation time linear in nnz(L) and < 10 s at nnz(L) = 50k.
- Dense kernels ≤2× LAPACK at n = 32–64.
- At least two non-PIQP examples: an SQP Newton step with a sparse KKT, and a Kalman update.

Kill criteria:
- If the generated sparse LDLᵀ stays > 3× QDLDL after one optimization round, stop and revisit the generated-only decision for this kernel.
- If dense kernels miss their gate, limit dense to unrolled n ≤ 16.

Draft PR sequence:

| PR | Content | Size | Status |
|---|---|---|---|
| T2-1 | Step index (T2.a) | S | done (C-101): `scan`/`while_loop(index=True)`, no table stored; int64 procedures scalarize |
| T2-2 | `take`/`put_add`/`put` (T2.b) | M | done (C-102): out-of-range lanes read `fill` or write a per-lane scratch slot |
| T2-3 | In-place proof at the scan (T2.c) | L | done (C-103): `in_place_steps` from the loop's index tables; `Aᵀy` at n = 2000 2283 → 51 µs |
| T2-4 | `SparseMatrix` (#11) | M–L | done (C-104): `scaly.linalg.SparseMatrix`, static CSC pattern + `Expr` values |
| T2-5 | Symbolic analysis (#12) | M | done (C-105): orderings incl. `auto`, etree, L pattern, left-looking lanes, DP segments; MMD fill = SuperLU's |
| T2-6 | Dense kernels (#17) | M–L | done (C-106): cholesky, ldl, trisolve; within 1.6× of OpenBLAS at `-O3` (C-107 open for `-O2`) |
| T2-7 | Sparse LDLᵀ and solves (#13–14) | L | done (C-108): `SparseLDL`, in-place column scans, implicit solve derivatives |
| T2-8 | Index-aware fusion and ragged loops (T2.d–e) | M–L | done (C-109): `ragged_add`/`ragged_dot`; factor 1.04–1.37× the C baseline |
| T2-9 | Health, refinement, schedule and options (#15, #16, #18); T2.f–g | M | done (C-110): unroll/scan schedules, fixed and adaptive refinement, `inertia`/`health`, `sc.options`, `custom_derivative(sparsity=)`, trajectory guard; T2.g assessed as not needed (C-112); SQP and Kalman examples |
| T2-R | Agent review round | M | done (C-113): 5 agents; 10 defects fixed (6 silent wrong-number), JIT call 4.2 → 3.2 µs, in-place adaptive refinement, C-114–C-117 recorded |

Per-PR reports: `internal/notes/tier2_pr{1..}_report.html`; timings: `internal/notes/perf_2026_09_26_tier2/`.

Results: `SparseLDL` factors within 1.04–1.5× of a QDLDL-class C factorization with loops, and 4–5× faster than it as straight-line code on small KKT systems (≤ 1000 operations). Solves run at 0.5–2.1× that baseline, with implicit derivatives to second order, refinement and inertia/health checks. Dense kernels are within 1.6× of OpenBLAS at `-O3`. Generation takes 1.2 s at nnz(L) = 46 k. The SQP Newton-step and Kalman-update examples pass. 1 460 tests pass. Summary: `internal/notes/tier2_review_report.html`.
Open Tier 2 items:
- C-107: the JIT default `-O3`, your decision.
- C-111, C-115, C-116: generation and compile time of unrolled graphs.
- C-112: loop-invariant `while_loop` inputs, needed by Tier 3.
- C-114: unpadded factor updates, 1.6–1.7× in a prototype.
- C-117: dense dot kernels and small solves.

## Tier 3 — generic primal–dual IPM machinery
| # | Item | Notes (corrections from the PIQP 0.6.2 sources) |
|---|---|---|
| 19 | Ruiz equilibration and unscaling | Early exit at max\|1−d\| ≤ 1e-3; `limit_scaling` (values < 1e-4 → 1, capped at 1e4); separate box scaling; cost scaling off by default; `while_loop` or a freezing scan |
| 20 | KKT assembly behind a KKT-solver interface | Box bounds fold into the x diagonal. Two-sided rows give one KKT row with W = [Σ 1/(s/z+δ)]⁻¹. Dense backend: condensed Cholesky C = P + reg + AᵀA/δ + GᵀW⁻¹G. Sparse backend: full KKT with LDLᵀ. PIQP's elimination modes are optional later |
| 21 | Fraction to boundary | τ·min(1, min over components with ds < 0 of −s/ds); no ε guard |
| 22 | Mehrotra predictor–corrector | σ = clip(·)³; one factorization and two solves per iteration |
| 23 | Residuals and termination | OR of absolute and relative tolerance; duality gap included; regularised and unregularised residuals both tracked |
| 24 | Initial point | One solve at ρ = 1e-6, δ = 1e-4; shift by max(0, −min); μ ≥ 1e-10; projection onto the central path; no margin |
| 25 | Infinite bounds | \|value\| ≥ 1e30. Rows infinite on both sides are kept (still count in μ). Masks via `where` only (1/z = ∞) |
| T3-0 | Harness and NumPy reference IPM | PIQP trace harness from the vendored headers; curated `tests/data` (small MM subset, infeasible LPs, SQP/MPC sets, Apache-2.0); a NumPy reference used as the test oracle. Can run in parallel with Tier 2 |

Gate: the NumPy reference reproduces vendored PIQP decision traces on ≥ 90% of the small MM subset (otherwise stop and diagnose). The Scaly IPM with the dense backend matches the reference.

## Tier 4 — PIQP-specific and delivery
| # | Item | Notes |
|---|---|---|
| 26 | Proximal ρ/δ and centre updates | Centres move only when a residual falls by ≥ 5%; `iter<5` and prox-infeasibility special cases; separate branch with no inequalities; fine-tune switch |
| 27 | Infeasibility detection | no-update counters, prox-inf threshold 0.9, statuses |
| 28 | Boundary shift | ε added to all box duals if any is below ε |
| 29 | Factorization retry and refinement | The first failure turns refinement on; later failures scale ρ and δ by 100, up to 10 times. Documented deviation: non-finite solves also trigger a retry |
| 30 | Settings folded to constants; results and status parity; native `SolverStats` | `backend="scaly"` as a plain Function, not a SOLVER_CALL |
| 31 | Drop-in backend | 0.6.2 has no warm start; match `restore_dual` |
| 32 | Differential tests and benchmarks | Decision traces plus tolerances (not bitwise); MM subset; SQP corpora; closed-loop swap; reference machine per `docs/results/fairness.md` |
| — | Sensitivities | Implicit QP derivatives reusing the final factorization |

Gates:
- Status parity on MM with n ≤ 100.
- Mean iteration increase < 10% on the SQP corpus.
- Default switch only with no status regressions.

## Tier 5 — structure exploitation (later)
- Multistage/Riccati KKT: periodicity detection, then a condensed KKT factored by a scan over stages with dense blocks.
- Supernodal sparse LDLᵀ.
- Gate: ≤ 1.5× vendored multistage on chain/race_cars.

## Tier 6 — platform selection and optional external kernels (later, only if justified)
- A target description plus a selector over the lowering options, with optional measured tuning cached per (pattern, size, target).
- External kernels (BLASFEO/LAPACK/QDLDL) as optional backends, only if a measured gap justifies a second deployment path.

## Benchmarks (all tiers)
- **Metrics:** native solver time and Google Benchmark kernels; iterations; build/render/compile time; code size; workspace; generation RSS.
- **Baselines** (not shipped): vendored PIQP in dense, sparse and multistage modes; QDLDL/PIQP LDLt; LAPACK; OSQP/Clarabel/HPIPM optionally.
- **Problem families:** random dense and sparse QPs; the MM subset; linear MPC (nx ∈ {4, 12, 27}, N ∈ {10…100}); SQP corpora (chain, race_cars, npmpc, unbumpercars).

## Decisions
**Taken (2026-09-26):**
- Generic sparse before multistage (multistage moves to Tier 5).
- Generated code only (external kernels move to Tier 6).
- Tier 2 is general sparse linear algebra, not PIQP-private.

**Deferred to the tier that needs them** (none blocks Tier 2):
- *Tier 2 default, no decision needed:* ordering by SciPy's SuperLU MMD; an AMD port only if measured fill demands it. Tier 2 benchmark baselines (QDLDL-class LDLt) are built from the vendored headers in the VM, so no Mac is needed.
- *Decide at the start of Tier 3:*
  - a NumPy reference IPM plus a curated Maros–Mészáros subset in `tests/data` (Tier 2 may reuse MM KKT patterns as test matrices without this);
  - the dense backend as condensed Cholesky on the Tier 2 dense kernels.
- *Decide at the start of Tier 4:*
  - parity semantics (decision traces plus tolerances);
  - Mac-only vendored baselines and the SQP corpus dump.
- *Independent Tier 1 carry-over:* the `segment` merge; it can be scheduled at any time.

## Change log
- 2026-09-26 v3: plan revised after Tier 1 and a three-agent review.
- 2026-09-26 v3.1: generic sparse early, multistage later.
- 2026-09-26 v4: generated-only; plan re-organised into Tiers 1–6; Tier 2 made general-purpose sparse linear algebra; git tag `tier1-complete` marks the end of Tier 1.
- 2026-09-26 v4.1: the PIQP-specific pending decisions moved to Tiers 3–4; nothing blocks Tier 2.
- 2026-09-26 v4.2: T2-1 … T2-5 landed (C-101 … C-105).
- 2026-09-26 v4.3: T2-6 … T2-9 and the T2-R review round landed (C-106 … C-113); Tier 2 closed with tag `tier2-complete`.

## References
- Schwan, Jiang, Kuhn, Jones, PIQP, CDC 2023 — https://arxiv.org/abs/2304.00290
- Mehrotra, SIAM J. Optim. 2(4), 1992 — https://doi.org/10.1137/0802028
- Davis, Algorithm 849 (LDL), ACM TOMS 31(4), 2005 — https://doi.org/10.1145/1114268.1114277
- Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006 — https://doi.org/10.1137/1.9780898718881
- Amestoy, Davis, Duff, AMD, SIAM J. Matrix Anal. Appl. 17(4), 1996 — https://doi.org/10.1137/S0895479894278952
- Stellato et al., OSQP/QDLDL, Math. Prog. Comp. 12, 2020 — https://doi.org/10.1007/s12532-020-00179-2
- Rao, Wright, Rawlings, JOTA 99(3), 1998 — https://doi.org/10.1023/A:1021711402723
- Frison, Diehl, HPIPM, IFAC 2020 — https://arxiv.org/abs/2003.02547
- Maros, Mészáros, QP test set, Optim. Methods Softw. 11, 1999 — https://doi.org/10.1080/10556789908805768
