# Unbumpercars seed-axis regression investigation

The neural-network derivative matrix products explain most of the unbumpercars loss at PR #166. A local change to matrix-product lowering recovers near-parent closed-loop times without changing the derivative traversal or scalarization budgets. The separate chain M=5 Jacobian loss remains.

## What is measured

All timings are on idle la015 with all sixteen governors at performance and boost disabled. Initial inspection found no tmux session, benchmark runner or measurement lock. The IPOPT and sequential quadratic programming (SQP) backends use these oracle procedures. Native controller C uses system Clang 20.1.8, `-O2 -march=native -fno-math-errno -mno-gather -mno-scatter -fveclib=libmvec`, and glibc vector math. The requested sweep cells use their ordinary Clang `-O3` recipe. The same IPOPT and PIQP binaries are pinned throughout, with the hashes from the earlier October 29 controls.

Direct oracle timings use C loops calling the actual raw functions from a solver-bearing module. They reuse the parent IPOPT episode's saved unbumpercars sample and keep inputs fixed. The base function returns objective and constraints together, and both solver requests call it. It is timed once as that combined function. Native output buffers are checked against the parent. Headline oracle means use three fresh executable processes, without internal clocks. Separate instrumented builds time procedure bodies and outer loops. Those inclusive clocks perturb compilation and runtime, so their values locate costs and are not substitutes for the uninstrumented timings.

The final validation uses two fresh benchmark processes and compilation caches per revision/cell, with reverse revision ordering in the second block. It includes only the requested controller modes and two Hessian sweep cells, plus the culprit revision for attribution. It is not a new full benchmark study. Coefficients of variation use sample standard deviation divided by the process mean. Two processes give little information about the tails of the timing distribution.

## Mechanism

The normalized parent neural forward helper contains two separate matrix-vector products for its two control seeds. The new helper contains `W @ D.T`, including a `(128,256) @ (256,2)` product. The parent's matrix-vector lowering computes four rows at once. The new matrix-matrix lowering instead gives each row only two ordered accumulators. The generated machine code includes a dependent two-double fused multiply-add chain. The matrix operation count for this forward helper stays exactly 103,680 products, so its Jacobian loss cannot be explained by extra structural-zero products.

The forward-over-adjoint helper additionally changes separate vector-matrix products into `(2,128) @ (128,256)` and `(4,128) @ (128,256)`. PR #166's four-column panel revisits the reduction for each small column panel. Diagnostic clocks locate the largest added costs in these contractions and the narrow two-column product. Four output rows at once for a narrow matrix and eight contiguous output columns for a wider matrix give the compiler more independent outputs per reduction. Each output still accumulates in increasing reduction-index order.

There is also a measured zero-work contribution. In the new hoisted helper, the constant `k97[12]` is a 4-by-3 identity with an all-zero fourth row. This row continues through the neural adjoint derivative. The normalized forward-over-adjoint helper rises from 308,736 to 343,424 contraction products, exactly one network's 34,688 products, or 11.2%. The prototype preserves it. Removing all such internal seed/entry zeros belongs to the larger support-aware lowering work. The measurements do not establish a 95% zero fraction for this controller.

The same procedures remain unscalarized on both sides. This agrees with the maintainer's decision trace and rules out an unbumpercars scalarization-budget crossing. Movement still materializes some seed-axis transposes; the measured hot loops are the contractions. The prototype changes no fusion, lane-width policy, workspace pass, derivative rule, or scalarization limit. Its causal evidence is the controlled change in just contraction lowering and the resulting direct and closed-loop recovery. Exact attribution to cache behavior or processor pipeline stalls has not been measured with hardware counters.

## Prototype and validation

The production diff is confined to `src/scaly/passes/lowering/contraction.py`. It gives narrow products four-row panels and wider products eight-column panels. Four extra rectangular and tail cases extend the existing exact ordered-reduction test in `tests/passes/lowering/test_lowering.py`; `tests/baseline/pytest_nodeids.txt` records those cases. An initial experimental tail form exposed top-level mutable initializers to invariant hoisting and failed C compilation. It was discarded before the final prototype. The final form keeps accumulators inside row loops.

The required full suite passed with 2,759 passed and 53 skipped. Pinned Ruff lint, formatting and `ty check --error-on-warning` passed. The new five-row/two-column case failed on array equality when the accumulator initializer was deliberately changed from zero to one, then passed after exact restoration. All direct oracle outputs agree with the parent within `rtol=1e-11, atol=1e-10`.

The wide-contiguous-loop-only trial recovered too little. A sixteen-column panel was slower than eight. These rejected variants were not kept. The prototype is an uncommitted three-file diff against current main. No commit, push, PR, published page or full study is part of this task.

## Chain M=5

The direct fixed-input loss is in the Jacobian, about 17.8 to 32.2 microseconds at the culprit. The Hessian stays around 30 microseconds. Its old `chain_eq_stage_M5_fwd_612c7f031f` is one scalarized stage. After #166 it calls `chain_step_M5_fwd_28ec9a5da7`, which invokes the ODE derivative three times and materializes several 24-by-21 tangent arrays. The general ODE derivative sees runtime tangent buffers, which prevents the old whole-stage constant/zero folding and cross-call reuse. The generated body has no neural-style matrix-matrix contractions for the prototype to improve.

The direct chain diagnostic uses deterministic `sample_inputs(5,12,seed=0)` and fixed multipliers from 0.1 to 1.0. It does not reuse `representative_fe_inputs.npz` from the old controller artifact: that file deliberately contains a horizon-40 sweep sample, while the closed-loop controller's horizon is 12. The required episode comparisons use the unchanged canonical controller.


The scalarization decision trace identifies the exact M=5 boundary. Before #166, `chain_step_M5_fwd_d0cbed49b3` has estimated expansion work 5,977 and scalarizes. Its 24 seeded calls then fold into the explicitly scalar stage helper. After #166, `chain_step_M5_fwd_28ec9a5da7` carries all 24 directions at once and has work 76,405, above `AUTO_EXPANSION_WORK_PER_PROC = 65,536`. The pass skips that automatic callee. The explicitly scalar stage then also cannot expand because its callee is ineligible. This is an expansion-work crossing, distinct from M=9's 4,096-operation limit and from unbumpercars' matrix schedules. The detailed JSON decision traces were not kept. No limit was raised.

## Procedure and loop clocks

These are inclusive diagnostic times in microseconds, separately instrumented and not the raw-oracle timing builds. Each neural helper row aggregates eight car calls during one oracle evaluation. They confirm where costs move, but instrumentation changes compiler choices and absolute totals.

| Procedure or loop | Parent | #166 | Prototype |
| --- | ---: | ---: | ---: |
| `ctdt_dt_mlp_step_adj_7027c40181_fwd_pack_07b778dce8_hoisted_2_4_5`, Hessian | 359.9 | 875.5 | 473.0 |
| `ctdt_dt_mlp_step_fwd_67b047f677_hoisted_2_3_4`, Jacobian | 219.8 | 306.1 | 159.3 |
| Narrow `(128,256) @ (256,2)`, Jacobian forward helper | Two matvec loops, 131.5 combined | 220.9 | 73.8 |
| Wide `(4,128) @ (128,256)`, Hessian helper | Separate vector products | 358.9 | 183.9 |

The inspected C, intermediate representation and assembly showed the operand layouts and loop schedules. Those investigation artifacts were not kept. Hardware performance counters were not collected, so the processor-level explanation about hiding ordered-accumulator latency is an inference from the emitted instructions and controlled lowering variants.

## Direct unbumpercars oracles

Mean microseconds and process coefficient of variation, three executable processes per cell. The gradient is below 0.01 microseconds; its relative variation is dominated by the short timing interval.

| Raw oracle | Parent | #166 | Current main | Prototype |
| --- | ---: | ---: | ---: | ---: |
| Objective and constraints | 75.962 (0.26%) | 76.318 (0.66%) | 75.977 (0.22%) | 75.978 (0.09%) |
| grad | 0.006 (3.21%) | 0.006 (0.47%) | 0.006 (0.71%) | 0.006 (2.67%) |
| jac | 282.282 (0.20%) | 356.942 (0.04%) | 355.078 (0.10%) | 223.362 (0.22%) |
| hess | 660.539 (0.35%) | 1390.067 (2.58%) | 1375.830 (2.49%) | 770.417 (0.11%) |

## Required controller and sweep cells

Means and process coefficients of variation, two fresh benchmark processes and caches per cell. Controller compiler is Clang, so these absolute numbers should not be spliced into the earlier GCC confirmations.

| Cell | Unit | Parent | #166 | Current main | Prototype |
| --- | --- | ---: | ---: | ---: | ---: |
| unbumpercars ipopt step | ms | 20.062 (0.66%) | 31.829 (7.28%) | 30.293 (0.14%) | 20.915 (0.56%) |
| unbumpercars sqp step | ms | 10.665 (0.76%) | 17.503 (0.31%) | 17.311 (0.08%) | 10.914 (0.02%) |
| chain ipopt function evaluation | ms | 0.399 (0.48%) | 0.543 (5.68%) | 0.518 (0.93%) | 0.530 (1.21%) |
| chain sqp function evaluation | ms | 0.173 (0.14%) | 0.215 (1.01%) | 0.219 (1.48%) | 0.222 (0.75%) |
| chain none hess | us | 287.881 (2.71%) | 4742.288 (0.14%) | 3835.478 (15.28%) | 3753.462 (12.52%) |
| race_cars none hess | us | 24.020 (5.42%) | 25.210 (0.97%) | 29.014 (3.77%) | 29.592 (0.91%) |

All 16 compiled-C sweep gates passed. All 4,640 solver steps in 32 episodes succeeded. The chain controller and both control sweep kernels remain slower than the parent. The prototype does not change their generated C; both Hessian kernel objects are byte-identical to current main. The noisy M=9 timings, with 12-15% current/prototype process variation, establish no improvement from this patch. The race-car source and object are also unchanged. The separate source and object hash record was not kept.

For unbumpercars IPOPT, prototype and parent iterations and all oracle counts agree on every step in both blocks. SQP differs on steps 30, 148 and 198. The equal-work means below exclude every step with any count or iteration mismatch.

| Block | Equal-work steps | Parent SQP step, ms | Prototype SQP step, ms |
| --- | ---: | ---: | ---: |
| 1 | 197 | 10.366 | 10.703 |
| 2 | 197 | 10.478 | 10.699 |

Neither prototype IPOPT nor SQP misses the 100 ms deadline in either episode. All prototype and parent chain episodes have equal iteration and oracle counts. The separate summary of work comparisons, complete process values and other revision checks was not kept.

## Build cost

The direct solver module includes all embedded oracles. Generation and native runner compilation are single observations, not build-time uncertainty estimates.

| Revision | Render, s | Native runner compile, s | C bytes | Workspace doubles |
| --- | ---: | ---: | ---: | ---: |
| parent | 3.523 | 2.038 | 545,910 | 17,248 |
| seed | 3.103 | 2.877 | 470,545 | 21,536 |
| main | 2.815 | 2.773 | 359,594 | 21,536 |
| fix | 2.813 | 2.721 | 364,846 | 20,512 |

The prototype adds 1.5% to current main's solver source, reduces workspace by 1,024 doubles, and has similar generation and compilation time. Actual cold controller construction and compilation also remain bounded.

| Controller | Parent mean build, s | Current main | Prototype |
| --- | ---: | ---: | ---: |
| unbumpercars ipopt | 6.264 | 5.925 | 5.982 |
| unbumpercars sqp | 10.558 | 9.343 | 9.279 |
| chain ipopt | 15.529 | 12.255 | 12.283 |
| chain sqp | 16.463 | 12.506 | 12.566 |

The separate per-process sweep generation and compilation table was not kept. No scalarization budget changed and no straight-line expansion was added.

## Recommendation and retained work

Use the local matrix-lowering fix to address the unbumpercars regression on the measured Clang setup. It preserves the single traversal and returns closed-loop steps close to the parent. The remaining raw Hessian gap, chain M=5 Jacobian loss, chain M=9 loss and race-car regression still need their own release decision. This result removes unbumpercars as a reason to revert the entire derivative traversal; it does not establish that all derivative-engine changes are suitable for 0.1.0.

The prototype remains uncommitted in a temporary worktree, based on `435410e175a7b26213d3973c594276e5904bb0e0`. The temporary patch was not kept. Its recorded SHA-256 was `5f0de7d19f5eeb10b91d1eaf06a2c0f341a5af1a172a04557c947827fb081799`. It touches only contraction lowering, the existing lowering test, and the pytest collection baseline. No commit, push or PR was made.

The investigation scripts, specifications, derived measurements, logs, validation proof and compressed code evidence were not kept. Raw study data remains in `benchmarks/results/`. The original maintained checkout and preview checkout were not used for implementation or benchmark output. Only these authorized uncommitted research notes were added to the maintained checkout.

The findings were posted on [issue #85](https://github.com/PREDICT-EPFL/scaly/issues/85#issuecomment-6497794659). All three comparison worktrees were removed; the prototype worktree and raw evidence directory remained at the end of the investigation. Final hardware checks still show performance governors, boost off, no tmux session, no benchmark process and no measurement lock. The preview checkout contains modified benchmark CSV assets belonging to other work; this task made no edits there.
