# October 29 compiler controls and regression attribution

All three investigated slowdowns remain with October 20's GCC controller configuration.
The measured commit boundaries are PR #212 for the race-car Hessian at N=400, and PR #166 for
unbumpercars step time and chain function evaluation. Switching the published controller compiler
to GCC would not remove the observed 100 ms deadline misses.

## Revisions, environment and method

October 20's provenance names the October 20 study revision and
`SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++`. Its controllers used GCC 13.3.0,
but its C sweep kernels already used Clang 20.1.8. Setting `SCALY_CC=gcc` does not change the
sweep compiler. I reproduced that historical environment and separately recompiled the retained
C kernels with GCC. Both providers used Clang in the release study.

The controls measured clean `a3cc544c4e952520c747b7f4c4cef59074303a61` in a detached scratch
worktree. Three paired blocks used the historical environment and
`SCALY_VECTOR_LIBM=glibc SCALY_CC=clang CC=clang CXX=clang++`. Every command retained
`--repetitions 1 --order-seed 0 --headline --boost off`. Sweeps retained a 0.5-second minimum.
Additional targeted repetitions investigated high variation. Each controller process had a fresh
compilation cache. Direct kernel checks used unchanged saved C, samples and correctness-gated
Google Benchmark wrappers, with three fresh timing processes per compiled kernel.

La015 initially had no tmux sessions, benchmark processes or measurement lock. All sixteen
governors were performance, boost was off, affinity covered CPUs 0 through 15, and load was
0.02/0.04/0.08. Named tmux jobs held the measurement lock and checked the CPU policy. No competing
tests, builds or measurements ran. Main stayed at `435410e175a7b26213d3973c594276e5904bb0e0`.

The bisections searched the 25 first-parent main commits after the historical SHA. The first,
`fd427d28`, has identical executable files to the historical study revision. The searches used
N=400 for race cars, and canonical Scaly SQP episodes for chain M=5 and unbumpercars C=8.
There were four race-car and five points for each controller after measuring both endpoints.
Every isolated commit and its parent were then measured twice each. Controller confirmations
also measured IPOPT twice per revision.

The revision comparisons pinned the exact existing IPOPT and PIQP libraries from the control
checkout through `SCALY_IPOPT_LIB` and `SCALY_PIQP_LIB`. All point provenance is clean and names
the intended SHA. Every recorded IPOPT binary has SHA-256
`55b716d797e76e52d8e86e36103879775d24ec202423a5c40ca2c76d6950f52e`.
The fixed PIQP binary has SHA-256
`ca7c537817a8dc4d0a4291e15b6a9dbb4e76563e1f213e35c726c5fcc60a004b`.
The IPOPT build is 3.14.19 with MUMPS 5.8.2, METIS 5.2.1 and OpenBLAS 0.3.28.

## Measured parent and commit confirmations

Numbers are means of two fresh processes. Parent and commit coefficients of variation are sample
standard deviation divided by the mean, expressed as percentages. They describe process
dispersion, not the changing work over an episode and not confidence intervals.

| Quantity | Parent | Commit | Parent mean | Commit mean | Change | Parent CV | Commit CV |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Race-car N=400 Hessian, µs | ff17923b | 71f8188d, [PR #212](https://github.com/PREDICT-EPFL/scaly/pull/212) | 25.507 | 28.168 | +10.4% | 0.96% | 1.26% |
| Unbumpercars C=8 IPOPT step, ms | dd77f3bc | c99ba3c5, [PR #166](https://github.com/PREDICT-EPFL/scaly/pull/166) | 20.624 | 37.435 | +81.5% | 0.39% | 0.21% |
| Unbumpercars C=8 SQP step, ms | dd77f3bc | c99ba3c5, PR #166 | 10.921 | 19.627 | +79.7% | 0.10% | 0.03% |
| Chain M=5 IPOPT function evaluation, ms | dd77f3bc | c99ba3c5, PR #166 | 0.402576 | 0.505411 | +25.5% | 0.31% | 0.02% |
| Chain M=5 SQP function evaluation, ms | dd77f3bc | c99ba3c5, PR #166 | 0.180790 | 0.214252 | +18.5% | 0.24% | 0.41% |

Full culprit SHAs are `71f8188d4702995d89722d35018347224fc7b966` and
`c99ba3c563a29c8348f8d1b2e3a9d2c43f8d43c7`. Full parents are
`ff17923baa1a01d20b0934109e33667334a6ced9` and `dd77f3bcd5870bde09f8ad396612ff3c41a2bcdd`.
The race-car kernel uses the historical Clang sweep compiler. All controller confirmations use GCC.

Chain has identical iterations and oracle counts across each confirmed pair. Unbumpercars IPOPT
does too. Unbumpercars SQP differs on steps 30, 148 and 198. On the other 197 equal-work steps,
the two parent means are 10.710 and 10.689 ms, and the commit means are 19.050 and 19.147 ms.
Its slowdown therefore persists after excluding every step with a work mismatch.

The race-car result establishes a repeatable contribution from #212 at N=400. It does not
establish that this commit alone explains the entire historical 32.0% gap, or independently
identify the culprit at every horizon. The fresh historical endpoint took 24.453 µs, rather than
the original 22.951 µs. The measured source and compiler comparisons establish a code regression;
the precise decomposition of the full historical gap remains unresolved.

## Compiler controls at the release revision

Every race-car horizon remains slower than October 20 under the historical environment. Both
columns below actually compile their sweep C with Clang. The labels describe the controller
compiler environment. Entries are mean µs followed by coefficient of variation.

| N | October 20, µs | GCC environment | Clang environment | Processes per environment |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.851 | 1.007, 1.13% | 1.001, 0.46% | 3 |
| 10 | 1.108 | 1.179, 0.58% | 1.193, 2.58% | 3 |
| 25 | 1.984 | 2.371, 0.84% | 2.381, 2.11% | 3 |
| 50 | 3.281 | 4.018, 2.29% | 4.095, 3.85% | 6 |
| 100 | 6.380 | 7.759, 0.40% | 7.840, 1.39% | 3 |
| 200 | 11.797 | 15.131, 4.54% | 15.206, 3.56% | 6 |
| 400 | 22.951 | 28.841, 3.87% | 30.712, 2.51% | 3 |

Directly compiling both retained N=400 kernels with GCC also preserves a loss. October 20's C
takes 25.271 µs with a 1.95% coefficient of variation, and release C takes 30.504 µs with 1.12%,
a 20.7% increase. Each uses three fresh timing processes with fixed saved inputs.

| Release-revision controller quantity | October 20, ms | GCC mean, ms | GCC CV | Clang mean, ms | Clang CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unbumpercars IPOPT total | 20.120 | 34.422 | 9.08% | 33.756 | 0.64% |
| Unbumpercars IPOPT function evaluation | 16.204 | 30.451 | 10.24% | 29.818 | 0.65% |
| Unbumpercars SQP total | 10.944 | 19.052 | 0.33% | 17.290 | 0.32% |
| Unbumpercars SQP function evaluation | 9.656 | 17.773 | 0.35% | 16.015 | 0.38% |
| Chain IPOPT function evaluation | 0.396415 | 0.511073 | 0.45% | 0.517447 | 1.08% |
| Chain SQP function evaluation | 0.180320 | 0.218511 | 0.49% | 0.219232 | 1.61% |

There are three processes per cell, except six for GCC unbumpercars IPOPT after its noise rerun.
That cell remains noisy, with total means between 32.342 and 38.463 ms. Its exact mean and the
IPOPT compiler ordering are not settled headline results. The large code regression is separately
confirmed by the low-dispersion parent/commit experiment. The fresh Clang IPOPT mean also differs
from the original release study's 30.365 ms. These controls do not explain every absolute shift
between launches. Original study records remain unchanged.

All generated Scaly controller C files match byte for byte between the compiler environments
and across repeated blocks. IPOPT work matches across compilers. Current unbumpercars SQP differs
between compiler environments in iterations on steps 148 and 198, and in objective/constraint
counts also on step 30. Its compiler comparison is an observed episode result with those small
work differences. Chain work matches throughout.

The headline unbumpercars Hessian does not reproduce its controller loss. A stricter fixed-input
GCC `-O2` check, using October 20's saved samples and reference for both C files, improves from
984.421 µs with a 0.13% coefficient of variation to 613.255 µs with 0.29%. That is why the code
search uses the full C=8 episode rather than the Hessian sweep.

## Deadlines and the excluded M=9 investigation

GCC does not remove the 100 ms deadline misses. All three release-revision GCC SQP episodes miss
on steps 92 and 169. Across them those steps take 108.762 to 156.110 ms. Clang also has two misses
per episode, at the same steps, with times between 101.614 and 137.621 ms. All steps succeed as
solver calls. A successful solve can still miss its sampling deadline.

The known M=9 regression persists under GCC. On the unchanged release C and saved samples,
GCC takes 6,460.681 µs with a 0.155% coefficient of variation, and a paired Clang recompile takes
3,420.274 µs with 0.031%, each over three fresh processes. GCC therefore changes the absolute
time and makes this kernel slower; it does not restore October 20's 282.288 µs. The original
4,305.881 µs Clang study sample and the initially repeated full-cell runs also show launch
variation. No M=9 code bisection or investigation of the already recorded #214 cause was done.

## Recommendation and limits

Measured conclusions are the persistence of the losses under GCC, the parent/commit changes
above, the fixed-input kernel results, work counts, gates and deadline misses. Attributing the
other race-car horizons to #212's common path, and choosing a compiler for future benchmark studies,
are inferences. This work does not prove that #212 accounts for the full historical race-car
gap or predict performance after a repair.

My recommendation is to retain the release study's Clang 20.1.8 configuration for both providers,
with glibc vector math, native targeting, `-O3` for sweep C and `-O2` for controllers, including the
supported no-gather/no-scatter controller tuning. GCC is not a remedy for these regressions.
Clang has the lower measured unbumpercars SQP mean and much less IPOPT dispersion in these
controls. This is a recommendation for the tested source, not a universal compiler ranking.
Before the next release, address or explicitly accept the code losses and run one complete study at
the final revision with the chosen configuration. Do not splice these partial control cells into
the existing release study or publish them as its replacement.

The audit covers 60 control sweep cells, two preflight sweep cells, ten bisection sweep cells,
36 fixed-source kernel launches and 57 controller episodes with 8,430 solver steps. All timed
compiled-C gates passed and every solver step succeeded. There were no timed-cell compilation
timeouts, runtime failures or skipped cells. Both compiler smoke runs passed. Five research
scripts pass the pinned Ruff formatting and lint checks.

The retained setup interruptions were an archived-path link failure in the supplementary job
and an explicitly stopped native-solver setup before the first bisection timing. The smaller
corrected jobs completed. Neither interruption is a correctness-gate failure or a regression
measurement.

## Reproduction and artifacts

All paths below are on la015. Outputs are outside the original study directory.

- `benchmarks/results/control-2026-10-29-gcc/` and `control-2026-10-29-clang/` contain full cells,
  fresh compilation caches, episode telemetry and provenance. Blocks 1 through 3 are paired.
  Blocks 4 through 6 hold the targeted noise checks.
- `benchmarks/results/control-2026-10-29-kernels/` contains fixed-source compiler checks.
- `benchmarks/results/control-2026-10-29-bisect-fixed/` contains every bisection point and the
  repeated confirmations, with exact commands, logs, provenance and library hashes.
- Launcher logs end in `control-2026-10-29-job.log`, `control-2026-10-29-kernels.log`,
  `control-2026-10-29-refine-job.log` and `control-2026-10-29-bisect-fixed.log`.
- Each job's `measurement.json` records commands, environments, timestamps, status and log paths.
  Each bisection's `bisection.json` records its complete range, points and confirmations.

The investigation scripts, specifications and derived audits in this note's directory were not
kept. The raw study files remain in `benchmarks/results/`.
No project source, published page, benchmark asset or comparison-history entry was changed.
No commit or pull request was created.

The findings were posted on
[issue #85](https://github.com/PREDICT-EPFL/scaly/issues/85#issuecomment-6496017593).
