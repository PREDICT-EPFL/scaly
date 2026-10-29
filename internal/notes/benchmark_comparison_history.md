# Benchmark comparison history

This note preserves the cross-study comparisons and configuration audits moved out of the public
results on 2026-09-24. The public results describe the latest complete study. Historical findings
below retain their original context and are not claims about the latest compiler.

The retained studies are `benchmarks/results/study-2026-09-23-c77-c79-final` for scalar libm and
`benchmarks/results/study-2026-09-23-c77-c79-libmvec` for vector libm. The September 10 study is the
comparison baseline used below. Actionable follow-up remains in [GitHub Issues](https://github.com/PREDICT-EPFL/scaly/issues).

## Final 0.1.0 study, October 29

`benchmarks/results/study-2026-10-29-final/` in the `ted-study-0.1.0-final` worktree measured
clean main at `06e10c16`, after #245, on la015. All 28 provenance records name that revision.
The study took 81.4 minutes. Both providers' controllers and the sweep used system Clang 20.1.8,
with `SCALY_VECTOR_LIBM=glibc SCALY_CC=clang CC=clang CXX=clang++`. The study options were
`--repetitions 1 --order-seed 0 --headline --boost off`. Controllers used `-O2`, sweep kernels
used `-O3`, and both used `-march=native`, glibc vector math and the `performance` governor.
The grids, 180-second compile limit, solver builds and timing boundaries follow the
[benchmark protocol](benchmark_protocol.md). This study's controller compiler supersedes the
GCC configuration described in that note's older studies.

All 106 sweep cells have outcomes, with 79 timed, 11 compilation timeouts and 16 larger cells
skipped after a timeout. There are no correctness or execution failures. All 21 Scaly cells
completed. All exclusions belong to CasADi. These counts match October 20. The earlier October
29 study had 77 timings and 18 skips because CasADi mapped SX did not compile neural-process
MPC at N=25 and N=50. In the final study those compiles take 173.7 and 177.1 seconds, near the
limit. That change in coverage does not establish a runtime improvement.

All 79 timed cells retain their compiled-C correctness success marker. All 15 closed-loop
episodes and 6,938 solver steps succeeded. The seven provider pairs have matching step numbers
and state/control differences below 4e-8. Six pairs match every iteration and oracle-call
count. Unbumpercars SQP differs in iterations at step 198 and in oracle-call counts at steps
30, 148 and 198, counting from zero. Its maximum state/control differences are 3.70e-8 and
2.82e-8. Those counts come from `stats.csv`, because unbumpercars `telemetry.csv` omits the
oracle-count columns and the standard report treats missing counts as equal. The published
step CSV includes those retained counts, and the pages qualify the SQP comparison.

The headline Hessian ratios divide the fastest completed CasADi time by Scaly's. Race cars
N=400 is 5.73×, neural-process MPC N=200 is 21.6× and unbumpercars C=8 is 25.5×. Chain is
1.20× at M=3, 0.85× at M=5 and 0.55× at M=9. No tested CasADi encoding compiles at
unbumpercars C=16 or C=32, so those sizes have no runtime ratio.

Scaly Hessian times against the previously published October 20 study:

| Problem | Size | October 20, µs | Final October 29, µs | Change, % |
|---|---:|---:|---:|---:|
| race_cars | 5 | 0.851 | 0.999 | +17.3 |
| race_cars | 10 | 1.108 | 1.180 | +6.5 |
| race_cars | 25 | 1.984 | 2.359 | +18.9 |
| race_cars | 50 | 3.281 | 4.107 | +25.2 |
| race_cars | 100 | 6.380 | 7.725 | +21.1 |
| race_cars | 200 | 11.797 | 16.136 | +36.8 |
| race_cars | 400 | 22.951 | 28.531 | +24.3 |
| npmpc | 6 | 10.532 | 7.790 | -26.0 |
| npmpc | 12 | 23.340 | 16.796 | -28.0 |
| npmpc | 25 | 48.643 | 32.230 | -33.7 |
| npmpc | 50 | 77.684 | 59.497 | -23.4 |
| npmpc | 100 | 162.526 | 107.083 | -34.1 |
| npmpc | 200 | 253.473 | 215.235 | -15.1 |
| unbumpercars | 2 | 227.132 | 113.667 | -50.0 |
| unbumpercars | 4 | 280.510 | 167.151 | -40.4 |
| unbumpercars | 8 | 575.297 | 415.742 | -27.7 |
| unbumpercars | 16 | 1462.377 | 1454.337 | -0.5 |
| unbumpercars | 32 | 5533.094 | 7644.897 | +38.2 |
| chain | 3 | 22.025 | 21.525 | -2.3 |
| chain | 5 | 100.177 | 96.314 | -3.9 |
| chain | 9 | 282.288 | 4317.256 | +1429.4 |

The maintainer accepted the chain losses for 0.1.0. At M=5, Scaly's mean function evaluation
rises from October 20's 0.396 to 0.518 ms with IPOPT and from 0.180 to 0.223 ms with SQP.
The native total means move from 6.848 to 6.977 ms and from 6.345 to 6.379 ms. Its Hessian
is slightly faster, so the controller loss is not a Hessian slowdown. The M=9 Hessian is
15.29× slower than October 20 and close to earlier October 29's 4,305.881 µs. The race-car
N=400 Hessian remains 24.3% slower than October 20. Its source and object are identical to
earlier October 29's, so the movement from 30.291 to 28.531 µs does not isolate a #245 gain.

The maintainer also accepted unbumpercars C=32 for 0.1.0. Its Hessian rose from 5,267.123 to
7,644.897 µs between the two October 29 studies, 45.1% slower with identical saved inputs.
It is outside the headline size and no tested CasADi encoding compiles there. The pages retain
the measured point without making the cross-study loss a talking point. At C=8, final Scaly
native solver means are 20.782 ms with IPOPT and 10.885 ms with SQP, against 30.365 and
17.297 ms earlier that day and 20.120 and 10.944 ms on October 20. October 20 used GCC
13.3.0 for controllers, so that historical comparison combines code and compiler changes.
Some harvested sweep inputs also differ across studies, including C=8 between the two October
29 runs. October 20 and the final C=8 inputs agree. The final study contains Hessians only,
with no Jacobian timings. Prototype derivative measurements do not enter the published data.

Final paired per-step ratios, as the median with tenth and ninetieth percentiles:

| Controller | Solver | Total-time ratio | Function-evaluation ratio |
|---|---|---:|---:|
| Race cars N=50 | IPOPT | 1.09×, 1.07 to 1.10 | 2.79×, 2.67 to 2.87 |
| Race cars N=50 | SQP | 1.04×, 1.01 to 1.05 | 3.04×, 2.79 to 3.16 |
| Neural-process MPC N=12 | IPOPT | 1.19×, 1.18 to 1.20 | 1.75×, 1.72 to 1.77 |
| Neural-process MPC N=12 | SQP | 1.26×, 1.25 to 1.29 | 1.80×, 1.76 to 1.85 |
| Unbumpercars C=8 | IPOPT | 8.10×, 7.94 to 8.22 | 9.70×, 9.50 to 9.80 |
| Unbumpercars C=8 | SQP | 9.01×, 8.61 to 9.25 | 10.09×, 9.62 to 10.36 |
| Chain M=5 | SQP | 0.99×, 0.98 to 0.99 | 0.59×, 0.55 to 0.60 |

Scaly misses no sampling deadline across its 3,514 steps. CasADi unbumpercars misses 195 of
200 steps with IPOPT and 55 with SQP, against October 20's 186 and 48. The final Scaly
unbumpercars maxima are 43.137 ms and 80.900 ms, below the 100 ms interval. Both providers
meet every deadline on the other problems. The largest distance of a paired tenth or ninetieth
percentile from its median is 7.99%. These ranges describe changing work over one episode,
not measurement noise or confidence intervals. One process per sweep cell gives no estimate
of variation between processes.

The five published CSV assets in `docs/assets/benchmarks/` come only from this final study.
The four raw sweep CSVs and their summaries are concatenated. The closed-loop summary is the
report's output, paired ratios use `numpy.percentile` at 10, 50 and 90, and step rows use the
report's telemetry normalization with six significant digits, supplemented by unbumpercars
`stats.csv` oracle counts. Regenerating the report on a temporary copy of the saved inputs
reproduces the retained report and summary exactly. The source study worktree stays unchanged.
The four benchmark pages and all nine data-driven figures use these assets.

## Seed-selector bit tables, October 20 study

`benchmarks/results/study-2026-10-20` measured the clean pushed merge of main into pull request
#181's branch, after "Store 0/1 seed selector tables as bits" (#184) and "Fold a constant cube and
check lane strides with NumPy" (#182). It took 82.9 minutes on la015 with October 15's grid,
toolchain, solver libraries and CPU and numerical policies. Of 106 attempts, 79 were timed, 11
reached the compile limit and 16 were skipped after a timeout. CasADi mapped SX on npmpc compiled
at N=25 and N=50 in 179.2 and 179.4 seconds, where October 15 stopped it at N=25 and skipped
N=50, and then timed out at N=100. All 21 Scaly cells completed, all timed kernels passed both
gates, and all 6,938 solver steps succeeded.

#184 changes only the npmpc and unbumpercars Scaly kernels. Every race-car and chain cell, and
every CasADi cell, keeps identical source and object sizes. Scaly's npmpc object file shrinks at
every horizon, and the shrinkage grows with it, as the per-entry `double` tables become bits:

| npmpc N | Object, kB, Oct 15 → Oct 20 | Hessian, µs, Oct 15 → Oct 20 |
|---:|---:|---:|
| 6 | 83 → 64 | 11.15 → 10.53 |
| 12 | 148 → 113 | 25.22 → 23.34 |
| 25 | 175 → 116 | 53.00 → 48.64 |
| 50 | 271 → 123 | 77.83 → 77.68 |
| 100 | 428 → 131 | 177.31 → 162.53 |
| 200 | 695 → 97 | 256.84 → 253.47 |

No npmpc size is slower. The pull request's 5% slowdown at N=6 on a Core Ultra 9 285K does not
appear on la015, where N=6 is 5.6% faster. Unbumpercars objects shrink by 1 to 38% except at C=4,
which grows from 133 to 143 kB. Its Hessian is 6% faster at C=8 and 19% faster at C=16, and
within 1% elsewhere. Across all matching timed cells the October 20/October 15 runtime ratio has a
median of 0.9997 and 10th/90th percentiles of 0.9619/1.0100. The unchanged cell with the largest
move is again NPMPC MX at N=25, 15% faster, inside the 85 to 99 µs spread October 15's
retained-binary diagnostic found.

The npmpc and unbumpercars Scaly controllers generate different C, while the race-car and chain
controllers produce byte-identical C and shared libraries. The unbumpercars SQP pair now differs
in iterations on two of 200 steps instead of three, with oracle-call counts still identical and
trajectories within 6×10⁻⁹. Paired medians move by at most 2.4%, on chain function evaluation,
except race-car SQP function evaluation, 3.122× to 2.813×. Scaly's identical library there took
a median of 59 rather than 54 µs per step, which is variation between processes. The largest 10th/90th percentile distance from a
paired median is 6.7%, so the page's bound for the middle 80% of step ratios returns from 11% to
7%. Forty-eight CasADi unbumpercars SQP steps exceed the sampling time, against 50 on October 15.

The [audit and complete comparison tables](perf_2026_10_20/analysis.md) come from the tracked
publication script. [Its README](perf_2026_10_20/README.md) gives the reproduction commands and
artifact locations.

## Race-car horizons and controller refresh, October 15 study

`benchmarks/results/study-2026-10-15` measured the clean pushed branch for pull request #181
after "Sweep the race car from N = 5 to 400 and close its loop at N = 50". It took 76.7 minutes
on la015 with October 14's toolchain, solver libraries, CPU and numerical policies. Every
provenance file names the same measured revision. The later figure/prose-only commit was pulled
after measurement, before publishing the results.

The race-car grid is N=5,10,25,50,100,200,400 and the controller horizon is N=50. The other
grids are unchanged. Of 106 attempts, 77 were timed, 11 reached the compile limit and 18 were
skipped after a timeout. All 21 Scaly cells completed. All timed kernels passed their symbolic
and compiled-C gates, and all 6,938 solver steps in 15 episodes succeeded. October 14's raw
rows contain 87 timings, 11 timeouts and 18 skipped cells across 116 attempts. The ten fewer
timings are the removed N=1 and N=40 race-car cells. Replacing N=500 with N=400 changes no outcome
count. The earlier prose's 12 timeouts and 17 skipped cells miscounted those raw rows.

The new N=400 race-car Hessian takes 22.926 µs with Scaly and 165.883 µs with the fastest
CasADi encoding, SX, a measured ratio of 7.24×. Scaly's C source is 78,071 bytes and its
object 43,399 bytes, against SX's 8,242,871 and 3,581,303 bytes. The old N=500 data had been
relabelled N=400 as a temporary figure input. Those placeholders are replaced by actual N=400
measurements, and they are not a baseline for a performance change.

The N=50 race-car provider pairs agree in every step's iterations and oracle-call counts.
The SQP total and function-evaluation paired medians are 1.038× and 3.122×, with 10th/90th
percentiles of 1.012/1.055× and 2.889/3.256×. IPOPT's are 1.083× and 2.720×, with percentiles
of 1.073/1.093× and 2.627/2.788×. Each episode still takes 1,367 steps. These are different
NLPs from the N=40 controllers, so their timings cannot establish a compiler speedup.

Every unchanged cell keeps the same generated-source and object sizes. Across matching timed
cells, the October 15/October 14 runtime ratio has a median of 1.0001 and 10th/90th percentiles
of 0.9840/1.0102. The two largest changes are NPMPC MX at N=25, up 17.8%, and Scaly at N=200,
down 7.9%. A separate diagnostic reran each day's retained binary in three fresh processes.
Both MX binaries alternate between about 85 and 99 µs, with coefficients of variation of
9.4% and 9.3%. Their means differ by 0.2%. Both Scaly binaries now take about 254 to 263 µs,
with coefficients of variation of 0.9% and 1.4%. These checks establish measurement variation,
not a compiler change. Their data remain outside the published study.

All unchanged closed-loop native means stay within 0.7% except NPMPC IPOPT/Scaly, up 3.7% in
total while function evaluation stays within 0.2%. Its iteration and oracle-call counts still
match, and the paired total speedup's 10th percentile is 10.5% below its median. The page
therefore gives an 11% bound for the middle 80% of paired step ratios, rather than retaining
the earlier 7% claim. The unbumpercars SQP pair retains three iteration mismatches and identical
oracle-call counts. Fifty CasADi SQP steps exceed the sampling time, against 48 previously.
All Scaly controllers still meet their sampling times. A single process per controller gives no
estimate of between-process noise.

The [audit and complete comparison tables](perf_2026_10_15/analysis.md) come from the tracked
publication script. [Its README](perf_2026_10_15/README.md) gives reproduction commands, diagnostic
limits and artifact locations.

## Compact code and fastest CasADi encodings, October 13 study

`benchmarks/results/study-2026-10-13-post-179` measured clean pushed main at `38bd5896`,
including the removal of inlined callees, compact lane helpers and the fastest-encoding
closed-loop changes. The machine, native compilers, solver libraries, CPU policy and glibc
vector-math policy match October 12. The protocol changed from five processes to one.
The run took 103 minutes, with 138 sweep attempts and 15 episodes. All 103 timed cells passed
their gates; 15 compilation attempts timed out and 20 cells were skipped after a timeout.
There were no correctness or execution failures. Every Scaly cell completed, and every solver
step succeeded.

The previous study had 515 timings, 17 timeouts and 158 skipped cells across 690 attempts.
These counts cannot be compared directly across five processes and one. CasADi's mapped SX
at neural-process MPC N=25 and N=50 completed in only some October 12 processes and did not
complete in this study. The compile limit still determines which encodings count.

The complete tables, step statistics, paired ratios and episode agreement are retained in
[the reproducible audit](perf_2026_10_13/analysis.md). Its script compares against the October 12
summary files at main's `d9755969` and writes the three current published CSV assets from this
study only.

Measured executable-source reductions are about 23% for the race car beyond N=1, 64 to 65%
for neural-process MPC, 58 to 64% for unbumpercars, and 22 to 40% for chain. These byte counts
are deterministic artifacts. Kernel compile times change by less than 6%. Runtime does not
improve uniformly: unbumpercars Hessians are 7 to 32% slower, neural-process MPC ranges from
16% faster to 28% slower, and chain stays within 0.4% of its previous means. Several slowdowns
are far outside the October 12 dispersion: neural-process MPC N=12 rises 12.4% against a 0.46%
coefficient of variation, and unbumpercars C=4 rises 32.4% against 0.12%. These are likely runtime
regressions and warrant targeted follow-up. One process gives no new noise estimate, and this
study measures the combined changes without isolating either code-size pass. The smaller
source alone does not establish a runtime improvement.

CasADi's SQP function evaluation changes most: chain falls from 8.121 to 0.160 ms per step,
race cars from 0.220 to 0.122 ms, and unbumpercars from 105.636 to 86.024 ms. Neural-process MPC
already used the selected encoding and stays at 1.026 ms. Scaly closed-loop totals stay within
1.7% of the previous means. Chain's median paired total ratio is 0.991×, with CasADi faster in
function evaluation. Race-car SQP's median total ratio is 1.038×, its 10th/90th percentiles are
1.019/1.047×, and its median function-evaluation ratio is 2.956×. Unbumpercars SQP's total ratio
is 8.103×, with 10th/90th percentiles 7.554/8.385×.

All seven provider pairs have matching step numbers and identical oracle-call counts. Six have
identical iteration counts too. Unbumpercars SQP has three iteration mismatches across 200 steps,
against 15 across 1,000 previously. Its maximum state/control differences are 6.09e-9/4.91e-9.
That pair remains a closed-loop comparison with slightly different work. The spread along an
episode is not a noise estimate or confidence interval, and the median per-step ratio is not
the ratio of provider means.

## Change since 0.1.0a1, October 12 study

`benchmarks/results/study-2026-10-12-post-0.1.0a1` reran the frozen grid at `1c955a07` with the
September 23 libmvec study's machine, toolchain, CPU policy and vector-math policy. Its baseline is
the September 23 summary files as `docs/assets/benchmarks/` held them at `1c955a07`, since that
study's raw files are gone.
`internal/notes/perf_2026_10_12/compare.py` prints the tables below.

Every timed cell passed its correctness check. Of 690 sweep attempts, 515 produced timings, 17
reached the compilation limit and 158 were skipped after a failure, against 520, 15 and 155 on
September 23. The difference is all in CasADi encodings whose compiles sit near the 180-second
limit: npmpc `casadi_map_sx` completed only 3, 1 and 1 of 5 processes at N=12, 25 and 50. Every
Scaly size completed all five processes. Every CasADi closed-loop total and function-evaluation
time is within 1.6% of September 23, which bounds the drift of the machine and of the PIQP 0.6.2 to
0.6.4 update between the two studies.

Scaly Hessian means in µs, with the change in executable source bytes and kernel compile time:

| Problem | Size | September 23 | CV, % | October 12 | CV, % | Change, % | Executable bytes, % | Kernel compile, % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | 3 | 21.810 | 1.02 | 21.672 | 0.24 | -0.6 | -17.4 | -0.5 |
| chain | 5 | 100.037 | 0.36 | 99.802 | 0.04 | -0.2 | -17.2 | -1.5 |
| chain | 9 | 282.649 | 0.06 | 282.448 | 0.08 | -0.1 | -15.8 | -0.6 |
| npmpc | 6 | 12.982 | 0.68 | 13.044 | 1.39 | +0.5 | -29.7 | -0.4 |
| npmpc | 12 | 22.786 | 4.97 | 22.342 | 0.46 | -1.9 | -29.9 | +0.6 |
| npmpc | 25 | 45.339 | 0.87 | 44.690 | 5.49 | -1.4 | -29.9 | -0.1 |
| npmpc | 50 | 93.701 | 0.80 | 90.509 | 2.97 | -3.4 | -29.9 | +0.1 |
| npmpc | 100 | 143.361 | 0.77 | 142.471 | 0.35 | -0.6 | -30.1 | +0.3 |
| npmpc | 200 | 235.705 | 1.20 | 245.132 | 0.88 | +4.0 | -30.1 | -0.1 |
| race_cars | 1 | 0.360 | 0.83 | 0.357 | 0.10 | -0.7 | +0.0 | +0.1 |
| race_cars | 5 | 0.861 | 0.78 | 0.864 | 1.12 | +0.4 | -15.9 | -1.4 |
| race_cars | 10 | 1.036 | 1.10 | 1.030 | 1.90 | -0.6 | -16.2 | -0.5 |
| race_cars | 25 | 2.015 | 2.34 | 1.977 | 1.01 | -1.9 | -16.2 | -0.1 |
| race_cars | 40 | 3.021 | 3.11 | 3.002 | 1.80 | -0.6 | -16.2 | -1.0 |
| race_cars | 50 | 3.495 | 1.50 | 3.471 | 2.47 | -0.7 | -16.2 | -0.7 |
| race_cars | 100 | 6.940 | 5.66 | 6.918 | 4.12 | -0.3 | -16.5 | +0.1 |
| race_cars | 200 | 12.502 | 1.69 | 12.649 | 2.11 | +1.2 | -16.5 | -0.8 |
| race_cars | 500 | 32.178 | 10.73 | 31.402 | 4.77 | -2.4 | -16.5 | -0.9 |
| unbumpercars | 2 | 185.986 | 0.74 | 185.320 | 0.22 | -0.4 | -32.9 | -5.0 |
| unbumpercars | 4 | 215.733 | 0.78 | 212.036 | 0.12 | -1.7 | -38.1 | -25.4 |
| unbumpercars | 8 | 484.377 | 0.60 | 487.540 | 0.46 | +0.7 | -47.8 | -61.1 |
| unbumpercars | 16 | 1422.562 | 1.67 | 1538.086 | 0.28 | +8.1 | -60.1 | -88.1 |
| unbumpercars | 32 | 6245.988 | 0.51 | 5167.377 | 2.25 | -17.3 | -72.2 | -91.1 |

Scaly closed loops, ms per step:

| Problem | Solver | September 23 total | October 12 total | CV, % | Change, % | September 23 FE | October 12 FE | Change, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chain | ipopt | 6.776 | 6.849 | 0.10 | +1.1 | 0.326 | 0.402 | +23.5 |
| chain | sqp | 6.267 | 6.306 | 0.26 | +0.6 | 0.153 | 0.181 | +18.1 |
| npmpc | ipopt | 3.609 | 3.531 | 0.78 | -2.2 | 0.966 | 0.963 | -0.3 |
| npmpc | sqp | 1.270 | 1.273 | 0.31 | +0.3 | 0.474 | 0.473 | -0.3 |
| race_cars | ipopt | 4.714 | 4.597 | 0.75 | -2.5 | 0.183 | 0.143 | -21.9 |
| race_cars | sqp | 2.489 | 2.470 | 0.47 | -0.8 | 0.053 | 0.040 | -23.7 |
| unbumpercars | ipopt | 21.106 | 20.349 | 1.47 | -3.6 | 17.022 | 16.405 | -3.6 |
| unbumpercars | sqp | 11.331 | 10.791 | 1.15 | -4.8 | 10.045 | 9.509 | -5.3 |

Measured: unbumpercars C=32 is 17.3% faster and C=16 8.1% slower, both well outside either study's
dispersion. Npmpc N=200 is 4.0% slower, outside its 0.9% and 1.2% CVs. Chain function evaluation
rises 23.5% under IPOPT and 18.1% under SQP although the chain Hessian kernels are unchanged. That
suggests, without an oracle-level measurement to show it, that the added time lies in another chain
oracle. Race-car function evaluation falls by about 22%. Unbumpercars
function evaluation is back to its September 10 level, 16.405 against 16.419 ms with IPOPT and 9.509
against 9.513 ms with SQP.

Executable source shrinks at every Scaly size except race-car N=1: about 16% for chain and
race-car, 30% for npmpc, and from 33% at C=2 to 72% at C=32 for unbumpercars. Unbumpercars kernel
compile time falls with it, by 61% at C=8 and 91% at C=32, from 40.9 to 3.6 seconds. The other
problems' compile times move by less than 2%. The harness's byte counting did not change between
the two revisions, CasADi's executable bytes are identical and its compile times are within 2.7%,
so these are compiler changes. Their cause was not isolated: the September 23 generated sources
are gone.

Inferred, not measured: the unbumpercars and npmpc kernel shifts come from the short-loop lane
change (#142), the change in the range aimed at lane widths of mapped loops.

The unbumpercars SQP pair now has 15 iteration mismatches across 1,000 paired steps, against 5, still with no oracle-count
mismatch and state differences below 4e-8.

## Change from the scalar-libm studies

These Scaly measurements keep the earlier studies separate from the current policy. The September 10
study is the previously published baseline. The completed September 23 scalar-libm study includes
the compiler changes before the final short-loop lane cap. The final column includes that cap and
libmvec, so the difference between the last two columns is not a pure libmvec comparison.

| Problem | Size | September 10 scalar libm, µs | September 23 scalar libm, µs | September 23 libmvec, µs |
|---|---:|---:|---:|---:|
| Race-car | N=200 | 89.722 | 29.286 | 12.502 |
| Neural-process MPC | N=200 | 595.213 | 307.330 | 235.705 |
| Chain | M=9 | 550.460 | 282.777 | 282.649 |
| Unbumpercars | C=2 | 207.925 | 345.619 | 185.986 |
| Unbumpercars | C=32 | 6872.175 | 5361.081 | 6245.988 |

All final Scaly Hessian means improve on the previously published baseline. Unbumpercars C=32 is
16.5% slower than the intermediate scalar-libm study, although it remains 9.1% faster than the
published baseline. The final study also has closed-loop regressions, listed below. These results
do not establish a regression-free compiler change.

## Closed-loop changes

Against the previously published Scaly results, unbumpercars function evaluation is slower by 3.7%
with IPOPT, from 16.419 to 17.022 ms, and 5.6% with SQP, from 9.513 to 10.045 ms. Total time is
also slower, from 20.375 to 21.106 ms with IPOPT and from 10.786 to 11.331 ms with SQP.
Neural-process MPC IPOPT total time rises 1.5%, from 3.557 to 3.609 ms, although function evaluation
improves from 1.006 to 0.966 ms. Its total CV is 2.01%, so the total-time difference is smaller than
that run's process dispersion. The remaining Scaly closed-loop totals improve on the published baseline.

## Study artifacts and policy changes

The September 10 study and the completed September 23 scalar-libm study used
`SCALY_VECTOR_LIBM=none` and `vector_libm="none"`. Their artifacts remain separate from
`study-2026-09-23-c77-c79-libmvec`. The short-loop lane cap also changed between the latest
scalar and vector studies, so that comparison does not isolate libmvec's effect.

## Remaining regressions and limits

Publishing the complete measurements does not establish a no-regression result. Compared with
September 10, unbumpercars function evaluation is 3.7% slower with IPOPT and 5.6% slower with
SQP. Its total solver times are 3.6% and 5.1% slower. Neural-process MPC function evaluation
improves, but its IPOPT total increases by 1.5%; the non-oracle portion accounts for the increase.

The vector-policy study also has slower unbumpercars C=16 and C=32 Hessians than the preceding
scalar-policy study: 1,422.562 versus 1,280.988 µs and 6,245.988 versus 5,361.081 µs. These
remain faster than September 10. Race-car N=1 also rises from 0.319 to 0.360 µs between
these two recent runs. The two recent runs changed both math policy and short-loop
lane caps, so they do not establish that libmvec caused the slowdown. Large unbumpercars mapped
Jacobian and Hessian helpers contain substantial matrix arithmetic without transcendental calls.
Their register and scratch-memory costs need separate measurement before changing the width policy.

Race-car N=500 has a 10.73% coefficient of variation across five processes. Keep that dispersion
visible rather than selecting the fastest process. The complete sweep also reports source size,
workspace, and compilation cost, which can increase despite faster evaluation. These measurements
cover the reference x86 machine; they do not establish Apple M4 performance.

## Why the CPU policy is fixed

The following audit experiment selected the CPU policy. It is not a headline benchmark result.

The harness protocol check on 2026-09-05 used `performance`, five fresh processes per backend,
and alternating backend order on the race-car Hessian at N=40. Google Benchmark ran for at least
0.5 seconds per cell. Raw rows and provenance are local, gitignored artifacts under
`benchmarks/results/protocol/boost_on/race.csv` and `benchmarks/results/protocol/boost_off/race.csv`.

| Backend | Boost-on mean, µs | Boost-off mean, µs | Boost-on coefficient of variation | Boost-off coefficient of variation |
|---|---:|---:|---:|---:|
| Scaly | 20.486 | 26.661 | 0.714% | 0.822% |
| CasADi mapped SX | 46.049 | 59.535 | 0.664% | 0.278% |

Boost remains disabled. It reduced the average coefficient of variation across these two columns,
though Scaly's dispersion increased slightly and both kernels became slower. Five samples on one
workload do not establish a general variance advantage. This check selects a machine configuration,
not a paper result. The completed study uses the selected configuration for every column.

## Why the native compiler flags are fair

The following flag ablation explains the shared compiler policy. It isolates compiler flags on one
kernel and does not replace the current sweep.

The historical flag ablation compiled both providers' kernels with
`-O3 -march=native -fno-math-errno`. The closed-loop CasADi IPOPT wrapper adds the same two
flags to the optimization level it shares with Scaly's JIT,
and the JIT applies them too. Scaly compiles on the machine it runs on, so the native target is the
deployment reality, and only binaries distributed to other machines, such as the solver plugin
wheels, stay at the portable x86-64 baseline. Both providers get identical flags, so the comparison
stays controlled, but the flags do not move both encodings equally, which is why the baseline is a
handicap and not a neutral choice. The portable target withholds fused multiply-add (FMA), and
clang's default `-ffp-contract=on` fuses only within one expression: Scaly renders compound
expressions and gets the contraction, while SX emits one operation per statement
(`a=(a*b); a=(a+c);`) and never forms an FMA. Race-car Hessian at N=50, µs, Scaly then SX: `-O3`
alone 32.9 and 21.3; `-mfma` alone 27.8 and 20.7; `-march=native` 26.9 and 20.7;
`-march=native -ffp-contract=off` 35.2 and 20.5. The gain is contraction, not vector width, and the
SX object in that historical experiment contains no FMA and no vector instruction at all.
The current vector-math study does emit vector calls in SX. `-fno-math-errno` lets `sqrt` and
the other libm calls inline instead of setting `errno` nothing reads: 31.8 against 33.4 for Scaly
and 20.6 against 21.3 for SX on the same cell. The completed study applies the native flags to both
providers. Its `.provenance.json` sidecars, `study.json`, and compile logs record the exact
commands.

## Controlled audit evidence

The experiments below isolate configuration variables that the complete study holds fixed. Their
absolute timings are audit measurements, not current benchmark headlines. The current study results
remain on the [overview](../../docs/benchmarks/index.md) and [scalability page](../../docs/benchmarks/scalability.md).

## The IPOPT build is a first-order confound

This finding undercuts the attribution of every pre-audit `ipopt+scaly` versus `ipopt+casadi` gap
in the suite.

The pre-audit columns did not run the same solver. Scaly's column loaded a locally built IPOPT
3.14.19 with MUMPS, METIS and OpenBLAS linked statically; CasADi calls the IPOPT 3.14.11 that ships
in its wheel, dynamically linked against `libcoinmumps`, `libcoinmetis` and `libcasadi-tp-openblas`.
The suite's own provenance records the CasADi version and the C compiler, and neither IPOPT version.

Scaly's generated wrapper carries `DT_NEEDED libipopt.so.3` and a `RUNPATH`, which `LD_LIBRARY_PATH`
overrides, so the same scaly oracles can be pointed at CasADi's IPOPT instead. Every row below is
the same generated C, the same warm starts and the same episode; only the `libipopt.so.3` the loader
resolves changes. Iteration counts are identical in every pair, and where the oracle timer is
comparable, so is function-evaluation time.

| problem | IPOPT build | solve | FE | non-FE | iterations |
|---|---|---:|---:|---:|---:|
| `npmpc` N=12, 100 steps | scaly 3.14.19 | 3.64 ms | 1.34 ms | **2.30 ms** | 9.99 |
| | CasADi 3.14.11 | 7.67 ms | 1.34 ms | **6.33 ms** | 9.99 |
| `chain` M=5 N=12, 60 steps | scaly 3.14.19 | 9.24 ms | 3.70 ms | **5.54 ms** | 8.05 |
| | CasADi 3.14.11 | 101.51 ms | 5.56 ms | **95.95 ms** | 8.05 |
| `race_cars` N=40, 150 steps | scaly 3.14.19 | 4.60 ms | 1.22 ms | **3.37 ms** | 9.70 |
| | CasADi 3.14.11 | 13.47 ms | 1.26 ms | **12.21 ms** | 9.70 |
| `unbumpercars` C=8, 40 steps | scaly 3.14.19 | 53.45 ms | 49.92 ms | **3.53 ms** | 14.10 |
| | CasADi 3.14.11 | 58.86 ms | 49.56 ms | **9.30 ms** | 14.10 |

Function evaluation is unchanged, as it must be, since the oracles are the same compiled C in both
rows. Everything else takes 2.7× longer on the smallest problem and 17× longer on the chain, where
the KKT system is largest and the linear solver dominates. Repeating the `npmpc` pair under
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` moves nothing, so this is not BLAS threading; it is a
property of the two builds.

Two caveats on that table. The chain row's FE also moves (3.70 to 5.56 ms), which it should not; on
a 96 ms solve that is most likely cache and frequency effects, and it is small next to the 90 ms it
sits beside. And `unbumpercars` is 93% function evaluation at C=8, so the confound is real there but
not where its claim lives.

The swap does not work in the other direction. CasADi's IPOPT plugin links IPOPT's C++ interface
built against the pre-C++11 `std::string` ABI, so preloading scaly's IPOPT fails to resolve
`Ipopt::StreamJournal`. Getting the fourth cell of the matrix needs the CasADi column
code-generated, which is possible.

## The CasADi column can be code-generated, all the way down

`ca.CodeGenerator().add(nlpsol_instance)` works for the IPOPT plugin. It emits self-contained C that
calls `IpStdCInterface.h` directly, with the oracles as generated C functions in the same
translation unit, and it creates and frees the IPOPT problem object per solve exactly as scaly's
wrapper does. Compiling it with scaly's compiler and flags, linking it against scaly's own
`libipopt.so`, and calling it through one `ctypes` call gives a CasADi column that shares scaly's
IPOPT build, its optimization level, its interface into IPOPT and its timer placement. A three-line
C shim wraps the generated entry point in `clock_gettime(CLOCK_MONOTONIC)`.

Two things are easy to get wrong here, and both produce wrong numbers that look right:

- CasADi's generated code uses the `res` array as scratch for nested calls, clobbering the leading
  output slots. Because `casadi_copy` skips a `NULL` destination without complaining, a `res` array
  set up once makes every call after the first return the first call's answer, with a success
  status. The output pointers have to be re-set on every call.
- A `nlpsol` cannot be code-generated and timed in the same process that built it. Constructing the
  `nlpsol` dlopens the wheel's `libipopt.so.3`, and every later `DT_NEEDED libipopt.so.3` resolves
  to that one regardless of the generated library's `RUNPATH`. Build in one process, time in another
  that only `ctypes`-loads the artifact.

### The controlled 2×2

`npmpc`, the canonical 100-step episode, both columns generated C behind one `ctypes` call, both
compiled by gcc 13.3 at `-O2`, both timed by `clock_gettime` inside C, identical bounds and warm
starts. State trajectories agree to 8.8e-13 across all four cells.

| oracle provider | IPOPT 3.14.19 (scaly's build) | IPOPT 3.14.11 (CasADi's wheel) |
|---|---:|---:|
| **scaly generated C** | **3.48 ms** | 7.04 ms |
| **CasADi generated C** | **3.93 ms** | 5.83 ms |

What the table says:

- Holding the IPOPT build fixed, scaly's oracles are worth 1.13× on total solve time (3.48 against
  3.93). That was the controlled oracle-provider result in this audit.
- Holding the oracle provider fixed, the IPOPT build is worth 2.02× for scaly's column and 1.48× for
  CasADi's, a larger effect than the thing being measured.
- The two effects are not additive: with CasADi's IPOPT, the CasADi column is faster than scaly's
  (5.83 against 7.04). Scaly's wrapper recreates the IPOPT problem and reapplies every option on
  each solve, and that costs more against the wheel build. This is not explained and should not be
  over-read; it is a reason to report the matrix and not one cell of it.
- The 1.13× here and the 1.22× from the already-fair `scaly-sqp` pair (1.29 against 1.57 ms) agree
  in magnitude, which they did not before. Two independent fair comparisons landing in the same
  place is the best evidence available that this is the real number.

## `expand` and `jit`, per problem

Neither had been swept per problem. Both matter, and they do not point the same way.

The tables in this historical configuration study are retained for their closed-loop evidence. The
kernel sweep now keeps its encoding labels literal: `expand=True` for `casadi_sx`, and `False` for
`casadi_mx`, `casadi_call_mx` and `casadi_map_sx`.

### `npmpc`, canonical episode, 100 steps

| column | build | solve | p95 | FE | Python-level | iterations |
|---|---:|---:|---:|---:|---:|---:|
| scaly (JIT `-O2`, C timer) | 0.3 s | **3.77 ms** | 6.26 | 1.49 ms | 3.85 ms | 9.99 |
| CasADi interp-SX, pre-audit | 0.5 s | 17.01 ms | 19.94 | 12.87 ms | 17.32 ms | 9.99 |
| CasADi interp-MX | 0.02 s | 7.79 ms | 10.14 | 3.85 ms | 8.01 ms | 9.99 |
| CasADi jit-MX `-O3` | 23.9 s | 5.23 ms | 6.55 | 1.47 ms | 5.43 ms | 9.99 |
| CasADi jit-SX `-O3` | **1017 s** | 7.25 ms | 9.28 | 3.15 ms | 7.55 ms | 9.99 |
| CasADi codegen-MX `-O2` (C timer) | 12.7 s | 5.56 ms | 6.48 | n/a | 5.60 ms | n/a |
| CasADi codegen-MX `-O3` (C timer) | 23.4 s | 5.15 ms | 6.83 | n/a | 5.18 ms | n/a |

The pre-audit column is the slowest of the six. Expanding to scalar SX is a trap on this problem: it
triples the interpreter's work, and compiled it costs a seventeen-minute build to land slower than
compiled MX. Compiled, CasADi's function evaluation (1.47 ms) is a wash against scaly's (1.49 ms),
which is what the kernel sweeps said, so the interpreted column's 8.6× gap was a configuration
artifact.

### `race_cars`, N=40, 149 steps of the canonical lap

| column | solve | p95 | FE | non-FE | iterations |
|---|---:|---:|---:|---:|---:|
| scaly (JIT `-O2`, C timer) | **4.60 ms** | 5.07 | 1.225 ms | 3.375 ms | 9.70 |
| CasADi interp-SX, pre-audit | 9.16 ms | 10.04 | 1.810 ms | 7.350 ms | 9.70 |
| CasADi interp-SX + `ca.cse` | 9.12 ms | 10.20 | 1.680 ms | 7.443 ms | 9.70 |
| CasADi interp-MX | 44.52 ms | 50.31 | 35.276 ms | 9.238 ms | 9.70 |
| CasADi jit-SX `-O3` | 7.95 ms | 8.72 | **0.398 ms** | 7.553 ms | 9.70 |
| CasADi jit-MX `-O3` | 8.03 ms | 8.72 | 0.643 ms | 7.385 ms | 9.70 |
| CasADi codegen-SX `-O2` (C timer) | 7.62 ms | 8.32 | n/a | n/a | n/a |
| CasADi codegen-MX `-O2` (C timer) | 8.08 ms | 9.11 | n/a | n/a | n/a |

Here `expand=True` is the right default by a wide margin. Interpreted MX is five times worse than
interpreted SX, because the MX evaluator's per-node overhead dominates a kernel this cheap. The
audit also found that once CasADi is compiled its function evaluation is 3.1× faster than scaly's,
0.398 against 1.225 ms. The sign of the oracle comparison reverses on this problem. Scaly's total is
still 1.7× better, but on this problem that is the IPOPT build, not the oracles.

`ca.cse` is worth 7% of function evaluation here and nothing on the total.

## Timers: what each column's number actually covers

Scaly's `t_total` is `clock_gettime` inside the generated C entry point, so it excludes the ctypes
call, the input coercion and the output allocation. The CasADi columns' `t_total` is
`time.perf_counter()` in Python around the SWIG call, so it includes the numpy to `DM` conversion of
every argument, and the difference is then booked as `t_solver`, which reads as IPOPT's time.

Measured on the `npmpc` episode, per step:

| column | Python-level | column's reported total | difference |
|---|---:|---:|---:|
| `ipopt+scaly` | 3.49 ms | 3.41 ms | 0.086 ms |
| `sqp+scaly` | 1.29 ms | 1.23 ms | 0.060 ms |
| `ipopt+casadi` | 17.81 ms | 17.43 ms | 0.380 ms |
| `sqp+casadi` | 1.51 ms | 1.45 ms | 0.060 ms |

Scaly's Python boundary costs 60 to 90 µs per solve, 2.5% of an IPOPT step and 5% of an SQP step.
For scaly, the columns are not measuring overhead.

The CasADi IPOPT column now uses the stronger fix measured in the audit. The harness code-generates
the complete `nlpsol`, calls it through `ctypes`, and measures `clock_gettime` inside the C
entry point. Both columns now have the same kind of boundary.

## Why chain uses common-subexpression elimination

The chain audit isolated CasADi common-subexpression elimination (CSE) before `nlpsol`. At M=5,
unrolled SX generated 215,190 lines with CSE and 309,030 without it. The version without CSE did not
compile inside the ten-minute audit budget. On race-car, the same preprocessing changed function
evaluation by 7% and did not change total solve time. The harness therefore applies `ca.cse` to the
chain nonlinear-program expressions only. This is a measured formulation choice, not a global
advantage assigned to either provider.

## Why CasADi transformation and GEMM variants remain in the sweep

The CasADi 3.8 audit isolated the passes behind `Function.transform()`. On the race-car SX Hessian
at N=25, CSE changed 15.0 µs to 8.2 µs. The other tested passes did not change the result. Generated
source fell by 15% to 30% across the audit cells. The current sweep therefore transforms every
CasADi oracle before code generation.

CasADi's BLAS selector does not propagate to matrix products created by automatic differentiation.
The audit also found that batching repeated network products into one general matrix multiplication
(GEMM) made the tested Hessians slower, although it let one larger NPMPC graph compile. The sweep
keeps the three GEMM encodings as recorded alternatives and reports them under the same compile
budget.

## Where the suite handicaps Scaly

- Per-solve IPOPT setup. Scaly's wrapper calls `CreateIpoptProblem`, every option setter and
  `FreeIpoptProblem` on every solve, because bounds are parameter-dependent, and all of it is
  inside its own timer. CasADi's `nlpsol` does that once at construction, outside every timer. Its
  generated C does it per solve, like scaly's, one more reason the code-generated column is the
  right comparison.
- `t_fe` is not the same quantity on both sides. Scaly's includes a bounds-evaluation kernel that
  CasADi has no analogue for; CasADi's is the sum of its `t_wall_nlp_*` callback timers, which may
  not cover the interface's sparse-matrix copies.
- Fused oracles. Scaly's `base` kernel computes `f` and `g` together and is called for both, so it
  evaluates `g` on every `f` call and vice versa, where CasADi evaluates them separately.

