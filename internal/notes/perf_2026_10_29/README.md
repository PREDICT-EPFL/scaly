# October 29 release study audit

The audit compared the completed study, October 20 comparison and October 12 repeated-process
dispersion. Its scripts and derived data were not kept. The findings remain in these notes,
and the raw study data remains in `benchmarks/results/`.

The study is `benchmarks/results/study-2026-10-29-release-0.1.0`, measured on la015 at
`a3cc544c4e952520c747b7f4c4cef59074303a61`. Its launch log is the adjacent
`study-2026-10-29-release-0.1.0-run.log`. The requested report was regenerated with
`uv run benchmarks/run.py report benchmarks/results/study-2026-10-29-release-0.1.0`.
That command automatically refreshed the pure-Python editable Scaly package. Subsequent
analysis uses `--no-sync` to avoid another refresh. No native build or measurement ran.

The audit checks all 106 planned sweep cells, all timed cells' saved compiled-C correctness
markers and gate order, the precompile sparsity/reference gate's control flow, retained samples,
compiler commands, revision provenance, CPU policy, 15 episodes and 6,938 solver steps.
It compares every matching sweep timing, source and object hash, numerical samples, controller
means, paired per-step ratios, oracle counts, iteration counts and sampling deadlines.

One process gives no estimate of current measurement dispersion. October 12's coefficients of
variation are historical context. Episode standard deviations and percentiles describe changing
solver work, not measurement noise. October 20 already used Clang for sweep kernels, but GCC
for controllers. The release study uses Clang 20.1.8 for both providers throughout. Cross-study
differences combine revision and toolchain changes and do not identify either cause.

The launch manifest and eight early provenance records name the launch revision. Nineteen later
records name `eca417896b026c06b88264678da71c848861bc36`. The checkout was updated during the run.
The script confirms identical source, plugin, benchmark and lockfile trees. The changed revision
contains documentation and one Windows package classifier. Preserve this qualification and the
original provenance, rather than rewriting the records to one revision.

The unbumpercars telemetry omits oracle counts. The script reads its separate `stats.csv` files
and finds three SQP steps with different provider counts, plus one iteration mismatch.
October 20's detailed stats have two affected steps, despite the old summary saying zero.
The script also records changed harvested inputs for the C=8 sweep. Its cross-study timing
cannot establish a fixed-input improvement. Within the current study the providers' normalized
inputs agree at absolute and relative tolerances of 1e-9.

Published pages, assets, the comparison history and project source remain unchanged.

## Compiler controls and regression attribution

The maintainer's follow-up investigation ran the measurements itself. Read
`regression_findings.md` for the measured compiler controls, parent/commit confirmations,
dispersion, deadline results, compiler recommendation and limitations. The original `analysis.md`
remains as the audit record.

The investigation scripts, specifications and derived tables were not kept. The raw
measurement directories begin `benchmarks/results/control-2026-10-29-` and preserve every command,
log and provenance record. The two setup interruptions and their corrected runs are described in
`compiler_control_plan.md` and the findings. The investigation made no commits or pull requests
and changed no published page.

The completed findings comment is on
[issue #85](https://github.com/PREDICT-EPFL/scaly/issues/85#issuecomment-6496017593).
