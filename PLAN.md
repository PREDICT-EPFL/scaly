# Ipopt native producer, issue 229

Base: `t3code/issue-228-native-bundle-builder`, PR #231. This branch owns producer workflows,
validation scripts and maintainer instructions. Builder/schema changes belong to #228 and
consumer pins and ordinary wheel setup belong to #230.

Inputs read: issue #229, issue #228's fixed design comment, PR #231 description,
`internal/notes/ipopt_native_bundle_contract.md`, `internal/notes/ipopt_native_controls.md`,
the architecture/codebase/conventions/contribution guides, existing wheel/test/release workflows,
`windows_install.py`, `solve_smoke.py`, the builder and solver tests.

- [x] Inspect scope, stack and existing interfaces. Project access lacks `read:project`.
- [x] Add failing checks for trigger selection, dependency/floor audits and publication gates.
- [x] Implement five-target candidate generation with manylinux image identities and retained diagnostics.
- [x] Implement isolated validation using existing solver-required tests and smoke commands, native C/C++
  linkage, load identities, runtime-removal control and retained-source numerical comparison.
- [x] Implement guarded publication of exact validated bytes, with a testable read-only preflight.
- [x] Document candidate/health/publication/rollback procedures and expected pins.
- [x] Run focused checks and `wt hook pre-merge --yes`; open a draft PR against the builder branch.
- [ ] Run all five candidates in this PR's CI. Iterate single targets when needed, especially Windows.
- [x] Run one cross-family review and one delta check. Both blocking findings are resolved.
- [ ] Record five-target CI evidence and the exact maintainer handoff.
- [ ] Remove this plan when complete, mark ready and watch the PR.

Dependencies: validation follows candidate generation; publication consumes only a completed full
main-branch run. Target builds are independent. Monthly health never publishes. No publication,
tag, merge, default-branch push or repository setting change is authorized here.

Local evidence: 20 admission/rejection tests pass. A mutation disabling the glibc floor rejection
makes its test fail. Both workflows pass actionlint. All new Python scripts pass pinned Ruff and ty.
The complete Windows validator passes with a diagnostic metadata repack of the retained #228
archive. It includes 125 existing solver tests and both closed-loop probes. This is validation
script evidence, not a successful new candidate build. The native bytes remain those of #228.

The real audit found #228's `objdump` import inventory is empty for the repaired Windows Ipopt
DLL. A minimal builder fix reads ordinary/delay imports through its already-required `pefile`.
The coordinator was told before committing, then asked to keep the fix out of #229's history.
The fix and regression landed in #228. The saved local diagnostic patch is dropped. The
#229-only commits are rebased onto `origin/t3code/issue-228-native-bundle-builder`, including
its later macOS floor-parser fix. No builder, schema or fixture changed here.

Draft PR: https://github.com/PREDICT-EPFL/scaly/pull/233. Initial producer run 40605435231 uses
the incomplete Windows inventory and was cancelled. The next run is on the pushed draft.
Full pre-merge type, format and lint passed. The suite has 2743 passed, 107 skipped and the known
workstation `test_native_recipe_render_policies_separate_cache_entries` failure from missing `cc`.
The benchmark smoke passes separately. The docs build exits 0 with no issues. The cold Windows
source build still takes about an hour.

Windows system function imports are checked against the retained source-control API requirements,
along with ordinary/delay DLL imports. Publication also retains the exact source-control wheels,
so monthly validation survives Actions artifact expiration. Before the initial native revision,
the producer bootstraps those wheels from the recorded #218 run. Consumer pins remain untouched.

Run 40606421460 exposed root-owned Linux output directories, fixed by creating the destination
before Docker starts. Both macOS builds reached packaging, then the builder mistook the linker
tool version for the minimum OS. Layer 1 now fixes that parser with regression tests, preserving
the 15.0 floor. The ordinary CI type failure is fixed with declared audit dependencies and native
module enumeration without psutil. A clean locked environment passes the root type check.

The independent cross-family review requested actual generated-solver bindings in the Python
coexistence checks and removal of the extra GitHub approval prerequisite. Direct/nested bindings
now come from generated-module handles or Windows import address tables. Four fresh Windows
candidate/baseline probes pass both load orders and verify CasADi's own Ipopt identity too.
Publication requires the maintainer's explicit reviewed commit and its merged PR. Environment
protections remain available without mandating a second account. Cheap review notes are resolved
by narrowing input paths, correcting thread-capacity prose and testing the actual manylinux
policy check without a duplicate floor check. Fable 5.1 at low effort approved the delta and
the verdict is recorded on PR #233. Two nonblocking notes remain, a missing filename in the
manylinux version error and a redundant always() in the cancellation condition.

After the macOS rebase, 68 producer/builder tests pass. The full candidate run is queued behind
the preserved earlier Windows build. Dispatch both macOS targets separately to exercise the
fixed packaging path and acceptance without waiting for another Windows compilation.
