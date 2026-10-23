# Windows Ipopt, issue #196

Ted chose native Windows builds through MSYS2 UCRT64 after the CasADi investigation.
Keep the pinned Ipopt, MUMPS, METIS, GKlib and OpenBLAS versions. Users install a self-contained
wheel and use packaged Zig for generated C. The investigation and decision live in
`internal/notes/ipopt_windows_build_options_2026_10_23.md`.

Read the Ipopt and PIQP hooks, solver library discovery, wheel/test/release workflows,
Windows installed-wheel assertions, `internal/notes/vendored_solvers.md` and the Windows
toolchain decisions. No prerequisite is open and no PR overlaps this task. Project access
is unavailable because the token lacks `read:project`.

## 1. Establish the native source build

- [x] Trace the released CasADi recipe and prove Zig can call its MinGW-built Ipopt.
- [x] Agree on native MSYS2 UCRT64 as the first route.
- [ ] Provision a workspace-local MSYS2 environment and record its package versions.
- [ ] Build current numerical sources and solve through Zig with only bundled DLLs available.
- [ ] Record required source/build adjustments and evidence in the investigation note.

## 2. Package and test

Depends on the native build. Keep changes in `plugins/scaly-ipopt/` unless discovery requires
a focused core adjustment.

- [ ] Extend the hook for native GNU builds, Windows cache locks and DLL dependencies.
- [ ] Package matching headers, import library, DLLs and notices.
- [ ] Add meaningful regression checks and demonstrate failure before the fix.
- [ ] Build and install the wheel into a clean environment, run all Ipopt numerical references,
      status/statistics checks and a missing-runtime negative control.

## 3. Integrate the installed wheel

Depends on a passing wheel. Workflow and documentation edits can be checked independently.

- [ ] Add the Windows Ipopt wheel to build/release matrices and require the backend in CI.
- [ ] Extend the isolated Windows installation checks to direct and generated Ipopt solves.
- [ ] Update platform documentation through `write-docs`.

## 4. Validate and review

- [ ] Run narrow checks, then `wt hook pre-merge --yes` and report platform limitations.
- [ ] Commit the task, push and open/link a draft PR.
- [ ] Run one cross-family review, fix findings and run the delta check.
- [ ] Verify PR CI, remove this temporary plan, mark ready, update the issue and watch the PR.
