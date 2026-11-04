# Reuse released native solver libraries

Inputs: issue #250, native-code decisions 2–4 in
`internal/notes/ci_and_native_builds_2026_11_02.md`, both plugin hooks,
`ci-build.yml`, contributing guide and vendored solver notes.

- [x] Inspect issue, prerequisites, worktree and overlapping work.
- [x] Implement and test native input stamps, verified newest-release lookup, cache separation and source forcing. Native-hooks agent owns plugin code and tests.
- [x] Wire preflight and conditional Fortran installation in ci-build.yml. Parent owns workflow and docs, in parallel with hook work; final wiring depends on preflight API.
- [ ] Run required checks and one independent cross-review, fix findings and check delta.
- [ ] Open PR, collect its CI evidence and toolchain step timings, update description and tracking.

No changes to uv cache settings, smoke layouts, zig or JIT. No merge, tags or main push.

Focused tests: 11 pass, SHA-256 perturbation fails as expected. Full suite: 2821 pass, 50 skip. Type, format and lint pass; smoke/docs pending. Tests were run after implementation, a deviation from test-first procedure. Baseline was corrected after an interleaved stderr line corrupted initial generation.
