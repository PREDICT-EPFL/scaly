# Reuse released native solver libraries

Inputs: issue #250, native-code decisions 2–4 in
`internal/notes/ci_and_native_builds_2026_11_02.md`, both plugin hooks,
`ci-build.yml`, contributing guide and vendored solver notes.

- [x] Inspect issue, prerequisites, worktree and overlapping work.
- [x] Implement and test native input stamps, verified newest-release lookup, cache separation and source forcing. Native-hooks agent owns plugin code and tests.
- [x] Wire preflight and conditional Fortran installation in ci-build.yml. Parent owns workflow and docs, in parallel with hook work; final wiring depends on preflight API.
- [x] Run required checks and one independent cross-review, fix findings and check delta.
- [ ] Open PR, collect its CI evidence and toolchain step timings, update description and tracking.

No changes to uv cache settings, smoke layouts, zig or JIT. No merge, tags or main push.

Focused tests: 11 pass, SHA-256 perturbation fails as expected. Full suite: 2821 pass, 50 skip. All pre-merge checks pass, including smoke/docs. Tests were run after implementation, a deviation from test-first procedure. Baseline was corrected after an interleaved stderr line corrupted initial generation.

Review by Fable 5.1 found workspace-glob warning and uncaught HTTPException. Both fixed; added two tests that failed before fix and pass afterwards. Environment-variable table updated. All pre-merge checks pass after fixes: 2823 passed, 50 skipped, 13 focused tests passed. Fable 5.1 delta check resolves both findings. First CI run 41810845064: cold macOS arm64 setup 14s, x86_64 36s, Windows setup 48s; remaining builds running. GitHub branch points at the fixes but PR head/CI synchronization is delayed; push this plan update to trigger synchronization. Warm timings pending. Linux re-repair with a real released native payload remains to verify once an artifact is available.
