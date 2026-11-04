# CI cache and smoke change

Scope: setup-uv policy across workflows and a separate Linux smoke job, following decisions 9, 11 and 12 in internal/notes/ci_and_native_builds_2026_11_02.md. No issue requested.

- [ ] Inspect pinned action cache semantics and implement policy and smoke split.
- [ ] Run local checks and open draft PR with temporary smoke failure.
- [ ] Verify failure propagation and no uv saves in PR logs, then remove perturbation.
- [ ] Cross-review, final CI timing comparison against main run 41803148595, update description and mark ready.

All phases are sequential. Preserve toolchain installs, build hooks and Windows warming.
