# #252 First JIT compile on a fresh machine

Brief: measure zig's first compile cold and warm on Linux and Windows, get the maintainer's choice
between documenting the delay and a warm-up mechanism, and cache zig's global cache in the Windows
CI test job. Out of scope: `setup-uv` settings and the smoke job (PR #254), `ci-build.yml` (PR #255).

- [x] Linux measurements (`.github/scripts/measure_first_compile.py`, temporary)
- [x] Zig cache restore/save around "Warm zig's cache" in `ci-test.yml`
- [ ] Windows measurements from the PR's CI, several runs for the CPU models and hit rate
- [ ] Post measurements and recommendation on #252, ask the coordinator for the maintainer's choice
- [ ] Note under `internal/notes/` with measurements, the generic-runtime answer and the decision
- [ ] Implement the chosen outcome (user-guide text, or warm-up in `scaly.codegen.toolchain`)
- [ ] Remove the measurement step and script, delete this plan, PR description with CI timings
