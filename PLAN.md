# #252 First JIT compile on a fresh machine

Brief: measure zig's first compile cold and warm on Linux and Windows, get the maintainer's choice
between documenting the delay and a warm-up mechanism, and cache zig's global cache in the Windows
CI test job. Out of scope: `setup-uv` settings and the smoke job (PR #254), `ci-build.yml` (PR #255).

- [x] Linux measurements (`.github/scripts/measure_first_compile.py`, temporary)
- [x] Zig cache restore/save around "Warm zig's cache" in `ci-test.yml`
- [x] Windows measurements from the PR's CI (runs 41811995716, 41823146142)
- [ ] Hit rate over several runs and reruns of `test / windows-x86_64`
- [x] Post measurements and recommendation on #252; maintainer chose: document, install job stays fresh
- [x] Note `internal/notes/first_jit_compile_2026_11_04.md`
- [x] Guide: `docs/guide/installation.md#compiler-selection`
- [x] Remove the measurement step and script
- [ ] Cross-review, delete this plan, PR description with CI timings
