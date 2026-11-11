# Windows JIT source changes, issue #395

The compiler run plan in `internal/notes/compiler_run_2026_11_10/plan.md` and
`~/.scaly-suite/RULES.md` govern this task. Base and pull request target are main.
The section 6 coordinator owns #295. The parallel simplification of `digest.py`
may require a rebase when it merges.

Scope is Windows source-change metadata in `src/scaly/function/digest.py` and
`tests/function/test_digest.py`. Read the #64 Foundations item in the compiler
roadmap, #395 and #321. Keep Linux/macOS rules, bytecode checking, clock slack,
and the exemption for Python creating `__pycache__` unchanged.

- [x] Read the brief, skills, run rules, project context and current implementation.
- [ ] Enable Windows regression tests for restored modification times and directory swaps.
- [ ] Read NTFS change times through ctypes only on Windows, using metadata handles.
- [ ] Prove Linux behaviour and that warm source lookups do not read file contents.
- [ ] Run all pre-merge checks through the detached shared suite runner.
- [ ] Commit, push once, open and link a draft pull request, obtain one correctness review.
- [ ] Settle review delegates, address findings, and confirm only any fixes.
- [ ] Get Windows CI evidence from one run. If the method cannot work, tell the
      coordinator before replacing it with the issue's documented-limitation path.
- [ ] Remove this plan, mark ready, queue squash auto-merge after review and green checks,
      report Windows evidence, open points and any source:agent follow-ups.

There is no usable Windows machine. Do not bypass thebeauty's SSH host-key failure.
No direct main push, tags, releases or deployment workflows.
