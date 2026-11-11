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
- [x] Enable Windows regression tests for restored modification times and directory swaps.
- [x] Read NTFS change times through ctypes only on Windows, using metadata handles.
- [x] Prove Linux behaviour and that warm source lookups do not read file contents.
- [x] Run all pre-merge checks through the detached shared suite runner.
- [x] Obtain one correctness review before the initial push so one CI run proves reviewed code.
- [ ] Remove the temporary plan, push once, open and link a ready pull request through REST.
- [ ] Settle review delegates, address findings, and confirm only any fixes.
- [ ] Get Windows CI evidence from one run. If the method cannot work, tell the
      coordinator before replacing it with the issue's documented-limitation path.
- [ ] Remove this plan, mark ready, queue squash auto-merge after review and green checks,
      report Windows evidence, open points and any source:agent follow-ups.

There is no usable Windows machine. Do not bypass thebeauty's SSH host-key failure.
No direct main push, tags, releases or deployment workflows.

Local evidence: 49 focused digest and index tests passed. Windows and Python 3.14
type checks passed. Replacing change time with modification time makes all three
change-time regressions fail. Clearing the content memo makes the warm-read test
fail. Both perturbations were process-local and are absent from the files.
Full checks are detached under `/tmp/scaly-395/pre-merge.log`, with their exit
code written to `/tmp/scaly-395/pre-merge.exit`.
The full runner exited 0: types, Ruff, 3679 passed and 55 skipped, benchmark smoke,
documentation and private-name checks passed. Test collection passed with 3734
node IDs, including the new test's required baseline entry.

Review by Opus high found one actionable issue: a zero native ChangeTime must
give no source digest. The two-line guard and a Windows regression are added.
A mocked native-call reproduction failed without the guard and passes with it,
including handle closure. The remaining review notes need the Windows CI run.
The first review delegate is settled. A short delta check remains before push.
