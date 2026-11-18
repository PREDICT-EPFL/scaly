# Correct the functions guide's integer leaves

Scope: issue #466, based independently on main. Only the extremum-reduction
and runtime-indexing passages in `docs/guide/functions.md`, their documentation
tests, and the test collection baseline may change. Preserve operation names
and every generated-C snapshot.

The issue's design is decided. The declarations and numerical boundary in
`src/scaly/function/tree.py`, typed dispatch in `src/scaly/codegen/jit.py`,
and `docs/guide/codegen.md#float32-and-integer-functions` confirm that integer
leaves already work. Follow `.agents/skills/write-docs/brief.md` and
`~/.scaly-suite/RULES.md`.

- [x] Read the issue, parent, dependencies, overlap, repository rules and implementation.
- [x] Add two documentation tests and reproduce the current examples' float64 leaves.
- [x] Edit only the two passages, execute their examples and verify the tests detect wrong outputs.
- [x] Run focused checks, then the detached pre-merge hook through the suite runner.
- [ ] Commit and push the completed change, open a draft PR against main, and link it in T3.
- [ ] Obtain one claude-opus-5-5 review at high effort. Fix findings and check only the fixes.
- [ ] Remove this temporary plan, mark the PR ready, update the tracker and report evidence.

The edits wait for the failing tests. Full validation waits for the final local
change. Review waits for passing checks. No merge, queue, tag, or push to main.

Focused validation: 40 documentation and snapshot tests pass with two workers.
Both new tests failed against the old guide and again with deliberately wrong
example outputs. Ruff format, lint, strict ty and complete test collection pass.
The collection baseline adds only the two new cases. Logs stay under the
gitignored `.pytest_cache/466/` directory.

The detached `~/.scaly-suite/run` pre-merge hook exited zero: strict types,
formatting, lint, 5,134 tests passed and 16 skipped, benchmark smoke, docs build
and private-name check. The linked print example runs with its stated output.
No generated-C snapshot changed. An obsolete float64-output sentence also
remains in the `sc.argmax` docstring, outside this task's permitted passages.
