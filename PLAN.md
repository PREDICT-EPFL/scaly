# Publish external C functions

Issue #480 publishes the decided `sc.extern` and `sc.CLibrary` interface. The base is
`t3code/479-numpy-names`, third in the stack above #482 and #486. The pull request targets
`t3code/479-numpy-names`. Follow `~/.scaly-suite/RULES.md`, the implement and write-docs skills,
and the roadmap's "Runtime indexing, in-place updates and external code" decision.

Scope is API entries and one short section in `docs/guide/codegen.md`, with a C library example
executed by `tests/test_docs_prose.py`. State the pointer signature, float64-only leaves,
unsupported structs, and absolute include/library paths in exported manifests. Preserve behavior,
the workspace decision and every generated-C snapshot. Avoid `docs/guide/functions.md`.

1. [x] Inspect the issue, parent, checkout, stacked prerequisites and documentation conventions.
2. [x] Add the focused documentation example test, observe its failure, write the section and API
   entries, and demonstrate that a wrong C result fails the NumPy assertion.
3. [ ] Run focused tests, render and inspect both API docstrings, and run the detached merge checks
   through `~/.scaly-suite/run`. Stop if any C snapshot changes.
4. [ ] Commit, push once and open a draft PR against the stack base. Get one Opus 5.5 high review
   for correctness and style, fix findings, and request only a delta check if needed.
5. [ ] Remove this plan, update the PR and issue, report criterion evidence and CI state, and watch
   the PR. Never merge or queue it.

GitHub Project REST access returned 404. Track status through issue comments without guessing
Project identifiers. Scratch files and logs live under ignored `build/issue480/`.

Focused validation: 62 documentation and external-library tests passed. The new test first failed
because its section was absent. Replacing the C square by `3*x` made all three NumPy comparisons
fail, then the original C was restored. The docs-only Zensical build passed with no issues and
contains both API anchors and docstrings. Existing docstrings need no edits.
