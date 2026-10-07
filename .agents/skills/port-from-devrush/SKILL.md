---
name: port-from-devrush
description: Port tests, examples or benchmark problems for a feature from the frozen origin/devrush branch to main, rewriting them against main's API. Use only when the user or another skill asks for it by name, never on your own initiative.
---

# Port from devrush

`origin/devrush` is frozen. Read it with `git grep` and `git show`, never check it out, merge it or
copy its implementation. Its todo ids name different items than main's, so cite its work by file
and title. The roadmap's section *Devrush as a source of tests and examples* is the rule this skill
follows.

1. **Find** what exercises the feature:
   `git grep -n -i "<terms>" origin/devrush -- tests examples bench`. Most of it sits in
   `tests/core`, `tests/linalg`, `tests/integration`, `examples/case_studies`, `examples/ocp`
   and `examples/opt`. Read a candidate with `git show origin/devrush:<path>`.
2. **Choose** the ones that show the feature working for a user. Skip the ones that test
   devrush's internals. A candidate that needs something main lacks becomes a follow-up issue,
   not a port of devrush's implementation.
3. **Port the problem and its reference values, not the code.** Rewrite against main's current
   API and its function-authoring conventions, using public names only.
   - A test goes under `tests/`, as `docs/dev/contributing.md` places it, and checks against NumPy
     or an unrolled reference.
   - An example goes under `examples/`, self-contained, and is linked from the guide with
     `write-docs`.
   - A benchmark problem goes under `benchmarks/problems/`, with its gates in `checks.py`, as the
     benchmarks README describes.
4. **Check the reference values.** Devrush's review reproduced 13 bugs, so recompute each
   reference independently where that is cheap rather than copying devrush's numbers. Keep the
   source and license of any vendored data.
5. **Prove the port.** It fails without the feature, or on a perturbation of it, and passes with it.
6. **Record** on the issue or pull request each ported file with its devrush source by path and
   title, and what was left out and why.
