# Integer and bool mean

Implement #430 on `t3code/430-integer-mean`, based on main. This is the first branch in the
stack followed by #479 and #480. The maintainer's 2026-11-18 issue comment fixes NumPy's dtype
rule. Read the issue and parent #296, the compiler roadmap's dtype section and item rules,
compiler maintenance context, project guides, and `~/.scaly-suite/RULES.md`.

Scope is `Expr.mean`, its docstring, the mean dtype sentence in the functions guide, and one
focused NumPy regression test. Preserve integer accumulation, cast its result before division,
count bool values in int64, preserve float32, and return correctly typed NaNs for empty domains.
Do not edit the other batch tasks' lines or regenerate any C snapshots. Stop if a snapshot moves.

- [x] Inspect task, dependencies, branch and neighboring code.
- [ ] Write and run a failing focused regression for int32, int64, bool, float32, an axis tuple,
      keepdims, and an empty reduction domain.
- [ ] Implement and document the fixed rule, then verify the test detects a perturbation.
- [ ] Run focused checks and the complete pre-merge checks detached through the suite runner.
- [ ] Commit and push the reviewed state, open a draft PR against main, and link it in T3.
- [ ] Obtain the requested claude-opus-5-5 high review, address findings, and check fixes if needed.
- [ ] Remove this temporary plan, mark the PR ready, update the task, and report criteria,
      review verdict, CI state, open points and follow-ups. Never merge or queue.

Each phase depends on the previous phase. The suite runs only on a state intended for pushing.
Focused pytest commands use at most two workers. GitHub reads use REST and pushes are batched.
