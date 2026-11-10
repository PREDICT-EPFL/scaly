# Sparse derivative result types, #356

Base: `origin/main`. The issue, merged PRs #328 and #354, the compiler-run plan,
`~/.scaly-suite/RULES.md`, the compiler roadmap's Signatures and templates and Sparse tensors
sections, and the frontend's existing arity ladders define this task.

Scope: static gradient, forward and adjoint result and seed types through concrete calls,
groups and maps. Runtime differentiation, layouts, reconstruction and Jacobian/Hessian types
stay unchanged. Nearby PRs #335 and #341 may merge; rebase only for a conflict.

- [x] Inspect the issue, prerequisites, run rules, architecture and implementation.
- [x] Add static regressions for the four reported cases, packed outputs from dense inputs,
      concrete sources/calls, groups/maps and declarations proving density. Show them failing.
- [x] Correct the wrappers' types and make the dense shape-hole tradeoff explicit. Keep
      uncertain patterns on the numerical union without adding a Tree type parameter.
- [x] Check runtime derivatives and inherited layouts against NumPy, including sparse maps;
      prove the static gate fails under a targeted annotation perturbation.
- [x] Run strict types on all CI targets, Ruff, snapshots/import checks and the detached full
      pre-merge hook through the shared suite slot. Record results here.
- [ ] Open and link the draft PR, obtain one cross-review and any required delta check, then
      record completion evidence and the typing tradeoff in the PR. Delete this plan before ready.
- [ ] With the verdict and green checks, queue the PR once with squash auto-merge. Settle review
      delegates and report the PR state, decisions, open points and follow-ups.

The checks precede review. Review fixes precede their focused checks and the single final push.
No performance claim, runtime compiler change or snapshot regeneration is planned.

Evidence so far: the new static cases failed with 57 diagnostics on the original annotations.
Restoring NumPy-only result annotations separately for gradient, forward and adjoint fails with
15, 15 and 10 diagnostics. The fixed repository passes strict ty for the default, Python 3.14
and Windows targets, plus both Ruff checks. The focused runtime run passes all 97 cases.
The roadmap removes the #356 gap and records the conservative typing choice, with no departure.
The broader focused run passed all 251 frontend/snapshot/import cases, and the final full
collection passed. The detached shared-slot pre-merge hook passed with exit 0: strict types,
both Ruff checks, 3,360 tests passed and 52 skipped, benchmark smoke, documentation build with
no issues, and the private-name check. Log path is in `/tmp/scaly-356-suite-path`.
