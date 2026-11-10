# Preserve mixed-tangent zeros, #167

This branch stacks on `origin/t3code/221-call-support`, PR #333. The compiler run plan in
`internal/notes/compiler_run_2026_11_10/plan.md` and `~/.scaly-suite/RULES.md` override the usual
implementation workflow. Scope is CALL/VMAP Jacobian sparsity and the minimal consumer change
needed for the mixed-tangent example. Do not edit `ir/support.py`, lower seed axes or add LOOP.

Design is the core compiler roadmap's Sparse tensors section, especially #40 and #221.
The section coordinator's evidence is on #167. Read architecture, codebase, conventions,
contributing, compiler maintenance context and the implement workflow before changing code.

- [x] Read the brief, tracker and existing code. Checkout is the prepared worktree at #333's tip.
- [x] Add regressions for top level, calls, maps, mixed selections and runtime-seed values;
      reproduce the remaining failure on #333.
- [x] Use argument-specialised support in Jacobian coefficient rules, keeping custom derivative
      declarations authoritative and nested callee walks iterative and shared by support context.
- [x] Run focused tests with at most two workers; perturb the new gates; update collection baseline.
- [x] Run the full pre-merge hook detached through `~/.scaly-suite/run`, poll inside this turn,
      and keep C snapshots byte-identical. Commit and push only the reviewable state.
- [ ] Open the stacked PR, obtain Opus cross-review plus the high-risk sol correctness review,
      fix findings and request only the delta checks required by the cross-review skill.
- [ ] When #333 merges, rebase onto main with its old tip as the exclusion boundary, retarget
      through REST, validate any conflict resolution and push with force-with-lease.
- [ ] Record review verdict and criterion evidence; queue once when the base is main.

Open PRs #320 and #313 touch `ad/sparsity.py`. Expect a conflict if either lands first. No roadmap
departure is needed. A devrush search found no mixed-tangent regression to port.

The implementation also needs two small forward-mode consumers: `forward.py` prunes an empty
movement tangent with the existing custom-derivative guard, and `calls.py` skips an empty gathered
seed before local-colour reconstruction creates a product. No callee cloning or support changes.
Focused tests passed, including all 17 new cases. Perturbations detected disabled support
specialisation, disabled zero pruning and doubled runtime-seed values. Full pre-merge hook passed, 2,962 tests and 52 skips, plus smoke, docs and private-name checks.
Log: `/tmp/scaly-167-suite.log`; exit file: `/tmp/scaly-167-suite.exit` contains 0.
