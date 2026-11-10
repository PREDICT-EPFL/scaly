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
Focused tests passed, including all 22 new cases. Perturbations detected disabled support
specialisation, disabled zero pruning and doubled runtime-seed values. The updated full pre-merge hook passed, 2,967 tests and 52 skips, plus smoke, docs and private-name checks.
Log: `/tmp/scaly-167-suite-review.log`; exit file: `/tmp/scaly-167-suite-review.exit` contains 0.


## Review handoff

Draft PR: https://github.com/PREDICT-EPFL/scaly/pull/347. Sol approved without findings. Opus's
xhigh review found one test gap: counting visits per support lookup allowed fresh contexts to
hide duplicate work. The test now requires exactly three contexts and fails when sharing is
perturbed. Also skipped inactive call specialisation and added CALL/VMAP cache-fallback and
inline/call/map custom-derivative regressions. All new gates were perturbed and failed.

The one Opus low-effort delta check failed before returning a verdict because the Claude API
hit its rate limit. No substitute reviewer was launched. Confirmation is still missing; keep the
PR draft. Once the provider resumes, request only the fix delta against the original review,
not another full review. Delete this plan in the last commit before ready.

The duplicated map-window grouping is maintenance follow-up #350, labelled source:agent.
No roadmap departure. #333 remains unmerged as of 19:36 UTC, so this PR still targets its branch.
The old base is the tip of origin/t3code/221-call-support at dispatch. On its squash merge,
exclude that old base when rebasing onto origin/main, then force-with-lease and retarget through
REST. Queue once after the base is main and the review is complete. Children do not watch PRs.
