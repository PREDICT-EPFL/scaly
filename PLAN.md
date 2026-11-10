# Keep select masks through calls, issue #339

The section 4 brief and `internal/notes/compiler_run_2026_11_10/plan.md` on main
govern this work. Base: `t3code/30-predicates-select-cast`, pull request #320.
The roadmap's masked partials and one AD engine decisions are fixed. The select's
condition must survive into the reverse callee body. A zero cotangent is not a mask.
Forward mode, float32 constants and lowering, and signed overflow are outside scope.

Read: the run-wide `~/.scaly-suite/RULES.md`, project guides, compiler maintenance
context, `ad/reverse.py`, `ad/rules.py`, `ad/calls.py`, `ad/helpers.py`, and the select tests.
Overlapping work: #312 in calls/reverse, #347 in calls, #349 in sparse lowering.

- [x] Inspect issue, prerequisite, parent, checkout and overlaps.
- [x] Turn the call regression around, add mapped and higher derivative differential
      cases, and observe the reported failures before changing production code.
- [x] Preserve the mask in ordinary substitutions and mapped adjoint helpers, keyed
      separately from unmasked helpers. Keep unmasked helper names unchanged.
- [x] Update the functions guide and `where` docstring, run focused tests and prove
      the regression checks detect a removed mask.
- [ ] Refresh collection baseline and run all pre-merge checks through the shared queue.
      Existing C snapshots must stay byte-identical.
- [ ] Open the pull request on #320's branch, complete cross-review with Opus and the
      required independent correctness review, and confirm any fixes once.
- [ ] After #320 merges, rebase onto main, retarget the pull request, validate, and
      queue with the review verdict. Handle queue failures and report the merge.

The phases are sequential. Independent checks and the two independent reviews may run
in parallel. Full test runs use the detached shared queue, and this turn stays alive
by polling its exit file. Push only locally validated review or merge candidates.
Delete this temporary plan before marking the pull request ready.

Focused validation: 1,261 AD, integration, snapshot and import tests passed.
Removing call masks fails both regressions; substituting cot != 0 fails both
zero-cotangent Hessians; removing the helper cache flag fails its regression.
Mapped helpers pass the condition as nondifferentiable 0/1 floats and return
input activity beside the adjoints, since #55 owns typed helper leaves. Nested
conditions also survive the structural transpose. The original #320 head must
be replaced with its rebased head before the first push, per the coordinator.
