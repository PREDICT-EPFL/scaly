# Add external C calls (#83)

The compiler run plan on main and ~/.scaly-suite/RULES.md override the normal implement workflow.
Base: origin/main. Branch: t3code/83-extern. PR base: main. No stack.

Read: #83 and its re-cut comment, parent #294, closed prerequisites #28/#11/#82/#80;
the compiler roadmap decisions 7 and 10, external-code contract and rules for every item;
architecture, codebase, conventions, contributing and compiler maintenance context.
The frozen devrush extern tests supply the compile/nest/reject cases, rewritten for this API.

- [x] Inspect ownership, prerequisites, overlap and the existing compiler paths.
- [x] Add failing end-to-end and boundary tests for invocation grouping, compiled source,
      named system linking, cache content, derivative rejection/custom rules and AOT manifest.
- [x] Add frozen library descriptions and concrete extern declarations, EXTERN verification,
      AD handling, lowering and rendering with workspace and sparse leaf layouts.
- [x] Compile external sources separately, resolve named libraries in the toolchain, include
      all native content in the JIT key, and export a portable AOT build manifest.
- [x] Add public exports/docstrings, explicitly exclude API names from published docs and write
      the internal section note and end-to-end example test. No guide page or plugin changes.
- [ ] Run focused checks, perturb new gates, then all pre-merge checks through the detached
      ~/.scaly-suite/run queue. Existing C snapshots must remain byte-identical.
- [ ] Commit, push once, open/link a draft PR and get one claude-opus-5-5 high review.
- [ ] Fix findings together, run relevant checks, get one short delta check on fixes only,
      update the review verdict, mark ready and queue once with --squash --auto.

Expected overlaps: JIT cache PRs #321/#310/#316 and frame budget #314; expression/lowering
PRs #320/#331/#349/#337/#329/#312. Rebase only for an actual conflict or queue ejection.
Public interface requires the maintainer's later section review. No run-time performance claims.

Focused evidence: 23 extern tests and 201 surrounding tests passed, including unchanged C snapshots.
All six completion gates failed under deliberate perturbation and were restored. Ruff/ty pass.
Full suite and all pre-merge checks are next, through the shared queue.
