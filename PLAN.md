# Reverse loop derivatives (#33)

Base: local `t3code/32-loop-forward-ad`, pull request #426. Implements the decided
reverse rule in `internal/notes/core_compiler_roadmap.md`, including trajectory
cotangents and per-loop storage limits. Run rules in compiler_run_2026_11_10/plan.md
and ~/.scaly-suite/RULES.md apply. No queue until instructed.

- [x] Read brief, run rules, base verdict and loop derivative implementation.
- [x] Add failing differential and structural tests; inspect and port devrush cases.
- [x] Implement shared backward loop, trajectory differentiation and max_trajectory.
- [x] Check higher derivatives, edge cases and RK4 sizes/counts/compile times.
- [x] Publish scan/while_loop guide and API; publish cond only if merged.
- [ ] Focused checks, byte-identical C snapshots, detached full suite.
- [ ] Commit, push review state, open/link stacked PR; Opus and sol review.
- [ ] Fix findings, short delta checks, final checks and ready-to-queue report.
