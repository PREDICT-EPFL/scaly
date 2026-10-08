# Function identity and generated names

Implements #10 and #114 against main as the bottom of a two-PR stack. #118 owns changes to lane outlining and emitted width counts.

Read the Foundations and Rules sections of internal/notes/core_compiler_roadmap.md, compiler_maintenance_context.md, the architecture, codebase, conventions and contribution checks. Source inspection covers lowering, program name allocation, scalar scheduling, ABI headers and solver rendering. Devrush's tests/core/codegen/test_name_clash.py supplies input/output collision cases; loop cases require features absent here.

- [x] Inspect issues, prerequisites, overlap and stack commands; adopt branch with gh stack.
- [x] Add regressions and reproduce failures.
- [x] Key lowering state by concrete Function identity and allocate symbols and local names with NameScope.
- [x] Route program-generated names, scalar temporaries and header names through NameScope.
- [x] Run targeted tests and perturb a correctness gate; keep C snapshots unchanged.
- [ ] Run full pre-merge checks, open draft PR, cross-review and check fixes.
- [ ] Remove this temporary plan, mark PR ready, update issue status and watch CI.

Implementation phases are sequential because they share name allocation. Independent validation commands may run in parallel. Review begins only after checks pass.
