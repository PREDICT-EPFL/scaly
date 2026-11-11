# Forward derivatives through LOOP

Implement issue #32 on the approved local #34 branch, following the fixed design in
`internal/notes/core_compiler_roadmap.md`. The compiler run plan and
`~/.scaly-suite/RULES.md` govern checks and tracker access.

- [x] Read the task, stack, design, implementation and devrush test sources.
- [x] Add differential tests and observe the current forward-mode rejection.
- [x] Extend the existing LOOP forward rule with primal and tangent carries.
- [x] Cover edge counts, slicing, parameters, dtypes, nesting and trajectories.
- [x] Check RK4 derivatives and record C sizes, operations and compile times.
- [x] Run focused checks and the detached suite through the shared runner.
- [ ] Open the stacked pull request when #34 has one; obtain Opus review and delta check.
- [ ] Resolve required stack rebases, validate, and report ready to queue.

Evidence so far: 75 focused tests passed. The original scan tests failed with the
existing forward rejection. Zeroing initial tangent carries makes the unrolled
comparison fail, and the rule was restored. RK4 at N=200 agrees with central
finite differences and the Python-unrolled graph.

On la015, generated gradient C at N=20/200/2000 has 22963/22995/23066 bytes
and 1008 program nodes at every length. C compilation took
0.1909/0.2241/0.3122 seconds, using the module recipe and JIT compile command.
These are compile times under concurrent work, not runtime performance claims.

The devrush sources ported are tests/core/integration/test_multiseed_loops.py
(scan edge lengths, shared inputs, symbolic seeds, nested scans/maps, nested forward)
and tests/core/integration/test_loop_edge_cases.py (nonfloating carry tangents).
Reverse-mode cases, trajectories in reverse, and local compression remain out of scope.

Full shared-runner check passed: 4787 passed, 56 skipped; type, format, lint,
benchmark smoke, docs build and private-name checks passed. C snapshots unchanged.
The first runner attempt stopped on the stale node-ID baseline, now refreshed.
#34 has pull request #425. Open this layer against t3code/34-fold-vmap.

Opus review found the custom-JVP nonfloating stacked output offset bug. Added
int64/bool reproductions, both failed, then fixed the tangent output list to use
the same floating indices as result selection. 100 focused loop/sugar tests pass.
Removed premature guide/API publication, retaining accurate source docstrings.
Kept the existing reverse error wording as explicitly required by the brief.
Documented n_carry in HelperKey and removed the misleading pack stem from
condition helper names. Delta check pending after the full runner.

Known cost for the PR: a forward Jacobian seeded in a sliced input materializes
a seed table growing with steps times seeds. Reverse mode (#33) and deferred
per-step compression own this limitation. No new follow-up issue is needed.
