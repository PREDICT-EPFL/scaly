# Static loop unrolling (#50)

Implement the roadmap's whole-body budget in LOOP lowering on the #31 stack.
Preserve the ordered lowered schedule, hints, static eligibility, and existing snapshots.

- [x] Add failing boundary, hint, numerical-order and triangle-folding tests.
- [x] Expand eligible static loops using the existing scalar interpreter and ordered program body.
- [ ] Run focused checks and record source size, operations, C compilation and RK4 gates.
- [ ] Run the detached suite, open the stacked pull request, obtain Opus review and delta check.
- [ ] Address base changes, checks and report readiness to the coordinator.

Focused proof: 76 tests passed including every existing C snapshot. Setting the budget to zero makes the new automatic-unroll gate fail.
Measurements on la015, GCC 13.3 -O2 C99, lanes=1: scalar sin/multiply/add body costs 3 RHS. N=84/85 unroll, 5665/5722 C bytes, 252/255 arithmetic occurrences, 0.050/0.062 s compilation. N=86/200 roll, 1228/1229 bytes, 6 arithmetic occurrences, 0.042/0.043 s compilation. Scalar N=86 overrides the budget, 5779 bytes, 258 operations, 0.049 s compilation. Timings are observations on the shared machine, not isolated performance comparisons.

First full suite passed: 3823 passed, 55 skipped; smoke, docs, types, Ruff and private-name checks passed. Added a reproduced duplicate-result registration fix so conflicting hints on the same numbered result obey block precedence. A second full suite will cover it.
Roadmap departure: preserve narrow stores, call boundaries and effects by retaining those loops even under a scalar hint. This keeps the existing scalar interpreter's safety limits and is explicitly recorded beside #50.
