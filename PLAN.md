# Sparse-sparse products, #47

Implement the decided sparse product kernel and stored-coordinate derivatives on top of #367,
branch `t3code/47-sparse-sparse`. The unattended compiler run plan and `~/.scaly-suite/RULES.md`
override the usual maintainer wait, snapshot serialization and run-time measurement rules.

Read: the compiler roadmap's decisions 4 and 5, Sparse tensors, release scope and item rules;
compiler maintenance context; section 1's sparse_values packet; #46's plans, mixed products,
stored tangents and sampled adjoints. Publish sparse documentation in #307, not here.

- [x] Read the brief, verify the clean linked checkout and approved parent, inspect issue #47
      and section #292. Project REST endpoints return 404, so project status is unavailable.
- [x] Add SciPy and finite-difference differential tests over empty, rectangular, banded, random
      and Gram products, in CSC, CSR and coordinate layouts. Observe the missing-kernel failure.
- [x] Add the column accumulator plan in lowering/plans.py, with compressed active-column
      tables bounded by operand and result stored entries, and direct layout writes.
- [x] Add sampled sparse-sparse reverse rules; reuse sparse tangents and the product kernel.
- [x] Port small MPC normal-equation and finite-element restriction problems from the frozen
      devrush notebooks, without their solver implementations.
- [x] Pin table and workspace growth for weighted Gram products and perturb each new gate.
      Record operation counts, generated C size and C compile time; defer execution timings.
- [ ] Run focused checks, regenerate test node IDs, then detached ~/.scaly-suite/run on the
      final review state. Dense C snapshots must remain unchanged.
- [ ] Commit and push once for the draft PR against t3code/46-sparse-ad; register with gh stack
      and link in T3. Run cross-review with Opus and sol; fix findings and get one delta check.
- [ ] Publish the verdict immediately, update the issue, mark ready. After #367 merges, retarget
      through REST to main without rebasing unless required by conflicts, then queue as RULES
      permits. Watch CI and address failures. Delete this temporary plan before ready.

Scope limits: no sparse derivative leaves, per-entry seeds, SELECT, pattern blocks, traversal
fusion, sparse factorization or public sparse documentation. Preserve reviewed parent code
except sparse-sparse cases. Never push the parent branches or main.
