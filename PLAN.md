# Sparse-sparse products, #47

Implement the decided sparse product kernel and stored-coordinate derivatives on top of #367,
branch `t3code/47-sparse-sparse`. The unattended compiler run plan and `~/.scaly-suite/RULES.md`
override the usual maintainer wait, snapshot serialization and run-time measurement rules.

Read: the compiler roadmap's decisions 4 and 5, Sparse tensors, release scope and item rules;
compiler maintenance context; section 1's sparse_values packet; #46's plans, mixed products,
stored tangents and sampled adjoints. Publish sparse documentation in #307, not here.

- [x] Read the brief, verify the clean linked checkout and approved parent, inspect issue #47
      and section #292. Project status is In progress, read and changed through projectsV2 REST.
- [x] Add SciPy and finite-difference differential tests over empty, rectangular, banded, random
      and Gram products, in CSC, CSR and coordinate layouts. Observe the missing-kernel failure.
- [x] Add the column accumulator plan in lowering/plans.py, with compressed active-column
      tables bounded by operand and result stored entries, and direct layout writes.
- [x] Add sampled sparse-sparse reverse rules; reuse sparse tangents and the product kernel.
- [x] Port small MPC normal-equation and finite-element restriction problems from the frozen
      devrush notebooks, without their solver implementations.
- [x] Pin table and workspace growth for weighted Gram products and perturb each new gate.
      Record operation counts, generated C size and C compile time; defer execution timings.
- [x] Run focused checks, regenerate test node IDs, then detached ~/.scaly-suite/run on the
      final review state. Dense C snapshots must remain unchanged.
- [x] Commit and push the draft PR #369 against t3code/46-sparse-ad, link in T3 and register
      #367 and #369 with gh stack, stack #370.
- [ ] Finish cross-review on the current base. Sol approved numerical correctness; Opus requested
      removing the unsampled reverse pattern and work, including Hessians. The fix carries a
      sample on MATMUL, with a stored-slot scratch row and sampled support/derivative analysis.
      New build, work, workspace, Hessian and analysis gates pass, with seven perturbations.
      Run full validation, then one delta check per reviewer. The current changes-needed verdict
      is already in #369. Publish the final verdict immediately when the checks finish.
- [ ] Wait for the coordinator's explicit notice that #367 merged. Do not restack or push
      for base changes before that notice, even if a parent rebases or main advances.
- [ ] After the notice, fetch and rebase only this task's commits onto origin/main, excluding
      the approved #367 parent. Run focused checks and one full suite through the shared runner,
      retarget #369 to main through REST, push once and queue once. Never use --auto on a
      non-main base. Watch CI and address failures. Delete this temporary plan before ready.

Scope limits: no sparse derivative leaves, per-entry seeds, SELECT, pattern blocks, traversal
fusion, sparse factorization or public sparse documentation. Preserve reviewed parent code
except sparse-sparse cases. Never push the parent branches or main.
