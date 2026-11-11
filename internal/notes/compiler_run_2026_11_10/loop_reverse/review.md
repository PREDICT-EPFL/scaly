# Reverse loop review

## Correctness review by gpt-6.1-sol

The reviewer reproduced four defects and requested changes. All four were reproduced
as failing tests before their fixes.

1. Swapping which parameters are broadcast or sliced reused an incompatible cached
   adjoint. The helper key now records the broadcast parameter indices. The regression
   checks both a swapped partition and a changed number of broadcast parameters.
   The neighboring case of a step-number formal versus an ordinary int64 slice
   also changes the helper input order. Its key includes the index flag, with a
   regression in both cache orders.
2. A scalar parameter accumulator was scattered with vector indices without first
   flattening its values. Its transpose produced a vector carry where a scalar was
   required. Flattening before the scatter preserves the scalar shape through a
   second reverse derivative. The regression checks the expected second derivative
   of three multiplications against forward over reverse.
3. An unused carry with a singular square-root partial contaminated unrelated
   gradients through its zero cotangent. The backward rule now follows the carry
   dependencies of seeded outputs to a fixed point and differentiates only those
   carries. Regressions check the singular unused carry and coupling between two
   carries across steps. Custom rules retain their declared derivative dependencies.
4. Result nodes with different lowering hints shared one invocation and output index,
   but the seed dictionary kept only the last cotangent. Seeds now accumulate by
   output index. The regression checks a result added to its scalar-hinted alias.

The reviewer also ran 78 focused tests and 20 differential cases successfully.
The parent reran the loop and snapshot tests after the fixes. C snapshots remained
unchanged, and the RK4 source sizes and program counts remained identical to the
recorded measurements. The short delta check follows the final combined fixes.

## Main review by claude-opus-5-5

The main reviewer requested changes, including the cache partition finding above.

- User trajectory limits were inherited by generated tangent and backward loops,
  so a Hessian failed at the limit that accepted its gradient. Derived loops now
  drop the limit, and the duplicate lowering check is removed. One reverse-rule
  check counts the user's primal trajectory. The roadmap states this scope in
  one line. The regression checks Hessians and reverse over reverse in the initial
  carry, sliced inputs and broadcast parameter at exactly the primal limit.
- Several seeded outputs of an independent map took the carried-loop adjoint path.
  Every map seed now uses the existing mapped adjoint specialization. The structural
  regression checks that both output adjoints retain mapped structure. No new
  sequential-map policy or run-time measurement claim was adopted.
- PLAN.md is removed before the delta check and queueing.
- The evidence now distinguishes the frozen devrush source path from the port's
  location, names both the cond issue and its pull request, and includes the
  Hessian-at-the-limit check. The mapped while-loop finite-difference experiment
  is kept as a regression with different trip counts per entry.
- The guide explains derivatives with a fixed step count, evaluates its second
  derivative, describes reuse of the primal values without internal adjoint terms,
  explains which forward-mode storage grows and the lack of per-step direction
  compression, and shows the repeated prints in a reverse gradient. It links the
  existing general printing rules. Neighboring guides link sequential recurrence
  discussions to scan. Rendered docstrings avoid semicolons, and the scan API entry
  displays the supported signature without the reserved in_place argument.

All changed guide fences were executed together, including their new Hessian and
forward/backward print outputs. The reviewer also confirmed mixed float32/float64
carries, nested while-loop Hessians, overlapping-window source size and trip ordering.

## Delta checks

Both short delta checks approved the combined fixes with no unresolved findings
or adjacent regressions. Opus checked the source and documentation diff and did
not rerun tests. Sol independently reran the 39 reverse-loop cases, all passing.
No additional review round was requested. All four delegate review threads were
read and settled after their results were used.
