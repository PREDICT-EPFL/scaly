# #22 `sc.custom_derivative` with residuals

Design: `internal/notes/core_compiler_roadmap.md`, *One AD engine*, [#22], and decision 7.
Base: `t3code/issue-21-intermediate-wrt` (#171).

## Shape of the change

- `ConcreteFunction.rules: CustomRules | None` (`function/concrete.py`): the bound JVP rule, the
  bound `bwd`, the residual expressions over the function's own inputs, and the declared pattern
  blocks (`[output][input]`, `None` for dense). `results = outputs + residuals`; a `CALL` output
  index ranges over `results`, so a residual is a result of the same invocation and lowering's
  `call_invocations` dedup computes it once with the primal.
- `sc.custom_derivative(fn, *, jvp=, fwd=, bwd=, sparsity=)` in `function/api.py`. Templates go
  through `lift`. With `fwd`, the body becomes `fwd`'s outputs; its residuals are the rest.
  `jvp(*primals, *tangents) -> tangents`; `fwd(*primals) -> (outputs, residuals)`;
  `bwd(residuals, cotangents) -> input cotangents`. `sparsity(of, wrt) -> pattern`.
- `ad/calls.py`: `body_tangents`/`body_cotangents` take the callee and output indices; a JVP rule
  is mapped over the seeds; `bwd` is called on residual `CALL`s of the callee. `read_tangents`
  tells the traversal which argument tangents a rule reads. `HelperKey.rules` names the rules.
- `ad/sparsity.py`: one `callee_mask` reads declared blocks or dense for rule-bearing callees.
- Lowering: a callee procedure writes its residuals when the program uses one.

## Steps

- [ ] Tests first (`tests/ad/test_custom_derivative.py`): linearity/duality, fwd-over-rev,
      fwd-over-fwd, map case, several seeds via one map, unread tangents, sparsity, solve reusing
      its factorization (node count and C), template rules, errors. Ported devrush cases.
- [ ] `CustomRules`, `results`, validation, `custom_derivative`, public name
- [ ] AD: body_tangents/body_cotangents, traversal call sites, HelperKey
- [ ] Sparsity blocks
- [ ] Lowering of residual results
- [ ] Guide page and API entries (write-docs)
- [ ] `wt hook pre-merge --yes`, C snapshots byte-identical, #15 harness
- [ ] PR, cross-review, description; delete this file

## Out of scope / follow-ups

- LOOP (#31): it will reach the rules through `body_tangents`/`body_cotangents` unchanged.
- Map adjoint lanes recompute residuals per lane instead of sharing the primal map (#73/#90).
