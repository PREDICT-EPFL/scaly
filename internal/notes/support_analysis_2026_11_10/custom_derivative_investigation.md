# Custom derivatives and value support in `ad/sparsity.py`

Independent investigation for PREDICT-EPFL/scaly#40, 2026-11-10. It asks whether Jacobian sparsity
may drop the rows of a node whose value is identically zero when a custom derivative rule sits
upstream. Line numbers refer to the branch `t3code/40-support` on this date.

Two variants of the last phase of `ad/sparsity.py::_jac_mask` are compared:

- **blanket filter** (earlier): `memo[key] = _supported_rows(_jac_mask_uncached(...), node.support)`,
  applied to every node.
- **current**: no filter on every node. The `INPUT` branch filters by `expr.support`
  (`ad/sparsity.py:139-141`), and the `MUL`, `DIV` and `MATMUL` rules filter one operand's relations
  by the other operand's support (`ad/sparsity.py:150-155`, `243-250`).

## Conclusion

1. A custom rule that is nonzero where the value is identically zero is a legitimate declaration
   under the recorded contract. Nothing recorded lets Scaly reject or ignore it.
2. The blanket filter does not implement "never `expr.size` rows". It keeps matrices of
   `expr.size` rows and removes entries. On graphs without custom rules it removes nothing. On
   graphs with custom rules it removes entries the derivative engine computes, and the entries it
   keeps then hold wrong values.
3. The current code is the smallest change the contract justifies, and it keeps all the pruning of
   the design on ordinary paths. No further code change is needed.

Design settled by existing contract. The one open point is wording in the roadmap that matters to
#41, described under [What #41 inherits](#what-41-inherits).

## 1. Is the declaration legitimate?

### What the repository states

| Source | Statement | Kind |
| --- | --- | --- |
| `function/api.py:458` (docstring of `custom_derivative`) | "Return `fn` with derivative rules that replace differentiating its body." | explicit |
| `function/api.py:472-474` | `sparsity(of, wrt, shape)` "gives the Jacobian pattern of output `of` in input `wrt`". "Without it, the pattern of a function with rules is dense." | explicit |
| `docs/guide/derivatives.md:397-399` | "Sparsity analysis cannot see the derivative a rule computes, so the Jacobian pattern of a function with rules is dense unless it is declared." | explicit |
| `docs/guide/derivatives.md:411-414` | "A declared pattern is taken as given. Scaly does not compare the declared pattern with the rule." | explicit |
| `ad/sparsity.py:262-267` (docstring of `_callee_mask`) | "A function with custom rules has the pattern it declares, or a dense one; its body need not describe the derivative its rules compute." | explicit |
| `core_compiler_roadmap.md:572-588`, #22 | "Declared sparsity becomes pattern blocks at construction, which `ad/sparsity.py` reads without importing Function; without a declaration the pattern is dense." Every derivative of a Function body goes through the rule, "so `CALL`, `LOOP` and the map case all honour the rule". | explicit, recorded decision |
| `core_compiler_roadmap.md:119-124`, decision 10 | "custom behaviour lives at Function boundaries", in JAX's `custom_vjp` form. | explicit, recorded decision |
| `docs/guide/derivatives.md:384-385` | "The outputs of `fwd` replace those of the body and should equal them." | explicit, about values only |

No recorded text says that a rule must equal the mathematical derivative of the body, and no code
checks it. `custom_derivative` validates only shapes and dtypes of the rules
(`function/api.py:494-512`) and the shape of a declared pattern (`function/api.py:522-526`).

### Mathematics against registered rules

If a function is identically zero in some entry, the mathematical derivative of that entry is zero.
A rule that returns something else there is not the derivative of the body. The contract does not
make this an error: the rule replaces differentiation, and the pattern is what the user declares or
dense. The pattern of a function with rules is therefore defined by the rule and its declaration,
never by the body. The value support of the body says nothing about it.

Two points support reading this as intended and not as an oversight. Both are inferences.

- `sc.stop_gradient` is already a derivative that deliberately differs from the derivative of the
  evaluated value: "Finite-difference checks perturb the input and recompute the stopped value, so
  they need not agree with this derivative" (`docs/guide/derivatives.md:300-302`).
- Decision 10 names JAX's `custom_vjp` as the form. JAX's own guide
  (<https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>) uses
  it for gradient clipping, where the rule is deliberately not the derivative of the identity
  function it decorates. This informs the reading and decides nothing.

### The requirement that does not depend on the reading

Whatever one thinks of such a rule, the dense and sparse Jacobians of one expression must agree,
and a pattern must contain every entry that may be nonzero
(`docs/guide/sparsity.md:49-53`: "`nnz` is the number of entries that may be nonzero", and extra
entries are allowed, missing ones are not). `ad/forward.py` and `ad/reverse.py` do not read
`support` at all, so the dense derivative always honours the rule:

```console
$ grep -rnE "\.support\b|broadcast_support" src/scaly/ad
src/scaly/ad/sparsity.py:141:    return _supported_rows(mask, expr.support)
src/scaly/ad/sparsity.py:152:      xm = _supported_rows(xm, broadcast_support(y, expr.shape))
src/scaly/ad/sparsity.py:154:      ym = _supported_rows(ym, broadcast_support(x, expr.shape))
src/scaly/ad/sparsity.py:243:  xs, ys = x.support, y.support
```

A pattern that drops those rows is wrong for the derivative Scaly itself computes. Making the two
agree the other way, by zeroing tangent rows outside the value support or by rejecting the
declaration, would be new behaviour that contradicts "taken as given" and "its body need not
describe the derivative its rules compute". That would be a new design decision. Keeping the rows
is not.

One existing bound on a custom rule is real and separate: an output with a sparse type keeps its
stored pattern, "A sparse output's stored pattern needs no rule: it is part of the declared output
type" (#22). That is a storage contract of the type (decision 4), not value support of the body.

## 2. What the blanket filter does

### The sentence in the design

> `ad/sparsity.py` consults support in `MUL`, `DIV`, `SELECT` and `MATMUL`, so a dense constant with
> zero blocks no longer gets a dense Jacobian, and keeps its relations only over each node's nonzero
> entries, never `expr.size` rows.

The sentence has two parts. The first is a rule per op: the coefficient of a relation is a value,
and where that value is identically zero the relation vanishes. The second is about how relations
are stored: over the nonzero entries of a node and not in a structure with `expr.size` rows. #41
names the same thing as a defect it fixes: "`ad/sparsity.py` allocates CSR row pointers of length
`expr.size`" (`core_compiler_roadmap.md:939-945`). The second part is a statement about the row
domain of the stored relation. It is not an instruction to delete derivative rows by value support.
This reading is an inference from the wording and from #41, since the sentence does not mention
custom rules.

### What the filter does

`_supported_rows` (`ad/sparsity.py:104-109`) converts the mask to coordinate form, tests each stored
row in a Python loop and rebuilds a CSR array with `shape=mask.shape`. So:

- the matrix still has `node.size` rows and a row pointer of length `node.size + 1`. The storage
  part of the sentence is not implemented by it, and under the stated scope it belongs to #41;
- it adds a pass over every stored entry of every node;
- it encodes the assumption "a row outside the value support has no relation". That holds for every
  op whose derivative comes from the engine's own table. It is false at a `CALL` or `VMAP` of a
  function with rules, and for every node downstream of one.

### It removes nothing on ordinary paths

Instrumenting the current code to record every node whose mask has an entry in a row outside
`node.support`, over the whole suite:

```python
# rowcheck.py, loaded with: ROWCHECK_LOG=log PYTHONPATH=<dir> uv run pytest -p rowcheck -n=auto tests plugins typing_playground
import os
import scaly.ad.sparsity as sp

_orig = sp._jac_mask_uncached

def _checked(node, variable, memo):
  mask = _orig(node, variable, memo)
  if node.support is not None and sp._supported_rows(mask, node.support).nnz != mask.nnz:
    with open(os.environ["ROWCHECK_LOG"], "a") as f:
      f.write(f"{os.environ.get('PYTEST_CURRENT_TEST', '?').split(' ')[0]}\t{node.op}\n")
  return mask

sp._jac_mask_uncached = _checked
```

Result: 2715 passed, 52 skipped, and 98 recorded nodes, all of them in
`tests/ad/test_custom_derivative.py::test_custom_derivative_rows_outside_primal_support_survive_composition`
(ops `call`, `vmap`, `reshape`, `sin`, `mul`, `div`, `matmul`, `concat`). No other test in the suite
produces a mask entry outside a node's support. The one failure,
`tests/test_op_coverage.py::test_every_expr_op_is_classified_in_every_pass`, comes from the
instrumentation replacing the function that test inspects. It passes without the plugin.

This matches an induction over the rules: with the `INPUT` base case and the `MUL`, `DIV` and
`MATMUL` coefficient filters, each rule maps operands whose rows lie inside their supports to a
result whose rows lie inside its support. `tests/ad/test_sparsity.py:57`
(`test_ordinary_derivative_relations_stay_within_each_nodes_support`) checks the same property.
The induction is an argument, not a proof over every op, which is why the suite run is reported.

### It gives wrong values with custom rules

The reproduction from the task, with later operations added, run once as is and once with the
blanket filter restored by wrapping `_jac_mask_uncached`:

```python
# repro.py: `uv run repro.py` for the current code, `uv run repro.py blanket` for the earlier filter
import sys
import numpy as np
import scaly as sc
import scaly.ad.sparsity as sp

_orig = sp._jac_mask_uncached
if sys.argv[1:] == ["blanket"]:
  sp._jac_mask_uncached = lambda node, variable, memo: sp._supported_rows(_orig(node, variable, memo), node.support)

@sc.function(sc.arg("x", 3), outputs=sc.arg("y"))
def masked(x):
  return x * sc.const([0, 1, 0])

@sc.function(sc.arg("x", 3), sc.arg("dx", 3), outputs=sc.arg("dy"))
def identity_jvp(x, dx):
  return dx

eye = lambda of, wrt, shape: np.eye(*shape, dtype=bool)
declared = sc.custom_derivative(masked, jvp=identity_jvp, sparsity=eye)
undeclared = sc.custom_derivative(masked, jvp=identity_jvp)

@sc.function(sc.arg("x", 3), outputs=sc.arg("y"))
def nested(x):
  return declared(x)

M = sc.const(np.arange(1.0, 10.0).reshape(3, 3))
cases = {
  "declared": lambda q: declared(q),
  "undeclared": lambda q: undeclared(q),
  "nested": lambda q: nested(q),
  "sin": lambda q: declared(q).sin(),
  "mul": lambda q: declared(q) * sc.const([2.0, 3.0, 4.0]),
  "div": lambda q: declared(q) / sc.const([2.0, 3.0, 4.0]),
  "matmul": lambda q: M @ declared(q),
  "sum": lambda q: declared(q).sum().reshape(1),
  "ordinary": lambda q: masked(q),
  "ordinary_sin": lambda q: masked(q).sin(),
}
for name, build in cases.items():
  @sc.function(sc.arg("q", 3), outputs=sc.group(sc.arg("dense"), sc.arg("sparse")))
  def derivatives(q):
    y = build(q)
    return sc.jacobian(y, q), sc.sparse_jacobian(y, q).to_dense()
  dense, sparse_ = derivatives(np.array([0.1, 0.2, 0.3]))
  print(f"{name:13s} agree={np.array_equal(dense, sparse_)} dense={dense.tolist()} sparse={sparse_.tolist()}")
```

| Case | Dense Jacobian, both variants | Sparse, current | Sparse, blanket filter |
| --- | --- | --- | --- |
| `declared`, `undeclared`, `nested` | `I` | `I` | `diag(0, 1, 0)` |
| `sin` | `diag(1, 0.98, 1)` | same as dense | `diag(0, 0.98, 0)` |
| `mul` by `[2, 3, 4]` | `diag(2, 3, 4)` | same as dense | `diag(0, 3, 0)` |
| `div` by `[2, 3, 4]` | `diag(1/2, 1/3, 1/4)` | same as dense | `diag(0, 1/3, 0)` |
| `M @ declared(q)` | `M` | `M` | columns 0 and 2 zero, column 1 `[6, 15, 24]` |
| `sum` | `[1, 1, 1]` | `[1, 1, 1]` | `[0, 3, 0]` |
| `ordinary`, `ordinary_sin` | `diag(0, 1, 0)`, `diag(0, 0.98, 0)` | same as dense | same as dense |

The last two custom rows are the important ones. With `M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]` the
true column 1 is `[2, 5, 8]`. The filtered pattern says columns 0 and 2 are empty, the compression
puts all three columns in one pass, and the entry it keeps receives their sum `[6, 15, 24]`. A
pattern that is too small does not only drop entries. It corrupts the ones it keeps.

The same holds at Function level and through a map: `sc.sparse_jacobian` of a Function returning
`ruled(q) * [2, 3, 4]` gives `[2, 3, 4]` now and `[3]` with the filter, and a two-stage
`sc.vmap(ruled, 2)` gives six stored ones now and two with the filter, while `sc.jacobian` gives
the six in both. A rule on a function whose value is zero everywhere (`x * sc.const([0, 0, 0])`)
also keeps dense and sparse derivatives equal under the current code through `sin`, `+`, `*`, `/`,
`@`, `sum`, a slice and `concat`, so the zero folding of `passes/expr.py:265-275` does not drop the
rule on these paths.

## 3. The smallest justified change

The current code is it. Its parts fall into two groups.

**Required by the contract.**

- No filter by the support of the node being differentiated, except where the node's derivative
  comes from the engine's own rule. Removing the blanket filter is the whole fix.
- `_callee_mask` returning the declared or dense pattern (`ad/sparsity.py:262-274`), which was
  already there.

**Precision the design asks for, sound with custom rules.**

- The `MUL`, `DIV` and `MATMUL` filters. They use the value support of the *coefficient*: in
  `d(x*y) = dx*y + x*dy` the term `dx*y` vanishes where the value `y` is zero, whatever rule
  produced `dx`. Values are not changed by custom rules, so these filters stay sound downstream of
  one. The `mul`, `div` and `matmul` rows above confirm it. They are what gives "a dense constant
  with zero blocks no longer gets a dense Jacobian" (`tests/ad/test_sparsity.py:16`, `:33`).
- The `INPUT` filter. An input's support is `None` unless its type has a stored pattern or it is
  empty (`ir/support.py:35-45`), and entries outside a stored pattern are identically zero by
  decision 4. Removing it fails exactly one test,
  `tests/ad/test_sparsity.py::test_sparse_inputs_keep_only_stored_derivative_relations`, and it is
  the base case that makes the blanket filter redundant elsewhere.

**Not needed.** Restoring any filter on `CALL` or `VMAP` for functions without rules, or on unary
ops, gains nothing: the suite run above shows no entry for it to remove. Rejecting such a rule at
declaration, or zeroing tangents outside the value support, is not justified by anything recorded
and would be a new decision.

`SELECT` is absent on this base. When it lands, its rule should follow the same principle as `MUL`:
filter by the constant condition, never by the support of the result.

## What #41 inherits

The second half of the #40 sentence, read literally, says relations are kept "only over each node's
nonzero entries". When #41 replaces the `expr.size`-row CSR arrays with a compressed row domain,
that domain cannot be the value support alone: for a `CALL` or `VMAP` of a function with rules it
has to include the rows of the declared (or dense) pattern, and the rows of every node downstream.
The existing contract already forces this, so it is not a new decision, but the roadmap sentence
does not say it. A sentence in #40 or #41 such as "the row domain of a relation is the value
support, extended by the declared rows of a function with custom rules" would stop the next
implementer from repeating the blanket filter in the new storage. That edit is the maintainer's;
this memo changes no roadmap text.

## Verdict

Design settled by existing contract.

Next action: keep `ad/sparsity.py` as it is on this branch, with no blanket filter, the `INPUT`
filter and the `MUL`, `DIV` and `MATMUL` coefficient filters. Propose to the maintainer the roadmap
sentence above for #41, which does not block #40.
