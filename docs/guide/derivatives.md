# Derivatives

Every derivative in alloy is another `Function`. It is built from the original graph, it compiles
the same way, it can be composed into a bigger graph, and it can be written out as C. There is no
tape and no separate derivative runtime.

## The quick way

```python
al.gradient(fn, "x", "f")     # d f / d x, for a scalar output f
al.jacobian(fn, "x", "y")     # dense Jacobian, shape (y.size, x.size)
al.hessian(fn, "x", "f")      # second derivatives of a scalar output
al.spjacobian(fn, "x", "y")   # compact nonzero Jacobian values, with the pattern
al.sphessian(fn, "x", "f")    # compact nonzero Hessian values, with the pattern
al.forward(fn, "x", "y")      # seeded forward mode: J @ seed
al.adjoint(fn, "x", "y")      # seeded reverse mode: J' @ cotangent
```

Each returns a one-output `Function`. Pass `name=` to choose its name, and `extra_inputs=` to carry
additional inputs through to the result — useful when the derivative has to be called with the same
parameters as the original.

The two seeded ones take an **extra input** for the seed: `fwd:x` for `al.forward`, `lam:y` for
`al.adjoint`. Check `input_names` before calling them.

## The factory

The wrappers above are convenience over one mechanism. When you want several derivatives out of one
function, ask for them together:

```python
combined = fn.factory(
    "fn_all",              # name of the new Function
    ["x", "p"],            # its inputs, by name
    ["f", al.grad("f", "x"), al.spjac("g", "x")],
)
combined.output_names      # ('f', 'grad_f_x', 'spjac_g_x')
```

This is worth doing rather than building three separate functions. The value and its derivatives
share subexpressions, and asking for them in one function means those are computed once, in one
compiled artifact, instead of three times in three.

An output request is either a **string**, naming an existing output to pass through, or a **spec
object** describing a derivative to build.

## The spec kinds

| Spec | Produces | Derived output name |
| --- | --- | --- |
| `al.jac(of, wrt)` | dense Jacobian | `jac_<of>_<wrt>` |
| `al.grad(of, wrt)` | gradient of a scalar output | `grad_<of>_<wrt>` |
| `al.hess(of, wrt[, wrt2])` | Hessian, or the mixed partial when `wrt2` differs | `hess_<of>_<wrt>_<wrt2>` |
| `al.spjac(of, wrt)` | compact nonzero Jacobian values, plus the pattern | `spjac_<of>_<wrt>` |
| `al.sphess(of, wrt[, wrt2])` | compact nonzero Hessian values, plus the pattern | `sphess_<of>_<wrt>_<wrt2>` |
| `al.fwd(of, wrt)` | `J(of, wrt) @ fwd:<wrt>` | `fwd_<of>_<wrt>` |
| `al.adj(of, wrt)` | `J(of, wrt)' @ lam:<of>` | `adj_<of>_<wrt>` |

These are typed objects, not strings: an editor can complete them, and there is no grammar to
learn or mis-spell. A spec constructs freely — the names are checked when `factory` resolves them
against the function, which raises `ValueError` naming the offending request. If you know CasADi's
`"jac:eq:z"` factory strings, this is the same idea with the parser removed — see
[the comparison](../how_it_works/comparison.md#casadi).

The derived names matter: they become the generated C symbols and the header's sparsity table
prefixes, so renaming an output moves symbols.

## Seeds and duals

The seeded modes need an extra input, and the factory creates it for you under a fixed naming
convention:

- `fwd:<input>` — a forward seed, the same shape as that input.
- `lam:<output>` — a dual (cotangent), the same shape as that output.

Ask for them by name in the inputs list:

```python
adj = fn.factory("fn_adj", ["x", "p", "lam:g"], [al.adj("g", "x")])
adj.input_names    # ('x', 'p', 'lam:g')
```

Input names stay strings on purpose — they name things that already exist, or follow one of these
two conventions, so there is nothing for a type to catch.

## Lagrangians

A Lagrangian is a weighted combination of outputs, which the factory builds through `aux`:

```python
lag = fn.factory(
    "fn_lag",
    ["x", "lam:f", "lam:g"],
    [al.sphess("gamma", "x")],
    aux={"gamma": ["f", "g"]},
)
```

`aux` declares a new output — here `gamma` — as `sum(lam:<name> * <name>)` over the listed outputs,
in that order. Then differentiate it like any other output. The order is load-bearing: it fixes
which dual multiplies which constraint block, which is what makes the resulting Hessian correct.

The wrappers cover the usual case:

```python
al.lagrangian_hessian(fn, "x", ["f", "g"])          # dense
al.sparse_lagrangian_hessian(fn, "x", ["f", "g"])   # compact, with the pattern
# inputs: ('x', 'lam:f', 'lam:g')   output: ('sphess_gamma_x_x',)
```

This is exactly what `al.nlp(...)` builds for IPOPT — there is no privileged internal path.

## Working on expressions directly

Below the named-function layer, the primitives operate on `Expr` graphs:

```python
seed = al.sym("seed", y.shape)
(vjp_x,) = al.vjp((y,), (x,), (seed,))       # reverse mode
tangent = al.jvp(y, x, seed)                 # forward mode

seeds = al.sym("seeds", (4, *x.shape))
batched = al.jvp_many(y, x, seeds)           # shape (4, *y.shape)

al.expr_jacobian(y, x)
al.expr_gradient(y, x)
al.expr_hessian(y, x)
```

Use these when you are building something the factory does not cover. For anything the factory does
cover, prefer the factory — it names the result, attaches sparsity metadata, and gives you one
compiled artifact.

## What it costs

`gradient` is one reverse sweep, so a scalar objective costs about one function evaluation
regardless of how many inputs it has.

`jacobian` is forward mode, batched: all `x.size` identity columns go through in a single pass, with
the expensive shared parts computed once. For a wide input this is much better than a loop of
sweeps, and still worse than exploiting sparsity — which is the next page.

`hessian` is `jacobian` of `gradient`: reverse, then batched forward.

Some operations have no multi-seed forward rule yet — `abs`, `asin`, `acos`, `atan`, `atan2`,
`minimum`, `maximum`, `floor`, `ceil`, and `transpose` producing rank 4 or higher — and any graph
containing one falls back to evaluating seeds one at a time. The fallback is correct and quiet. Set
`ALLOY_STRICT_JVP_MANY=1` to make it raise instead, which is what you want when you are
investigating why a Jacobian is slower than you expected.

Derivatives through `call` and `map_` preserve the structure rather than expanding it — see
[how differentiation works](../how_it_works/autodiff.md). Derivatives through a
`solver_call` are zero; implicit differentiation of a solve is future work.
