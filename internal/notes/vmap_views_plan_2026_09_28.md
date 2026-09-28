# `sc.vmap` from views: argument shapes at the call site (2026-09-28, v1)

Status: **proposed**. Addresses todo API-70 and the template half of the `NotConcrete` error that
`sc.vmap` raises on a bare `@sc.function` callee. Scope is `function/sugar.py` and the public
surface only: `ExprOp.VMAP`, its attributes, AD, sparsity and lowering do not change.

## 1. The problem

`vmap(callee, length, [(outer, start, stride), ...])` fixes where each trip's slice starts but not
its width; the width is `callee.inputs[k].size`. A template callee has no width, so it cannot be
mapped, and the call site does not determine one (overlap and gaps are both legal, so
`(z, 0, 3)` with `|z| = 30`, `length = 10` admits widths 1, 2 and 3).

## 2. What `origin/dev` already does

Tudor's "Infer vmap slicing from sizes and call solvers with parameters only" (2026-09-22, on
`origin/dev` only: not in `main`, not in `claude/integrators-mpc`) lets an entry be a bare rank-1
tensor instead of a tuple, and infers the **stride from the width**:

- `outer.size == length * formal.size` → `(outer, 0, formal.size)`, contiguous chunks;
- `outer.size == formal.size` → `(outer, 0, 0)`, broadcast;
- otherwise an error; the tuple stays for overlapping or offset windows.

It still reads `formal.size`, so the callee must be concrete, and Rosenbrock or an interleaved
`[x0 u0 x1 u1 …]` still need tuples. This plan infers the **width from the argument** instead, and
keeps dev's two rules as fallbacks for concrete callees, so both branches converge on one rule set.

## 3. Design

### 3.1 Call surface

```python
sc.vmap(term, N - 1, [x[:-1], x[1:]])            # Rosenbrock, template callee
Z = z.reshape((N, 3))
sc.vmap(f, N, [Z[:, :2], Z[:, 2]])               # interleaved stages: x_k ∈ ℝ², u_k ∈ ℝ
sc.vmap(g, N, [X, sc.broadcast(p)])              # p passed whole to every trip
sc.vmap(h, N, [(x, 0, 1), (x, 1, 1)])            # the tuple form, unchanged
```

The signature stays `vmap(callee, length, inputs, output=0)`, so every existing call is valid. A
mapped argument's leading axis is the trip; the rest of its shape is the callee's argument shape.
`sc.broadcast(e)` is a small marker (not an `Expr`) meaning "the whole of `e`, every trip"; it is
needed because a template cannot tell a mapped `(N,)` from a broadcast `(N,)`.

### 3.2 Resolving one entry (first match wins)

`s` is the formal shape when the callee is concrete; `w = prod(s)`.

| # | entry | condition | spec |
|---|---|---|---|
| 1 | `(outer, start, stride)` | callee concrete | as today |
| 2 | `sc.broadcast(e)` | — | window of `e`, stride 0; template formal `e.shape` |
| 3 | `Expr` / array | `a.shape[0] == length` and (template, or `a.shape[1:] == s`) | window of `a`; template formal `a.shape[1:]` |
| 4 | rank-1 `Expr` / array | concrete, `a.size == length * w` | window of `a.reshape((length, *s))` (dev rule) |
| 5 | `Expr` / array | concrete, `a.size == w` | as `sc.broadcast(a)` (dev rule) |
| — | anything else | | error naming the readings tried and the fix |

A tuple with a template callee is an error that says to instantiate or use views. Rule 3 before
4/5 is deterministic: for `s = ()` rules 3 and 4 give the same node; for `a.shape == s == (length,)`
rule 3 fails its shape test and rule 5 applies.

### 3.3 Reading a view in place

A view is a chain of `SLICE`, `RESHAPE` and `TRANSPOSE` over some base expression. The helper
mirrors that chain on `np.arange(base.size).reshape(base.shape)`, which gives the flat base index
of every entry of the view (NumPy is the specification; no index algebra is written by hand). With
`rows = I.reshape(length, w)`:

- if `rows[0]` is `start … start + w - 1` and `rows[k] = rows[0] + k * stride` with `stride ≥ 0`,
  the spec is `(base_flat, start, stride)`, where `base_flat` is `base` or `base.reshape((size,))`;
- otherwise the **fallback** `(a.reshape((a.size,)), 0, w)`: correct, but lowering materializes the
  view (a copy) and the structured VMAP Jacobian may not apply. Example: mapping over the columns
  of a row-major matrix, `A.T`.

Because expressions are hash-consed, the in-place case produces **the same `Expr` object** as the
hand-written tuple, so the generated C, sparsity and derivatives are unchanged by construction.
Peeling also sees through slices the user took for other reasons (the README's `zs = w[:2N+2]`
is read straight from `w`). Cost is O(base size) NumPy work at graph construction.

### 3.4 Template callees

After resolution, a template callee is instantiated once with
`callee.instantiate(*TensorType(shape_k, dtype_k, diff=diff_k))`, and the call continues exactly as
for a concrete callee. Instances are cached on the template, so repeated maps share one procedure.

### 3.5 Prototype (2026-09-28, staged copy of `claude/integrators-mpc`, CPython 3.14)

About 40 lines outside the package, calling today's `sc.vmap`:

- Rosenbrock, `[x[:-1], x[1:]]` with a bare `@sc.function` term: `is` the node of
  `sc.vmap(term.instantiate((), ()), 9, [(x, 0, 1), (x, 1, 1)])`.
- Interleaved `[Z[:, :2], Z[:, 2]]`: `is` the node of `[(z, 0, 3), (z, 2, 3)]`, starts `(0, 2)`,
  strides `(3, 3)`.
- `sc.broadcast(p)`: stride 0. A rank-2 leaf as base: outer is its flat `RESHAPE`.
- `A.T` column sums: fallback taken; JIT result equals `A.sum(0)`.

## 4. Steps

- **V0. Base branch.** Land on top of dev's change if it merges first (its bare-tensor rules become
  rules 4 and 5 here), otherwise implement rules 1–5 directly on `main`. `sugar.py` has diverged
  (`scan` was added on this branch), so expect a small conflict in `vmap` only.
- **V1. Helpers in `function/sugar.py`:** `broadcast` (marker class, public), `_view_window(a,
  length, bcast)`, `_resolve_inputs(callee, length, inputs) -> (callee_instance, specs)`. `vmap`
  calls `_resolve_inputs` and is otherwise untouched. Mapping inputs (by callee name) resolve the
  same way.
- **V2. Tests** (§5).
- **V3. Surface:** export `sc.broadcast`; update `tests/test_import_boundaries.py`; a typed overload
  for `inputs` in `typing_playground/`.
- **V4. Docs and examples:** `docs/guide/functions.md` ("Regular repetition") leads with views and
  explains the tuple form as the general case; README's defects become
  `Z = zs.reshape((N + 1, 2)); sc.vmap(defect, N, [Z[:-1], us, Z[1:]])`; migrate examples only
  where it reads better. Internal callers in `ad/`, `linalg/`, `mpc/` keep tuples.
- **V5. todo:** close API-70; add the `scan` follow-up (§7).

## 5. Tests (`tests/function/test_vmap_views.py`)

- **Identity:** for Rosenbrock, interleaved, broadcast, rank-2 formals (`B.reshape((N, nx, nu))`)
  and a slice-of-a-slice base, the view spelling `is` the tuple spelling's node.
- **Generated C:** rendered source of both spellings is byte-identical for one case of each kind.
- **Templates:** bare `@sc.function` callee through `sc.gradient`, `sc.jacobian` and
  `sc.sparse_hessian`, against NumPy; one procedure per distinct instance.
- **Fallback:** `A.T`, negative steps (`x[::-1]`), non-contiguous trip rows: value and gradient
  against NumPy.
- **Property test:** random chains of slice/reshape/transpose on random bases; the vmapped identity
  equals the NumPy view, and the in-place path is taken exactly when the rows are affine and
  contiguous.
- **Errors:** leading axis ≠ `length`; concrete shape mismatch; tuple with a template; unmatched
  sizes (message lists rules 3–5).
- **Unchanged:** the existing `tests/ad/test_vmap.py` and `tests/codegen/test_vmap.py` pass as they
  are; full suite, since the node is shared with AD and lowering.

## 6. Files touched

`src/scaly/function/sugar.py` (about 80 lines), `src/scaly/__init__.py`,
`tests/function/test_vmap_views.py` (new), `tests/test_import_boundaries.py`,
`tests/baseline/pytest_nodeids.txt`, `typing_playground/`, `docs/guide/functions.md`, `README.md`,
`internal/todo.md`. No new module, so no `IMPORT_LAYERS` entry.

## 7. Deferred

- **`scan` with views:** the same `_resolve_inputs` answers API-79's "a slice's size is not its
  stride" for `xs`; a separate small change.
- **Inferring `length`** from the mapped arguments (keeping it positional for now is compatible
  with every existing call).
- **Output shape** `(length, *out.shape)` instead of flat, and returning the callee's output tree:
  API-2.
- **Mapping over an axis other than 0** (`in_axes` à la JAX); a transpose view covers it today via
  the fallback.

## 8. Questions for sign-off

1. The broadcast marker's name: `sc.broadcast(p)` (proposed), or JAX-style `in_axes=(0, None)`.
2. The fallback: silent (proposed), a warning, or an error under `strict=True`.
3. Base branch for V0: wait for dev to merge, or land on `main` first.
