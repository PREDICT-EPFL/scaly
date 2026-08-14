# Replacing the string factory grammar with typed derivative specs

Design note, 2026-08-14. **Status: proposal — authorizes no code changes yet.**
Intended to fold into the ongoing core-restructuring plan
(`restructuring_opus.md` / `restructuring_sol.md`).

## Rationale

`Function.factory` encodes derivative requests as strings: `"jac:y:x"`, `"grad:f:x"`,
`"hess:gamma:x:x"`, `"spjac:eq:z"`. The encoding was inherited from CasADi, where it exists
because factory requests cross a SWIG boundary into C++ and a string is the cheapest thing to
pass uniformly from Python, MATLAB, and Octave. Alloy is pure Python; that rationale does not
transfer, and nobody has been observed writing the strings by choice. What the grammar costs us:

- **No static checking.** A typo in a spec is a runtime `ValueError` at trace time. The type
  checker and the IDE cannot see into the grammar — no completion, no refactoring support.
  `tests/alloy/test_api.py` spends a parametrized test enumerating malformed strings; that tests
  the parser, not the automatic differentiation.
- **A hand-rolled parser.** `Function._factory_output` is ~60 lines of `split(":")` dispatch
  re-implementing what Python function dispatch gives for free, with an irregular arity
  (3 parts for `jac`/`grad`/`fwd`/`adj`/`spjac`, 4 for `hess`/`sphess`, bare names pass through,
  `aux` is a separate kwarg).
- **`:` is overloaded.** In an output spec it is grammar; in an input name (`"lam:f"`, `"fwd:x"`)
  it is a naming convention for auto-created symbols. Two meanings, one character.
- **Two APIs for one thing.** `api.py` has nine named wrappers documented as "sugar" with the
  strings declared canonical, yet the wrappers are too weak for real use: `solvers/nlp.py`
  cannot use `sparse_lagrangian_hessian` because it needs extra pass-through parameter inputs
  the wrapper does not accept, so the solver layer falls back to raw strings and hardcodes the
  `"lam:f"` convention.

The two things the factory genuinely provides — multi-output requests that share one graph and
one common-subexpression pool (e.g. `["jac:eq:z", "jac:eq:xnext"]` in
`benchmarks/problems/chain/__init__.py`), and the auto-created dual/seed inputs with the `aux`
Lagrangian mechanism — do not need a string grammar. They need a structured multi-output
request.

Dropping the grammar is an explicit break with the "CasADi-style factory strings" identity in
`AGENTS.md`, `README.md`, and `docs/spec.md`. That break is intentional: alloy's goal is a
principled improvement on CasADi's core, and this grammar is one of the papercuts.

## Proposed API

Output requests become small typed spec objects (frozen dataclasses or named tuples) exported
from `alloy`:

```python
al.jac(of="eq", wrt="z")        # dense Jacobian
al.grad("f", "x")               # gradient of a scalar output
al.hess("f", "x")               # Hessian; optional wrt2 defaults to wrt
al.fwd("y", "x")                # seeded forward mode (J @ fwd:x)
al.adj("y", "x")                # seeded reverse mode (J.T @ lam:y)
al.spjac("eq", "z")             # compact nonzero Jacobian values + sparsity
al.sphess("gamma", "x")         # compact nonzero Hessian values + sparsity
```

`Function.factory` keeps its shape; only the output encoding changes. A bare string output is
still a passthrough of an original output:

```python
# before
fn.factory("chain_ref", ["z", "xnext", "params"], ["jac:eq:z", "jac:eq:xnext"])
nlp.factory("h", ["x", "lam:f", "lam:g"], ["sphess:gamma:x:x"], aux={"gamma": ["f", "g"]})

# after
fn.factory("chain_ref", ["z", "xnext", "params"], [al.jac("eq", "z"), al.jac("eq", "xnext")])
nlp.factory("h", ["x", "lam:f", "lam:g"], [al.sphess("gamma", "x")], aux={"gamma": ["f", "g"]})
```

Scope boundaries, chosen to keep the diff small and the generated ABI unchanged:

- **Input names stay strings**, including the auto-created `"lam:<output>"` and `"fwd:<input>"`
  names. A name is a name; it is grammar-in-strings that goes. Renaming the dual/seed convention
  (and its leak into `nlp.py`) is a separate decision, out of scope here.
- **`aux` stays as it is** — `aux={"gamma": ["f", "g"]}` is already structured data.
- **Output names of the derived function are unchanged.** Today they come from
  `spec.replace(":", "_")` → `jac_eq_z`; each typed spec derives the same name
  (`f"{kind}_{of}_{wrt}"`). No churn in generated C symbol or output naming.
- Each spec type carries its own build method (spec → `(Expr, SparsityType | None)` given the
  input/output maps), replacing the `_factory_output` if-chain with dispatch on type. Unknown
  names still raise the same `ValueError`s; malformed *requests* become unrepresentable.

The named wrappers in `api.py` (`al.gradient`, `al.jacobian`, …) stay and become the canonical
convenience layer, each a one-line call into `factory` with a typed spec. One real gap gets
fixed while we are there: the wrappers accept the full input-name list (or extra pass-through
inputs), so `solvers/nlp.py` can use `sparse_lagrangian_hessian` instead of raw factory calls.

No compatibility shim for the string form: nothing external consumes the strings (the laopt
adapter is gone; generated C glue does not carry factory specs), so the string parser is deleted
outright.

## Modifications inventory

Code:

- `src/alloy/function.py` — `factory` accepts `Sequence[str | DerivSpec]`; delete the
  `_factory_output` string parser; keep the input/output-name validation helpers.
- `src/alloy/api.py` — define and export the spec constructors; update the nine wrappers to pass
  typed specs; extend `lagrangian_hessian` / `sparse_lagrangian_hessian` (and friends as needed)
  to take the input list.
- `src/alloy/solvers/nlp.py` — 4 call sites; the two Hessian ones switch to the fixed wrapper.

Tests (mechanical unless noted):

- `tests/alloy/test_factory_casadi.py` — update and rename to `test_factory.py`; the CasADi
  cross-checks keep their value (same math, new request encoding).
- `tests/alloy/test_api.py` — the malformed-string cases disappear (unrepresentable); keep the
  unknown-name and aux-shadowing error tests.
- ~8 further test files with raw `.factory(` calls (`test_map.py`, `test_alloy_sparsity.py`,
  `test_stage_transcription.py`, `test_program_migration.py`, `test_passes.py`,
  `test_reverse_ad.py`, …).

Benchmarks: `problems/chain/{__init__,checks}.py`, `problems/race_cars/checks.py`,
`harness/sweep.py` — ~6 call sites.

Docs: `docs/spec.md` ("Canonical factory API" section and the CasADi-comparison table row),
`AGENTS.md` and `README.md` project-objective lines describing "CasADi-style derivative
factories (`jac:*` / `grad:*` / …)", `docs/roadmap.md` where it echoes that phrasing.

## Interaction with the restructuring plan

The Sol plan carves out `function/factory.py` described as "`jac:`/`grad:`/`hess:`/`lam:`
request *parsing* and construction". With this change that module becomes *spec types and
construction* — there is no parsing. Recommended sequencing: fold this into the same phase that
splits `function.py`, so the string parser is never moved only to be deleted. The change is
behavior-preserving at the graph level (same derived graphs, same output names, same sparsity
handling), so it does not disturb the plan's "no semantic redesign during moves" rule beyond the
request encoding itself — but it should land as its own commit/phase step with the full test
suite as the gate.

## Open decisions

1. **Rename `factory` itself?** The name is CasADi jargon; `Function.derive` would say what it
   does. Cheap to do in the same pass, but purely cosmetic — default is to keep `factory`.
2. **Spec constructor casing/location:** lowercase constructors (`al.jac`) read like the ops they
   are; the underlying types can live in `function/factory.py` per the restructuring layout.
