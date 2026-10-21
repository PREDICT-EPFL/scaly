# Typing audit, 2026-10-21

This note records the audit before the follow-up implementation. The maintainer approved the improvements in sections 1 through 6, which are implemented on this branch. Section 7 is a separate change to be kept uncommitted after the main improvements are pushed. The larger GitHub issues remain out of scope.

The follow-up adds a variadic group overload, so the temporary explicit declarations in the nested-group test are no longer needed. That test uses its original generated declarations and tuple slices again.

## Update and validation

Updated Ruff from 0.15.13 to [0.17.1](https://pypi.org/project/ruff/0.17.1/) and ty from 0.0.77 to [0.0.90](https://pypi.org/project/ty/0.0.90/), the latest stable releases on PyPI on this date. Only these two package versions changed. uv also updated the lockfile format.

The old strict type check passed before the upgrade. The new version reported six diagnostics, now resolved:

- Forward derivative helper input names now use the same string construction as their seed declarations, avoiding the optional `Expr.name` field.
- The print and visualization context managers return `Generator` in their annotations.
- The HTTP handler casts its server at the point that needs `VizServer`, instead of narrowing an inherited mutable attribute.
- The mask test gives `sum` an array start value.
- The nested-group test spells out its fixed declarations because ty resolves a star-unpacked iterable against `group`'s one-argument overload and infers one-element groups. It also spells out the numerical argument structure so the tuple lengths are known.

Ruff's newer defaults reported 385 findings across the repository. The configuration now explicitly selects the previous `E4`, `E7`, `E9`, `F` rules and retains the existing `E731` exclusion. Formatting required no repository-wide changes.

Validation passed: Ruff lint and format, `ty check --error-on-warning`, the same type check targeting Python 3.12 and 3.13, and `uv lock --check`. The full suite completed with 1,557 passed and 13 skipped on macOS with Python 3.14.6. The focused pass, visualization, and differentiation tests also passed.

## Recommended order

### 1. Ship the package's type information

There is no `src/scaly/py.typed`. Building the core wheel with `uv build --wheel --package scaly` and inspecting its ZIP entries confirmed that the wheel contains no `py.typed` either. Checking the source checkout does not test how an installed consumer discovers the annotations.

Add the marker and verify its presence in the wheel. Add a small consumer check against the installed wheel, outside the source import path, covering a valid declared call and a rejected mixed symbolic/numerical call. The [typing distribution specification](https://typing.python.org/en/latest/spec/distributing.html#packaging-type-information) requires the marker for packages distributing inline types. No existing Scaly issue mentioning `py.typed` was found.

### 2. Stop losing known types at public forwarding methods

`src/scaly/function/model.py::Function.solver_stats` returns `Any`, while `ConcreteFunction.solver_stats` and the compiled implementation return `SolverStats`. A ty probe confirmed the difference and accepted a nonexistent method on the declaration's statistics result.

Annotate the forwarding method with `SolverStats`, using a type-checking import as the concrete implementation already does. Add an `assert_type` and an expected error in `tests/typing/test_arity.py`. This is a small improvement with no runtime or API design change.

### 3. Check the benchmarks routinely

`pyproject.toml` excludes all of `benchmarks/` from ty. This also excludes the positive `assert_type` checks in `benchmarks/problems/npmpc/__init__.py`. An explicit check with the upgraded tool reports 17 diagnostics:

```sh
uv run ty check $(rg --files benchmarks -g '*.py') --error-on-warning --output-format concise
```

The findings include generic type loss, optional values that need narrowing, dynamic CasADi attributes, and string values passed to literal-typed settings. They are not 17 demonstrated runtime failures. Start by checking the shared benchmark code and Scaly formulations. Resolve third-party typing mismatches individually before removing the blanket exclusion.

Two particularly small fixes have clear evidence:

- `benchmarks/harness/recording.py::Recorder._log` accepts `type[BaseModel]` and returns `BaseModel`. Making it generic over `M: BaseModel`, with `model: type[M]` and return type `M`, fixes the two return-type diagnostics in `record_metadata` and `record_telemetry`. The delegated audit verified this in a temporary copy. The existing mypy-style `return-value` ignores do not match ty's `invalid-return-type` diagnostic.
- `benchmarks/harness/timing.py::SolveTiming.summary` returns `dict[str, object]`; `mode_rows` takes bare `dict` and returns `list[dict]`. Use small `TypedDict` records for these known fields. Probes accepted a string `build_ms` and the misspelled key `time_to_frist_solve_ms` under the current annotations.

The 17 diagnostics occur in `closed_loop.py`, `hyperdual.py`, `provenance.py`, `recording.py`, `sweep.py`, the chain, race-car and unbumpercars problems, and `benchmarks/run.py`. The last file has two symbolic bodies where an untyped parameter tree makes the solver overload resolve to a numerical result.

### 4. Annotate test fixtures, callbacks, and function builders

The default ty check covers tests, but unknown parameter types still let mistakes pass. The delegated audit demonstrated these cases:

- `tests/benchmarks/test_closed_loop.py` leaves the parametrized `runner` untyped. A temporary copy with `smoke=True` changed to `smkoe=True` still passed ty. A small callable protocol catches the keyword typo.
- `tests/benchmarks/test_timing.py` has untyped `tmp_path` arguments. Use `Path` for paths and `pytest.MonkeyPatch` for monkeypatch fixtures where they occur. Give custom fixtures their concrete result types.
- `tests/integration/test_stage_transcription.py::bicycle_eq_function` returns bare `sc.Function`. Ty reveals four unknown type parameters and accepts `bicycle_eq_function(2)("wrong", "extra")`. Preserve its exact declared function type. The same file also retrieves derivative methods through `cast(Any, ...)`; retain the typed declarations where possible.

Prioritize shared fixtures and builders, which affect many callers. Keep deliberate runtime rejection tests dynamic where necessary. Add static positive and negative assertions for real API contracts rather than annotating every local variable.

### 5. Type internal derivative and compilation caches

`src/scaly/function/concrete.py::_Memo` stores `compiled`, Jacobian-vector product (JVP) caches, and mapped adjoint caches as `Any`. The call-rule helpers in `src/scaly/ad/forward.py` and `reverse.py` accept an `Any` callee and return tuples containing `Any`, although they construct `ConcreteFunction` objects and fixed tuples of integer indices.

Start by annotating the callee and return values with `ConcreteFunction` and typing each cache's actual key/value structure. Annotate the compiled handle with `CompiledFunction | None` through a type-checking import, preserving the one runtime import through `_jit`. Use tuple aliases where the same structure repeats. This would let ty check the helper construction and unpacking that currently crosses these unchecked regions. No new runtime protocol or registry is needed.

The playground has a smaller version of the same problem. `typing_playground/trees.py` uses `Any` for its internal skeleton transformations. A recursive alias such as `LeafKey | tuple[Skeleton, ...]` would reject `skeleton_shapes("bad")`, which currently passes ty. Treat this as secondary to the production code.

### 6. Handle star-unpacked group declarations honestly

The independent reviewer checked the original nested-group test and reproduced a public typing problem in `src/scaly/function/tree.py::group`. Ty infers `Tree[tuple[Expr], tuple[Array]]` for `sc.group(*(sc.arg(f"x{i}", 1) for i in range(8)))`. The runtime group contains eight leaves, so the inferred one-element structure is wrong and rejects correct calls. The original test's only diagnostic was at its numerical call; declaration construction itself passed.

Investigate an overload for dynamically sized groups that preserves the known leaf types without asserting a false tuple length. Keep exact inference for explicit fixed groups and keep the runtime width limit. Add a static reproduction for `sc.group(*parts)` alongside the existing exact-width assertions. Until that is resolved, spelling fixed declarations explicitly is preferable to casting away errors at their call sites. This is a separate library improvement, not a reason to weaken the updated test.

### 7. Align numerical input annotations with accepted values

`src/scaly/function/tree.py` aliases `Array` to bare `np.ndarray` and uses it for both numerical inputs and outputs. Yet `flatten_numerical` deliberately calls `np.asarray`, and a declared function accepted `[1.0, 2.0]` at runtime while ty rejected that same call.

Settle the scalar and dtype policy in [#25](https://github.com/PREDICT-EPFL/scaly/issues/25) first, then distinguish accepted array-like inputs from NumPy array outputs. Consider `numpy.typing.ArrayLike` for inputs, with precise output array types where guaranteed. Preserve rejection of mixed symbolic/numerical argument trees. Coordinate output dtype annotations with [#24](https://github.com/PREDICT-EPFL/scaly/issues/24), [#55](https://github.com/PREDICT-EPFL/scaly/issues/55), and [#56](https://github.com/PREDICT-EPFL/scaly/issues/56); avoid a shape-generic redesign for this.

The expression operators and `Expr.const` also accept `Any`. A probe confirmed that `Expr + {"wrong": object()}` passes ty. Narrow these inputs when the coercion policy is settled, so invalid values fail statically without excluding valid NumPy values.

## Deliberate limits to keep

The typed decorators, derivative wrappers, solvers, and `vmap` already have positive assertions and diagnostic-specific expected errors. Keep the `TYPE_CHECKING` blocks, `assert_type`, and `# ty: ignore[diagnostic]` convention. With warnings treated as errors, an expected error that disappears fails as an unused ignore.

The `Any` numerical outputs for bare functions and inferred structured outputs are documented in `function/api.py` and asserted in `tests/typing/test_arity.py`. Python typing does not provide the general recursive symbolic-to-numerical type mapping these modes need. For examples that require precise numerical results, declare `outputs=`. Do not add speculative overload ladders for arbitrary inferred structures.

## GitHub findings

Read both open and closed Scaly issues, including body searches for typing, annotations, protocols, ty, and Ruff. Some search hits use "type" to mean an expression dtype or generated C type, rather than Python annotations.

- Open [#59](https://github.com/PREDICT-EPFL/scaly/issues/59) covers the remaining npmpc template conversion. Its body says the public typed decorators and exact `Function` annotations already landed.
- Open [#25](https://github.com/PREDICT-EPFL/scaly/issues/25) owns scalar coercion, which affects the correct Python input annotations.
- Open [#168](https://github.com/PREDICT-EPFL/scaly/issues/168) covers float32 constants in structural derivative rules. It is a graph dtype problem, separate from Python annotation coverage.
- Open [#24](https://github.com/PREDICT-EPFL/scaly/issues/24), [#55](https://github.com/PREDICT-EPFL/scaly/issues/55), and [#56](https://github.com/PREDICT-EPFL/scaly/issues/56) cover dtype preservation, lowering, and typed generated entry points.
- Closed [#6](https://github.com/PREDICT-EPFL/scaly/issues/6), [#8](https://github.com/PREDICT-EPFL/scaly/issues/8), [#13](https://github.com/PREDICT-EPFL/scaly/issues/13), and [#115](https://github.com/PREDICT-EPFL/scaly/issues/115) record the typed Function, parameter-list, mapped-call, and zero-parameter contracts. Recommendations should preserve these decisions.

No dedicated open issue was found for the missing package marker, statistics return annotation, blanket benchmark exclusion, or unknown test fixtures/builders. No issues were created or changed.
