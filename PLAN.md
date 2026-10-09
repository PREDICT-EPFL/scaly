# Frozen lazy records

Apply the frozen-lazy-record pattern
(https://gist.github.com/tudoroancea/cca107e4dc47743b4062c26291a9655e) where it fits, as agreed in
the design discussion. No issue. The motivation is that everything a record derives can be inspected,
which the revived viz tool can use.

Decided:

- `ConcreteFunction`: cache the node order (`nodes`) and the effective lowering hint, and move the
  forward and adjoint helper caches from module-level weak tables in `ad/` into its `_memo`.
  `recompile()` stays the one documented reset of `_memo.compiled`.
- `Function`: a frozen `eq=False` dataclass. Instance caches in fields built empty. Fully declared
  functions are still traced where they are defined (#11), now by the factories (`function`,
  `lift`, `vmap`), so `__post_init__` stays cheap and `_from_instance` uses the generated
  `__init__`. `_Mapped` becomes a `_Derived`. The benchmarks' `_benchmark_base` attribute moves to
  a weak-keyed registry in `benchmarks/harness`.
- `Problem`: `_cache` is replaced by a cached frozen record holding the stacked NLP form, which
  `nlp.py` and `qp.py` read through typed attributes. The record lives in `problem.py` so no import
  cycle forms.
- Left alone: `Expr`, `ProgramNode` (interned, slotted, too many nodes), `CompiledFunction`
  (public, compiles in its constructor).
- A scoped rule in `docs/dev/conventions.md`.

## Phases

- [x] 1. `ConcreteFunction` caches and the AD tables (tests first)
- [x] 2. Frozen `Function`, `_Derived`, `_Mapped`, the benchmark evaluator registry
- [x] 3. `Problem`'s NLP form record
- [x] 4. Conventions rule
- [ ] 5. `wt hook pre-merge`, C snapshots unchanged, PR, cross-review

Phases 1 to 3 touch different files and could run in parallel, but they are small enough to do in
order. Phase 4 follows them so the rule matches the code.
