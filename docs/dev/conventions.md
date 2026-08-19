# Conventions

Small rules, written down because they are the kind that erode quietly. `AGENTS.md` at the
repository root points here.

## Naming

**Derivative suffixes are abbreviated.** `_grad`, `_jac`, `_hess` — never `_gradient`,
`_jacobian`, `_hessian` in an identifier. The long forms exist as the user-facing wrapper names
(`al.gradient`, `al.jacobian`, `al.hessian`) and nowhere else.

**Multistage stage variables are `zprev`, `z`, `znext`.** Never `zm`/`zp`. The abbreviation saves
two characters and costs a reader a guess about whether `m` means minus or measured.

**Dialect vocabulary follows the dialect.** Expression-side names are `Expr*` — `ExprOp`,
`verify_expr`, `spec_expr` — and print with an `expr.` prefix. Program-side names are `Program*` —
`ProgramOp`, `ProgramNode`, `verify_program` — and print with `prog.`. There is no third
vocabulary; "semantic IR" and the old `P`-prefixed spellings are gone and are not coming back.

**Derived output names are load-bearing.** A derivative output is named `{kind}_{of}_{wrt}`, or
`{kind}_{of}_{wrt}_{wrt2}` for the two Hessian kinds. These become generated C symbols and the
prefixes of the sparsity tables in the header, so renaming one moves symbols in everyone's build.

## Code style

Alloy is meant to read like tinygrad: small, dense, every line earning its place. Concretely:

- Two-space indentation, 150-column lines. `uv run ruff format` decides; do not argue with it.
- No speculative abstractions. A configuration option with one caller, a base class with one
  subclass, or a hook nothing uses is a defect, not foresight.
- Validate at boundaries — user input, file formats, plugin protocols. Trust internal code and
  language guarantees where an invalid state cannot arise.
- Delete what a change makes obsolete rather than leaving a renamed placeholder, a dead branch or
  a comment saying something used to be here. Git remembers.
- Comments explain *why*, and only when the why is not evident. Code that needs a comment to say
  what it does usually needs different code.

## Where a module belongs

Every module has a layer, and a module may import its own layer or below and never above. The table
and its two sanctioned exceptions are in [the architecture](../how_it_works/architecture.md#layers),
and `tests/test_layering.py` enforces them.

Every module also carries a one-line docstring saying what it owns. Directory names do not keep a
package coherent; that sentence does, because it is what makes an incoherent addition obvious.

## Tests against benchmarks

Correctness checks have two homes, and each check belongs in exactly one.

**`tests/` covers alloy itself** — the intermediate representations, differentiation, code
generation, solver plumbing. It mirrors `src/alloy/` directory for directory. It must never import
`benchmarks.problems`, and imports `benchmarks.harness` only to test the harness.

**`benchmarks/problems/<problem>/checks.py` covers that problem** — its input data, its
formulation, its parameter layout, its constant pins, and agreement between backends. Those checks
gate the measurement they belong to.

The test when it could go either way: *would retiring this problem make the check meaningless?* If
yes, it belongs to the problem. Benchmarks churn with the workload roadmap; tests must not.

One rule that follows from this: **never let a benchmark be the only thing exercising a compiler
path.** When it is, copy a small self-contained reproduction into `tests/` — differential against
an unrolled or NumPy reference — *before* changing or retiring the benchmark.

And prove a new gate can fail. Perturb what it checks and watch it go red; a gate that cannot fail
is worse than no gate, because it reads like coverage.

## Documentation

- Spell out an acronym on first use unless it is standard outside the field. HTTP is fine; HOCBF is
  not.
- Prefer the plainest accurate words. "The number stops the check from ever failing" beats "the
  gate is vacuous"; "the harness gives an agent more time" beats "resumption eligibility". Reach
  for a technical term when it carries real meaning, not for tone.
- Update `docs/` in the same change as the behaviour it describes, when that behaviour is
  user-facing or reaches the generated ABI.
- Do not write down a commit hash made on a working branch. Branches land squashed, so the hash
  stops existing the moment the work merges. Name the file, the function or the change instead.
  Hashes already on the default branch are safe to cite.
