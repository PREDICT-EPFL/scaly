# Conventions

Follow these conventions when adding code, tests, or documentation. They keep names consistent
between the Python API, generated C, and examples.

## Naming

Derivative suffixes are abbreviated: `_grad`, `_jac`, `_hess`, never `_gradient`, `_jacobian`,
`_hessian` in an identifier. The long forms are the user-facing wrapper family (`sc.gradient`,
`sc.jacobian`, `sc.hessian`, `sc.sparse_jacobian`, `sc.sparse_hessian`, `sc.lagrangian_hessian`,
`sc.sparse_lagrangian_hessian`).

Use `zprev`, `z`, and `znext` for the previous, current, and next stage variables.
Avoid `zm` and `zp`, whose meaning is unclear without context.

Dialect vocabulary follows the dialect. Expression-side names are `Expr*` (`ExprOp`, `verify_expr`,
`spec_expr`) and print with an `expr.` prefix. Program-side names are `Program*` (`ProgramOp`,
`ProgramNode`, `verify_program`) and print with `prog.`.

A derivative output is named `{kind}_{of}_{wrt}`, or `{kind}_{of}_{wrt}_{wrt}` for the two Hessian
kinds. These become generated C symbols and the prefixes of the sparsity tables in the header, so
renaming one moves symbols in everyone's build.

## Code style

Prefer small, direct implementations that follow the surrounding code.

- Two-space indentation, 150-column lines. `uv run ruff format` decides.
- Add abstractions and configuration options when an existing use requires them. Avoid unused hooks
  and interfaces intended only for possible future features.
- Validate at boundaries (user input, file formats, plugin protocols). Trust internal code and
  language guarantees where an invalid state cannot arise.
- Delete code that a change makes obsolete, including unused branches and placeholders.
- Comments explain why, and only when the why is not evident. Code that needs a comment to say
  what it does usually needs different code.

## Where a module belongs

Every module has an import layer. A module may import its own import layer or a lower one, never a
higher one. The table and its two sanctioned exceptions are in
[The codebase](codebase.md#import-layers), and
`tests/test_import_layering.py` enforces them.

Give every module a one-line docstring stating its responsibility. Add code to the module that
owns the relevant concept.

## Tests against benchmarks

Correctness checks have two homes, and each check belongs in exactly one.

`tests/` covers scaly itself: the intermediate representations, differentiation, code generation
and solver plumbing. It mirrors `src/scaly/` directory for directory. It never imports
`benchmarks.problems`, and it imports `benchmarks.harness` only in `tests/benchmarks/` and the viz
recording tests.

`benchmarks/problems/<problem>/checks.py` covers that problem: its input data, formulation, parameter
layout, constant pins, and agreement between backends. Those checks gate the measurement they belong
to.

When a check could go either way, ask whether retiring the problem would make it meaningless. If
yes, it belongs to the problem. Benchmarks churn with the workload roadmap, but tests must not.

Never let a benchmark be the only thing exercising a compiler path. When it is, copy a small
self-contained reproduction into `tests/`, differential against an unrolled or NumPy reference,
before changing or retiring the benchmark.

Check that a new correctness test detects the error it is intended to catch. Temporarily perturb
the result or implementation, verify that the test fails, then remove the perturbation.

## Documentation

- Open a page with one to three sentences saying what it covers. End without a summary.
- Say what the code does. Cut a sentence that could sit unchanged in another project's docs, and
  cut commentary about the page or the argument itself.
- Active voice, plain words, sentence-case headings. Spell out an acronym on first use unless it is
  standard outside the field. HTTP is fine, HOCBF is not.
- No em or en dashes, and avoid semicolons. End the sentence or use a comma. Colons only before a
  list, a code block or an example.
- Read the implementation before describing behaviour, and run each example to check its output.
  Show the actual exception or wrong result in a failure example.
- Keep maintenance plans, migration history and future work out of user pages. They belong in
  `internal/`.
- Use an admonition for a limitation that deserves visual separation, such as a silent wrong result.
  Keep the main explanation in prose.
- No bold-label bullets that restate the line. Write prose or a plain bullet.
- State a performance number only where the [benchmark pages](../benchmarks/index.md) hold it, and link
  there.
- Update `docs/` in the same change as the behaviour it describes, when that behaviour is
  user-facing or reaches the generated ABI.
- Do not write down a commit hash made on a working branch. Branches land squashed, so the hash
  stops existing when the work merges. Name the file, the function or the change instead. Hashes
  already on the default branch are safe to cite.
