# Writing the scaly user guides

This is the maintained writing brief agreed with Ted during the user-guide review. The edited
[Getting started](../../../docs/guide/getting_started.md) is the reference for tone and depth.
These preferences take precedence over generic writing-skill advice, including advice to make
every tutorial a sequence of commands or separate every explanation from an example.

## Readers and purpose

Assume familiarity with Python, NumPy, and the mathematics relevant to the topic. Control examples
can assume basic optimal control and model predictive control. Do not assume experience with
symbolic modelling, CasADi, or compiler implementation. A short prerequisite sentence is useful
when it tells readers what background a page needs. It should not become boilerplate on every page.

Teach how scaly represents a calculation and how its public objects behave. `Expr` is a symbolic
value, `Function` is the unit of composition, `Problem` describes an optimization problem, and
`Solver` provides its numerical and symbolic solve interface. Introduce the relevant distinction
before relying on it in code. Explain when Python executes, when a graph is built, and when
numerical values are evaluated.

Use NumPy as a familiar point of comparison for shapes, indexing, broadcasting, and operators.
State the limits of that comparison. Similar syntax does not imply the entire NumPy API exists.
Readers familiar with CasADi should recognize the purpose of the features without everyone else
having to learn CasADi terminology first.

## Independent topics within a coherent guide

Getting started gives the overall workflow. The other user guides collect details about using
particular parts of the library. They should be useful when opened directly to answer a question,
not only when read as chapters of one long tutorial.

The structural reference is [JAX's discussion of common gotchas](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html):
focused topics, concrete examples, observable behavior, and explanations of surprising results.
Borrow that organization, not its branding or JAX-specific semantics.

A section should identify the behavior or question, show a small example where useful, and explain
what follows from it. Not every section needs the same format. Tables suit mappings and options.
Equations suit mathematical definitions. Prose suits the relationship between concepts.

Keep dependencies between examples local and explicit. An advanced example may reuse a named
model from an earlier section on the same page, but a reader should know where it came from.
Avoid undefined `fn`, `problem`, or `x_value` in examples presented as runnable. Long option or
operation inventories belong in the API reference, with links from the guide.

## Examples and mathematical notation

- Put the relevant equations next to the code when they clarify a model, derivative, or
  optimization problem. Explain the mapping between mathematical symbols and declared names.
- Annotate symbolic function arguments and results, including `ProblemSpec` builders. Inside a
  problem builder, costs, constraints, and bounds are expressions, not evaluated NumPy values.
- Explain what annotations buy: an IDE's type checker can compare tuple structures and symbolic
  or numerical types against the declarations. Annotations are optional and do not change
  evaluation. Do not claim that Python annotations statically check array dimensions.
- Explain `L` as leaf and `G` as group when introducing input/output trees. Distinguish an array
  from a tuple containing an array, and a scalar from a one-element vector.
- Prefer complete small examples with outputs or an explicit description of the result.
  Failure examples should show the actual exception or wrong result and explain the cause.
- Use descriptive names such as `model` for a dynamics function. Preserve the user's chosen names.
- Keep examples focused on the library concept. Do not teach elementary Python, motivate arbitrary
  control weights, or rehearse how an MPC loop works unless it is necessary to explain the API.
- Do not prescribe saving snippets under filenames or explain how to run a Python script.
  Filenames are useful when they are part of the operation being taught, such as a code-generation
  command's `module:attribute` argument or a C compilation command.
- Generated-code sketches may explain structure without reproducing actual compiler output.
  Label them as sketches. Distinguish constant loop-body size from storage, runtime work, and
  sparsity-table sizes that grow with the horizon.
- Explain why export is useful, such as integration into a C++ controller or ROS application,
  along with the actual deployment requirements. Link the generated API details.

A few self-contained files in `examples/` complement the inline snippets and the larger benchmark
workflows. Choose important constructs, show numerical results, and export C when useful. Link
these from the relevant guide. Use repository branch links until a stable release reference exists,
not commit permalinks to a branch that will be squashed. Do not turn every snippet into a file.

## Prose and formatting

Describe what the example or feature does. Avoid empty imperatives such as "Build a controller"
or "Follow Installation first". Commands are appropriate when the user needs to perform a specific
operation, but they are not a substitute for an informative introduction or heading.

Use connected, natural prose. Avoid semicolons, em dashes, slogans, inflated claims, and dense
strings of implementation terms. Do not replace them mechanically with another repeated mannerism.
Vary sentence length when that makes the explanation easier to read.

Use admonitions for limitations or side details that merit visual separation, especially silent
wrong-result risks such as missing solver sensitivities. Keep the main explanation in ordinary
prose and avoid turning every section into a callout.

Lists and bold text are useful when they support scanning. Removing bold from an unchanged list
is not simplification. Rewrite a repetitive label-and-description list into a better explanation,
or keep the formatting when it genuinely helps. Do not force prose into lists or lists into prose.

Keep user-facing behavior in the guide. Internal module ownership, maintenance instructions,
migration history, future implementation plans, and agent handover notes belong in `internal/`
or the appropriate contributor documentation, not in a user explanation.

## Accuracy and review

Read the current implementation before describing behavior. Execute examples and check their
outputs. Check annotations when making typing claims. A feature missing from scaly is a limitation
to explain or report, not an invitation to invent a familiar API from another library.

Distinguish ordinary function calls from mapped repetition. Ordinary calls may expand during
differentiation or compilation. `vmap` records independent calls as mapped structure and does not
replace a sequential recurrence. Do not promise a particular final C spelling after optimization.

Document important surprises near the behavior they affect: symbolic Python conditions, fixed
shapes, sparse-value ordering, unsupported derivatives, warm-start differences, and solver status.
Keep such notes specific and proportionate rather than surrounding every example with warnings.

A documentation task does not authorize fixing the API. Record verified gaps separately, and
preserve user edits and files outside the agreed scope.
