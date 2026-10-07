---
name: write-docs
description: Write, revise or review Scaly's published prose (pages under docs/, the README, rendered docstrings) in the maintainer's style. Use only when the user or another skill asks for it by name, never on your own initiative.
---

# Write Scaly documentation

Read these before writing, every time:

- [The writing brief](brief.md), which takes precedence over generic writing advice, including
  the `unslop` skill.
- The *Documentation* section of [the conventions](../../../docs/dev/conventions.md).
- The publication rules in [`AGENTS.md`](../../../AGENTS.md): only finished features, nothing from
  GitHub Issues or the roadmap in a user page or a rendered docstring, maintenance material in
  `internal/`.
- The page you are changing, in full, and one reference page for tone:
  [Getting started](../../../docs/guide/getting_started.md). The maintainer's own rewrite of the
  README and the home page, `git show ffece649 -- README.md docs/index.md`, is the clearest
  example of the style they want.

## Where content goes

User guide pages answer how to use one part of the library and should be useful when opened
directly. *How it works* explains the compiler to a curious user, not to a maintainer. Benchmark
pages give the main results, the decisions that shape them and their limits. The API reference
comes from docstrings. Module ownership, migration history, plans, full result tables and notes
for the next agent go to `internal/`. When unsure, leave it out of `docs/`.

## Writing

1. Read the implementation of everything you describe, then write.
2. When editing an existing page, change what the behaviour change requires and keep the page's
   voice. A rewrite that keeps the meaning but changes the style is a regression.
3. Write documentation, not a story. State what an object is and what it does. A sentence such as
   "`features` is now a `Function`." narrates, while "`features` is a `Function` of ..." informs.
4. Keep the tone factual and modest. Scaly is young: neither commercial nor apologetic, and no
   claims a benchmark page does not support.
5. Write for four readers at once: numerical experts new to the library, newcomers, CasADi users,
   and students who are learning model predictive control.
6. Examples use the API as it is meant to be used. When the API offers a lighter form, such as
   separate arguments instead of a tuple unpacked in the body, every example uses it.
7. Fix a mannerism by rewriting the sentence, never by a mechanical substitution. Removing bold from
   a label list, or trading dashes for colons, produces a new mannerism.

## Checks

1. Run every code fence you added or changed, with the imports the page supplies, and compare the
   output the page states. Run the `examples/` scripts the page links to.
2. Search the diff for em and en dashes, semicolons, bold-label bullets and roadmap words.
3. Unfinished names stay filtered out of the API pages, as `docs/api/core.md` does.
4. Build the site: `uv run --only-group docs zensical build`.
5. For a change of more than a few sentences, ask a model other than the writer for one style
   review. Give it the brief, the reference page and the diff, and ask it to flag sentences that
   break the brief, not to rewrite. Apply what it flags once. Do not loop.

## Scope

A documentation task does not authorize fixing the API. List each API gap you find in the report.
File issues for them only when the user or the skill that called this one asks for it. Preserve edits by the maintainer and pages
outside the task.

Report the pages changed, the fences and examples run, the build result and the gaps found.
