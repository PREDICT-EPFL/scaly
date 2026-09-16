# Visualization

Scaly can record everything it does to a function — the expression graph you built, the lowered
program, the result of each optimization pass, and the generated C — and serve it in a browser.
It is the fastest way to answer "why is my generated code like that?"

## Recording a function

```python
from scaly.viz import visualize, serve

visualize(fn)     # mark this exact Function
fn(x_value)       # compiling it now records every stage
serve()           # browse at http://127.0.0.1:8000
```

Recording is **opt-in per function object**, not global. Nothing is captured unless you marked that
specific `Function`, so leaving `visualize` in a script costs nothing for everything else in it.
Importing `scaly.viz` is what arms the machinery at all; a plain `import scaly` does not.

`serve(host="0.0.0.0", port=8000)` to expose it beyond localhost. `unvisualize_function(fn)` stops
recording it.

## What you get

The browser view has a sidebar of recorded functions and, for each, the stages in order:

1. the expression graph, as `expr.*` assembly;
2. the lowered program, as `prog.*` assembly;
3. one snapshot after each optimization pass — `fuse_elementwise`, `unroll_unit_loops`,
   `pack_workspace`;
4. the generated C.

Stepping through the passes is the useful part. If a loop you expected to survive got unrolled, or
a temporary you expected to be fused is still a buffer, the step where it changed is right there.

## Where recordings go

By default `$XDG_CACHE_HOME/scaly/viz/recordings.json`, or `~/.cache/scaly/viz/recordings.json`.
Set `SCALY_VIZ_DIR` to move it, or pass `--recording-path` to the server.

```bash
uv run scaly_viz --recording-path path/to/recordings.json --browser
```

That serves a recording produced elsewhere — a benchmark run, a colleague's bug report — without
re-running anything.

```python
from scaly.viz import recordings, clear_recordings
recordings()          # the recorded data, as Python objects
clear_recordings()    # start clean
```

## Text without the browser

For a quick look, or for a test:

```python
expr.debug()                          # topological dump with %0, %1, ... names
sc.format_expr(expr)                  # the same, as a string
sc.render_expr_assembly(fn)           # the expr.* assembly
sc.render_program_assembly(prog)      # the prog.* assembly
sc.expr_graph(expr)                   # nodes and edges as JSON
sc.program_graph(prog)
```

The assembly forms are the stable ones. They are meant to be diffed, pasted into an issue and
asserted on in tests; the graph JSON is for building your own tooling.

## How it stays out of the way

The visualizer registers itself into an observer hook that the code generator owns, so nothing in
the compiler imports the visualizer or knows it exists. That is one of the two deliberate exceptions
in Scaly's import-layer rules, and it is why recording can be genuinely zero-cost when you have not asked for
it — see [the architecture](../how_it_works/architecture.md#the-two-sanctioned-exceptions).
