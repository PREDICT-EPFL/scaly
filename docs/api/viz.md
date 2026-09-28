# Visualization

`scaly.viz` records what the compiler does to chosen functions: the expression graph, the program
after each optimization pass, and the generated C. It is opt-in. Importing `scaly` does not import
it, and nothing is recorded for a function that was not marked.

```python
import scaly as sc
from scaly.codegen import render_c_module
from scaly.viz import visualize

@sc.function(sc.L("x", 3), sc.L("energy", ...))
def energy(x: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x)

visualize(energy, label="Energy model")
render_c_module(energy)
```

Every render of a marked function, whether ahead of time or through the JIT, adds one recording.
Run `uv run scaly_viz --browser` to browse them. `visualize` is the same function as
`visualize_function`.

## Recording

::: scaly.viz.recording.visualize_function

::: scaly.viz.recording.unvisualize_function

::: scaly.viz.recording.capture

## Reading recordings

::: scaly.viz.recording.recordings

::: scaly.viz.recording.load_recordings

::: scaly.viz.recording.clear_recordings

::: scaly.viz.recording.recording_dir

::: scaly.viz.recording.recording_path

## Browsing

::: scaly.viz.serve.serve
