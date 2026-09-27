# Installation

Scaly requires Python 3.12 or newer, on Linux and macOS, and only requires
a C compiler to be installed system wide. Using [uv](https://docs.astral.sh/uv):

```bash
# install just the core library
uv add scaly
# or with an additional solver interface
uv add "scaly[ipopt]"
# or with all solvers
uv add "scaly[solvers]"
```

You can of course also use pip by replacing `uv add` with `pip install`.

The solver plugins ship prebuilt libraries. A solve raises `SolverLibraryError` when the library it
needs is not there. If no wheel matches your platform, or you want a checkout of the repository,
build the solvers from source as described in [Contributing](../dev/contributing.md#setup).

## C compiler

Scaly looks for `cc` on your `PATH`, the POSIX name for the system default C compiler on Linux,
macOS and the BSDs. Set the `SCALY_CC` environment variable to override it.

To check what C/C++ compiler scaly uses, you can run the `scaly_toolchain` script via

```bash
# with uv
uv run scaly_toolchain
# or if you have activated the virtual environment simply
scaly_toolchain
```

This script also prints other useful information to debug errors at the C
compilation level: the JIT's cache directory, the JIT compilation flags, and the
status of each solver plugin.

## Checking it works

You can run the following small example:

```bash
uv run python -c "
import numpy as np
import scaly as sc

@sc.function(sc.L('x', 3), output=sc.L('y', ...))
def f(x):
    return (x.sin() + x * x).sum()

np.testing.assert_allclose(5.524412954423689, f(np.ones(3)))
"
```

If an assertion error is raised, something went wrong.
