"""``python -m scaly.codegen`` — the AOT command line.

The shim lives here rather than in ``aot`` because ``codegen/__init__`` imports ``.aot``: running
``python -m scaly.codegen.aot`` executes that module a second time under the name ``__main__``,
which warns and leaves two copies of ``_RENDER_OBSERVERS``, so a target arming ``scaly.viz`` would
register against the copy the render does not consult.
"""

from scaly.codegen.aot import main

if __name__ == "__main__":
  main()
