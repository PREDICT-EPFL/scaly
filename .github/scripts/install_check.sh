#!/bin/sh
# Install the wheels in dist/ into a fresh environment of the Python given as $1, then solve with
# every backend, first with the system C compiler and then through zig cc.
set -eux
"$1" -m venv /tmp/env
/tmp/env/bin/pip install --only-binary :all: dist/*.whl
/tmp/env/bin/python .github/scripts/solve_smoke.py
/tmp/env/bin/pip install --only-binary :all: "$(echo dist/scaly-[0-9]*.whl)[toolchain]"
/tmp/env/bin/scaly_toolchain | grep '(zig-bin)'
/tmp/env/bin/python .github/scripts/solve_smoke.py
