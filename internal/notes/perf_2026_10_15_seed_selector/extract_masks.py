"""Write the four selector masks of a generated npmpc Hessian kernel to masks_N<stages>.bin for bench.cpp.

    uv run internal/notes/perf_2026_10_15_seed_selector/extract_masks.py <kernel.c> <stages> <out-dir>

The masks are the four ``static const double`` tables multiplied into the first selector loop
(``s4[i] = k0[i]*s1[v] + k6[i]*s2[v] + ...``), each 7 directions x stages x 9 entries.
"""

from pathlib import Path
import re
import sys

import numpy as np

source, stages, out = Path(sys.argv[1]).read_text(), int(sys.argv[2]), Path(sys.argv[3])
loop = re.search(r"\(\(\(\((k\d+)\[\w+\] \* \w+\[\w+\]\) \+ \((k\d+)\[\w+\] \* \w+\[\w+\]\)\) \+ \((k\d+)\[\w+\] \* \w+\[\w+\]\)\) \+ \((k\d+)\[\w+\] \*", source)
masks = [np.array([float(x) for x in re.search(rf"static const double {k}\[\d+\] = \{{([^}}]*)\}}", source).group(1).split(",")]) for k in loop.groups()]
assert all(m.size == 7 * stages * 9 and set(np.unique(m)) <= {0.0, 1.0} for m in masks)
np.concatenate(masks).tofile(out / f"masks_N{stages}.bin")
print(f"{loop.groups()} -> {out / f'masks_N{stages}.bin'}")
