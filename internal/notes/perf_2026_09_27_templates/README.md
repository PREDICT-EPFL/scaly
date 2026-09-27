# Function templates, call overhead, 2026-09-27

Timing behind the `templates_p*_report.html` reports (todo API-1). Run from the repository root; the
numbers come from an Apple M3 Max with Python 3.14 and are indicative, not reference-machine
results.

| Script | Measures |
| --- | --- |
| `trivial_call.py` | the Python-side cost of calling a compiled Function with one leaf, one group of four leaves, and four parameters (when the checkout has parameter lists): fastest of 7 rounds of 100 000 calls, in µs. Run it in a base worktree (`PYTHONPATH=<worktree>/src uv run --no-sync ...`) and on the branch, interleaved, and compare minima |

| Phase | One leaf | One group of 4 | 4 parameters |
| --- | --- | --- | --- |
| P1b-i (base) | 2.83 | 5.81 | n/a |
| P1b-ii | 2.44 | 5.01 | 4.44 |
