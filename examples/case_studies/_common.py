"""What the case studies share: waiting for a quiet machine before a timed run.

The studies time on a workstation that other jobs share, not on the reference machine of
docs/results/fairness.md. `wait_for_quiet` holds each timed cell until the one-minute load average
drops below a threshold and returns the load it started at, which the cell records next to its
timings, so a number taken under load can be recognized and rerun.
"""

from __future__ import annotations

import os
import sys
import time


def wait_for_quiet(max_load: float | None = None, timeout: float = 3600.0, poll: float = 5.0) -> float:
  """Block until the one-minute load average is below ``max_load`` (or ``timeout`` passes); return it.
  The threshold defaults to ``CASE_STUDY_MAX_LOAD``, else 6: on a 16-core workstation whose own
  background (backups, indexing) holds the load near 4, that still leaves most cores idle."""
  max_load = float(os.environ.get("CASE_STUDY_MAX_LOAD", "6")) if max_load is None else max_load
  start = time.monotonic()
  announced = False
  while True:
    load = os.getloadavg()[0]
    if load < max_load or time.monotonic() - start > timeout:
      return load
    if not announced:
      print(f"waiting for the load average ({load:.1f}) to drop below {max_load}", file=sys.stderr, flush=True)
      announced = True
    time.sleep(poll)
