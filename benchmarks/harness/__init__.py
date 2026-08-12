"""Benchmark generation, correctness, and sweep helpers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
CLOSED_LOOP_RESULTS = RESULTS / "closed-loop"
SMOKE_RESULTS = RESULTS / "smoke"
SWEEP_RESULTS = RESULTS / "sweep"
