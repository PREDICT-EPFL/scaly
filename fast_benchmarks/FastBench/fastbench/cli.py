"""FastBench command-line interface.

    fastbench list
    fastbench run    [--problems ...] [--solvers ...] [--episodes N] [--seed S] [--out DIR]
    fastbench report [--results results/results.json] [--out DIR]
    fastbench doctor      # show which solvers are available on this machine
    fastbench clean  [--all] [--dry-run]   # remove regenerable build/run artifacts
"""
from __future__ import annotations

import argparse
import sys

from fastbench.core.registry import (get_solver, list_problems, list_solvers)


def _cmd_list(args):
    print("Problems:")
    for p in list_problems():
        print(f"  - {p}")
    print("Solvers:")
    for s in list_solvers():
        print(f"  - {s}")


def _cmd_doctor(args):
    print("Solver availability on this machine:")
    for s in list_solvers():
        try:
            ok = get_solver(s).available()
        except Exception as e:
            ok, e = False, e
        print(f"  {'OK ' if ok else '-- '} {s}")


def _cmd_run(args):
    from fastbench.core.runner import run_benchmark
    problems = args.problems or list_problems()
    # by default exclude internal reference-only solvers (used for suboptimality)
    solvers = args.solvers or [s for s in list_solvers() if not s.endswith("_ref")]
    run_benchmark(problems, solvers, n_episodes=args.episodes,
                  seed=args.seed, outdir=args.out, ref_costs=not args.no_ref)
    if not args.no_report:
        from fastbench.report import build_report
        build_report(f"{args.out}/results.json")


def _cmd_report(args):
    from fastbench.report import build_report
    build_report(args.results, args.out)


def _cmd_clean(args):
    from fastbench.core.clean import clean
    targets = clean(deep=args.all, dry_run=args.dry_run)
    verb = "would remove" if args.dry_run else "removed"
    for t in targets:
        print(f"  {verb}: {t}")
    scope = "build/run artifacts" + (" + venv/acados/examples" if args.all else "")
    print(f"{verb}: {len(targets)} item(s) ({scope})")
    if args.dry_run:
        print("(dry run - nothing deleted; re-run without --dry-run to apply)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fastbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=_cmd_list)
    sub.add_parser("doctor").set_defaults(func=_cmd_doctor)

    r = sub.add_parser("run")
    r.add_argument("--problems", nargs="*", default=None)
    r.add_argument("--solvers", nargs="*", default=None)
    r.add_argument("--episodes", type=int, default=5)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="results")
    r.add_argument("--no-ref", action="store_true", help="skip IPOPT reference cost")
    r.add_argument("--no-report", action="store_true")
    r.set_defaults(func=_cmd_run)

    rp = sub.add_parser("report")
    rp.add_argument("--results", default="results/results.json")
    rp.add_argument("--out", default=None)
    rp.set_defaults(func=_cmd_report)

    cl = sub.add_parser("clean", help="remove regenerable build/run artifacts")
    cl.add_argument("--all", action="store_true",
                    help="also remove .venv, third_party (acados) and examples")
    cl.add_argument("--dry-run", "-n", action="store_true",
                    help="list what would be removed without deleting")
    cl.set_defaults(func=_cmd_clean)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
