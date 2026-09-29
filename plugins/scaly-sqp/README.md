# scaly-sqp

A sequential quadratic programming (SQP) solver backend for nonlinear programs in
[scaly](https://pypi.org/project/scaly/). The solver is generated in C together with the problem and
solves its subproblems with [scaly-piqp](https://pypi.org/project/scaly-piqp/).

Install it through the scaly extra, with [uv](https://docs.astral.sh/uv/) or pip:

```bash
uv add "scaly[sqp]"
```

and select it with `sc.solver(problem, "sqp")`. See the
[documentation](https://github.com/PREDICT-EPFL/scaly/blob/main/docs/guide/solver_backends.md) for
how the solver backends differ and the [scaly repository](https://github.com/PREDICT-EPFL/scaly) for
everything else.

## Acknowledgements

Scaly is developed by Tudor A. Oancea (main developer) and Colin N. Jones (methods and math), at the
[Predictive control lab](https://www.epfl.ch/labs/la3/) from EPFL. This project is funded by the
[Swiss National Science Foundation](https://www.snf.ch) through the [NCCR
Automation](https://nccr-automation.ch) (grant agreement 51NF40_180545).

## AI usage disclosure

This project has been substantially developed using AI coding agents (Claude, Codex and others),
under the direction and review of the authors, who are responsible for the result.

## License

BSD-2-Clause. See
[LICENSE.md](https://github.com/PREDICT-EPFL/scaly/blob/main/plugins/scaly-sqp/LICENSE.md).
