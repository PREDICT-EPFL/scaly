# scaly-piqp

The [PIQP](https://github.com/PREDICT-EPFL/piqp) quadratic programming solver backend for
[scaly](https://pypi.org/project/scaly/). The wheels bundle a prebuilt PIQP, so nothing else needs
to be installed.

Install it through the scaly extra, with [uv](https://docs.astral.sh/uv/) or pip:

```bash
uv add "scaly[piqp]"
```

and select it with `sc.solver(problem, "piqp")`. See the
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
[LICENSE.md](https://github.com/PREDICT-EPFL/scaly/blob/main/plugins/scaly-piqp/LICENSE.md).

The wheels bundle PIQP (BSD-2-Clause), Eigen (MPL-2.0), BLASFEO (BSD-2-Clause) and LDL from
SuiteSparse (LGPL-2.1-or-later). Their license texts ship in the wheel under `scaly_piqp/licenses/`.
