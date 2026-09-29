# scaly-ipopt

The [IPOPT](https://github.com/coin-or/Ipopt) nonlinear programming solver backend for
[scaly](https://pypi.org/project/scaly/). The wheels bundle a prebuilt IPOPT with MUMPS and METIS,
so nothing else needs to be installed.

Install it through the scaly extra, with [uv](https://docs.astral.sh/uv/) or pip:

```bash
uv add "scaly[ipopt]"
```

and select it with `sc.solver(problem, "ipopt")`. See the
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
[LICENSE.md](https://github.com/PREDICT-EPFL/scaly/blob/main/plugins/scaly-ipopt/LICENSE.md).

The wheels bundle IPOPT (EPL-2.0), MUMPS (CeCILL-C), METIS and GKlib (Apache-2.0), the GCC Fortran
runtime (GPL-3.0-or-later with the GCC runtime library exception) and, on Linux, OpenBLAS
(BSD-3-Clause). Their license texts ship in the wheel under `scaly_ipopt/licenses/`.
