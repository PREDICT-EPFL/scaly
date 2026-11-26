# Checking the port of `extract_arrow_structure` against PIQP, 2026-11-26

`block_info.cpp` subclasses PIQP 0.7.0's `MultistageKKT` to print its `block_info`; `oracle.sh`
builds it (Eigen from the system, BLASFEO from CasADi's wheel, only so that the multistage sources
compile). `compare.py` runs it beside `examples/piqp_multistage` on five `mpc_qp` problems and on
PIQP's own test problems in `tests/data` of its repository at `v0.7.0`:

```
mpc_qp(5, 3, 2, 0, seed=0)   n =   28  blocks =   6  arrow =   0  same
mpc_qp(8, 4, 2, 2, seed=1)   n =   54  blocks =   9  arrow =   2  same
mpc_qp(20, 6, 3, 3, seed=2)  n =  189  blocks =  21  arrow =   4  same
mpc_qp(10, 2, 1, 0, seed=3)  n =   32  blocks =  11  arrow =   0  same
mpc_qp(40, 5, 2, 1, seed=4)  n =  286  blocks =  41  arrow =   1  same
chain_mass_sqp               n =  982  blocks =  41  arrow =   4  same
race_line                    skipped, n = 8816
robot_arm_sqp                n =  649  blocks =  31  arrow =   8  same
robot_arm_sqp_constr_perm    n =  649  blocks =  31  arrow =   8  same
robot_arm_sqp_no_global      n =  644  blocks =  31  arrow =   0  same
scenario_mpc                 n =  632  blocks =  43  arrow =  14  same
scenario_mpc_small           n =   33  blocks =   5  arrow =   5  same
small_dense                  n =   20  blocks =   2  arrow =   4  same
small_sparse_dual_inf        n =    2  blocks =   3  arrow =   0  same
```

A sweep over 216 `banded_qp` problems (n in 16, 24, 32; bandwidth 2, 4, 6; density 0.1 to 0.3;
0 or 2 shared variables; four seeds) agreed on every one. The stage-regular problems above,
PIQP's included, never reach two decisions: the ratio that closes a block
(`diag >= 2 * off_diag`) and the rounding up of a block's halves. Changing either leaves all of
them equal to PIQP's, so the test pins two `banded_qp` problems that each change breaks.

PIQP's merge loop steps past a block it has just merged, so a block is never merged twice in a
row. The port keeps that, and no problem tried here tells the two behaviours apart.
