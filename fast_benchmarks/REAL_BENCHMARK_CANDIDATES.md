# Real, complex benchmark candidates for FastBench

The 14 problems currently in FastBench are deliberately small cross-validation
unit tests. This note collects **genuinely complex, real-data / hardware-validated**
optimization-in-control problems that are still reproducible, and assesses how
each would integrate.

A recurring theme: the credible "real" benchmarks pair a **high-fidelity
external simulator as the plant** with a **control-oriented model in the OCP**.
FastBench already separates `plant_step` from the prediction model, so the main
new piece of machinery needed is an *external-simulator plant adapter* (a
`plant_step` that calls OpenFAST / a BOPTEST REST endpoint / the Tennessee
Eastman binary / a high-fidelity Modelica FMU). The CasADi/acados-native ones
(quadrotor, AWE, PMSM) are closer to drop-in.

## Shortlist (ranked by value/effort for this benchmark)

| # | problem | domain | complexity | real data | toolchain | FastBench fit |
|---|---------|--------|-----------|-----------|-----------|---------------|
| 1 | **Agile quadrotor + learned aero (uzh-rpg)** | aerial robotics | ~13 states, real-time NMPC + GP/NN residual | flight logs to 14 m/s, >4 g | acados + CasADi | **High** — acados-native; ship the real trajectory + GP/NN residual |
| 2 | **BOPTEST building HVAC** | building energy | Modelica emulator, slow + stochastic, economic+comfort KPIs | real Brussels building, real weather years | Docker + REST; any MPC | **High via external-plant adapter** — emulator is the plant, RC model in OCP |
| 3 | **AWEbox airborne wind energy** | renewable energy | 20–30+ states (6-DoF kite + tether), periodic "pumping" orbits, unstable | physically-grounded AWE models | **CasADi-native** (IPOPT; codegen MPC) | **High** — pull the model straight out of awebox |
| 4 | **PMSM / electric-drive NMPC** | power electronics | stiff, **sub-millisecond** sampling, dq dynamics + constraints | hardware-validated (dSPACE) params | acados | **Medium-High** — embedded showcase for FastSQP/acados timing |
| 5 | **Whole-body legged MPC (OCS2 / ANYmal)** | legged robotics | **~24–49 states, ~24–55 inputs**, contact switching, ROS | sim2real on real quadrupeds | OCS2 (C++) / Crocoddyl | **Medium** — biggest NMPC; needs contact handling + external dynamics |
| 6 | **OpenFAST + ROSCO (NREL 5 MW turbine)** | wind energy | aeroelastic, many flexible DOF, turbulent wind | NREL reference turbine, real wind fields | OpenFAST (Fortran) + Python | **Medium via external-plant adapter** — high-fidelity plant, reduced model in OCP |
| 7 | **Tennessee Eastman process** | process industry | **~50 states**, 41 meas / 12 inputs, 20+ fault modes | de-facto industrial benchmark | Simulink / C / NIST | **Medium via external-plant adapter** — plant-wide economic MPC + faults |
| — | **PGLib-OPF (AC-OPF)** | power systems | up to **9241-bus PEGASE / 6470-bus RTE**, nonconvex | real European/French grids | solver-agnostic | **Different track** — large *static* optimization, not closed-loop MPC |

## Detail and integration notes

**1. Agile quadrotor with learned aerodynamics (UZH RPG).** Real-time NMPC for a
quadrotor where a Gaussian-process / neural residual corrects the aerodynamic
model; demonstrated on physical flights up to 14 m/s and >4 g with ~70 % lower
tracking error than a nominal model. acados-based, public code and the real
reference trajectories. *Fit:* highest value — it is exactly the embedded-NMPC
+ learning regime FastBench targets, slots into the acados adapter, and the
"real data" is the flown trajectory plus the identified residual. Pairs
naturally with the existing `l4casadi`/Neural-MPC tooling.

**2. BOPTEST building HVAC.** Containerized Modelica emulators (e.g.
`bestest_hydronic_heat_pump`: a real Brussels dwelling with a 15 kW
air-to-water heat pump and floor heating) behind a REST API, driven by real
weather, scored on standardized energy/comfort KPIs, with an annual challenge.
*Fit:* the emulator becomes the `plant_step` (FastBench calls the REST endpoint
each step) while the OCP uses a low-order RC model — a textbook real-MPC setup,
and a strong test of economic/long-horizon MPC rather than fast NMPC.

**3. AWEbox (airborne wind energy).** Single- and multi-kite optimal control
with flexible tethers, solving periodic power-cycle orbits; highly nonlinear,
unstable, 20–30+ states, and it can code-generate the MPC solver. *Fit:* it is
already CasADi/IPOPT-native, so the model and constraints can be lifted into a
FastBench problem with comparatively little glue — the most complex *nonlinear*
OCP that is still nearly drop-in. LGPL-3.

**4. PMSM / electric-drive NMPC.** Continuous-control-set NMPC of a permanent-
magnet (or reluctance) synchronous machine: stiff dq dynamics, voltage/current
constraints, sampling in the tens-of-µs to sub-ms range, validated on dSPACE
hardware. *Fit:* the best *embedded-speed* real problem — it stresses
FastSQP/acados latency where it matters and has real, citable machine
parameters.

**5. Whole-body legged MPC (OCS2 / `legged_control`, ANYmal-class).** Nonlinear
whole-body or single-rigid-body MPC with contact scheduling; reported
formulations reach ~49 states / ~55 inputs, run on real quadrupeds (sim2real),
and are open-source (C++/ROS, OCS2 or Crocoddyl). *Fit:* by far the largest
NMPC and a serious test of an SQP solver, but contact-implicit dynamics and the
C++/ROS stack make it the heaviest lift; a reduced centroidal model is the
pragmatic entry point.

**6. OpenFAST + ROSCO (NREL 5 MW reference turbine).** High-fidelity aeroelastic
simulation (flexible blades/tower, turbulent inflow) of a real reference
turbine, with the open ROSCO controller as a baseline. *Fit:* OpenFAST is the
high-fidelity plant (external-simulator adapter); the OCP uses a reduced
2–4-state control model. Real wind fields and turbine data; the comparison of
interest is load/power trade-offs over realistic wind.

**7. Tennessee Eastman process.** The canonical plant-wide industrial benchmark:
~50 states, 41 measurements, 12 manipulated variables, and 20+ programmed fault
scenarios. *Fit:* economic/plant-wide MPC with the TE simulator as the plant
(several maintained ports, incl. NIST). Excellent for fault-tolerance and
constraint-handling metrics; the prediction model is a reduced/identified
subset.

**PGLib-OPF (separate track).** Not a closed-loop MPC problem but the standard
large-scale **AC optimal power flow** benchmark — real European (PEGASE, up to
9241 buses) and French (RTE, up to ~6500 buses) grids, nonconvex, CC-BY. Worth
adding as a distinct *static large-scale NLP* track to exercise IPOPT/solver
scaling on real data, clearly separated from the dynamic-MPC problems.

## Recommendation

Three give the most coverage for the least glue, and I'd add them first:

1. **Agile quadrotor + learned aero** — real flight data, acados-native, the
   embedded-NMPC + learning sweet spot.
2. **AWEbox** — the hardest *nonlinear* OCP that is still CasADi-native (nearly
   drop-in), real renewable-energy application.
3. **BOPTEST** — introduces the *external-simulator plant adapter*, unlocking
   real high-fidelity plants (and later OpenFAST and Tennessee Eastman reuse the
   same mechanism).

That sequence also delivers a reusable capability — the external-plant adapter —
that turns FastBench from "toy models" into a harness that can drive
industry-grade simulators while keeping the same metrics and reporting.

## References

- Torrente, Kaufmann, Föhn, Scaramuzza, *Data-Driven MPC for Quadrotors*, RA-L 2021 — https://arxiv.org/abs/2102.05773 · code: https://github.com/uzh-rpg/data_driven_mpc
- Kaufmann et al., *Computationally Efficient Data-Driven MPC for Agile Quadrotor Flight*, 2023 — https://arxiv.org/abs/2305.17254
- Blum, Arroyo, Huang et al., *BOPTEST*, J. Building Performance Simulation 2021 — https://doi.org/10.1080/19401493.2021.1986574 · code: https://github.com/ibpsa/project1-boptest
- De Schutter, Leuthold, Bronnenmeyer, Malz, Gros, Diehl, *AWEbox*, Energies 16(4), 2023 — https://www.mdpi.com/1996-1073/16/4/1900 · code: https://github.com/awebox/awebox
- Englert et al., *Continuous Control Set NMPC of Reluctance Synchronous Machines*, IEEE TCST 2021 — https://ieeexplore.ieee.org/document/9360312
- Sleiman et al., *A Unified MPC Framework for Whole-Body Dynamic Locomotion and Manipulation*, RA-L 2021; OCS2 — https://github.com/leggedrobotics/ocs2 · `legged_control` — https://github.com/qiayuanl/legged_control
- Grimminger/NREL, *OpenFAST* — https://github.com/OpenFAST/openfast · *ROSCO* — https://github.com/NREL/ROSCO
- Bathelt, Ricker, Jelali, *Revision of the Tennessee Eastman Process Model*, 2015 — https://doi.org/10.1016/j.ifacol.2015.08.199 · NIST port: https://github.com/usnistgov/tesim
- Babaeinejadsarookolaee et al., *The Power Grid Library (PGLib-OPF)*, 2019 — https://arxiv.org/abs/1908.02788 · code: https://github.com/power-grid-lib/pglib-opf
